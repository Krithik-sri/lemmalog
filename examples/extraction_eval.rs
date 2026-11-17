//! Score LLM-free extraction (gline-rs + NegEx + linking) on a gold set
//! with MCC. GLiNER runs once per sentence (cached); the linkers are then
//! compared on the same model output:
//!
//! 1. the gold alias table (perfect linking: GLiNER + NegEx alone),
//! 2. glinker-rs alone, at each `--sweep` confidence threshold,
//! 3. exact unambiguous aliases from the glinker-rs entities file first,
//!    glinker-rs for the rest, at each threshold.
//!
//! Picking a threshold on the same gold set it is scored on overfits; use
//! the sweep to see the trade-off, not to report a final number.
//!
//! ```text
//! export ORT_DYLIB_PATH=/path/to/libonnxruntime.so   # onnxruntime 1.20.x
//! cargo run --release --example extraction_eval --features gliner-load-dynamic,glinker -- \
//!     models/gliner-multitask-large-v0.5/tokenizer.json \
//!     models/gliner-multitask-large-v0.5/onnx/model.onnx \
//!     benchmarks/extraction_gold.tsv [glinker/pipeline.json [entities.jsonl]] \
//!     [--pipeline NAME=pipeline.json]... [--entities entities.jsonl] [--sweep 0.1,0.3,0.5] [--spans]
//! ```
//! `--pipeline` (repeatable) scores several glinker-rs configs on the same
//! GLiNER output, e.g. entity databases with type-only and with rich
//! descriptions. `--entities` is the exact-alias table for the cascade runs.
//! `--spans` prints GLiNER's raw mentions and relations for every episode
//! that any detailed run got wrong.
//! Set the pipeline's `l3.threshold` at or below the lowest sweep value:
//! glinker-rs drops predictions under it before lemmalog sees them.
use lemmalog::extract::eval::{score, Confusion, Gold, Miss};
use lemmalog::extract::gline::GlineBackend;
use lemmalog::extract::link::GlinkerLinker;
use lemmalog::extract::{
    split_sentences, CachedBackend, CascadeLinker, EntityLinker, GlinerExtractor, Rejection, RejectReason, TableLinker,
};
use std::collections::{BTreeMap, BTreeSet};

fn reason(r: &RejectReason) -> String {
    match r {
        RejectReason::Unlinked(m) => format!("Unlinked({m})"),
        RejectReason::BadEntity(_) => "BadEntity".into(),
        other => format!("{other:?}"),
    }
}

fn report(name: &str, c: &Confusion, rejected: &[Rejection], detail: bool) {
    println!("\n== {name}");
    println!(
        "TP={} TN={} FP={} FN={}  precision={:.3} recall={:.3}  MCC={:.3}",
        c.tp, c.tn, c.fp, c.fn_, c.precision(), c.recall(), c.mcc()
    );
    if !detail {
        return;
    }
    for m in &c.misses {
        match m {
            Miss::Missed(e, t) => println!("  missed   {e}: {} --{}--> {}", t.0, t.1, t.2),
            Miss::Asserted(e, t) => println!("  asserted {e}: {} --{}--> {}", t.0, t.1, t.2),
            Miss::Spurious(e, t) => println!("  spurious {e}: {} --{}--> {}", t.0, t.1, t.2),
        }
    }
    let mut why: BTreeMap<String, usize> = BTreeMap::new();
    for r in rejected {
        let k = reason(&r.reason);
        *why.entry(k.split('(').next().unwrap().to_string()).or_default() += 1;
        println!("  rejected {}: {} --{}--> {}  [{k}]", r.episode, r.subject, r.relation, r.object);
    }
    println!("rejections: {why:?}");
}

fn run<L: EntityLinker>(name: &str, backend: &mut CachedBackend<GlineBackend>, gold: &Gold, linker: L, detail: bool) -> Confusion {
    let mut x = GlinerExtractor::new(&mut *backend, linker);
    let c = score(&mut x, gold);
    report(name, &c, x.rejected(), detail);
    c
}

/// Take every `flag VALUE` pair out of `args`.
fn take_all(args: &mut Vec<String>, flag: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    while let Some(i) = args.iter().position(|x| x == flag) {
        let v = args.get(i + 1).ok_or(format!("{flag} needs a value"))?.clone();
        args.drain(i..=i + 1);
        out.push(v);
    }
    Ok(out)
}

fn print_spans(backend: &CachedBackend<GlineBackend>, gold: &Gold, episodes: &BTreeSet<String>) {
    println!("\n== GLiNER raw output for episodes with errors");
    for g in gold.episodes.iter().filter(|g| episodes.contains(&g.episode.id)) {
        println!("{}: {}", g.episode.id, g.episode.text);
        for s in split_sentences(&g.episode.text) {
            let Some(a) = backend.analysis(&s) else { continue };
            let ms: Vec<String> = a.mentions.iter().map(|m| format!("{}<{}>{:.2}", m.text, m.label, m.score)).collect();
            println!("  mentions:  {}", if ms.is_empty() { "(none)".to_string() } else { ms.join(", ") });
            if a.relations.is_empty() && a.unresolved.is_empty() {
                println!("  relations: (none)");
            }
            for r in &a.relations {
                println!("  relation:  {} --{}--> {}  {:.2}", r.subject.text, r.label, r.object.text, r.score);
            }
            for (sj, l, o) in &a.unresolved {
                println!("  unresolved: {sj} --{l}--> {o}");
            }
        }
    }
}

fn main() -> Result<(), String> {
    let mut a: Vec<String> = std::env::args().collect();
    let spans = a.iter().any(|x| x == "--spans");
    a.retain(|x| x != "--spans");
    let sweep: Vec<f32> = match take_all(&mut a, "--sweep")?.pop() {
        Some(list) => list.split(',').map(|v| v.trim().parse::<f32>().map_err(|e| format!("--sweep {v}: {e}"))).collect::<Result<_, _>>()?,
        None => vec![0.5],
    };
    let mut pipelines: Vec<(String, String)> = Vec::new();
    for p in take_all(&mut a, "--pipeline")? {
        let (n, path) = p.split_once('=').ok_or(format!("--pipeline {p}: expected NAME=PATH"))?;
        pipelines.push((n.to_string(), path.to_string()));
    }
    let mut entities = take_all(&mut a, "--entities")?.pop();
    if a.len() < 4 {
        return Err("usage: extraction_eval <tokenizer.json> <model.onnx> <gold.tsv> [pipeline.json [entities.jsonl]] [--pipeline NAME=PATH]... [--entities PATH] [--sweep t1,t2,..] [--spans]".into());
    }
    if let Some(p) = a.get(4) {
        pipelines.insert(0, ("glinker-rs".into(), p.clone()));
    }
    if let Some(e) = a.get(5) {
        entities.get_or_insert(e.clone());
    }
    let gold = Gold::parse(&std::fs::read_to_string(&a[3]).map_err(|e| e.to_string())?)?;
    let mut backend = CachedBackend::new(GlineBackend::new(&a[1], &a[2])?);
    let mut wrong: BTreeSet<String> = BTreeSet::new();
    let note = |c: &Confusion, wrong: &mut BTreeSet<String>| {
        for m in &c.misses {
            let (Miss::Missed(e, _) | Miss::Asserted(e, _) | Miss::Spurious(e, _)) = m;
            wrong.insert(e.clone());
        }
    };
    let c = run("gold alias table (GLiNER + NegEx only)", &mut backend, &gold, gold.table_linker(), true);
    note(&c, &mut wrong);
    let exact = match &entities {
        Some(p) => Some(TableLinker::from_glinker_entities(p)?),
        None => None,
    };
    let mut table: Vec<(String, f64)> = Vec::new();
    for (pname, cfg) in &pipelines {
        let mut glinker = GlinkerLinker::from_config_file(cfg)?;
        let last = sweep.len() - 1;
        for (i, &t) in sweep.iter().enumerate() {
            glinker.min_confidence = t;
            let name = format!("{pname} @ {t:.2}");
            let c = run(&name, &mut backend, &gold, &mut glinker, i == last);
            if i == last {
                note(&c, &mut wrong);
            }
            table.push((name, c.mcc()));
            if let Some(ex) = &exact {
                let name = format!("exact alias -> {pname} @ {t:.2}");
                let c = run(&name, &mut backend, &gold, CascadeLinker::new(ex.clone(), &mut glinker), i == last);
                table.push((name, c.mcc()));
            }
        }
        if glinker.failures > 0 {
            println!("\n{pname} failures: {}", glinker.failures);
        }
    }
    if table.len() > 1 {
        println!("\nthreshold sweep (MCC):");
        for (n, m) in &table {
            println!("  {n:<44} {m:.3}");
        }
    }
    if spans {
        print_spans(&backend, &gold, &wrong);
    }
    println!("\nGLiNER sentences analysed: {}; entity spans dropped by max_entities: {}", backend.misses, backend.inner.truncated);
    Ok(())
}
