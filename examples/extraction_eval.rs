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
//!     benchmarks/extraction_gold.tsv [glinker/pipeline.json [entities.jsonl]] [--sweep 0.1,0.3,0.5]
//! ```
//! Set the pipeline's `l3.threshold` at or below the lowest sweep value:
//! glinker-rs drops predictions under it before lemmalog sees them.
use lemmalog::extract::eval::{score, Confusion, Gold, Miss};
use lemmalog::extract::gline::GlineBackend;
use lemmalog::extract::link::GlinkerLinker;
use lemmalog::extract::{CachedBackend, CascadeLinker, EntityLinker, GlinerExtractor, Rejection, RejectReason, TableLinker};
use std::collections::BTreeMap;

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

fn run<L: EntityLinker>(name: &str, backend: &mut CachedBackend<GlineBackend>, gold: &Gold, linker: L, detail: bool) -> f64 {
    let mut x = GlinerExtractor::new(&mut *backend, linker);
    let c = score(&mut x, gold);
    report(name, &c, x.rejected(), detail);
    c.mcc()
}

fn main() -> Result<(), String> {
    let mut a: Vec<String> = std::env::args().collect();
    let mut sweep = vec![0.5f32];
    if let Some(i) = a.iter().position(|x| x == "--sweep") {
        let list = a.get(i + 1).ok_or("--sweep needs a list, e.g. 0.1,0.3,0.5")?.clone();
        sweep = list.split(',').map(|v| v.trim().parse::<f32>().map_err(|e| format!("--sweep {v}: {e}"))).collect::<Result<_, _>>()?;
        a.drain(i..=i + 1);
    }
    if a.len() < 4 {
        return Err("usage: extraction_eval <tokenizer.json> <model.onnx> <gold.tsv> [pipeline.json [entities.jsonl]] [--sweep t1,t2,..]".into());
    }
    let gold = Gold::parse(&std::fs::read_to_string(&a[3]).map_err(|e| e.to_string())?)?;
    let mut backend = CachedBackend::new(GlineBackend::new(&a[1], &a[2])?);
    run("gold alias table (GLiNER + NegEx only)", &mut backend, &gold, gold.table_linker(), true);
    let mut table: Vec<(String, f64)> = Vec::new();
    if let Some(cfg) = a.get(4) {
        let mut glinker = GlinkerLinker::from_config_file(cfg)?;
        let exact = match a.get(5) {
            Some(p) => Some(TableLinker::from_glinker_entities(p)?),
            None => None,
        };
        let last = sweep.len() - 1;
        for (i, &t) in sweep.iter().enumerate() {
            glinker.min_confidence = t;
            let name = format!("glinker-rs @ {t:.2}");
            let m = run(&name, &mut backend, &gold, &mut glinker, i == last);
            table.push((name, m));
            if let Some(ex) = &exact {
                let name = format!("exact alias -> glinker-rs @ {t:.2}");
                let m = run(&name, &mut backend, &gold, CascadeLinker::new(ex.clone(), &mut glinker), i == last);
                table.push((name, m));
            }
        }
        if glinker.failures > 0 {
            println!("\nglinker-rs failures: {}", glinker.failures);
        }
    }
    if table.len() > 1 {
        println!("\nthreshold sweep (MCC):");
        for (n, m) in &table {
            println!("  {n:<36} {m:.3}");
        }
    }
    println!("\nGLiNER sentences analysed: {}; entity spans dropped by max_entities: {}", backend.misses, backend.inner.truncated);
    Ok(())
}
