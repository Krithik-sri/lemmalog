//! Score LLM-free extraction (gline-rs + NegEx + linking) on a gold set
//! with MCC. Runs twice: with the gold alias table as the linker (isolates
//! GLiNER NER/relations + NegEx), then, if a glinker-rs pipeline config is
//! given, with glinker-rs linking.
//!
//! ```text
//! export ORT_DYLIB_PATH=/path/to/libonnxruntime.so   # onnxruntime 1.20.x
//! cargo run --release --example extraction_eval --features gliner-load-dynamic,glinker -- \
//!     models/gliner-multitask-large-v0.5/tokenizer.json \
//!     models/gliner-multitask-large-v0.5/onnx/model.onnx \
//!     benchmarks/extraction_gold.tsv [glinker/pipeline.json]
//! ```
use lemmalog::extract::eval::{score, Confusion, Gold, Miss};
use lemmalog::extract::gline::GlineBackend;
use lemmalog::extract::link::GlinkerLinker;
use lemmalog::extract::{EntityLinker, GlinerExtractor, RejectReason};
use std::collections::BTreeMap;

fn report(name: &str, c: &Confusion, rejected: &[lemmalog::extract::Rejection]) {
    println!("\n== {name}");
    println!(
        "TP={} TN={} FP={} FN={}  precision={:.3} recall={:.3}  MCC={:.3}",
        c.tp, c.tn, c.fp, c.fn_, c.precision(), c.recall(), c.mcc()
    );
    for m in &c.misses {
        match m {
            Miss::Missed(e, t) => println!("  missed   {e}: {} --{}--> {}", t.0, t.1, t.2),
            Miss::Asserted(e, t) => println!("  asserted {e}: {} --{}--> {}", t.0, t.1, t.2),
            Miss::Spurious(e, t) => println!("  spurious {e}: {} --{}--> {}", t.0, t.1, t.2),
        }
    }
    let mut why: BTreeMap<String, usize> = BTreeMap::new();
    for r in rejected {
        let k = match &r.reason {
            RejectReason::Unlinked(_) => "Unlinked".to_string(),
            RejectReason::BadEntity(_) => "BadEntity".to_string(),
            other => format!("{other:?}"),
        };
        *why.entry(k).or_default() += 1;
    }
    println!("rejections: {why:?}");
}

fn run<L: EntityLinker>(name: &str, backend: &mut GlineBackend, gold: &Gold, linker: L) {
    let before = backend.truncated;
    let mut x = GlinerExtractor::new(&mut *backend, linker);
    let c = score(&mut x, gold);
    report(name, &c, x.rejected());
    drop(x);
    println!("entity spans dropped by max_entities: {}", backend.truncated - before);
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        return Err("usage: extraction_eval <tokenizer.json> <model.onnx> <gold.tsv> [glinker-pipeline.json]".into());
    }
    let gold = Gold::parse(&std::fs::read_to_string(&a[3]).map_err(|e| e.to_string())?)?;
    let mut backend = GlineBackend::new(&a[1], &a[2])?;
    run("gold alias table (GLiNER + NegEx only)", &mut backend, &gold, gold.table_linker());
    if let Some(cfg) = a.get(4) {
        run("glinker-rs linking (end to end)", &mut backend, &gold, GlinkerLinker::from_config_file(cfg)?);
    }
    Ok(())
}
