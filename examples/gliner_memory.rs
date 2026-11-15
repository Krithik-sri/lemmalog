//! Agent memory with LLM-free extraction: gline-rs + NegEx + GLinker.
//!
//! ```text
//! export ORT_DYLIB_PATH=/path/to/libonnxruntime.so   # onnxruntime 1.20.x
//! cargo run --example gliner_memory --features gliner,glinker -- \
//!     models/gliner-multitask-large-v0.5/tokenizer.json \
//!     models/gliner-multitask-large-v0.5/onnx/model.onnx glinker/pipeline.json
//! ```
//!
//! `pipeline.json` is a glinker-rs pipeline config with `"l1": "none"`
//! (mentions come from gline-rs), an `entities` JSONL file and an `l3` model
//! dir; see the glinker-rs README.
use lemmalog::agent::{Episode, Extractor};
use lemmalog::extract::gline::GlineBackend;
use lemmalog::extract::link::GlinkerLinker;
use lemmalog::extract::GlinerExtractor;

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        return Err("usage: gliner_memory <tokenizer.json> <model.onnx> <glinker-pipeline.json>".into());
    }
    let backend = GlineBackend::new(&a[1], &a[2])?;
    let mut x = GlinerExtractor::new(backend, GlinkerLinker::from_config_file(&a[3])?);
    for (i, text) in [
        "Bob Smith works at Acme Corp.",
        "Bob doesn't work at Initech anymore.",
        "I work at Acme.",
    ]
    .iter()
    .enumerate()
    {
        let ep = Episode { id: format!("e{i}"), text: text.to_string(), ts: i as i64, speaker: Some("Dana".into()) };
        for f in x.extract(&ep) {
            println!("{} --{}--> {}  ({:.2})", f.subj, f.pred, f.obj, f.confidence);
        }
    }
    for r in x.rejected() {
        println!("rejected {:?}: {} --{}--> {}", r.reason, r.subject, r.relation, r.object);
    }
    Ok(())
}
