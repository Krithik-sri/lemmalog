//! Agent memory with LLM-free extraction: gline-rs + NegEx + GLinker.
//!
//! ```text
//! python scripts/glinker_sidecar.py --entities data/entities.jsonl &
//! cargo run --example gliner_memory --features gliner,glinker -- \
//!     models/gliner-multitask-large-v0.5/tokenizer.json \
//!     models/gliner-multitask-large-v0.5/onnx/model.onnx http://127.0.0.1:8765
//! ```
use lemmalog::agent::{Episode, Extractor};
use lemmalog::extract::gline::GlineBackend;
use lemmalog::extract::link::GlinkerHttpLinker;
use lemmalog::extract::GlinerExtractor;

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        return Err("usage: gliner_memory <tokenizer.json> <model.onnx> <glinker-url>".into());
    }
    let backend = GlineBackend::new(&a[1], &a[2])?;
    let mut x = GlinerExtractor::new(backend, GlinkerHttpLinker::new(&a[3]));
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
