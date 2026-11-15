//! End-to-end check of the in-process glinker-rs linker on a real ONNX
//! session. Needs a glinker-rs pipeline dir (config + entities + model), e.g.
//! the synthetic one from glinker-rs `tools/make_synthetic.py`:
//!
//! ```text
//! python glinker-rs/tools/make_synthetic.py $DIR/models/linker
//! cp glinker-rs/tests/fixtures/{simple.json,entities.jsonl} $DIR
//! GLINKER_E2E_CONFIG=$DIR/simple.json ORT_DYLIB_PATH=... \
//!     cargo test --features glinker --test glinker_e2e_test
//! ```
//!
//! Without `GLINKER_E2E_CONFIG` the tests return early.
#![cfg(feature = "glinker")]

use lemmalog::agent::{Episode, Extractor};
use lemmalog::extract::{EntityLinker, FixtureBackend, GlinerExtractor, GlinkerLinker, Link, Mention, RejectReason};

fn linker() -> Option<GlinkerLinker> {
    let cfg = std::env::var("GLINKER_E2E_CONFIG").ok()?;
    Some(GlinkerLinker::from_config_file(&cfg).expect("load glinker-rs pipeline"))
}

fn m(s: &str, text: &str) -> Mention {
    let b = s.find(text).unwrap();
    Mention { text: text.into(), label: "thing".into(), start: b, end: b + text.len(), score: 1.0 }
}

#[test]
fn links_provided_spans_in_process() {
    let Some(mut lk) = linker() else { return eprintln!("GLINKER_E2E_CONFIG unset: skipped") };
    let s = "Insulin regulates glucose in patients with diabetes.";
    let links = lk.link(s, &[m(s, "Insulin"), m(s, "glucose"), m(s, "diabetes")]);
    let ids: Vec<_> = links
        .iter()
        .map(|l| match l {
            Link::Canonical { id, .. } => id.as_str(),
            Link::Unlinked => "-",
        })
        .collect();
    assert_eq!(ids, ["Q1", "Q4", "Q3"], "{links:?}");
    assert_eq!(lk.failures, 0);
}

#[test]
fn extractor_with_glinker_rs_and_negex() {
    let Some(lk) = linker() else { return eprintln!("GLINKER_E2E_CONFIG unset: skipped") };
    let backend = FixtureBackend::new().relation("Insulin", "thing", "likes", "glucose", "thing", 0.9);
    let mut x = GlinerExtractor::new(backend, lk);
    let facts = x.extract(&Episode { id: "a".into(), text: "Insulin regulates glucose.".into(), ts: 0, speaker: None });
    let t: Vec<_> = facts.iter().map(|f| (f.subj.as_str(), f.pred.as_str(), f.obj.as_str())).collect();
    assert_eq!(t, [("Q1", "likes", "Q4")]);
    let none = x.extract(&Episode { id: "b".into(), text: "Insulin no longer regulates glucose.".into(), ts: 1, speaker: None });
    assert!(none.is_empty());
    assert_eq!(x.rejected().last().unwrap().reason, RejectReason::Negated);
}
