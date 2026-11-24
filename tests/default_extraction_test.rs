//! LLM-free extraction as the default: the review fixes to
//! `GlinerExtractor` (SPEC-7.1-O3/O4/O5 amendments), `NameLinker`,
//! `LazyExtractor`, `GlinerConfig` and `AgentMemory::observe_text`.
//! No model is loaded: the backend is a fixture.

use lemmalog::agent::{AgentMemory, CandidateFact, Episode, Extractor};
use lemmalog::extract::{
    EntityLinker, FixtureBackend, GlinerConfig, GlinerExtractor, LazyExtractor, Link, Mention, NameLinker, NegEx,
    RawRelation, RejectReason, RelationBackend, SentenceAnalysis, TableLinker,
};
use std::cell::Cell;
use std::rc::Rc;

fn ep(id: &str, text: &str) -> Episode {
    Episode { id: id.into(), text: text.into(), ts: 0, speaker: None }
}

fn linker() -> TableLinker {
    TableLinker::new()
        .entity("person:bob", Some("person"), &["Bob", "Robert"])
        .entity("org:acme", Some("organization"), &["Acme"])
        .entity("org:initech", Some("organization"), &["Initech"])
}

fn triples(f: &[CandidateFact]) -> Vec<(&str, &str, &str)> {
    f.iter().map(|c| (c.subj.as_str(), c.pred.as_str(), c.obj.as_str())).collect()
}

fn m(text: &str, label: &str, start: usize, end: usize) -> Mention {
    Mention { text: text.into(), label: label.into(), start, end, score: 0.9 }
}

/// Replays one fixed analysis for every sentence.
struct Canned(SentenceAnalysis);

impl RelationBackend for Canned {
    fn analyse(&mut self, sentences: &[String]) -> Result<Vec<SentenceAnalysis>, String> {
        Ok(sentences.iter().map(|_| self.0.clone()).collect())
    }
}

// ------------------------------------------------------------ review fixes

// obligation: SPEC-7.1-O3
#[test]
fn nan_relation_score_is_rejected() {
    // `NaN < min_score` is false, and `f32::min` drops a NaN argument: the
    // old check let this through with the link score (1.0) as confidence
    let backend = FixtureBackend::new().relation("Bob", "person", "works at", "Acme", "organization", f32::NAN);
    let mut x = GlinerExtractor::new(backend, linker());
    assert!(x.extract(&ep("e", "Bob works at Acme.")).is_empty());
    assert_eq!(x.rejected()[0].reason, RejectReason::LowScore);
}

// obligation: SPEC-7.1-O3
#[test]
fn infinite_relation_score_is_rejected() {
    let backend = FixtureBackend::new().relation("Bob", "person", "works at", "Acme", "organization", f32::INFINITY);
    let mut x = GlinerExtractor::new(backend, linker());
    assert!(x.extract(&ep("e", "Bob works at Acme.")).is_empty());
    assert_eq!(x.rejected()[0].reason, RejectReason::LowScore);
}

struct NanLinker;

impl EntityLinker for NanLinker {
    fn link(&mut self, _s: &str, ms: &[Mention]) -> Vec<Link> {
        ms.iter().map(|m| Link::Canonical { id: m.text.clone(), score: f32::NAN }).collect()
    }
}

// obligation: SPEC-7.1-O3
#[test]
fn nan_link_score_is_rejected() {
    let backend = FixtureBackend::new().relation("Bob", "person", "works at", "Acme", "organization", 0.9);
    let mut x = GlinerExtractor::new(backend, NanLinker);
    assert!(x.extract(&ep("e", "Bob works at Acme.")).is_empty());
    assert_eq!(x.rejected()[0].reason, RejectReason::LowScore);
}

/// Fails the first `fail` calls, then works.
struct Flaky {
    fail: usize,
    calls: usize,
}

impl RelationBackend for Flaky {
    fn analyse(&mut self, sentences: &[String]) -> Result<Vec<SentenceAnalysis>, String> {
        self.calls += 1;
        if self.calls <= self.fail {
            return Err("onnx runtime: transient failure".into());
        }
        FixtureBackend::new().relation("Bob", "person", "works at", "Acme", "organization", 0.9).analyse(sentences)
    }
}

// obligation: SPEC-7.1-O3
#[test]
fn failed_episode_is_retried_not_cached_empty() {
    let mut x = GlinerExtractor::new(Flaky { fail: 1, calls: 0 }, linker());
    let e = ep("e", "Bob works at Acme.");
    assert!(x.extract(&e).is_empty(), "first call fails");
    assert_eq!(x.errors().len(), 1);
    assert_eq!(triples(&x.extract(&e)), vec![("person:bob", "works_at", "org:acme")], "second call retries");
    assert_eq!(x.stats(), (2, 1));
    // a success is memoized: the backend is not called again
    x.extract(&e);
    assert_eq!(x.backend().calls, 2);
    assert_eq!(x.stats(), (2, 1));
}

// obligation: SPEC-7.1-O5
#[test]
fn mention_text_is_taken_from_its_span() {
    // the model's text says Initech, the span covers Acme: NegEx reads the
    // span, so the linker must read the same words
    let s = "Bob works at Acme.";
    let r = RawRelation {
        subject: m("Bob", "person", 0, 3),
        label: "works at".into(),
        object: m("Initech", "organization", 13, 17),
        score: 0.9,
    };
    let mut x = GlinerExtractor::new(Canned(SentenceAnalysis { relations: vec![r], ..Default::default() }), linker());
    assert_eq!(triples(&x.extract(&ep("e", s))), vec![("person:bob", "works_at", "org:acme")]);
}

// obligation: SPEC-7.1-O5
#[test]
fn empty_span_is_a_bad_span() {
    let r = RawRelation {
        subject: m("Bob", "person", 0, 3),
        label: "works at".into(),
        object: m("Acme", "organization", 13, 13),
        score: 0.9,
    };
    let mut x = GlinerExtractor::new(Canned(SentenceAnalysis { relations: vec![r], ..Default::default() }), linker());
    assert!(x.extract(&ep("e", "Bob works at Acme.")).is_empty());
    assert_eq!(x.rejected()[0].reason, RejectReason::BadSpan);
}

// obligation: SPEC-7.1-O4
#[test]
fn subject_and_object_linking_to_one_entity_is_a_self_loop() {
    let backend = FixtureBackend::new().relation("Bob", "person", "reports to", "Robert", "person", 0.9);
    let mut x = GlinerExtractor::new(backend, linker());
    assert!(x.extract(&ep("e", "Bob reports to Robert.")).is_empty());
    assert_eq!(x.rejected()[0].reason, RejectReason::SelfLoop);
}

// --------------------------------------------------------------- NameLinker

#[test]
fn name_linker_normalises_names() {
    assert_eq!(NameLinker::canonical("Acme").as_deref(), Some("Acme"));
    assert_eq!(NameLinker::canonical("  Bob   Smith ").as_deref(), Some("Bob Smith"));
    assert_eq!(NameLinker::canonical("Acme's").as_deref(), Some("Acme"));
    assert_eq!(NameLinker::canonical("Acme Corp.").as_deref(), Some("Acme Corp"));
    assert_eq!(NameLinker::canonical("(Initech),").as_deref(), Some("Initech"));
    assert_eq!(NameLinker::canonical("..."), None);
    assert_eq!(NameLinker::canonical(""), None);
}

#[test]
fn name_linker_ids_match_protocol_names() {
    let backend = FixtureBackend::new().relation("Bob", "person", "works at", "Acme", "organization", 0.8);
    let mut x = GlinerExtractor::new(backend, NameLinker);
    let f = x.extract(&ep("e", "Bob works at Acme."));
    assert_eq!(triples(&f), vec![("Bob", "works_at", "Acme")]);
    assert!((f[0].confidence - 0.8).abs() < 1e-6, "confidence is the relation score");
}

// ------------------------------------------------------------ LazyExtractor

fn fixture_extractor() -> Box<dyn Extractor> {
    let backend = FixtureBackend::new()
        .relation("I", "person", "works at", "Initech", "organization", 0.9)
        .relation("Bob", "person", "works at", "Globex", "organization", 0.9);
    Box::new(GlinerExtractor::new(backend, NameLinker))
}

#[test]
fn lazy_extractor_builds_once_on_first_use() {
    let built = Rc::new(Cell::new(0));
    let b = built.clone();
    let mut x = LazyExtractor::new(
        "gliner: fixture",
        Box::new(move || {
            b.set(b.get() + 1);
            Ok(fixture_extractor())
        }),
    );
    assert!(x.enabled());
    assert_eq!(built.get(), 0, "nothing is loaded before the first episode");
    assert_eq!(x.extract(&ep("e1", "Bob works at Globex.")).len(), 1);
    x.extract(&ep("e2", "Bob works at Globex."));
    assert_eq!(built.get(), 1);
    assert_eq!(x.stats(), (2, 0));
}

#[test]
fn lazy_extractor_off_extracts_nothing_and_says_why() {
    let mut x = LazyExtractor::off("LEMMALOG_EXTRACTOR=off");
    assert!(!x.enabled());
    assert!(x.extract(&ep("e", "Bob works at Globex.")).is_empty());
    assert_eq!(x.status(), "off: LEMMALOG_EXTRACTOR=off");
}

#[test]
fn lazy_extractor_reports_a_failed_load() {
    let mut x = LazyExtractor::new("gliner: missing", Box::new(|| Err("model.onnx: no such file".to_string())));
    assert!(x.extract(&ep("e", "Bob works at Globex.")).is_empty());
    assert_eq!(x.error(), Some("model.onnx: no such file"));
    assert_eq!(x.stats(), (0, 1));
    assert!(x.status().starts_with("failed:"));
    assert!(!x.enabled(), "a failed load is not retried on every turn");
}

// ------------------------------------------------------------- GlinerConfig

fn vars<'a>(kv: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
    move |k| kv.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string())
}

#[test]
fn config_defaults_to_gliner_under_home() {
    let c = GlinerConfig::from_vars(vars(&[("HOME", "/h")])).unwrap();
    assert_eq!(c.tokenizer, "/h/.lemmalog/models/gliner-multitask-large-v0.5/tokenizer.json");
    assert_eq!(c.model, "/h/.lemmalog/models/gliner-multitask-large-v0.5/onnx/model.onnx");
    assert_eq!(c.glinker_pipeline, None);
    assert_eq!(c.missing().len(), 2);
}

#[test]
fn config_honours_overrides_and_off() {
    let c = GlinerConfig::from_vars(vars(&[("LEMMALOG_GLINER_DIR", "/m"), ("LEMMALOG_GLINKER_PIPELINE", "/p.json")]))
        .unwrap();
    assert_eq!((c.model.as_str(), c.glinker_pipeline.as_deref()), ("/m/onnx/model.onnx", Some("/p.json")));
    assert!(GlinerConfig::from_vars(vars(&[("LEMMALOG_EXTRACTOR", "off")])).is_err());
    assert!(GlinerConfig::from_vars(vars(&[("LEMMALOG_EXTRACTOR", "GLiNER")])).is_ok());
    assert!(GlinerConfig::from_vars(vars(&[("LEMMALOG_EXTRACTOR", "llama")])).is_err());
}

// ---------------------------------------------------- AgentMemory::observe_text

fn current(m: &AgentMemory<LazyExtractor>) -> Vec<(String, String, String)> {
    let mut v: Vec<_> = m
        .engine
        .relation_keys("current")
        .iter()
        .filter(|k| k.len() == 3)
        .map(|k| {
            let d = |i: usize| m.engine.interner.display(&k[i]).to_string();
            (d(0), d(1), d(2))
        })
        .collect();
    v.sort();
    v
}

#[test]
fn observe_text_takes_protocol_lines_and_free_text() {
    let mut mem = AgentMemory::new(LazyExtractor::ready("gliner: fixture", fixture_extractor()), "").unwrap();
    let r = mem.observe_text(
        "Alice --works_at--> Acme\nI work at Initech.\nBob doesn't work at Globex.",
        10,
        Some("Dana"),
    );
    assert_eq!((r.prose_lines, r.prose_facts, r.report.added), (2, 1, 2));
    assert!(r.dropped.is_empty());
    mem.maintain(10);
    let t = |a: &str, b: &str, c: &str| (a.to_string(), b.to_string(), c.to_string());
    // "I" is the speaker; the negated sentence asserts nothing
    assert_eq!(current(&mem), vec![t("Alice", "works_at", "Acme"), t("Dana", "works_at", "Initech")]);
}

#[test]
fn observe_text_with_extraction_off_still_takes_protocol_lines() {
    let mut mem = AgentMemory::new(LazyExtractor::off("test"), "").unwrap();
    let r = mem.observe_text("Alice --works_at--> Acme\nI work at Initech.", 10, Some("Dana"));
    assert_eq!((r.prose_lines, r.prose_facts, r.report.added), (1, 0, 1));
}

// ------------------------------------------------------ NegEx: known gaps

/// Sentences NegEx gets wrong today (from the extraction review). Tracked
/// here so a fix can be checked off; run with `--ignored`.
#[test]
#[ignore = "known NegEx gaps; see the extraction review"]
fn negex_known_gaps() {
    // (sentence, subject, object, other mentions, really negated?)
    let cases: &[(&str, &str, &str, &[&str], bool)] = &[
        ("Neither Bob nor Dana works at Acme.", "Bob", "Acme", &["Dana"], true),
        ("Neither Bob nor Dana works at Acme.", "Dana", "Acme", &["Bob"], true),
        ("Not Bob but Dana works at Acme.", "Bob", "Acme", &["Dana"], true),
        ("Bob doesn't, however, work at Acme.", "Bob", "Acme", &[], true),
        ("Bob does not now work at Acme.", "Bob", "Acme", &[], true),
        ("Bob dislikes sushi.", "Bob", "sushi", &[], true),
        ("Bob hates Acme.", "Bob", "Acme", &[], true),
        ("Bob works at Acme, which is not true of Dana.", "Bob", "Acme", &["Dana"], false),
        ("Bob left his laptop at Acme.", "Bob", "Acme", &[], false),
        ("Bob has no doubts and works at Acme.", "Bob", "Acme", &[], false),
    ];
    let n = NegEx::default();
    let wrong: Vec<&str> = cases
        .iter()
        .filter(|(s, subj, obj, others, gold)| {
            let o = s.find(obj).unwrap();
            let others: Vec<(usize, usize)> = others.iter().map(|x| { let i = s.find(x).unwrap(); (i, i + x.len()) }).collect();
            n.negated_with(s, (o, o + obj.len()), s.find(subj).unwrap(), &others) != *gold
        })
        .map(|c| c.0)
        .collect();
    assert!(wrong.is_empty(), "NegEx still wrong on: {wrong:#?}");
}
