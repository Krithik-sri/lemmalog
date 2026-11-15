//! Checks for spec/extraction.md (SPEC-7.1). Each test is tagged with the
//! obligation it checks; obligations.lock pins these tests.

use lemmalog::agent::{CandidateFact, Episode, Extractor};
use lemmalog::extract::{
    EntityLinker, FixtureBackend, GlinerExtractor, Link, Mention, RejectReason, RelationBackend,
    SentenceAnalysis, TableLinker,
};

fn ep(id: &str, text: &str) -> Episode {
    Episode { id: id.into(), text: text.into(), ts: 0, speaker: Some("Dana".into()) }
}

fn linker() -> TableLinker {
    TableLinker::new()
        .entity("person:bob", Some("person"), &["Bob", "Bob Smith"])
        .entity("person:dana", Some("person"), &["Dana"])
        .entity("org:acme", Some("organization"), &["Acme", "Acme Corp"])
        .entity("org:initech", Some("organization"), &["Initech"])
}

fn works(subj: &str, obj: &str) -> FixtureBackend {
    FixtureBackend::new().relation(subj, "person", "works at", obj, "organization", 0.9)
}

fn triples(f: &[CandidateFact]) -> Vec<(String, String, String)> {
    f.iter().map(|c| (c.subj.clone(), c.pred.clone(), c.obj.clone())).collect()
}

fn t(s: &str, p: &str, o: &str) -> (String, String, String) {
    (s.into(), p.into(), o.into())
}

// obligation: SPEC-7.1-O1
#[test]
fn negated_relations_produce_no_fact() {
    for text in [
        "Bob doesn't work at Acme.",
        "Bob no longer works at Acme.",
        "Bob never worked at Acme.",
        "Bob quit Acme.",
        "Bob works at Acme no more.",
        "Bob is not at Acme anymore.",
    ] {
        let mut x = GlinerExtractor::new(works("Bob", "Acme"), linker());
        let facts = x.extract(&ep("e", text));
        assert!(facts.is_empty(), "{text:?} gave {facts:?}");
        assert_eq!(x.rejected().len(), 1, "{text:?}");
        assert_eq!(x.rejected()[0].reason, RejectReason::Negated, "{text:?}");
    }
}

// obligation: SPEC-7.1-O1
#[test]
fn pseudo_triggers_and_terminated_scopes_do_not_negate() {
    for text in [
        "Bob works at Acme.",
        "Bob not only works at Acme, he runs the lab.",
        "Bob left Initech but works at Acme.",
        "Bob doesn't like tea; Bob works at Acme.",
        "Not much changed. Bob works at Acme.",
    ] {
        let mut x = GlinerExtractor::new(works("Bob", "Acme"), linker());
        let facts = x.extract(&ep("e", text));
        assert_eq!(triples(&facts), vec![t("person:bob", "works_at", "org:acme")], "{text:?}");
    }
}

// obligation: SPEC-7.1-O1
#[test]
fn negation_is_per_relation_within_a_sentence() {
    // one sentence, two relations: only the negated one is dropped
    let backend = works("Bob", "Initech").relation("Bob", "person", "works at", "Acme", "organization", 0.9);
    let mut x = GlinerExtractor::new(backend, linker());
    let facts = x.extract(&ep("e", "Bob doesn't work at Initech anymore, but Bob works at Acme."));
    assert_eq!(triples(&facts), vec![t("person:bob", "works_at", "org:acme")]);
    let negated: Vec<_> =
        x.rejected().iter().filter(|r| r.reason == RejectReason::Negated).map(|r| r.object.clone()).collect();
    assert_eq!(negated, vec!["Initech".to_string()]);
}

// obligation: SPEC-7.1-O2
#[test]
fn facts_use_canonical_ids_from_the_linker() {
    let mut x = GlinerExtractor::new(works("Bob Smith", "Acme Corp"), linker());
    let facts = x.extract(&ep("e", "Bob Smith works at Acme Corp."));
    assert_eq!(triples(&facts), vec![t("person:bob", "works_at", "org:acme")]);
    for f in &facts {
        assert!(f.subj != "Bob Smith" && f.obj != "Acme Corp", "raw mention leaked: {f:?}");
    }
}

// obligation: SPEC-7.1-O2
#[test]
fn unlinked_mentions_produce_no_fact() {
    // Globex is not in the table; a label mismatch also fails to link
    let backend = works("Bob", "Globex")
        .relation("Acme", "person", "works at", "Initech", "organization", 0.9);
    let mut x = GlinerExtractor::new(backend, linker());
    let facts = x.extract(&ep("e", "Bob works at Globex. Acme works at Initech."));
    assert!(facts.is_empty(), "{facts:?}");
    let reasons: Vec<_> = x.rejected().iter().map(|r| r.reason.clone()).collect();
    assert_eq!(
        reasons,
        vec![RejectReason::Unlinked("Globex".into()), RejectReason::Unlinked("Acme".into())]
    );
}

// obligation: SPEC-7.1-O2
#[test]
fn first_person_mentions_link_as_the_speaker() {
    let backend = FixtureBackend::new().relation("I", "person", "works at", "Acme", "organization", 0.8);
    let mut x = GlinerExtractor::new(backend, linker());
    let facts = x.extract(&ep("e", "I work at Acme."));
    assert_eq!(triples(&facts), vec![t("person:dana", "works_at", "org:acme")]);
}

/// Counts calls; can be told to fail.
struct Flaky {
    inner: FixtureBackend,
    fail: bool,
    calls: usize,
}

impl RelationBackend for Flaky {
    fn analyse(&mut self, s: &[String]) -> Result<Vec<SentenceAnalysis>, String> {
        self.calls += 1;
        if self.fail {
            Err("onnx session failed".into())
        } else {
            self.inner.analyse(s)
        }
    }
}

/// A linker that counts calls, to prove memoization covers linking too.
struct Counting(TableLinker, usize);

impl EntityLinker for Counting {
    fn link(&mut self, s: &str, m: &[Mention]) -> Vec<Link> {
        self.1 += 1;
        self.0.link(s, m)
    }
}

// obligation: SPEC-7.1-O3
#[test]
fn extraction_is_memoized_and_repeatable() {
    let backend = Flaky { inner: works("Bob", "Acme"), fail: false, calls: 0 };
    let mut lk = Counting(linker(), 0);
    let mut x = GlinerExtractor::new(backend, &mut lk);
    let e = ep("e1", "Bob works at Acme.");
    let first = x.extract(&e);
    let second = x.extract(&e);
    assert_eq!(first, second);
    assert_eq!(x.backend().calls, 1);
    assert_eq!(x.stats(), (1, 0));
    drop(x);
    assert_eq!(lk.1, 1);

    // a fresh extractor on the same input gives the same facts
    let mut y = GlinerExtractor::new(works("Bob", "Acme"), linker());
    assert_eq!(y.extract(&e), first);
}

// obligation: SPEC-7.1-O3
#[test]
fn output_is_deduplicated_and_sorted() {
    let backend = FixtureBackend::new()
        .relation("Bob", "person", "works at", "Initech", "organization", 0.6)
        .relation("Bob", "person", "works at", "Acme", "organization", 0.7)
        .relation("Dana", "person", "works at", "Acme", "organization", 0.9);
    let mut x = GlinerExtractor::new(backend, linker());
    let facts = x.extract(&ep("e", "Dana works at Acme. Bob works at Acme. Bob works at Initech. Bob works at Acme!"));
    assert_eq!(
        triples(&facts),
        vec![
            t("person:bob", "works_at", "org:acme"),
            t("person:bob", "works_at", "org:initech"),
            t("person:dana", "works_at", "org:acme"),
        ]
    );
    assert!((facts[0].confidence - 0.7).abs() < 1e-6);
}

// obligation: SPEC-7.1-O3
#[test]
fn backend_failure_yields_no_facts_and_is_counted() {
    let backend = Flaky { inner: works("Bob", "Acme"), fail: true, calls: 0 };
    let mut x = GlinerExtractor::new(backend, linker());
    assert!(x.extract(&ep("e", "Bob works at Acme.")).is_empty());
    assert_eq!(x.stats(), (1, 1));
}
