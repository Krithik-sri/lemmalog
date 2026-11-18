//! Checks for spec/extraction.md (SPEC-7.1). Each test is tagged with the
//! obligation it checks; obligations.lock pins these tests.

use lemmalog::agent::{CandidateFact, Episode, Extractor};
use lemmalog::extract::pipeline::{default_entity_labels, split_sentences};
use lemmalog::extract::{
    EntityLinker, FixtureBackend, GlinerExtractor, Link, Mention, NegEx, RejectReason,
    RelationBackend, SentenceAnalysis, TableLinker,
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

// obligation: SPEC-7.1-O1
#[test]
fn negation_window_counts_hyphenated_words_once() {
    // six words between trigger and object: inside the window only if
    // "well-known" and friends are one token each
    let text = "Bob doesn't work at the well-known long-standing big-name Acme.";
    let mut x = GlinerExtractor::new(works("Bob", "Acme"), linker());
    assert!(x.extract(&ep("e", text)).is_empty());
    assert_eq!(x.rejected()[0].reason, RejectReason::Negated);
}

// obligation: SPEC-7.1-O2
#[test]
fn table_alias_without_label_links_any_label() {
    let lk = TableLinker::new().entity("org:acme", None, &["Acme"]).entity("person:bob", None, &["Bob"]);
    let backend = FixtureBackend::new().relation("Bob", "person", "works at", "Acme", "thing", 0.9);
    let mut x = GlinerExtractor::new(backend, lk);
    let facts = x.extract(&ep("e", "Bob works at Acme."));
    assert_eq!(triples(&facts), vec![t("person:bob", "works_at", "org:acme")]);
}

// obligation: SPEC-7.1-O2
#[test]
fn byte_offsets_convert_to_char_offsets() {
    use lemmalog::extract::link::byte_to_char;
    let s = "José works at Acme";
    assert_eq!(byte_to_char(s, 0), 0);
    assert_eq!(byte_to_char(s, s.find("works").unwrap()), 5); // 'é' is 2 bytes
    assert_eq!(byte_to_char(s, s.len()), 18);
}

#[cfg(feature = "glinker")]
fn l0(start: usize, end: usize, id: Option<(&str, f32)>) -> glinker_rs::L0Entity {
    glinker_rs::L0Entity {
        mention_text: String::new(),
        label: None,
        mention_start: start,
        mention_end: end,
        left_context: String::new(),
        right_context: String::new(),
        candidates: vec![],
        num_candidates: 0,
        linked_entity: id.map(|(i, c)| glinker_rs::LinkedEntity {
            entity_id: i.into(),
            label: i.into(),
            confidence: c,
            start,
            end,
            matched_text: String::new(),
        }),
        is_linked: id.is_some(),
        candidate_scores: vec![],
        pipeline_stage: glinker_rs::Stage::L3Linked,
    }
}

#[cfg(feature = "glinker")]
fn mention(s: &str, text: &str) -> Mention {
    let b = s.find(text).unwrap();
    Mention { text: text.into(), label: "x".into(), start: b, end: b + text.len(), score: 1.0 }
}

// obligation: SPEC-7.1-O2
#[cfg(feature = "glinker")]
#[test]
fn glinker_output_maps_back_by_char_span() {
    use lemmalog::extract::link::links_from_l0;
    let s = "José works at Acme and Globex.";
    let ms = [mention(s, "José"), mention(s, "Acme"), mention(s, "Globex")];
    // Acme's char span is 14..18, its byte span 15..19: a byte match would miss it
    let found = [
        l0(0, 4, Some(("person:jose", 0.9))),
        l0(14, 18, Some(("org:acme", 0.5))),
        l0(23, 29, Some(("org:globex", 0.49))),
    ];
    assert_eq!(
        links_from_l0(s, &ms, &found, 0.5),
        vec![
            Link::Canonical { id: "person:jose".into(), score: 0.9 },
            Link::Canonical { id: "org:acme".into(), score: 0.5 },
            Link::Unlinked,
        ]
    );
}

// obligation: SPEC-7.1-O2
#[cfg(feature = "glinker")]
#[test]
fn glinker_unknown_unlinked_or_missing_spans_do_not_link() {
    use lemmalog::extract::link::links_from_l0;
    let s = "Bob met Ann and Cy.";
    let ms = [mention(s, "Bob"), mention(s, "Ann"), mention(s, "Cy")];
    let mut not_linked = l0(8, 11, Some(("person:ann", 0.9)));
    not_linked.is_linked = false;
    let found = [l0(0, 3, Some(("unknown", 0.99))), not_linked, l0(16, 17, Some(("person:cy", 0.9)))];
    assert_eq!(links_from_l0(s, &ms, &found, 0.0), vec![Link::Unlinked; 3]);
}

/// NegEx directly: is `obj` negated for a relation whose subject is `subj`?
fn negd(neg: &NegEx, s: &str, subj: &str, obj: &str) -> bool {
    let o = s.rfind(obj).unwrap();
    neg.negated(s, (o, o + obj.len()), s.find(subj).unwrap())
}

// obligation: SPEC-7.1-O1
#[test]
fn pre_negation_window_edge() {
    let n = NegEx::default(); // window 6
    assert!(negd(&n, "Bob doesn't aa bb cc dd ee ff Acme", "Bob", "Acme"), "6 words between");
    assert!(!negd(&n, "Bob doesn't aa bb cc dd ee ff gg Acme", "Bob", "Acme"), "7 words between");
    assert!(negd(&n, "Bob quit Acme", "Bob", "Acme"), "adjacent");
    let narrow = NegEx::default().with_window(2);
    assert!(negd(&n, "Bob doesn't aa bb cc Acme", "Bob", "Acme"));
    assert!(!negd(&narrow, "Bob doesn't aa bb cc Acme", "Bob", "Acme"));
    assert!(negd(&narrow, "Bob doesn't aa bb Acme", "Bob", "Acme"));
}

// obligation: SPEC-7.1-O1
#[test]
fn post_negation_window_edge() {
    let n = NegEx::default();
    assert!(negd(&n, "Bob at Acme aa bb cc dd ee ff anymore", "Bob", "Acme"), "6 words between");
    assert!(!negd(&n, "Bob at Acme aa bb cc dd ee ff gg anymore", "Bob", "Acme"), "7 words between");
    // a multi-word trigger that ends the text (no full stop) still matches
    assert!(negd(&n, "Bob works at Acme no more", "Bob", "Acme"));
    assert!(negd(&n, "Bob works at Acme anymore", "Bob", "Acme"));
}

// obligation: SPEC-7.1-O1
#[test]
fn window_counts_tokens_around_the_exact_object_span() {
    let n = NegEx::default();
    // a quote right before the object is a token of its own: 7 tokens between
    assert!(!negd(&n, "Bob doesn't aa bb cc dd ee ff \"Acme", "Bob", "Acme"));
    // an object span that starts with punctuation begins at that punctuation
    assert!(negd(&n, "Bob doesn't aa bb cc dd ee ff (Acme)", "Bob", "(Acme)"));
    // a quote right after the object is between object and trigger: 7 tokens
    // (not a comma: commas are list punctuation and do not count, see O1)
    assert!(!negd(&n, "Bob works at Acme\" aa bb cc dd ee ff anymore", "Bob", "Acme"));
    assert!(negd(&n, "Bob works at Acme, aa bb cc dd ee ff anymore", "Bob", "Acme"));
}

// obligation: SPEC-7.1-O1
#[test]
fn trigger_and_termination_words_inside_the_object_are_part_of_the_name() {
    let n = NegEx::default();
    assert!(!negd(&n, "Bob works at Never Again Ltd", "Bob", "Never Again Ltd"));
    assert!(!negd(&n, "Bob works at Studio Anymore", "Bob", "Studio Anymore"));
    // "But" and "Now" inside the name do not cut a scope that reaches the object
    assert!(negd(&n, "Bob doesn't work at But Labs", "Bob", "But Labs"));
    assert!(negd(&n, "Bob works at Studio Now anymore", "Bob", "Studio Now"));
}

// obligation: SPEC-7.1-O3
#[test]
fn relations_below_min_score_are_rejected() {
    let backend = FixtureBackend::new()
        .relation("Bob", "person", "works at", "Acme", "organization", 0.5)
        .relation("Dana", "person", "works at", "Initech", "organization", 0.49);
    let mut x = GlinerExtractor::new(backend, linker()); // min_score 0.5
    let facts = x.extract(&ep("e", "Bob works at Acme. Dana works at Initech."));
    assert_eq!(triples(&facts), vec![t("person:bob", "works_at", "org:acme")]);
    assert_eq!(x.rejected().len(), 1);
    assert_eq!(x.rejected()[0].reason, RejectReason::LowScore);
}

/// Returns one analysis too few.
struct Short(FixtureBackend);

impl RelationBackend for Short {
    fn analyse(&mut self, s: &[String]) -> Result<Vec<SentenceAnalysis>, String> {
        let mut v = self.0.analyse(s)?;
        v.pop();
        Ok(v)
    }
}

// obligation: SPEC-7.1-O3
#[test]
fn backend_reply_that_does_not_match_the_sentences_is_a_failure() {
    let mut x = GlinerExtractor::new(Short(works("Bob", "Acme")), linker());
    assert!(x.extract(&ep("e", "Bob works at Acme. Bob works at Acme!")).is_empty());
    assert_eq!(x.stats(), (1, 1));
}

// obligation: SPEC-7.1-O3
#[test]
fn sentence_split_drops_blank_pieces() {
    assert_eq!(
        split_sentences("Bob works at Acme.\n\n   \nDana works!  "),
        vec!["Bob works at Acme.".to_string(), "Dana works!".to_string()]
    );
    assert!(split_sentences(" \n ").is_empty());
}

// obligation: SPEC-7.1-O3
#[test]
fn default_entity_labels_cover_the_relation_set() {
    assert_eq!(
        default_entity_labels(),
        ["activity", "food", "job title", "location", "organization", "person", "team", "thing"]
    );
}

// obligation: SPEC-7.1-O3
#[test]
fn fixture_backend_reports_exact_spans_and_counts_calls() {
    let mut b = works("Bob", "Acme");
    let a = b.analyse(&["Bob works at Acme".to_string()]).unwrap();
    let r = &a[0].relations[0];
    assert_eq!((r.subject.start, r.subject.end), (0, 3));
    assert_eq!((r.object.start, r.object.end), (13, 17));
    assert_eq!(a[0].mentions.len(), 2);
    assert_eq!(b.calls, 1);
    // subject not at the start: offsets are absolute, not relative to the subject
    let a = b.analyse(&["Then Bob works at Acme".to_string()]).unwrap();
    let r = &a[0].relations[0];
    assert_eq!((r.subject.start, r.object.start, r.object.end), (5, 18, 22));
    assert_eq!(b.calls, 2);
}
