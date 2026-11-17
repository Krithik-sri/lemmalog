//! Checks for the fixes from the GLiNER/GLinker extraction review
//! (spec/extraction.md SPEC-7.1-O1, O2, O4, O5). Run with `--nocapture` to
//! see the NegEx scores.

use lemmalog::agent::{CandidateFact, Episode, Extractor};
use lemmalog::extract::{
    resolve_relations, split_sentences, FixtureBackend, GlinerExtractor, Mention, ModelRelation, NegEx, RawRelation,
    RejectReason, RelationBackend, SentenceAnalysis, TableLinker,
};

fn ep(id: &str, text: &str) -> Episode {
    Episode { id: id.into(), text: text.into(), ts: 0, speaker: Some("Dana".into()) }
}

fn linker() -> TableLinker {
    TableLinker::new()
        .entity("person:bob", Some("person"), &["Bob"])
        .entity("person:dana", Some("person"), &["Dana"])
        .entity("org:acme", Some("organization"), &["Acme"])
        .entity("org:initech", Some("organization"), &["Initech"])
}

fn triples(f: &[CandidateFact]) -> Vec<(&str, &str, &str)> {
    f.iter().map(|c| (c.subj.as_str(), c.pred.as_str(), c.obj.as_str())).collect()
}

// ---------------------------------------------------------------- NegEx, MCC

struct Score {
    tp: f64,
    tn: f64,
    fp: f64,
    fn_: f64,
}

impl Score {
    fn mcc(&self) -> f64 {
        let (tp, tn, fp, fn_) = (self.tp, self.tn, self.fp, self.fn_);
        let d = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
        if d == 0.0 { 0.0 } else { (tp * tn - fp * fn_) / d }
    }
}

fn negex_score(file: &str) -> Score {
    let n = NegEx::default();
    let text = std::fs::read_to_string(format!("{}/tests/{file}", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let mut sc = Score { tp: 0.0, tn: 0.0, fp: 0.0, fn_: 0.0 };
    println!("\n{file}");
    for line in text.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
        let f: Vec<&str> = line.split('\t').collect();
        let (s, subj, obj, gold) = (f[0], f[1], f[2], f[3] == "1");
        let o = s.find(obj).unwrap();
        let others: Vec<(usize, usize)> = f
            .get(4)
            .map(|x| x.split('|').filter(|m| !m.is_empty()).map(|m| { let i = s.find(m).unwrap(); (i, i + m.len()) }).collect())
            .unwrap_or_default();
        let pred = n.negated_with(s, (o, o + obj.len()), s.find(subj).unwrap(), &others);
        match (gold, pred) {
            (true, true) => sc.tp += 1.0,
            (false, false) => sc.tn += 1.0,
            (false, true) => { sc.fp += 1.0; println!("  FP {s}") }
            (true, false) => { sc.fn_ += 1.0; println!("  FN {s}") }
        }
    }
    println!("  TP={} TN={} FP={} FN={} MCC={:.3}", sc.tp, sc.tn, sc.fp, sc.fn_, sc.mcc());
    sc
}

// obligation: SPEC-7.1-O1
#[test]
fn negex_mcc_on_labelled_sets() {
    // measured when written: tuning 0.951, held-out 0.938 (the rules before
    // this change: 0.090 and 0.044). Floors sit a little below, so a rule
    // change may not make either set worse.
    assert!(negex_score("negex_tuning.tsv").mcc() >= 0.90);
    assert!(negex_score("negex_heldout.tsv").mcc() >= 0.85);
}

// obligation: SPEC-7.1-O1
#[test]
fn negex_cessation_idioms_and_scope_blocking() {
    let n = NegEx::default();
    let neg = |s: &str, subj: &str, obj: &str, others: &[&str]| {
        let o = s.find(obj).unwrap();
        let os: Vec<_> = others.iter().map(|m| { let i = s.find(m).unwrap(); (i, i + m.len()) }).collect();
        n.negated_with(s, (o, o + obj.len()), s.find(subj).unwrap(), &os)
    };
    // cessation, apostrophe-less contractions, sentential negation
    assert!(neg("Bob left Acme last year.", "Bob", "Acme", &[]));
    assert!(neg("Bob doesnt work at Acme.", "Bob", "Acme", &[]));
    assert!(neg("It's not true that Bob works at Acme.", "Bob", "Acme", &[]));
    // negated-positive idioms and double negatives
    assert!(!neg("Bob hasn't left Acme.", "Bob", "Acme", &[]));
    assert!(!neg("Bob never stopped working at Acme.", "Bob", "Acme", &[]));
    assert!(!neg("Bob doesn't mind working at Acme.", "Bob", "Acme", &[]));
    // another mention takes the trigger's scope ...
    assert!(!neg("Bob quit Initech to join Acme.", "Bob", "Acme", &["Initech"]));
    // ... unless it is coordinated with the object
    assert!(neg("Bob doesn't work at Acme, Initech, or Globex.", "Bob", "Globex", &["Acme", "Initech"]));
    // a trigger inside a comma-enclosed aside does not govern the object
    assert!(!neg("Bob, who has no car, works at Acme.", "Bob", "Acme", &[]));
    assert!(neg("Bob works at Acme, not Initech.", "Bob", "Initech", &["Acme"]));
}

// ------------------------------------------------------- sentence splitting

// obligation: SPEC-7.1-O5
#[test]
fn sentence_splitter_keeps_abbreviations_numbers_and_ellipses() {
    let cases: &[(&str, &[&str])] = &[
        ("Dr. Smith works at Acme Corp. in Austin.", &["Dr. Smith works at Acme Corp. in Austin."]),
        ("Bob moved to the U.S. last year and works at Acme.", &["Bob moved to the U.S. last year and works at Acme."]),
        ("Bob works at Acme Inc. He likes it.", &["Bob works at Acme Inc.", "He likes it."]),
        ("Version 2.5 shipped. Bob works at Acme.", &["Version 2.5 shipped.", "Bob works at Acme."]),
        ("Bob works at Acme... maybe.", &["Bob works at Acme... maybe."]),
        ("Really?! Bob works at Acme.", &["Really?!", "Bob works at Acme."]),
        ("see acme.com for details. bob works there.", &["see acme.com for details.", "bob works there."]),
        ("Bob works at Acme\nDana works at Initech", &["Bob works at Acme", "Dana works at Initech"]),
    ];
    for (text, want) in cases {
        assert_eq!(split_sentences(text), *want, "{text:?}");
    }
}

// ----------------------------------------------------- relation direction

// obligation: SPEC-7.1-O4
#[test]
fn manager_relations_read_as_reports_to() {
    // lemmalog: X --manager--> Y means X reports to Y
    let backend = FixtureBackend::new()
        .relation("Dana", "person", "manager of", "Bob", "person", 0.9)
        .relation("Bob", "person", "reports to", "Dana", "person", 0.9);
    let mut x = GlinerExtractor::new(backend, linker());
    let a = x.extract(&ep("a", "Dana is the manager of Bob."));
    assert_eq!(triples(&a), [("person:bob", "manager", "person:dana")]);
    let b = x.extract(&ep("b", "Bob reports to Dana."));
    assert_eq!(triples(&b), [("person:bob", "manager", "person:dana")]);
}

// ------------------------------------------------------- span integrity

/// A backend that returns whatever analysis it is given, for every sentence.
struct Canned(SentenceAnalysis);

impl RelationBackend for Canned {
    fn analyse(&mut self, s: &[String]) -> Result<Vec<SentenceAnalysis>, String> {
        Ok(vec![self.0.clone(); s.len()])
    }
}

fn m(text: &str, label: &str, start: usize) -> Mention {
    Mention { text: text.into(), label: label.into(), start, end: start + text.len(), score: 0.9 }
}

// obligation: SPEC-7.1-O5
#[test]
fn out_of_range_spans_are_rejected_not_trusted() {
    // the pre-fix gline backend produced object spans 56 bytes past the
    // sentence: such a relation must neither skip NegEx nor reach the linker
    let s = "Bob doesn't work at Acme.";
    let bad = RawRelation { subject: m("Bob", "person", 0), label: "works at".into(), object: m("Acme", "organization", 76), score: 0.9 };
    let mut x = GlinerExtractor::new(Canned(SentenceAnalysis { relations: vec![bad], ..Default::default() }), linker());
    assert!(x.extract(&ep("e", s)).is_empty());
    assert_eq!(x.rejected()[0].reason, RejectReason::BadSpan);
}

// obligation: SPEC-7.1-O5
#[test]
fn resolved_relations_are_negation_checked() {
    let s = "Bob doesn't work at Acme.";
    let mentions = vec![m("Bob", "person", 0), m("Acme", "organization", 20)];
    // object span relative to some prompt: ignored, object found by text
    let rel = ModelRelation { subject: "Bob".into(), label: "works at".into(), object: "Acme".into(), object_span: Some((76, 80)), score: 0.9 };
    let mut x = GlinerExtractor::new(Canned(resolve_relations(s, mentions, &[rel])), linker());
    assert!(x.extract(&ep("e", s)).is_empty());
    assert_eq!(x.rejected()[0].reason, RejectReason::Negated);
}

// ----------------------------------------------------- linking & speaker

// obligation: SPEC-7.1-O2
#[test]
fn unresolved_model_relations_are_reported() {
    let s = "Bob works at Acme.";
    let rel = ModelRelation { subject: "Dana".into(), label: "works at".into(), object: "Acme".into(), object_span: None, score: 0.9 };
    let a = resolve_relations(s, vec![m("Bob", "person", 0), m("Acme", "organization", 13)], &[rel]);
    let mut x = GlinerExtractor::new(Canned(a), linker());
    assert!(x.extract(&ep("e", s)).is_empty());
    let r = &x.rejected()[0];
    assert_eq!((r.subject.as_str(), r.object.as_str(), &r.reason), ("Dana", "Acme", &RejectReason::Unresolved));
}

// obligation: SPEC-7.1-O2
#[test]
fn first_person_objects_link_as_the_speaker() {
    let backend = FixtureBackend::new().relation("Bob", "person", "reports to", "me", "person", 0.9);
    let mut x = GlinerExtractor::new(backend, linker());
    let f = x.extract(&ep("e", "Bob reports to me."));
    assert_eq!(triples(&f), [("person:bob", "manager", "person:dana")]);
}

// --------------------------------------------------------------- gold set

#[test]
fn gold_set_parses_and_uses_known_ids() {
    use lemmalog::extract::eval::Gold;
    let text = std::fs::read_to_string(format!("{}/benchmarks/extraction_gold.tsv", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let g = Gold::parse(&text).unwrap();
    let ids: Vec<&str> = g.entities.iter().map(|e| e.id.as_str()).collect();
    assert!(g.episodes.len() >= 30);
    for e in &g.episodes {
        assert!(!e.positive.is_empty() || !e.negative.is_empty(), "{} has no labels", e.episode.id);
        for (s, _, o) in e.positive.iter().chain(&e.negative) {
            assert!(ids.contains(&s.as_str()) && ids.contains(&o.as_str()), "{}: unknown id", e.episode.id);
        }
    }
}
