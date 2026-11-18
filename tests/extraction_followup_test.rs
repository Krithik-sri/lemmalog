//! Checks for the fixes from the first real-model run of
//! notebooks/extraction_eval.ipynb (gold set, GLiNER multitask large v0.5,
//! glinker-rs with modern-gliner-bi-base-v1.0).

use lemmalog::agent::{CandidateFact, Episode, Extractor};
use lemmalog::extract::{
    strip_title, CachedBackend, CascadeLinker, EntityLinker, FixtureBackend, GlinerExtractor, Link, Mention,
    RejectReason, TableLinker,
};

fn ep(id: &str, text: &str) -> Episode {
    Episode { id: id.into(), text: text.into(), ts: 0, speaker: Some("Dana".into()) }
}

fn linker() -> TableLinker {
    TableLinker::new()
        .entity("person:bob", Some("person"), &["Bob", "Smith", "Bob Smith"])
        .entity("person:dana", Some("person"), &["Dana"])
        .entity("person:priya", Some("person"), &["Priya"])
        .entity("person:sam", Some("person"), &["Sam"])
        .entity("org:acme", Some("organization"), &["Acme"])
        .entity("org:globex", Some("organization"), &["Globex"])
}

fn triples(f: &[CandidateFact]) -> Vec<(&str, &str, &str)> {
    f.iter().map(|c| (c.subj.as_str(), c.pred.as_str(), c.obj.as_str())).collect()
}

fn mention(s: &str, text: &str, label: &str) -> Mention {
    let i = s.find(text).unwrap();
    Mention { text: text.into(), label: label.into(), start: i, end: i + text.len(), score: 0.9 }
}

// ------------------------------------------- 1. one direction per sentence

// obligation: SPEC-7.1-O4
#[test]
fn both_manager_directions_keep_the_one_the_sentence_words() {
    // e05: GLiNER fired "reports to" and "manager of" on the same pair; the
    // wrong one scored higher, the sentence's own wording decides
    let backend = FixtureBackend::new()
        .relation("Sam", "person", "reports to", "Priya", "person", 0.6)
        .relation("Sam", "person", "manager of", "Priya", "person", 0.9);
    let mut x = GlinerExtractor::new(backend, linker());
    let f = x.extract(&ep("e05", "Sam reports to Priya."));
    assert_eq!(triples(&f), [("person:sam", "manager", "person:priya")]);
    assert_eq!(x.rejected().iter().map(|r| &r.reason).collect::<Vec<_>>(), [&RejectReason::Reversed]);

    // e06: the other wording, the other winner
    let backend = FixtureBackend::new()
        .relation("Priya", "person", "manager of", "Sam", "person", 0.6)
        .relation("Priya", "person", "reports to", "Sam", "person", 0.9);
    let mut x = GlinerExtractor::new(backend, linker());
    let f = x.extract(&ep("e06", "Priya is the manager of Sam."));
    assert_eq!(triples(&f), [("person:sam", "manager", "person:priya")]);
}

// obligation: SPEC-7.1-O4
#[test]
fn without_a_cue_the_higher_confidence_direction_wins() {
    let backend = FixtureBackend::new()
        .relation("Sam", "person", "reports to", "Priya", "person", 0.9)
        .relation("Sam", "person", "manager of", "Priya", "person", 0.6);
    let mut x = GlinerExtractor::new(backend, linker());
    let f = x.extract(&ep("e", "Sam works under Priya."));
    assert_eq!(triples(&f), [("person:sam", "manager", "person:priya")]);
    // equal evidence: still exactly one survives, the same one every time
    let tie = || {
        let backend = FixtureBackend::new()
            .relation("Sam", "person", "reports to", "Priya", "person", 0.7)
            .relation("Sam", "person", "manager of", "Priya", "person", 0.7);
        let mut x = GlinerExtractor::new(backend, linker());
        x.extract(&ep("e", "Sam works under Priya."))
    };
    assert_eq!(tie().len(), 1);
    assert_eq!(tie(), tie());
}

// obligation: SPEC-7.1-O4
#[test]
fn directions_from_different_sentences_both_stand() {
    // only one sentence asserting both directions is a contradiction
    let backend = FixtureBackend::new()
        .relation("Sam", "person", "reports to", "Priya", "person", 0.9)
        .relation("Priya", "person", "reports to", "Sam", "person", 0.9);
    let mut x = GlinerExtractor::new(backend, linker());
    let f = x.extract(&ep("e", "Sam reports to Priya. Later Priya reports to Sam."));
    assert_eq!(f.len(), 2);
}

// ------------------------------------------------ 2. speaker linked by name

/// Records what the linker was asked, answering from a table.
struct Recording(TableLinker, Vec<(String, Vec<String>)>);

impl EntityLinker for Recording {
    fn link(&mut self, s: &str, m: &[Mention]) -> Vec<Link> {
        self.1.push((s.to_string(), m.iter().map(|x| x.text.clone()).collect()));
        self.0.link(s, m)
    }
}

// obligation: SPEC-7.1-O2
#[test]
fn speaker_is_linked_by_name_once_per_episode() {
    let backend = FixtureBackend::new()
        .relation("I", "person", "works at", "Globex", "organization", 0.9)
        .relation("Priya", "person", "reports to", "me", "person", 0.9);
    let mut lk = Recording(linker(), vec![]);
    let mut x = GlinerExtractor::new(backend, &mut lk);
    let f = x.extract(&ep("e", "I work at Globex. Priya reports to me."));
    assert_eq!(triples(&f), [("person:dana", "works_at", "org:globex"), ("person:priya", "manager", "person:dana")]);
    drop(x);
    // the name alone, as its own context, exactly once
    let speaker_calls: Vec<_> = lk.1.iter().filter(|(s, _)| s == "Dana").collect();
    assert_eq!(speaker_calls.len(), 1);
    // pronouns never reach the linker
    assert!(lk.1.iter().all(|(_, ms)| ms.iter().all(|m| m != "I" && m != "me")), "{:?}", lk.1);
}

// obligation: SPEC-7.1-O2
#[test]
fn first_person_without_a_speaker_does_not_link() {
    let backend = FixtureBackend::new().relation("I", "person", "works at", "Globex", "organization", 0.9);
    let mut x = GlinerExtractor::new(backend, linker());
    let f = x.extract(&Episode { id: "e".into(), text: "I work at Globex.".into(), ts: 0, speaker: None });
    assert!(f.is_empty());
    assert!(matches!(x.rejected()[0].reason, RejectReason::Unlinked(_)));
}

// ------------------------------------------------------- 3. title stripping

// obligation: SPEC-7.1-O2
#[test]
fn titles_are_stripped_before_linking() {
    let s = "Dr. Smith works at Acme.";
    let backend = FixtureBackend::new().relation("Dr. Smith", "person", "works at", "Acme", "organization", 0.9);
    let mut x = GlinerExtractor::new(backend, linker());
    assert_eq!(triples(&x.extract(&ep("e16", s))), [("person:bob", "works_at", "org:acme")]);

    for (sentence, text, want) in [
        ("Dr. Smith left.", "Dr. Smith", "Smith"),
        ("Dr Smith left.", "Dr Smith", "Smith"),
        ("Mrs. Lee Park left.", "Mrs. Lee Park", "Lee Park"),
        ("Drake left.", "Drake", "Drake"),
        ("The Dr. left.", "Dr.", "Dr."),
    ] {
        let m = mention(sentence, text, "person");
        let got = strip_title(sentence, &m);
        assert_eq!(got.text, want, "{sentence:?}");
        assert_eq!(&sentence[got.start..got.end], want, "span follows the name");
    }
}

// ------------------------------------------------ 4. exact alias, then model

// obligation: SPEC-7.1-O2
#[test]
fn shared_aliases_are_ambiguous_in_the_table() {
    let mut t = TableLinker::new()
        .entity("animal:jaguar", Some("animal"), &["jaguar"])
        .entity("org:jaguar-cars", Some("organization"), &["Jaguar", "Jaguar Cars"])
        .entity("org:jlr", None, &["JLR"])
        .entity("org:jaguar-cars", None, &["JLR"]);
    let s = "Jaguar Cars and JLR";
    let ms = [
        Mention { label: "thing".into(), ..mention(s, "Jaguar", "") },
        Mention { label: "organization".into(), ..mention(s, "Jaguar", "") },
        mention(s, "Jaguar Cars", "organization"),
        mention(s, "JLR", "organization"),
    ];
    let got: Vec<Option<String>> = t
        .link(s, &ms)
        .into_iter()
        .map(|l| match l { Link::Canonical { id, .. } => Some(id), Link::Unlinked => None })
        .collect();
    assert_eq!(got, [None, Some("org:jaguar-cars".into()), Some("org:jaguar-cars".into()), None]);
}

/// Links everything it is asked to one id, recording the mentions.
struct Fallback(Vec<String>);

impl EntityLinker for Fallback {
    fn link(&mut self, _s: &str, m: &[Mention]) -> Vec<Link> {
        self.0.extend(m.iter().map(|x| x.text.clone()));
        m.iter().map(|_| Link::Canonical { id: "model:x".into(), score: 0.4 }).collect()
    }
}

// obligation: SPEC-7.1-O2
#[test]
fn cascade_asks_the_model_only_for_what_the_table_left() {
    let s = "Bob met Globex Inc and Acme.";
    let ms = [mention(s, "Bob", "person"), mention(s, "Globex Inc", "organization"), mention(s, "Acme", "organization")];
    let mut fb = Fallback(vec![]);
    let mut c = CascadeLinker::new(linker(), &mut fb);
    let ids: Vec<String> = c
        .link(s, &ms)
        .into_iter()
        .map(|l| match l { Link::Canonical { id, .. } => id, Link::Unlinked => "-".into() })
        .collect();
    assert_eq!(ids, ["person:bob", "model:x", "org:acme"]);
    assert_eq!(fb.0, ["Globex Inc"]);
}

// ------------------------------------------------------ cached model output

#[test]
fn cached_backend_runs_the_model_once_per_sentence() {
    let mut cb = CachedBackend::new(FixtureBackend::new().relation("Bob", "person", "works at", "Acme", "organization", 0.9));
    for t in [0.3, 0.5] {
        let mut lk = linker();
        let mut x = GlinerExtractor::new(&mut cb, &mut lk);
        x.min_score = t;
        assert_eq!(x.extract(&ep("e", "Bob works at Acme. Bob works at Acme.")).len(), 1);
    }
    assert_eq!(cb.misses, 1);
    assert_eq!(cb.inner.calls, 1);
}

// ---------------------- direction tie-break under a low-confidence linker

/// Replays fixed relations for every sentence.
struct Canned(lemmalog::extract::SentenceAnalysis);

impl lemmalog::extract::RelationBackend for Canned {
    fn analyse(&mut self, s: &[String]) -> Result<Vec<lemmalog::extract::SentenceAnalysis>, String> {
        Ok(vec![self.0.clone(); s.len()])
    }
}

/// A table linker whose every link has the same low score, like glinker-rs
/// on short names.
struct Low(TableLinker);

impl EntityLinker for Low {
    fn link(&mut self, s: &str, m: &[Mention]) -> Vec<Link> {
        self.0
            .link(s, m)
            .into_iter()
            .map(|l| match l { Link::Canonical { id, .. } => Link::Canonical { id, score: 0.4 }, u => u })
            .collect()
    }
}

// obligation: SPEC-7.1-O4
#[test]
fn same_label_both_ways_is_decided_by_wording_position_not_link_scores() {
    // e05 under glinker-rs: "reports to" fired both ways; link scores (0.4)
    // capped both confidences to a tie, and the triple order picked the
    // wrong direction
    use lemmalog::extract::{RawRelation, SentenceAnalysis};
    let s = "Sam reports to Priya.";
    let (sam, priya) = (mention(s, "Sam", "person"), mention(s, "Priya", "person"));
    let rel = |a: &Mention, b: &Mention, score| RawRelation { subject: a.clone(), label: "reports to".into(), object: b.clone(), score };
    for (right, wrong) in [(0.6, 0.9), (0.9, 0.6), (0.7, 0.7)] {
        let a = SentenceAnalysis {
            mentions: vec![sam.clone(), priya.clone()],
            relations: vec![rel(&sam, &priya, right), rel(&priya, &sam, wrong)],
            ..Default::default()
        };
        let mut x = GlinerExtractor::new(Canned(a), Low(linker()));
        let f = x.extract(&ep("e05", s));
        assert_eq!(triples(&f), [("person:sam", "manager", "person:priya")], "scores {right}/{wrong}");
        assert!((f[0].confidence - 0.4).abs() < 1e-6, "confidence still includes the link score");
    }
}

// obligation: SPEC-7.1-O4
#[test]
fn without_wording_the_relation_score_decides_not_the_link_score() {
    use lemmalog::extract::{RawRelation, SentenceAnalysis};
    let s = "Sam works under Priya.";
    let (sam, priya) = (mention(s, "Sam", "person"), mention(s, "Priya", "person"));
    let rel = |a: &Mention, b: &Mention, score| RawRelation { subject: a.clone(), label: "reports to".into(), object: b.clone(), score };
    for (sp, ps, want) in [(0.9, 0.6, ("person:sam", "person:priya")), (0.6, 0.9, ("person:priya", "person:sam"))] {
        let a = SentenceAnalysis {
            mentions: vec![sam.clone(), priya.clone()],
            relations: vec![rel(&sam, &priya, sp), rel(&priya, &sam, ps)],
            ..Default::default()
        };
        let mut x = GlinerExtractor::new(Canned(a), Low(linker()));
        let f = x.extract(&ep("e", s));
        assert_eq!(triples(&f), [(want.0, "manager", want.1)]);
    }
}

#[test]
fn worded_between_needs_subject_label_object_order() {
    use lemmalog::extract::{worded_between, RawRelation};
    let s = "Priya is the manager of Sam.";
    let r = |a: &str, label: &str, b: &str| RawRelation {
        subject: mention(s, a, "person"),
        label: label.into(),
        object: mention(s, b, "person"),
        score: 0.9,
    };
    assert!(worded_between(s, &r("Priya", "manager of", "Sam")));
    assert!(!worded_between(s, &r("Sam", "manager of", "Priya")), "object before subject");
    assert!(!worded_between(s, &r("Priya", "reports to", "Sam")), "label not written");
}

// ------------------------------------------------ backend errors are visible

struct Failing;

impl lemmalog::extract::RelationBackend for Failing {
    fn analyse(&mut self, s: &[String]) -> Result<Vec<lemmalog::extract::SentenceAnalysis>, String> {
        Err(format!("relation extraction failed on {:?}: invalid input: empty texts and/or entities", s[0]))
    }
}

// obligation: SPEC-7.1-O3
#[test]
fn backend_errors_keep_their_message() {
    let mut x = GlinerExtractor::new(Failing, linker());
    assert!(x.extract(&ep("e25", "Acme is located in Berlin.")).is_empty());
    assert_eq!(x.stats(), (1, 1));
    let (id, msg) = &x.errors()[0];
    assert_eq!(id, "e25");
    assert!(msg.contains("Acme is located in Berlin.") && msg.contains("empty texts"), "{msg}");
}
