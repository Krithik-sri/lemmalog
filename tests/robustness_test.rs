//! Pipeline robustness: edited-episode memoization, per-sentence fallback,
//! bounded caches and logs (SPEC-7.1-O3), sentence splitting around
//! "St."/"Jr."/"Sr." (SPEC-7.1-O5), and hedged relations (SPEC-7.1-O6).

use lemmalog::agent::{CandidateFact, Episode, Extractor};
use lemmalog::extract::{
    split_sentences, CachedBackend, FixtureBackend, GlinerExtractor, NameLinker, NegEx, RejectReason,
    RelationBackend, SentenceAnalysis,
};

fn ep(id: &str, text: &str) -> Episode {
    Episode { id: id.into(), text: text.into(), ts: 0, speaker: None }
}

fn triples(f: &[CandidateFact]) -> Vec<(&str, &str, &str)> {
    f.iter().map(|c| (c.subj.as_str(), c.pred.as_str(), c.obj.as_str())).collect()
}

fn works() -> FixtureBackend {
    FixtureBackend::new()
        .relation("Bob", "person", "works at", "Acme", "organization", 0.9)
        .relation("Dana", "person", "works at", "Initech", "organization", 0.9)
}

/// Counts calls and sentences; fails any batch containing a sentence with
/// `poison` in it.
struct Poisoned {
    inner: FixtureBackend,
    poison: &'static str,
    calls: usize,
}

impl RelationBackend for Poisoned {
    fn analyse(&mut self, s: &[String]) -> Result<Vec<SentenceAnalysis>, String> {
        self.calls += 1;
        if s.iter().any(|x| x.contains(self.poison)) {
            return Err("onnx: input too long".into());
        }
        self.inner.analyse(s)
    }
}

// ------------------------------------------------------------------- O3

// obligation: SPEC-7.1-O3
#[test]
fn an_edited_episode_is_extracted_again() {
    let mut x = GlinerExtractor::new(works(), NameLinker);
    assert_eq!(triples(&x.extract(&ep("e", "Bob works at Acme."))), vec![("Bob", "works_at", "Acme")]);
    // same id, same text: memoized
    x.extract(&ep("e", "Bob works at Acme."));
    assert_eq!(x.stats(), (1, 0));
    // same id, new text: extracted again, the new facts replace the old
    assert_eq!(triples(&x.extract(&ep("e", "Dana works at Initech."))), vec![("Dana", "works_at", "Initech")]);
    assert_eq!(x.stats(), (2, 0));
    // same id and text, new speaker: extracted again
    let mut spoken = ep("e", "Dana works at Initech.");
    spoken.speaker = Some("Priya".into());
    x.extract(&spoken);
    assert_eq!(x.stats(), (3, 0));
}

// obligation: SPEC-7.1-O3
#[test]
fn one_bad_sentence_costs_only_its_own_facts() {
    let backend = Poisoned { inner: works(), poison: "@@@", calls: 0 };
    let mut x = GlinerExtractor::new(backend, NameLinker);
    let e = ep("e", "Bob works at Acme. @@@ garbled @@@. Dana works at Initech.");
    let f = x.extract(&e);
    assert_eq!(triples(&f), vec![("Bob", "works_at", "Acme"), ("Dana", "works_at", "Initech")]);
    // one batch call, then one call per sentence
    assert_eq!(x.backend().calls, 4);
    // one failure for the episode; the batch error and the bad sentence's
    // error are both kept
    assert_eq!(x.stats(), (1, 1));
    assert_eq!(x.errors().len(), 2);
    assert!(x.errors()[1].1.starts_with("sentence 2:"), "{:?}", x.errors());
    // partial failure is not memoized: the next extract tries again
    x.extract(&e);
    assert_eq!(x.stats(), (2, 2));
}

// obligation: SPEC-7.1-O3
#[test]
fn memo_and_logs_are_bounded() {
    let mut x = GlinerExtractor::new(works(), NameLinker);
    x.memo_capacity = 2;
    x.log_capacity = 4;
    for i in 0..3 {
        x.extract(&ep(&format!("e{i}"), "Bob works at Acme."));
    }
    // e0 was evicted (oldest), e2 is still memoized
    x.extract(&ep("e2", "Bob works at Acme."));
    assert_eq!(x.stats(), (3, 0));
    x.extract(&ep("e0", "Bob works at Acme."));
    assert_eq!(x.stats(), (4, 0));
    // rejections beyond the capacity drop the oldest, in batches
    for i in 0..20 {
        x.extract(&ep(&format!("n{i}"), "Bob doesn't work at Acme."));
    }
    assert!(x.rejected().len() <= 5, "{}", x.rejected().len());
    assert_eq!(x.rejected().last().unwrap().episode, "n19");
}

// obligation: SPEC-7.1-O3
#[test]
fn cached_backend_is_bounded_and_answers_whole_batches() {
    let mut c = CachedBackend::new(works());
    c.capacity = 2;
    let s = |t: &str| t.to_string();
    // a batch bigger than the capacity still gets every analysis back
    let got = c.analyse(&[s("Bob works at Acme."), s("Dana works at Initech."), s("Bob is here.")]).unwrap();
    assert_eq!(got.len(), 3);
    assert_eq!(c.len(), 2);
    assert_eq!(c.misses, 3);
    // the oldest sentence was evicted and is analysed again
    c.analyse(&[s("Bob works at Acme.")]).unwrap();
    assert_eq!(c.misses, 4);
}

// ------------------------------------------------------------------- O5

// obligation: SPEC-7.1-O5
#[test]
fn street_saint_junior_and_senior_split_correctly() {
    let split = |t: &str| split_sentences(t);
    // "Jr."/"Sr." end a sentence before a capitalised word ...
    assert_eq!(split("Bob Smith Jr. Dana works at Acme."), vec!["Bob Smith Jr.", "Dana works at Acme."]);
    assert_eq!(split("I met Ann Lee Sr. She works at Acme."), vec!["I met Ann Lee Sr.", "She works at Acme."]);
    // ... but not before a lower-case one
    assert_eq!(split("Bob Smith Jr. works at Acme."), vec!["Bob Smith Jr. works at Acme."]);
    // "Main St." is a street: it ends a sentence before a capitalised word
    assert_eq!(split("Bob lives on Main St. Dana works at Acme."), vec!["Bob lives on Main St.", "Dana works at Acme."]);
    assert_eq!(split("Bob lives on Main St. in Austin."), vec!["Bob lives on Main St. in Austin."]);
    // "St. Louis" is a saint: never split after it
    assert_eq!(split("Bob moved to St. Louis last year."), vec!["Bob moved to St. Louis last year."]);
    assert_eq!(split("St. Mary's hired Bob."), vec!["St. Mary's hired Bob."]);
    // titles still never split
    assert_eq!(split("Dr. Smith works at Acme."), vec!["Dr. Smith works at Acme."]);
}

// ------------------------------------------------------------------- O6

// obligation: SPEC-7.1-O6
#[test]
fn modal_intent_and_conditional_relations_are_hedged() {
    // (sentence, subject, object, other mentions, hedged?)
    let cases: &[(&str, &str, &str, &[&str], bool)] = &[
        ("Bob might join Acme.", "Bob", "Acme", &[], true),
        ("Bob may work at Acme.", "Bob", "Acme", &[], true),
        ("Bob will join Acme next month.", "Bob", "Acme", &[], true),
        ("Bob plans to join Acme.", "Bob", "Acme", &[], true),
        ("Bob is considering Acme.", "Bob", "Acme", &[], true),
        ("Bob applied to Acme.", "Bob", "Acme", &[], true),
        ("Bob probably works at Acme.", "Bob", "Acme", &[], true),
        ("If Bob joins Acme, he moves to Austin.", "Bob", "Acme", &[], true),
        ("Dana asked whether Bob works at Acme.", "Bob", "Acme", &[], true),
        // asserted
        ("Bob works at Acme.", "Bob", "Acme", &[], false),
        ("Bob joined Acme in May.", "Bob", "Acme", &[], false),
        ("In May Bob joined Acme.", "Bob", "Acme", &[], false),
        // the hedge belongs to another clause or entity
        ("Bob, who might retire, works at Acme.", "Bob", "Acme", &[], false),
        ("Bob wants to leave Initech for Acme.", "Bob", "Acme", &["Initech"], false),
        ("Bob might quit; he works at Acme.", "Bob", "Acme", &[], false),
        ("Dana will join Initech, but Bob works at Acme.", "Bob", "Acme", &["Dana", "Initech"], false),
        ("Bob works at Acme if I recall correctly.", "Bob", "Acme", &[], false),
    ];
    let n = NegEx::default();
    let wrong: Vec<&str> = cases
        .iter()
        .filter(|(s, subj, obj, others, gold)| {
            let o = s.find(obj).unwrap();
            let os: Vec<(usize, usize)> = others.iter().map(|x| { let i = s.find(x).unwrap(); (i, i + x.len()) }).collect();
            n.hedged_with(s, (o, o + obj.len()), s.find(subj).unwrap(), &os) != *gold
        })
        .map(|c| c.0)
        .collect();
    assert!(wrong.is_empty(), "hedging wrong on: {wrong:#?}");
}

// obligation: SPEC-7.1-O6
#[test]
fn hedged_relations_produce_no_fact_and_negation_wins() {
    let mut x = GlinerExtractor::new(works(), NameLinker);
    assert!(x.extract(&ep("h", "Bob might join Acme.")).is_empty());
    assert_eq!(x.rejected().last().unwrap().reason, RejectReason::Hedged);
    // negated and hedged: reported as Negated
    assert!(x.extract(&ep("n", "Bob might not work at Acme.")).is_empty());
    assert_eq!(x.rejected().last().unwrap().reason, RejectReason::Negated);
    // the hedge on Dana's clause leaves Bob's fact standing
    let f = x.extract(&ep("m", "Dana will join Initech, but Bob works at Acme."));
    assert_eq!(triples(&f), vec![("Bob", "works_at", "Acme")]);
}
