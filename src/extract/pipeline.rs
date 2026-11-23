//! `GlinerExtractor`: extraction without an LLM.
//!
//! ```text
//! episode text ── sentences ──▶ RelationBackend (gline-rs: NER + relations)
//!                                  │ mentions + raw relations, byte spans
//!                                  ▼
//!                               NegEx ── negated? ──▶ rejected (Negated)
//!                                  ▼
//!                               EntityLinker (GLinker) ── unlinked? ──▶ rejected
//!                                  ▼
//!                               CandidateFact(subj_id, pred, obj_id, confidence)
//! ```
//!
//! The backend and the linker are traits, so the gate's tests run with
//! [`FixtureBackend`] and [`TableLinker`](super::link::TableLinker) and need no
//! ONNX model or sidecar. The real backend is
//! [`GlineBackend`](super::gline::GlineBackend) (feature `gliner`).

use super::link::{EntityLinker, Link};
use super::negex::NegEx;
use crate::agent::{entity_token_problem, CandidateFact, Episode, Extractor};
use std::collections::{BTreeMap, HashMap};

/// An entity mention: text, GLiNER label and byte span in its sentence.
#[derive(Debug, Clone, PartialEq)]
pub struct Mention {
    pub text: String,
    pub label: String,
    pub start: usize,
    pub end: usize,
    pub score: f32,
}

/// A relation as the model reports it, before negation and linking.
#[derive(Debug, Clone, PartialEq)]
pub struct RawRelation {
    pub subject: Mention,
    /// The model's relation label, e.g. "works at".
    pub label: String,
    pub object: Mention,
    pub score: f32,
}

/// What the backend found in one sentence.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SentenceAnalysis {
    pub mentions: Vec<Mention>,
    pub relations: Vec<RawRelation>,
    /// Relations the model reported whose subject or object is not one of
    /// this sentence's mentions: (subject text, label, object text).
    /// Reported as [`RejectReason::Unresolved`], never dropped silently.
    pub unresolved: Vec<(String, String, String)>,
}

/// A relation as a model reports it before it is tied to mention spans:
/// GLiNER multitask gives the subject as text only, and the object as text
/// plus a span that may be relative to the model's prompt, not the sentence.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelRelation {
    pub subject: String,
    pub label: String,
    pub object: String,
    /// The object's byte span in the sentence, if known. Ignored unless it
    /// lies in the sentence and covers exactly `object`.
    pub object_span: Option<(usize, usize)>,
    pub score: f32,
}

/// Tie model relations to this sentence's mentions. The object is the
/// mention at `object_span` when that span is valid, otherwise the mention
/// with the object's text nearest the subject; the subject is the mention
/// with the subject's text nearest the object (never the object itself).
/// A relation with no matching subject or object mention is unresolved.
pub fn resolve_relations(sentence: &str, mentions: Vec<Mention>, rels: &[ModelRelation]) -> SentenceAnalysis {
    let mut out = SentenceAnalysis { mentions, ..Default::default() };
    let dist = |a: &Mention, b: &Mention| (a.start as i64 - b.start as i64).abs();
    for r in rels {
        let hinted = r
            .object_span
            .filter(|&(a, b)| sentence.get(a..b) == Some(r.object.as_str()))
            .and_then(|(a, b)| out.mentions.iter().find(|m| m.start == a && m.end == b));
        let objects: Vec<&Mention> = match hinted {
            Some(m) => vec![m],
            None => out.mentions.iter().filter(|m| m.text == r.object).collect(),
        };
        let best = objects
            .iter()
            .flat_map(|o| {
                out.mentions
                    .iter()
                    .filter(|s| s.text == r.subject && (s.start, s.end) != (o.start, o.end))
                    .map(move |s| (dist(s, o), s.start, o.start, s, *o))
            })
            .min_by_key(|(d, ss, os, _, _)| (*d, *ss, *os));
        match best {
            Some((_, _, _, s, o)) => out.relations.push(RawRelation {
                subject: s.clone(),
                label: r.label.clone(),
                object: o.clone(),
                score: r.score,
            }),
            None => out.unresolved.push((r.subject.clone(), r.label.clone(), r.object.clone())),
        }
    }
    out
}

/// NER + relation extraction over a batch of sentences.
pub trait RelationBackend {
    /// One analysis per input sentence, same order.
    fn analyse(&mut self, sentences: &[String]) -> Result<Vec<SentenceAnalysis>, String>;
}

impl<B: RelationBackend + ?Sized> RelationBackend for &mut B {
    fn analyse(&mut self, sentences: &[String]) -> Result<Vec<SentenceAnalysis>, String> {
        (**self).analyse(sentences)
    }
}

/// Remembers each sentence's analysis, so re-running extraction over the
/// same text (a threshold sweep, a second linker) calls the model once.
pub struct CachedBackend<B: RelationBackend> {
    pub inner: B,
    cache: HashMap<String, SentenceAnalysis>,
    /// Sentences sent to `inner`.
    pub misses: usize,
}

impl<B: RelationBackend> CachedBackend<B> {
    pub fn new(inner: B) -> Self {
        CachedBackend { inner, cache: HashMap::new(), misses: 0 }
    }

    /// Forget every cached analysis (after changing how `inner` analyses).
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// The cached model output for a sentence, if it was analysed.
    pub fn analysis(&self, sentence: &str) -> Option<&SentenceAnalysis> {
        self.cache.get(sentence)
    }
}

impl<B: RelationBackend> RelationBackend for CachedBackend<B> {
    fn analyse(&mut self, sentences: &[String]) -> Result<Vec<SentenceAnalysis>, String> {
        let mut todo: Vec<String> = Vec::new();
        for s in sentences {
            if !self.cache.contains_key(s) && !todo.contains(s) {
                todo.push(s.clone());
            }
        }
        if !todo.is_empty() {
            let got = self.inner.analyse(&todo)?;
            if got.len() != todo.len() {
                return Err("backend returned the wrong number of analyses".into());
            }
            self.misses += todo.len();
            self.cache.extend(todo.into_iter().zip(got));
        }
        Ok(sentences.iter().map(|s| self.cache[s].clone()).collect())
    }
}

/// Why a raw relation did not become a fact.
#[derive(Debug, Clone, PartialEq)]
pub enum RejectReason {
    Negated,
    Unlinked(String),
    BadEntity(String),
    LowScore,
    UnknownRelation,
    /// The model named a subject or object that is not a mention of the
    /// sentence it was found in.
    Unresolved,
    /// A mention span is not a valid byte range of its sentence.
    BadSpan,
    /// The same sentence gave this fact in the other direction with stronger
    /// evidence (see SPEC-7.1-O4).
    Reversed,
    /// Subject and object linked to the same entity ("Bob reports to
    /// Robert" with both names linking to `person:bob`).
    SelfLoop,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Rejection {
    pub episode: String,
    pub subject: String,
    pub relation: String,
    pub object: String,
    pub reason: RejectReason,
}

/// Map from the model's relation labels to lemmalog predicates, matching
/// the vocabulary of [`EXTRACTION_PROMPT`](crate::agent::EXTRACTION_PROMPT).
/// Each entry: (GLiNER label, predicate, subject labels, object labels,
/// inverse). `inverse` swaps subject and object: lemmalog reads
/// `X --manager--> Y` as "X reports to Y" (`reports_to(X,Y) :-
/// edge(X,"manager",Y,..)`), so "Dana is the manager of Bob" becomes
/// `Bob --manager--> Dana`.
pub const DEFAULT_RELATIONS: &[(&str, &str, &[&str], &[&str], bool)] = &[
    ("works at", "works_at", &["person"], &["organization"], false),
    ("reports to", "manager", &["person"], &["person"], false),
    ("manager of", "manager", &["person"], &["person"], true),
    ("likes", "likes", &["person"], &["thing", "food", "activity", "organization"], false),
    ("job title", "job_title", &["person"], &["job title"], false),
    ("member of", "member_of", &["person"], &["organization", "team"], false),
    ("located in", "located_in", &["person", "organization"], &["location"], false),
];

/// The GLiNER entity labels the default relation set needs.
pub fn default_entity_labels() -> Vec<String> {
    let mut v: Vec<String> = DEFAULT_RELATIONS
        .iter()
        .flat_map(|(_, _, s, o, _)| s.iter().chain(o.iter()))
        .map(|s| s.to_string())
        .collect();
    v.sort();
    v.dedup();
    v
}

/// Words that end in "." without ending a sentence when the next word is
/// capitalised (titles, initials-style abbreviations).
const TITLES: &[&str] = &["dr", "mr", "mrs", "ms", "prof", "st", "sr", "jr", "rev", "gen", "col", "capt", "lt", "sgt", "hon"];
/// Abbreviations that end a sentence only if the next word is capitalised.
const ABBREVS: &[&str] = &[
    "inc", "corp", "ltd", "co", "llc", "plc", "bros", "dept", "univ", "vs", "etc", "e.g", "i.e",
    "approx", "no", "vol", "fig", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec", "mon", "tue", "wed", "thu", "fri", "sat", "sun", "u.s", "u.k", "a.m", "p.m",
];

/// Split into sentences on `.`, `!`, `?` and newlines, keeping the
/// terminator with its sentence. Empty pieces are dropped. A run of
/// terminators ("...", "?!") is one boundary. A "." does not split inside a
/// number ("2.5"), after a title ("Dr. Smith"), or after an abbreviation or
/// single letter ("Acme Corp. in Austin", "the U.S. last year") unless the
/// next word starts with a capital letter. An ellipsis followed by a
/// lower-case word does not split either.
pub fn split_sentences(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut cur = String::new();
    let push = |cur: &mut String, out: &mut Vec<String>| {
        if !cur.trim().is_empty() {
            out.push(cur.trim().to_string());
        }
        cur.clear();
    };
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '\n' {
            push(&mut cur, &mut out);
            i += 1;
            continue;
        }
        if !matches!(c, '.' | '!' | '?') {
            cur.push(c);
            i += 1;
            continue;
        }
        // the whole run of terminators
        let mut j = i;
        while j < chars.len() && matches!(chars[j], '.' | '!' | '?') {
            cur.push(chars[j]);
            j += 1;
        }
        let run = j - i;
        let prev = if i > 0 { Some(chars[i - 1]) } else { None };
        let next = chars.get(j).copied();
        let next_word = chars[j..].iter().find(|c| !c.is_whitespace()).copied();
        let next_upper = next_word.map_or(true, |c| c.is_uppercase() || !c.is_alphabetic());
        // the word just before the terminator, lower-cased, keeping inner dots
        let word: String = {
            let w: String = cur[..cur.len() - run]
                .chars()
                .rev()
                .take_while(|c| c.is_alphanumeric() || *c == '.')
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            w.to_lowercase()
        };
        let only_dots = chars[i..j].iter().all(|c| *c == '.');
        let keep = if run == 1 && c == '.' {
            let in_number = prev.map_or(false, |p| p.is_ascii_digit()) && next.map_or(false, |n| n.is_ascii_digit());
            let single_letter = word.len() == 1 && word.chars().all(char::is_alphabetic)
                || word.split('.').all(|p| p.chars().count() == 1 && p.chars().all(char::is_alphabetic));
            in_number
                || next.map_or(false, |n| !n.is_whitespace()) && next.map_or(false, |n| n.is_alphanumeric())
                || TITLES.contains(&word.as_str())
                || (ABBREVS.contains(&word.as_str()) || single_letter) && !next_upper
        } else {
            // "..." before a lower-case word continues the sentence
            only_dots && run > 1 && !next_upper
        };
        if !keep {
            push(&mut cur, &mut out);
        }
        i = j;
    }
    push(&mut cur, &mut out);
    out
}

/// Is the relation's label written between its subject and its object, in
/// that order ("Sam [reports to] Priya", "Priya is the [manager of] Sam")?
/// This is the sentence's own evidence for the relation's direction; the
/// same label read the other way round ("Priya reports to Sam") has none.
pub fn worded_between(sentence: &str, r: &RawRelation) -> bool {
    r.subject.end <= r.object.start
        && sentence
            .get(r.subject.end..r.object.start)
            .map_or(false, |between| between.to_lowercase().contains(&r.label.to_lowercase()))
}

/// Drop a leading title ("Dr. Smith" -> "Smith", "Mrs Lee" -> "Lee") from
/// a mention before linking, keeping the span on the remaining name. A
/// mention that is only a title is returned unchanged.
pub fn strip_title(sentence: &str, m: &Mention) -> Mention {
    let text = sentence.get(m.start..m.end).unwrap_or(&m.text);
    let word: String = text.chars().take_while(|c| c.is_alphabetic()).collect();
    if word.is_empty() || !TITLES.contains(&word.to_lowercase().as_str()) {
        return m.clone();
    }
    let rest = &text[word.len()..];
    let rest_trim = rest.trim_start_matches('.').trim_start();
    if rest_trim.is_empty() || rest_trim.len() == rest.len() {
        // nothing after the title, or no separator ("Drake"): not a title
        return m.clone();
    }
    let cut = text.len() - rest_trim.len();
    Mention { text: rest_trim.to_string(), start: m.start + cut, ..m.clone() }
}

pub struct GlinerExtractor<B: RelationBackend, L: EntityLinker> {
    backend: B,
    linker: L,
    pub negex: NegEx,
    relations: HashMap<String, (String, bool)>,
    /// Relations scoring below this are dropped.
    pub min_score: f32,
    seen: HashMap<String, Vec<CandidateFact>>,
    rejected: Vec<Rejection>,
    calls: usize,
    failures: usize,
    errors: Vec<(String, String)>,
}

impl<B: RelationBackend, L: EntityLinker> GlinerExtractor<B, L> {
    pub fn new(backend: B, linker: L) -> Self {
        GlinerExtractor {
            backend,
            linker,
            negex: NegEx::default(),
            relations: DEFAULT_RELATIONS
                .iter()
                .map(|(l, p, _, _, inv)| (l.to_string(), (p.to_string(), *inv)))
                .collect(),
            min_score: 0.5,
            seen: HashMap::new(),
            rejected: Vec::new(),
            calls: 0,
            failures: 0,
            errors: Vec::new(),
        }
    }

    /// Relations the extractor saw but did not turn into facts.
    pub fn rejected(&self) -> &[Rejection] {
        &self.rejected
    }

    /// Backend failures, as (episode id, error message): each one is an
    /// episode that yielded no facts (a degraded turn).
    pub fn errors(&self) -> &[(String, String)] {
        &self.errors
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    fn reject(&mut self, ep: &Episode, r: &RawRelation, reason: RejectReason) {
        self.rejected.push(Rejection {
            episode: ep.id.clone(),
            subject: r.subject.text.clone(),
            relation: r.label.clone(),
            object: r.object.text.clone(),
            reason,
        });
    }

    /// Is this mention a first-person pronoun ("I", "me", "my", "myself")?
    fn first_person(m: &Mention) -> bool {
        matches!(m.text.to_lowercase().as_str(), "i" | "me" | "my" | "myself")
    }

    /// Link the episode's speaker by name, once per episode: the name with
    /// no sentence context, so a span-scoring linker scores "Dana", not the
    /// word "I" it stands for.
    fn speaker_link(&mut self, ep: &Episode, cache: &mut Option<Link>) -> Link {
        if let Some(l) = cache {
            return l.clone();
        }
        let l = match &ep.speaker {
            Some(name) => {
                let m = Mention { text: name.clone(), label: "person".into(), start: 0, end: name.len(), score: 1.0 };
                self.linker.link(name, &[m]).into_iter().next().unwrap_or(Link::Unlinked)
            }
            None => Link::Unlinked,
        };
        *cache = Some(l.clone());
        l
    }

    fn run(&mut self, ep: &Episode) -> Vec<CandidateFact> {
        let sentences = split_sentences(&ep.text);
        if sentences.is_empty() {
            return Vec::new();
        }
        let analyses = match self.backend.analyse(&sentences) {
            Ok(a) if a.len() == sentences.len() => a,
            Ok(a) => {
                self.failures += 1;
                let msg = format!("backend returned {} analyses for {} sentences", a.len(), sentences.len());
                self.errors.push((ep.id.clone(), msg));
                return Vec::new(); // degraded turn: no facts, no poison
            }
            Err(e) => {
                self.failures += 1;
                self.errors.push((ep.id.clone(), e));
                return Vec::new(); // degraded turn: no facts, no poison
            }
        };
        // (subj, pred, obj) -> best confidence; BTreeMap keeps output order stable
        let mut facts: BTreeMap<(String, String, String), f64> = BTreeMap::new();
        let mut speaker: Option<Link> = None;
        for (sentence, a) in sentences.iter().zip(analyses) {
            for (subject, relation, object) in &a.unresolved {
                self.rejected.push(Rejection {
                    episode: ep.id.clone(),
                    subject: subject.clone(),
                    relation: relation.clone(),
                    object: object.clone(),
                    reason: RejectReason::Unresolved,
                });
            }
            // a non-empty byte range of this sentence (`get` checks bounds
            // and char boundaries, and never panics)
            let valid = |m: &Mention| m.start < m.end && sentence.get(m.start..m.end).is_some();
            // NegEx reads the span and the linker reads the text: make them
            // the same, so negation and linking judge the same words
            let at_span = |m: &Mention| Mention { text: sentence[m.start..m.end].to_string(), ..m.clone() };
            // keep the relations that survive the cheap checks, then link
            // their mentions in one linker call per sentence
            let mut kept: Vec<(RawRelation, String, bool)> = Vec::new();
            for r in &a.relations {
                let Some((pred, inverse)) = self.relations.get(&r.label).cloned() else {
                    self.reject(ep, r, RejectReason::UnknownRelation);
                    continue;
                };
                if !valid(&r.subject) || !valid(&r.object) {
                    self.reject(ep, r, RejectReason::BadSpan);
                    continue;
                }
                // written so that a NaN score fails the check: `NaN < x` is
                // false, so `r.score < self.min_score` would let NaN through
                if !(r.score >= self.min_score) || !r.score.is_finite() {
                    self.reject(ep, r, RejectReason::LowScore);
                    continue;
                }
                let r = &RawRelation { subject: at_span(&r.subject), object: at_span(&r.object), ..r.clone() };
                let span = |m: &Mention| (m.start, m.end);
                let others: Vec<(usize, usize)> = a
                    .mentions
                    .iter()
                    .filter(|m| valid(m))
                    .map(span)
                    .filter(|s| *s != span(&r.subject) && *s != span(&r.object))
                    .collect();
                if self.negex.negated_with(sentence, span(&r.object), r.subject.start, &others) {
                    self.reject(ep, r, RejectReason::Negated);
                    continue;
                }
                kept.push((r.clone(), pred, inverse));
            }
            if kept.is_empty() {
                continue;
            }
            // first-person mentions link as the speaker (by name); the rest go
            // to the linker in one call, titles stripped ("Dr. Smith" -> "Smith")
            let all: Vec<&Mention> = kept.iter().flat_map(|(r, _, _)| [&r.subject, &r.object]).collect();
            let mut links: Vec<Link> = vec![Link::Unlinked; all.len()];
            let mut ask: Vec<(usize, Mention)> = Vec::new();
            for (i, m) in all.iter().enumerate() {
                if Self::first_person(m) {
                    links[i] = self.speaker_link(ep, &mut speaker);
                } else {
                    ask.push((i, strip_title(sentence, m)));
                }
            }
            if !ask.is_empty() {
                let ms: Vec<Mention> = ask.iter().map(|(_, m)| m.clone()).collect();
                for ((i, _), l) in ask.iter().zip(self.linker.link(sentence, &ms)) {
                    links[*i] = l;
                }
            }
            // this sentence's facts: triple -> (cue, relation score, best
            // confidence, relation)
            let mut found: BTreeMap<(String, String, String), (bool, f32, f64, RawRelation)> = BTreeMap::new();
            for (i, (r, pred, inverse)) in kept.iter().enumerate() {
                let (s, o) = (links.get(2 * i), links.get(2 * i + 1));
                let (Some(Link::Canonical { id: sid, score: ss }), Some(Link::Canonical { id: oid, score: os })) = (s, o) else {
                    let which = match s {
                        Some(Link::Canonical { .. }) => r.object.text.clone(),
                        _ => r.subject.text.clone(),
                    };
                    self.reject(ep, r, RejectReason::Unlinked(which));
                    continue;
                };
                if let Some(p) = entity_token_problem(sid).or_else(|| entity_token_problem(oid)) {
                    self.reject(ep, r, RejectReason::BadEntity(p));
                    continue;
                }
                // `f32::min` ignores a NaN argument, so a NaN link score would
                // silently drop out of the confidence: treat it as no link
                if !ss.is_finite() || !os.is_finite() {
                    self.reject(ep, r, RejectReason::LowScore);
                    continue;
                }
                if sid == oid {
                    self.reject(ep, r, RejectReason::SelfLoop);
                    continue;
                }
                let (sid, oid) = if *inverse { (oid, sid) } else { (sid, oid) };
                let conf = r.score.min(*ss).min(*os) as f64;
                let cue = worded_between(sentence, r);
                let key = (sid.clone(), pred.clone(), oid.clone());
                match found.get_mut(&key) {
                    Some(e) => {
                        e.2 = e.2.max(conf);
                        if (cue, r.score) > (e.0, e.1) {
                            (e.0, e.1, e.3) = (cue, r.score, r.clone());
                        }
                    }
                    None => {
                        found.insert(key, (cue, r.score, conf, r.clone()));
                    }
                }
            }
            // one sentence never asserts a fact in both directions: keep the
            // one worded in the sentence, then the higher relation score, then
            // the smaller triple (a strict order, so exactly one survives).
            // Link scores are not compared: both directions link the same two
            // mentions, so they say nothing about direction.
            let losers: Vec<(String, String, String)> = found
                .iter()
                .filter(|((a, p, b), (c, f, _, _))| {
                    a != b
                        && found.get(&(b.clone(), p.clone(), a.clone())).map_or(false, |(c2, f2, _, _)| {
                            let me = (*c, *f);
                            let other = (*c2, *f2);
                            me < other || (me == other && (a, b) > (b, a))
                        })
                })
                .map(|(k, _)| k.clone())
                .collect();
            for k in &losers {
                let (_, _, _, r) = found.remove(k).unwrap();
                self.reject(ep, &r, RejectReason::Reversed);
            }
            for ((subj, pred, obj), (_, _, conf, _)) in found {
                let e = facts.entry((subj, pred, obj)).or_insert(0.0);
                if conf > *e {
                    *e = conf;
                }
            }
        }
        facts
            .into_iter()
            .map(|((subj, pred, obj), confidence)| CandidateFact { subj, pred, obj, confidence })
            .collect()
    }
}

impl<B: RelationBackend, L: EntityLinker> Extractor for GlinerExtractor<B, L> {
    fn extract(&mut self, episode: &Episode) -> Vec<CandidateFact> {
        if let Some(cached) = self.seen.get(&episode.id) {
            return cached.clone();
        }
        self.calls += 1;
        let failures = self.failures;
        let out = self.run(episode);
        // memoize successes only: a failed turn (backend error) is retried
        // the next time the episode is extracted, instead of staying empty
        if self.failures == failures {
            self.seen.insert(episode.id.clone(), out.clone());
        }
        out
    }

    /// (model calls, failed calls)
    fn stats(&self) -> (usize, usize) {
        (self.calls, self.failures)
    }
}

/// A backend that replays fixed analyses: for tests and offline replays.
/// It finds mentions by text, so fixtures stay readable:
/// `FixtureBackend::new().relation("Bob", "person", "works at", "Acme", "organization", 0.9)`.
#[derive(Debug, Clone, Default)]
pub struct FixtureBackend {
    rules: Vec<(String, String, String, String, String, f32)>,
    /// How many times `analyse` ran.
    pub calls: usize,
}

impl FixtureBackend {
    pub fn new() -> Self {
        Self::default()
    }

    /// Report `subj --label--> obj` in every sentence containing both mentions
    /// (subject before object).
    pub fn relation(mut self, subj: &str, sl: &str, label: &str, obj: &str, ol: &str, score: f32) -> Self {
        self.rules.push((subj.into(), sl.into(), label.into(), obj.into(), ol.into(), score));
        self
    }
}

impl RelationBackend for FixtureBackend {
    fn analyse(&mut self, sentences: &[String]) -> Result<Vec<SentenceAnalysis>, String> {
        self.calls += 1;
        Ok(sentences
            .iter()
            .map(|s| {
                let mut a = SentenceAnalysis::default();
                for (subj, sl, label, obj, ol, score) in &self.rules {
                    let Some(si) = s.find(subj.as_str()) else { continue };
                    let Some(oi) = s[si + subj.len()..].find(obj.as_str()).map(|i| i + si + subj.len()) else {
                        continue;
                    };
                    let sm = Mention { text: subj.clone(), label: sl.clone(), start: si, end: si + subj.len(), score: *score };
                    let om = Mention { text: obj.clone(), label: ol.clone(), start: oi, end: oi + obj.len(), score: *score };
                    a.mentions.push(sm.clone());
                    a.mentions.push(om.clone());
                    a.relations.push(RawRelation { subject: sm, label: label.clone(), object: om, score: *score });
                }
                a
            })
            .collect())
    }
}
