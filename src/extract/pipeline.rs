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
}

/// NER + relation extraction over a batch of sentences.
pub trait RelationBackend {
    /// One analysis per input sentence, same order.
    fn analyse(&mut self, sentences: &[String]) -> Result<Vec<SentenceAnalysis>, String>;
}

/// Why a raw relation did not become a fact.
#[derive(Debug, Clone, PartialEq)]
pub enum RejectReason {
    Negated,
    Unlinked(String),
    BadEntity(String),
    LowScore,
    UnknownRelation,
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
/// Each entry: (GLiNER label, predicate, subject labels, object labels).
pub const DEFAULT_RELATIONS: &[(&str, &str, &[&str], &[&str])] = &[
    ("works at", "works_at", &["person"], &["organization"]),
    ("manager of", "manager", &["person"], &["person"]),
    ("likes", "likes", &["person"], &["thing", "food", "activity", "organization"]),
    ("job title", "job_title", &["person"], &["job title"]),
    ("member of", "member_of", &["person"], &["organization", "team"]),
    ("located in", "located_in", &["person", "organization"], &["location"]),
];

/// The GLiNER entity labels the default relation set needs.
pub fn default_entity_labels() -> Vec<String> {
    let mut v: Vec<String> = DEFAULT_RELATIONS
        .iter()
        .flat_map(|(_, _, s, o)| s.iter().chain(o.iter()))
        .map(|s| s.to_string())
        .collect();
    v.sort();
    v.dedup();
    v
}

/// Split into sentences on `.`, `!`, `?` and newlines, keeping the
/// terminator with its sentence. Empty pieces are dropped.
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        if c == '\n' {
            if !cur.trim().is_empty() {
                out.push(cur.trim().to_string());
            }
            cur.clear();
            continue;
        }
        cur.push(c);
        if matches!(c, '.' | '!' | '?') {
            if !cur.trim().is_empty() {
                out.push(cur.trim().to_string());
            }
            cur.clear();
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

pub struct GlinerExtractor<B: RelationBackend, L: EntityLinker> {
    backend: B,
    linker: L,
    pub negex: NegEx,
    relations: HashMap<String, String>,
    /// Relations scoring below this are dropped.
    pub min_score: f32,
    seen: HashMap<String, Vec<CandidateFact>>,
    rejected: Vec<Rejection>,
    calls: usize,
    failures: usize,
}

impl<B: RelationBackend, L: EntityLinker> GlinerExtractor<B, L> {
    pub fn new(backend: B, linker: L) -> Self {
        GlinerExtractor {
            backend,
            linker,
            negex: NegEx::default(),
            relations: DEFAULT_RELATIONS
                .iter()
                .map(|(l, p, _, _)| (l.to_string(), p.to_string()))
                .collect(),
            min_score: 0.5,
            seen: HashMap::new(),
            rejected: Vec::new(),
            calls: 0,
            failures: 0,
        }
    }

    /// Relations the extractor saw but did not turn into facts.
    pub fn rejected(&self) -> &[Rejection] {
        &self.rejected
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

    /// First-person mentions ("I", "me", "my") stand for the speaker.
    fn resolve_speaker(m: &Mention, ep: &Episode) -> Mention {
        let lower = m.text.to_lowercase();
        match &ep.speaker {
            Some(s) if matches!(lower.as_str(), "i" | "me" | "my" | "myself") => {
                Mention { text: s.clone(), ..m.clone() }
            }
            _ => m.clone(),
        }
    }

    fn run(&mut self, ep: &Episode) -> Vec<CandidateFact> {
        let sentences = split_sentences(&ep.text);
        if sentences.is_empty() {
            return Vec::new();
        }
        let analyses = match self.backend.analyse(&sentences) {
            Ok(a) if a.len() == sentences.len() => a,
            _ => {
                self.failures += 1;
                return Vec::new(); // degraded turn: no facts, no poison
            }
        };
        // (subj, pred, obj) -> best confidence; BTreeMap keeps output order stable
        let mut facts: BTreeMap<(String, String, String), f64> = BTreeMap::new();
        for (sentence, a) in sentences.iter().zip(analyses) {
            // keep the relations that survive the cheap checks, then link
            // their mentions in one linker call per sentence
            let mut kept: Vec<(RawRelation, String)> = Vec::new();
            for r in a.relations {
                let Some(pred) = self.relations.get(&r.label).cloned() else {
                    self.reject(ep, &r, RejectReason::UnknownRelation);
                    continue;
                };
                if r.score < self.min_score {
                    self.reject(ep, &r, RejectReason::LowScore);
                    continue;
                }
                if self.negex.negated(sentence, (r.object.start, r.object.end), r.subject.start)
                {
                    self.reject(ep, &r, RejectReason::Negated);
                    continue;
                }
                kept.push((r, pred));
            }
            if kept.is_empty() {
                continue;
            }
            let mentions: Vec<Mention> = kept
                .iter()
                .flat_map(|(r, _)| [Self::resolve_speaker(&r.subject, ep), r.object.clone()])
                .collect();
            let links = self.linker.link(sentence, &mentions);
            for (i, (r, pred)) in kept.iter().enumerate() {
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
                let conf = r.score.min(*ss).min(*os) as f64;
                let e = facts.entry((sid.clone(), pred.clone(), oid.clone())).or_insert(0.0);
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
        let out = self.run(episode);
        self.seen.insert(episode.id.clone(), out.clone());
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
