//! Entity linking: turn a mention ("Acme Corp", "acme") into a canonical
//! entity id ("org:acme") or say it could not be linked.
//!
//! [`TableLinker`] is a deterministic alias table, used in tests and for
//! small closed vocabularies. [`GlinkerLinker`] (feature `glinker`) runs
//! glinker-rs, the Rust port of GLinker, in-process: the spans come from
//! gline-rs, and glinker-rs's L2 candidate search plus L3/L4 GLiNER linking
//! only choose the entity (`Executor::link_with_mentions`, which skips L1).

use super::pipeline::Mention;
use std::collections::HashMap;

/// The outcome of linking one mention.
#[derive(Debug, Clone, PartialEq)]
pub enum Link {
    Canonical { id: String, score: f32 },
    Unlinked,
}

pub trait EntityLinker {
    /// Link every mention found in `sentence`. Returns one [`Link`] per
    /// mention, in the same order.
    fn link(&mut self, sentence: &str, mentions: &[Mention]) -> Vec<Link>;
}

impl<L: EntityLinker + ?Sized> EntityLinker for &mut L {
    fn link(&mut self, sentence: &str, mentions: &[Mention]) -> Vec<Link> {
        (**self).link(sentence, mentions)
    }
}

impl EntityLinker for Box<dyn EntityLinker> {
    fn link(&mut self, sentence: &str, mentions: &[Mention]) -> Vec<Link> {
        (**self).link(sentence, mentions)
    }
}

/// The linker for an open vocabulary with no entity database: an entity's
/// canonical id is its name as written, with whitespace collapsed, a
/// trailing possessive ("Acme's") and trailing punctuation removed. Free-text
/// facts then join line-protocol facts that name the same entity
/// ("Alice --works_at--> Acme"); spelling variants are reconciled later with
/// alias edges (`lemmalog_canonicalize`). A mention with no letters or
/// digits does not link.
#[derive(Debug, Clone, Copy, Default)]
pub struct NameLinker;

impl NameLinker {
    /// The canonical id for a name, or `None` if nothing nameable is left.
    pub fn canonical(name: &str) -> Option<String> {
        let mut s = name.split_whitespace().collect::<Vec<_>>().join(" ");
        loop {
            let t = s.trim_end_matches(|c: char| matches!(c, '.' | ',' | ';' | ':' | '!' | '?' | '"' | ')'));
            let t = t.strip_suffix("'s").or_else(|| t.strip_suffix("\u{2019}s")).unwrap_or(t);
            if t.len() == s.len() {
                break;
            }
            s = t.trim_end().to_string();
        }
        let s = s.trim_start_matches(|c: char| matches!(c, '"' | '(')).to_string();
        s.chars().any(char::is_alphanumeric).then_some(s)
    }
}

impl EntityLinker for NameLinker {
    fn link(&mut self, _sentence: &str, mentions: &[Mention]) -> Vec<Link> {
        mentions
            .iter()
            .map(|m| match Self::canonical(&m.text) {
                Some(id) => Link::Canonical { id, score: 1.0 },
                None => Link::Unlinked,
            })
            .collect()
    }
}

/// Case-insensitive alias table: `alias -> [(id, optional required label)]`.
/// An alias that more than one entity claims (for the mention's label) is
/// ambiguous and does not link: the table only answers when the name alone
/// settles the entity.
#[derive(Debug, Clone, Default)]
pub struct TableLinker {
    aliases: HashMap<String, Vec<(String, Option<String>)>>,
}

impl TableLinker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `id` under each alias. If `label` is set, only mentions of
    /// that GLiNER label link to it.
    pub fn entity(mut self, id: &str, label: Option<&str>, aliases: &[&str]) -> Self {
        for a in aliases {
            let v = self.aliases.entry(a.to_lowercase()).or_default();
            let e = (id.to_string(), label.map(str::to_string));
            if !v.contains(&e) {
                v.push(e);
            }
        }
        self
    }

    /// Number of distinct aliases.
    pub fn len(&self) -> usize {
        self.aliases.len()
    }

    pub fn is_empty(&self) -> bool {
        self.aliases.is_empty()
    }

    /// Build the table from a glinker-rs entities file (JSON lines with
    /// `entity_id`, `label` and `aliases`), so exact names can be linked
    /// before the model is asked. Labels are not required to match.
    #[cfg(feature = "glinker")]
    pub fn from_glinker_entities(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let mut t = TableLinker::new();
        for (n, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
            let v: serde_json::Value = serde_json::from_str(line).map_err(|e| format!("{path}:{}: {e}", n + 1))?;
            let id = v["entity_id"].as_str().ok_or_else(|| format!("{path}:{}: no entity_id", n + 1))?;
            let mut names: Vec<&str> = v["aliases"].as_array().into_iter().flatten().filter_map(|a| a.as_str()).collect();
            if let Some(l) = v["label"].as_str() {
                names.push(l);
            }
            t = t.entity(id, None, &names);
        }
        Ok(t)
    }
}

impl EntityLinker for TableLinker {
    fn link(&mut self, _sentence: &str, mentions: &[Mention]) -> Vec<Link> {
        mentions
            .iter()
            .map(|m| {
                let ids: Vec<&String> = self
                    .aliases
                    .get(&m.text.to_lowercase())
                    .into_iter()
                    .flatten()
                    .filter(|(_, l)| l.as_ref().map_or(true, |l| *l == m.label))
                    .map(|(id, _)| id)
                    .collect();
                match ids.as_slice() {
                    [id] => Link::Canonical { id: (*id).clone(), score: 1.0 },
                    [first, rest @ ..] if rest.iter().all(|i| i == first) => {
                        Link::Canonical { id: (*first).clone(), score: 1.0 }
                    }
                    _ => Link::Unlinked,
                }
            })
            .collect()
    }
}

/// Link with `first`; mentions it leaves unlinked go to `then`, in one call.
/// Typical use: an exact, unambiguous alias table first, a model linker
/// (glinker-rs) for everything else.
pub struct CascadeLinker<A: EntityLinker, B: EntityLinker> {
    pub first: A,
    pub then: B,
}

impl<A: EntityLinker, B: EntityLinker> CascadeLinker<A, B> {
    pub fn new(first: A, then: B) -> Self {
        CascadeLinker { first, then }
    }
}

impl<A: EntityLinker, B: EntityLinker> EntityLinker for CascadeLinker<A, B> {
    fn link(&mut self, sentence: &str, mentions: &[Mention]) -> Vec<Link> {
        let mut out = self.first.link(sentence, mentions);
        let rest: Vec<usize> = (0..mentions.len()).filter(|&i| !matches!(out.get(i), Some(Link::Canonical { .. }))).collect();
        if rest.is_empty() {
            return out;
        }
        out.resize(mentions.len(), Link::Unlinked);
        let ms: Vec<Mention> = rest.iter().map(|&i| mentions[i].clone()).collect();
        for (i, l) in rest.into_iter().zip(self.then.link(sentence, &ms)) {
            out[i] = l;
        }
        out
    }
}

/// Convert a byte offset into a char offset (glinker-rs, like Python GLinker,
/// uses char offsets; gline-rs uses bytes).
pub fn byte_to_char(s: &str, byte: usize) -> usize {
    s.char_indices().take_while(|(i, _)| *i < byte).count()
}

/// Turn glinker-rs L0 output for one sentence into one [`Link`] per mention.
/// Matching is by exact char span; a mention with no linked entity, or one
/// below `min_confidence`, is [`Link::Unlinked`].
#[cfg(feature = "glinker")]
pub fn links_from_l0(
    sentence: &str,
    mentions: &[Mention],
    found: &[glinker_rs::L0Entity],
    min_confidence: f32,
) -> Vec<Link> {
    mentions
        .iter()
        .map(|m| {
            let (s, e) = (byte_to_char(sentence, m.start), byte_to_char(sentence, m.end));
            found
                .iter()
                .find(|x| x.mention_start == s && x.mention_end == e)
                .and_then(|x| x.linked_entity.as_ref().filter(|_| x.is_linked))
                .filter(|le| le.entity_id != "unknown" && !le.entity_id.is_empty())
                .filter(|le| le.confidence >= min_confidence)
                .map(|le| Link::Canonical { id: le.entity_id.clone(), score: le.confidence })
                .unwrap_or(Link::Unlinked)
        })
        .collect()
}

/// In-process GLinker: a glinker-rs [`Executor`](glinker_rs::Executor) built
/// from a pipeline config (`pipeline.json`: entities file, L2 search, L3 model
/// dir, optional L4 reranker). No sidecar, no HTTP.
#[cfg(feature = "glinker")]
pub struct GlinkerLinker {
    executor: glinker_rs::Executor,
    /// Links below this confidence are treated as [`Link::Unlinked`].
    pub min_confidence: f32,
    pub failures: usize,
}

#[cfg(feature = "glinker")]
impl GlinkerLinker {
    pub fn new(executor: glinker_rs::Executor) -> Self {
        GlinkerLinker { executor, min_confidence: 0.5, failures: 0 }
    }

    /// Load the executor from a glinker-rs pipeline config file.
    pub fn from_config_file(path: &str) -> Result<Self, String> {
        glinker_rs::Executor::from_config_file(path).map(Self::new).map_err(|e| e.to_string())
    }

    pub fn executor_mut(&mut self) -> &mut glinker_rs::Executor {
        &mut self.executor
    }
}

#[cfg(feature = "glinker")]
impl EntityLinker for GlinkerLinker {
    fn link(&mut self, sentence: &str, mentions: &[Mention]) -> Vec<Link> {
        if mentions.is_empty() {
            return Vec::new();
        }
        let gm: Vec<glinker_rs::Mention> = mentions
            .iter()
            .map(|m| {
                let (s, e) = (byte_to_char(sentence, m.start), byte_to_char(sentence, m.end));
                glinker_rs::Mention {
                    text: m.text.clone(),
                    label: Some(m.label.clone()),
                    start: s,
                    end: e,
                    // never panic on a bad span: the pipeline rejects those
                    // first, but the linker is public
                    left_context: sentence.get(..m.start).unwrap_or("").to_string(),
                    right_context: sentence.get(m.end..).unwrap_or("").to_string(),
                }
            })
            .collect();
        match self.executor.link_with_mentions(&[sentence.to_string()], vec![gm]) {
            Ok(out) => {
                let found = out.entities.first().map(Vec::as_slice).unwrap_or(&[]);
                links_from_l0(sentence, mentions, found, self.min_confidence)
            }
            Err(_) => {
                // degraded: nothing links, so nothing is asserted
                self.failures += 1;
                vec![Link::Unlinked; mentions.len()]
            }
        }
    }
}

#[cfg(all(test, feature = "glinker"))]
mod tests {
    use super::*;
    use glinker_rs::{L0Entity, LinkedEntity, Stage};

    fn l0(start: usize, end: usize, id: Option<(&str, f32)>) -> L0Entity {
        L0Entity {
            mention_text: String::new(),
            label: None,
            mention_start: start,
            mention_end: end,
            left_context: String::new(),
            right_context: String::new(),
            candidates: vec![],
            num_candidates: 0,
            linked_entity: id.map(|(i, c)| LinkedEntity {
                entity_id: i.into(),
                label: i.into(),
                confidence: c,
                start,
                end,
                matched_text: String::new(),
            }),
            is_linked: id.is_some(),
            candidate_scores: vec![],
            pipeline_stage: Stage::L3Linked,
        }
    }

    fn m(text: &str, s: &str) -> Mention {
        let b = s.find(text).unwrap();
        Mention { text: text.into(), label: "x".into(), start: b, end: b + text.len(), score: 1.0 }
    }

    #[test]
    fn maps_char_spans_and_thresholds() {
        // "é" is two bytes: byte offsets and char offsets differ after it
        let s = "José works at Acme and Globex.";
        let ms = [m("José", s), m("Acme", s), m("Globex", s)];
        let found = [l0(0, 4, Some(("person:jose", 0.9))), l0(14, 18, Some(("org:acme", 0.8))), l0(23, 29, Some(("org:globex", 0.2)))];
        assert_eq!(
            links_from_l0(s, &ms, &found, 0.5),
            vec![
                Link::Canonical { id: "person:jose".into(), score: 0.9 },
                Link::Canonical { id: "org:acme".into(), score: 0.8 },
                Link::Unlinked,
            ]
        );
    }

    #[test]
    fn unknown_or_missing_is_unlinked() {
        let s = "Bob works at Acme.";
        let ms = [m("Bob", s), m("Acme", s)];
        let found = [l0(0, 3, Some(("unknown", 0.99)))];
        assert_eq!(links_from_l0(s, &ms, &found, 0.0), vec![Link::Unlinked, Link::Unlinked]);
    }
}
