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

/// Case-insensitive alias table: `alias -> (id, optional required label)`.
#[derive(Debug, Clone, Default)]
pub struct TableLinker {
    aliases: HashMap<String, (String, Option<String>)>,
}

impl TableLinker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `id` under each alias. If `label` is set, only mentions of
    /// that GLiNER label link to it.
    pub fn entity(mut self, id: &str, label: Option<&str>, aliases: &[&str]) -> Self {
        for a in aliases {
            self.aliases
                .insert(a.to_lowercase(), (id.to_string(), label.map(str::to_string)));
        }
        self
    }
}

impl EntityLinker for TableLinker {
    fn link(&mut self, _sentence: &str, mentions: &[Mention]) -> Vec<Link> {
        mentions
            .iter()
            .map(|m| match self.aliases.get(&m.text.to_lowercase()) {
                Some((id, None)) => Link::Canonical { id: id.clone(), score: 1.0 },
                Some((id, Some(l))) if *l == m.label => {
                    Link::Canonical { id: id.clone(), score: 1.0 }
                }
                _ => Link::Unlinked,
            })
            .collect()
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
                    left_context: sentence[..m.start].to_string(),
                    right_context: sentence[m.end..].to_string(),
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
