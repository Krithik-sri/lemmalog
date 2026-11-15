//! Entity linking: turn a mention ("Acme Corp", "acme") into a canonical
//! entity id ("org:acme") or say it could not be linked.
//!
//! [`TableLinker`] is a deterministic alias table, used in tests and for
//! small closed vocabularies. [`GlinkerHttpLinker`] (feature `glinker`) sends
//! the mentions to a GLinker sidecar (`scripts/glinker_sidecar.py`), which runs
//! GLinker's linking pipeline in external-entities mode: the spans come from
//! gline-rs, GLinker only chooses the entity.

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

/// Convert a byte offset into a char offset (Python indexes strings by code
/// point; Rust and gline-rs use bytes).
pub fn byte_to_char(s: &str, byte: usize) -> usize {
    s.char_indices().take_while(|(i, _)| *i < byte).count()
}

/// GLinker sidecar client.
///
/// Request:  `POST {base}/link`
/// `{"texts": [sentence], "entities": [[{"text", "start", "end", "label"}]]}`
/// with `start`/`end` in **characters**.
/// Response: `{"links": [[{"entity_id": "Q1" | "unknown" | null, "confidence": 0.93}]]}`.
#[cfg(feature = "glinker")]
pub struct GlinkerHttpLinker {
    base: String,
    /// Links below this confidence are treated as [`Link::Unlinked`].
    pub min_confidence: f32,
    pub failures: usize,
}

#[cfg(feature = "glinker")]
impl GlinkerHttpLinker {
    pub fn new(base: &str) -> Self {
        GlinkerHttpLinker {
            base: base.trim_end_matches('/').to_string(),
            min_confidence: 0.5,
            failures: 0,
        }
    }

    fn request(&self, sentence: &str, mentions: &[Mention]) -> Result<Vec<Link>, String> {
        use serde_json::{json, Value};
        let ents: Vec<Value> = mentions
            .iter()
            .map(|m| {
                json!({"text": m.text, "label": m.label,
                       "start": byte_to_char(sentence, m.start),
                       "end": byte_to_char(sentence, m.end)})
            })
            .collect();
        let body = json!({"texts": [sentence], "entities": [ents]});
        let text = ureq::post(&format!("{}/link", self.base))
            .set("Content-Type", "application/json")
            .send_string(&body.to_string())
            .map_err(|e| e.to_string())?
            .into_string()
            .map_err(|e| e.to_string())?;
        let resp: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        let row = resp["links"][0].as_array().ok_or("missing links[0]")?;
        if row.len() != mentions.len() {
            return Err(format!("{} links for {} mentions", row.len(), mentions.len()));
        }
        Ok(row
            .iter()
            .map(|l| {
                let id = l["entity_id"].as_str().unwrap_or("unknown");
                let score = l["confidence"].as_f64().unwrap_or(0.0) as f32;
                if id == "unknown" || id.is_empty() || score < self.min_confidence {
                    Link::Unlinked
                } else {
                    Link::Canonical { id: id.to_string(), score }
                }
            })
            .collect())
    }
}

#[cfg(feature = "glinker")]
impl EntityLinker for GlinkerHttpLinker {
    fn link(&mut self, sentence: &str, mentions: &[Mention]) -> Vec<Link> {
        if mentions.is_empty() {
            return Vec::new();
        }
        match self.request(sentence, mentions) {
            Ok(v) => v,
            Err(_) => {
                // degraded: nothing links, so nothing is asserted
                self.failures += 1;
                vec![Link::Unlinked; mentions.len()]
            }
        }
    }
}
