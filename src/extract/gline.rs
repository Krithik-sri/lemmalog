//! [`GlineBackend`]: the real [`RelationBackend`], running a GLiNER
//! multitask model (token mode) through gline-rs. NER first, then relation
//! extraction over the NER spans, both on the same ONNX session.
//!
//! Model files are not shipped. Get them from Hugging Face, e.g.
//! `onnx-community/gliner-multitask-large-v0.5`:
//!
//! ```text
//! models/gliner-multitask-large-v0.5/tokenizer.json
//! models/gliner-multitask-large-v0.5/onnx/model.onnx
//! ```
//!
//! Two gline-rs behaviours shape this backend:
//!
//! * gline-rs runs relation extraction on a prompt, `"{PREFIX} {sentence}"`,
//!   and reports object spans against that prompt, not the sentence. The
//!   span is mapped back by [`RELATION_PROMPT_OFFSET`] and used only if it
//!   then covers the object's text exactly; otherwise the object is found by
//!   text ([`resolve_relations`]).
//! * gline-rs builds one relation label per (subject entity, relation) over
//!   the whole batch. The backend therefore runs one sentence per call, so
//!   labels never mix sentences and grow with one sentence's entities only,
//!   capped by [`GlineBackend::max_entities`].

use super::pipeline::{
    default_entity_labels, resolve_relations, Mention, ModelRelation, RelationBackend, SentenceAnalysis,
    DEFAULT_RELATIONS,
};
use gliner::model::input::relation::schema::RelationSchema;
use gliner::model::input::text::TextInput;
use gliner::model::output::decoded::SpanOutput;
use gliner::model::output::relation::Relation;
use gliner::model::params::Parameters;
use gliner::model::pipeline::{relation::RelationPipeline, token::TokenPipeline};
use gliner::text::span::Span;
use gliner::text::splitter::RegexSplitter;
use gliner::text::tokenizer::HFTokenizer;
use orp::model::Model;
use orp::params::RuntimeParameters;

type Ner = TokenPipeline<RegexSplitter, HFTokenizer>;
type Rel = RelationPipeline<'static, RegexSplitter, HFTokenizer>;

/// gline-rs's relation prompt prefix (`PROMPT_PREFIX` in
/// `gline-rs/src/model/input/relation/mod.rs`), private to that crate.
pub const RELATION_PROMPT_PREFIX: &str = "Extract relationships between entities from the text: \n";
/// Bytes between the start of the relation prompt and the sentence:
/// gline-rs formats the prompt as `"{prefix} {text}"`.
pub const RELATION_PROMPT_OFFSET: usize = RELATION_PROMPT_PREFIX.len() + 1;

pub struct GlineBackend {
    // The relation pipeline borrows its schema and `Model::inference` ties the
    // pipeline lifetime to the model borrow, so both live for the process.
    // A backend is built once per process; this leaks one model, on purpose.
    model: &'static Model,
    ner: Ner,
    rel: Rel,
    params: Parameters,
    labels: Vec<String>,
    /// At most this many entity spans (highest score first) per sentence go
    /// into relation extraction, bounding the label prompt. Default 12.
    pub max_entities: usize,
    /// Add "I"/"me"/"myself" as `person` spans when NER missed them, so
    /// first-person relations reach the speaker. Default true.
    pub first_person: bool,
    /// Entity spans dropped by `max_entities` so far.
    pub truncated: usize,
}

impl GlineBackend {
    /// Load the model with the default relation set (see [`DEFAULT_RELATIONS`]).
    pub fn new(tokenizer_path: &str, model_path: &str) -> Result<Self, String> {
        let mut schema = RelationSchema::new();
        for (label, _, subj, obj, _) in DEFAULT_RELATIONS {
            schema.push_with_allowed_labels(label, subj, obj);
        }
        Self::with_schema(tokenizer_path, model_path, default_entity_labels(), schema)
    }

    pub fn with_schema(
        tokenizer_path: &str,
        model_path: &str,
        labels: Vec<String>,
        schema: RelationSchema,
    ) -> Result<Self, String> {
        let e = |x: Box<dyn std::error::Error + Send + Sync>| x.to_string();
        let model: &'static Model =
            Box::leak(Box::new(Model::new(model_path, RuntimeParameters::default()).map_err(e)?));
        let schema: &'static RelationSchema = Box::leak(Box::new(schema));
        Ok(GlineBackend {
            model,
            ner: TokenPipeline::new(tokenizer_path).map_err(e)?,
            rel: RelationPipeline::default(tokenizer_path, schema).map_err(e)?,
            params: Parameters::default(),
            labels,
            max_entities: 12,
            first_person: true,
            truncated: 0,
        })
    }

    pub fn with_threshold(mut self, t: f32) -> Self {
        self.params = self.params.with_threshold(t);
        self
    }

    fn analyse_one(&mut self, sentence: &str) -> Result<SentenceAnalysis, String> {
        let e = |x: Box<dyn std::error::Error + Send + Sync>| x.to_string();
        let input = TextInput::new(vec![sentence.to_string()], self.labels.clone()).map_err(e)?;
        let out: SpanOutput = self.model.inference(input, &self.ner, &self.params).map_err(e)?;
        let mut spans: Vec<Span> = out.spans.into_iter().next().unwrap_or_default();
        if self.first_person && self.labels.iter().any(|l| l == "person") {
            spans.extend(first_person_spans(sentence, &spans));
        }
        spans.sort_by(|a, b| b.probability().total_cmp(&a.probability()));
        if spans.len() > self.max_entities {
            self.truncated += spans.len() - self.max_entities;
            spans.truncate(self.max_entities);
        }
        spans.sort_by_key(|s| s.offsets());
        let mentions: Vec<Mention> = spans.iter().map(mention).collect();
        let again = SpanOutput::new(vec![sentence.to_string()], out.entities, vec![spans]);
        let rels = self.model.inference(again, &self.rel, &self.params).map_err(e)?;
        let rels: Vec<Relation> = rels.relations.into_iter().flatten().collect();
        Ok(map_relations(sentence, mentions, &rels))
    }
}

fn mention(s: &Span) -> Mention {
    let (start, end) = s.offsets();
    Mention { text: s.text().to_string(), label: s.class().to_string(), start, end, score: s.probability() }
}

/// "I", "me" and "myself" (as whole words) not already covered by a span.
pub fn first_person_spans(sentence: &str, existing: &[Span]) -> Vec<Span> {
    super::negex::tokenize(sentence)
        .into_iter()
        .filter(|t| {
            let raw = &sentence[t.start..t.end];
            raw == "I" || t.lower == "me" || t.lower == "myself"
        })
        .filter(|t| existing.iter().all(|s| s.offsets().1 <= t.start || s.offsets().0 >= t.end))
        .map(|t| Span::new(0, t.start, t.end, sentence[t.start..t.end].to_string(), "person".into(), 1.0))
        .collect()
}

/// Turn gline-rs relations for one sentence into a [`SentenceAnalysis`],
/// mapping prompt-relative object spans back to the sentence.
pub fn map_relations(sentence: &str, mentions: Vec<Mention>, rels: &[Relation]) -> SentenceAnalysis {
    let model: Vec<ModelRelation> = rels
        .iter()
        .map(|r| {
            let (a, b) = r.offsets();
            ModelRelation {
                subject: r.subject().to_string(),
                label: r.class().to_string(),
                object: r.object().to_string(),
                object_span: a
                    .checked_sub(RELATION_PROMPT_OFFSET)
                    .zip(b.checked_sub(RELATION_PROMPT_OFFSET)),
                score: r.probability(),
            }
        })
        .collect();
    resolve_relations(sentence, mentions, &model)
}

impl RelationBackend for GlineBackend {
    fn analyse(&mut self, sentences: &[String]) -> Result<Vec<SentenceAnalysis>, String> {
        sentences.iter().map(|s| self.analyse_one(s)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What gline-rs's relation decoder emits: a span over the prompt whose
    /// class is "<subject> <> <relation>".
    fn rel(sentence: &str, subj: &str, label: &str, obj: &str, nth: usize) -> Relation {
        let at = sentence.match_indices(obj).nth(nth).unwrap().0;
        let prompt = format!("{RELATION_PROMPT_PREFIX} {sentence}");
        let (a, b) = (at + RELATION_PROMPT_OFFSET, at + RELATION_PROMPT_OFFSET + obj.len());
        assert_eq!(&prompt[a..b], obj, "offset constant matches the prompt format");
        Relation::from(Span::new(0, a, b, obj.into(), format!("{subj} <> {label}"), 0.9)).unwrap()
    }

    fn ms(sentence: &str, items: &[(&str, &str)]) -> Vec<Mention> {
        let mut out = Vec::new();
        for (text, label) in items {
            for (i, _) in sentence.match_indices(text) {
                out.push(Mention { text: (*text).into(), label: (*label).into(), start: i, end: i + text.len(), score: 0.9 });
            }
        }
        out
    }

    #[test]
    fn prompt_relative_object_spans_map_to_the_sentence() {
        let s = "Bob doesn't work at Acme.";
        let a = map_relations(s, ms(s, &[("Bob", "person"), ("Acme", "organization")]), &[rel(s, "Bob", "works at", "Acme", 0)]);
        let r = &a.relations[0];
        assert_eq!((&s[r.object.start..r.object.end], r.object.label.as_str()), ("Acme", "organization"));
        assert_eq!(&s[r.subject.start..r.subject.end], "Bob");
        assert!(a.unresolved.is_empty());
    }

    #[test]
    fn repeated_object_text_uses_the_mapped_span() {
        let s = "Acme sued Bob, and Bob works at Acme.";
        let a = map_relations(s, ms(s, &[("Bob", "person"), ("Acme", "organization")]), &[rel(s, "Bob", "works at", "Acme", 1)]);
        let r = &a.relations[0];
        assert_eq!(r.object.start, s.rfind("Acme").unwrap());
        assert_eq!(r.subject.start, s.rfind("Bob").unwrap(), "subject nearest the object");
    }

    #[test]
    fn subject_not_in_sentence_is_unresolved() {
        let s = "Bob works at Acme.";
        let a = map_relations(s, ms(s, &[("Bob", "person"), ("Acme", "organization")]), &[rel(s, "Dana", "works at", "Acme", 0)]);
        assert!(a.relations.is_empty());
        assert_eq!(a.unresolved, vec![("Dana".into(), "works at".into(), "Acme".into())]);
    }

    #[test]
    fn first_person_spans_skip_covered_words() {
        let s = "I told Imogen about me and myself.";
        let got: Vec<_> = first_person_spans(s, &[]).iter().map(|x| x.text().to_string()).collect();
        assert_eq!(got, ["I", "me", "myself"]);
        let covered = [Span::new(0, 0, 1, "I".into(), "person".into(), 0.8)];
        assert_eq!(first_person_spans(s, &covered).len(), 2);
    }
}
