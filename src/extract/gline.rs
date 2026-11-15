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

use super::pipeline::{default_entity_labels, Mention, RawRelation, RelationBackend, SentenceAnalysis, DEFAULT_RELATIONS};
use gliner::model::input::relation::schema::RelationSchema;
use gliner::model::input::text::TextInput;
use gliner::model::output::decoded::SpanOutput;
use gliner::model::params::Parameters;
use gliner::model::pipeline::{relation::RelationPipeline, token::TokenPipeline};
use gliner::text::splitter::RegexSplitter;
use gliner::text::tokenizer::HFTokenizer;
use orp::model::Model;
use orp::params::RuntimeParameters;

type Ner = TokenPipeline<RegexSplitter, HFTokenizer>;
type Rel = RelationPipeline<'static, RegexSplitter, HFTokenizer>;

pub struct GlineBackend {
    // The relation pipeline borrows its schema and `Model::inference` ties the
    // pipeline lifetime to the model borrow, so both live for the process.
    // A backend is built once per process; this leaks one model, on purpose.
    model: &'static Model,
    ner: Ner,
    rel: Rel,
    params: Parameters,
    labels: Vec<String>,
}

impl GlineBackend {
    /// Load the model with the default relation set (see [`DEFAULT_RELATIONS`]).
    pub fn new(tokenizer_path: &str, model_path: &str) -> Result<Self, String> {
        let mut schema = RelationSchema::new();
        for (label, _, subj, obj) in DEFAULT_RELATIONS {
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
        })
    }

    pub fn with_threshold(mut self, t: f32) -> Self {
        self.params = self.params.with_threshold(t);
        self
    }
}

fn mention(text: &str, class: &str, start: usize, end: usize, p: f32) -> Mention {
    Mention { text: text.to_string(), label: class.to_string(), start, end, score: p }
}

impl RelationBackend for GlineBackend {
    fn analyse(&mut self, sentences: &[String]) -> Result<Vec<SentenceAnalysis>, String> {
        let e = |x: Box<dyn std::error::Error + Send + Sync>| x.to_string();
        let input = TextInput::new(sentences.to_vec(), self.labels.clone()).map_err(e)?;
        let spans: SpanOutput = self.model.inference(input, &self.ner, &self.params).map_err(e)?;
        let mut out: Vec<SentenceAnalysis> = vec![SentenceAnalysis::default(); sentences.len()];
        for seq in &spans.spans {
            for s in seq {
                let (a, b) = s.offsets();
                out[s.sequence()].mentions.push(mention(s.text(), s.class(), a, b, s.probability()));
            }
        }
        let again = SpanOutput::new(spans.texts.clone(), spans.entities.clone(), spans.spans.clone());
        let rels = self.model.inference(again, &self.rel, &self.params).map_err(e)?;
        for seq in &rels.relations {
            for r in seq {
                let i = r.sequence();
                let (a, b) = r.offsets();
                let object = out[i]
                    .mentions
                    .iter()
                    .find(|m| m.start == a && m.end == b)
                    .cloned()
                    .unwrap_or_else(|| mention(r.object(), "", a, b, r.probability()));
                // gline-rs gives the subject's text only: take the NER span with
                // that text nearest to the object
                let subject = out[i]
                    .mentions
                    .iter()
                    .filter(|m| m.text == r.subject())
                    .min_by_key(|m| (m.start as i64 - a as i64).abs())
                    .cloned();
                let Some(subject) = subject else { continue };
                out[i].relations.push(RawRelation {
                    subject,
                    label: r.class().to_string(),
                    object,
                    score: r.probability(),
                });
            }
        }
        Ok(out)
    }
}
