//! LLM-free fact extraction: gline-rs (GLiNER NER + relation extraction),
//! NegEx negation detection and GLinker entity linking.
//!
//! See `spec/extraction.md` for the obligations this module must meet.

pub mod link;
pub mod negex;
pub mod pipeline;
#[cfg(feature = "gliner")]
pub mod gline;

pub use link::{EntityLinker, Link, TableLinker};
pub use negex::NegEx;
pub use pipeline::{
    FixtureBackend, GlinerExtractor, Mention, RawRelation, RejectReason, Rejection,
    RelationBackend, SentenceAnalysis, DEFAULT_RELATIONS,
};
