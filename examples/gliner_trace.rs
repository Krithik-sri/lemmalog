//! One sentence -> FOL, stage by stage: GLiNER mentions and relations, the
//! NegEx verdict, GLinker canonical ids, LemmaLog candidate facts, and the
//! AgentMemory update decision per fact (ADD / UPDATE / NOOP / ESCALATE,
//! with the bi-temporal `edge(...)` rows it wrote). The rules the store
//! runs (`DEFAULT_RULES`) print once at the end, with the final open edges,
//! the derived `current(...)` view, a `why()` proof, and one retrieval
//! query.
//!
//! Sentences are argv entries; an optional `Speaker|` prefix resolves
//! first-person mentions to that entity ("Dana|I work at Globex.").
//! Linking is the tested cascade: the exact-alias table first, glinker-rs
//! for the rest (`min_confidence` 0.10 — the bi-encoder's 0.5 default
//! rejects almost everything).
//!
//! ```text
//! export ORT_DYLIB_PATH=/path/to/libonnxruntime.so   # onnxruntime 1.20.x
//! cargo run --release --example gliner_trace --features gliner-load-dynamic,glinker -- \
//!     <tokenizer.json> <model.onnx> <pipeline.json> <entities.jsonl> \
//!     "Bob Smith works at Acme Corp." "Dana|I work at Globex."
//! ```

use lemmalog::agent::{
    AgentMemory, Episode, Extractor, MockExtractor, DEFAULT_RULES,
};
use lemmalog::extract::gline::GlineBackend;
use lemmalog::extract::link::{
    CascadeLinker, EntityLinker, GlinkerLinker, Link, TableLinker,
};
use lemmalog::extract::{
    split_sentences, strip_title, CachedBackend, GlinerExtractor, Mention, RejectReason,
};
use lemmalog::{Engine, Value};

fn reason(r: &RejectReason) -> String {
    match r {
        RejectReason::Unlinked(m) => format!("Unlinked({m})"),
        other => format!("{other:?}"),
    }
}

/// The pipeline treats "I", "me", "my", "myself" as the speaker.
fn first_person(m: &Mention) -> bool {
    matches!(m.text.to_lowercase().as_str(), "i" | "me" | "my" | "myself")
}

fn show_link(l: &Link) -> String {
    match l {
        Link::Canonical { id, score } => format!("{id} ({score:.2})"),
        Link::Unlinked => "Unlinked".into(),
    }
}

/// `edge(S, P, O, valid_from, valid_to, asserted_at)` with i64::MAX shown as inf.
fn show_edge(e: &Engine, k: &[Value]) -> String {
    let d = |v: &Value| e.interner.display(v);
    let vt = if k[4].as_int() == Some(i64::MAX) {
        "inf".to_string()
    } else {
        d(&k[4])
    };
    format!(
        "edge({}, {}, {}, valid_from={}, valid_to={}, asserted_at={})",
        d(&k[0]),
        d(&k[1]),
        d(&k[2]),
        d(&k[3]),
        vt,
        d(&k[5])
    )
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 5 {
        return Err("usage: gliner_trace <tokenizer.json> <model.onnx> <pipeline.json> \
                    <entities.jsonl> [\"Speaker|SENTENCE\" ...]".into());
    }
    let mut glinker = GlinkerLinker::from_config_file(&a[3])?;
    glinker.min_confidence = 0.1;
    let exact = TableLinker::from_glinker_entities(&a[4])?;
    let mut linker = CascadeLinker::new(exact, glinker);
    let mut backend = CachedBackend::new(GlineBackend::new(&a[1], &a[2])?);
    // The extractor is rebuilt per episode so the shared backend/linker
    // borrows end and can be inspected between episodes; the backend's
    // per-sentence cache keeps this at one model call per sentence.
    let mut mem = AgentMemory::new(MockExtractor::new(0.9), "").map_err(|e| e.to_string())?;
    // Program-embedded facts (the exclusivity table) are minted into the fact
    // store by run(); mint them now so exclusive("works_at") governs from the
    // very first observation.
    let base_ts = 1_700_000_000i64;
    println!("primed: maintain derived {} facts (program facts minted)\n", mem.maintain(base_ts - 1));

    let mut first_fact: Option<String> = None;
    for (i, raw) in a.iter().skip(5).enumerate() {
        let (speaker, text) = match raw.split_once('|') {
            Some((s, t)) => (Some(s.to_string()), t.to_string()),
            None => (None, raw.clone()),
        };
        let ep = Episode {
            id: format!("demo{}", i + 1),
            text: text.clone(),
            ts: base_ts + i as i64,
            speaker: speaker.clone(),
        };
        println!("== episode {}  speaker={:?}", ep.id, speaker);
        println!("   \"{}\"", text);

        // [1] what GLiNER saw, per sentence (raw model output, cached)
        let sentences = split_sentences(&text);
        for s in &sentences {
            if sentences.len() > 1 {
                println!("   sentence: \"{}\"", s);
            }
            let Some(an) = backend.analysis(s) else { continue };
            for m in &an.mentions {
                println!("   [GLiNER ] mention   \"{}\" <{}> @{}..{} ({:.2})", m.text, m.label, m.start, m.end, m.score);
            }
            for r in &an.relations {
                println!("   [GLiNER ] relation  {} --{}--> {} ({:.2})", r.subject.text, r.label, r.object.text, r.score);
            }
            for (s2, l, o) in &an.unresolved {
                println!("   [GLiNER ] unresolved {} --{}--> {}", s2, l, o);
            }
        }

        // [2..4] run the real pipeline over the episode: NegEx -> linking ->
        // direction tie-break. Rejections say which stage dropped what.
        let (facts, rejections) = {
            let mut x = GlinerExtractor::new(&mut backend, &mut linker);
            let facts = x.extract(&ep);
            let rejections = x.rejected().to_vec();
            (facts, rejections)
        };
        for r in &rejections {
            println!("   [drop   ] {} --{}--> {}  [{}]", r.subject, r.relation, r.object, reason(&r.reason));
        }
        if facts.is_empty() && rejections.is_empty() {
            println!("   [fact   ] (none)");
        }

        // [5] how each mention links (the same cascade, for reference);
        // first-person mentions resolve as the speaker's name, exactly like
        // the pipeline's speaker_link.
        let speaker_link: Option<Link> = match &speaker {
            Some(name) => {
                let m = Mention { text: name.clone(), label: "person".into(), start: 0, end: name.len(), score: 1.0 };
                linker.link(name, &[m]).into_iter().next()
            }
            None => None,
        };
        for s in &sentences {
            let Some(an) = backend.analysis(s) else { continue };
            let mut ask: Vec<Mention> = Vec::new();
            let mut display: Vec<(String, bool)> = Vec::new(); // (mention text, first-person?)
            for m in &an.mentions {
                if first_person(m) {
                    display.push((m.text.clone(), true));
                } else {
                    let stripped = strip_title(s, m);
                    let shown = if stripped.text != m.text {
                        format!("{} (as \"{}\")", m.text, stripped.text)
                    } else {
                        m.text.clone()
                    };
                    display.push((shown, false));
                    ask.push(stripped);
                }
            }
            if ask.is_empty() {
                continue;
            }
            let links = linker.link(s, &ask);
            let mut li = 0;
            for (shown, fp) in &display {
                if *fp {
                    let to = match &speaker_link {
                        Some(l) => show_link(l),
                        None => "Unlinked (no speaker)".into(),
                    };
                    println!("   [linker ] \"{}\" -> {} (first-person => speaker)", shown, to);
                } else {
                    println!("   [linker ] \"{}\" -> {}", shown, show_link(&links[li]));
                    li += 1;
                }
            }
        }

        // [6] the candidate facts the store actually received
        for f in &facts {
            println!("   [fact   ] {} --{}--> {} [{:.2}]", f.subj, f.pred, f.obj, f.confidence);
            if first_fact.is_none() {
                first_fact = Some(format!("current({}, {}, {})", f.subj, f.pred, f.obj));
            }
        }

        // [7] the update policy's decision, with the edge rows it wrote
        let lines: Vec<String> = facts
            .iter()
            // the confidence suffix belongs on the relation, never the object
            .map(|f| format!("{} --{}[{:.2}]--> {}", f.subj, f.pred, f.confidence, f.obj))
            .collect();
        let (report, dropped) = mem.observe_extracted(&lines.join("\n"), base_ts + i as i64);
        for d in &dropped {
            println!("   [memory ] dropped protocol line {:?}", d);
        }
        println!(
            "   [memory ] ADD={} UPDATE={} NOOP={} ESCALATE={}",
            report.added, report.updated, report.noop, report.escalations.len()
        );
        for e in &report.escalations {
            println!("   [memory ] escalation: {}", e);
        }
        for f in &facts {
            let subj = mem.engine.sym(&f.subj);
            let pred = mem.engine.sym(&f.pred);
            let rows = mem.engine.query("edge", &[Some(subj), Some(pred), None, None, None, None]);
            for (k, _) in &rows {
                if k[4].as_int() == Some(i64::MAX) {
                    println!("   [store  ] {}  <- open", show_edge(&mem.engine, k));
                }
            }
            for (k, _) in &rows {
                if k[4].as_int() != Some(i64::MAX) {
                    println!("   [store  ] {}  <- closed (superseded)", show_edge(&mem.engine, k));
                }
            }
        }
        println!();
    }

    // [8] the rules the store runs, the final state, the derived view, a
    // proof, and retrieval
    let derived = mem.maintain(base_ts + (a.len() - 5) as i64 + 1);
    println!("== the rules the store runs (DEFAULT_RULES)");
    for l in DEFAULT_RULES.trim().lines() {
        println!("   {}", l);
    }
    println!("\n== final open edges ({} derived facts)", derived);
    let mut open: Vec<String> = mem
        .engine
        .query("edge", &[None, None, None, None, None, None])
        .into_iter()
        .filter(|(k, _)| k[4].as_int() == Some(i64::MAX))
        .map(|(k, _)| show_edge(&mem.engine, &k))
        .collect();
    open.sort();
    for e in &open {
        println!("   {}", e);
    }
    println!("\n== derived current() view");
    for row in mem.ask("current(E, R, O)").unwrap_or_default() {
        println!("   {}", row);
    }
    if let Some(fact) = &first_fact {
        println!("\n== why({})", fact);
        println!("{}", mem.why(fact));
    }
    println!("\n== retrieval: context_for_query(\"Where does Bob work?\", 256)");
    println!("{}", mem.context_for_query("Where does Bob work?", 256));
    Ok(())
}
