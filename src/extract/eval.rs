//! Scoring extraction against a labelled gold set with MCC.
//!
//! Gold format (tab-separated, `#` comments), see
//! `benchmarks/extraction_gold.tsv`:
//!
//! ```text
//! @   entity id   GLiNER label   alias|alias     entity table
//! ep  episode id  speaker        text            an episode
//! +   subject     predicate      object          must be extracted
//! -   subject     predicate      object          must not be extracted
//! ```
//!
//! Every `+`/`-` line is one labelled candidate. A candidate is a true
//! positive when it is `+` and extracted, a true negative when it is `-` and
//! not extracted, and so on. An extracted triple with no gold line counts as
//! a false positive. MCC is computed over those four counts.

use super::link::TableLinker;
use crate::agent::{Episode, Extractor};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq)]
pub struct GoldEntity {
    pub id: String,
    pub label: String,
    pub aliases: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct GoldEpisode {
    pub episode: Episode,
    pub positive: Vec<(String, String, String)>,
    pub negative: Vec<(String, String, String)>,
}

#[derive(Debug, Clone, Default)]
pub struct Gold {
    pub entities: Vec<GoldEntity>,
    pub episodes: Vec<GoldEpisode>,
}

impl Gold {
    pub fn parse(text: &str) -> Result<Gold, String> {
        let mut g = Gold::default();
        for (n, line) in text.lines().enumerate() {
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }
            let f: Vec<&str> = line.split('\t').collect();
            let err = || format!("line {}: {line:?}", n + 1);
            match f.as_slice() {
                ["@", id, label, aliases] => g.entities.push(GoldEntity {
                    id: id.to_string(),
                    label: label.to_string(),
                    aliases: aliases.split('|').map(str::to_string).collect(),
                }),
                ["ep", id, speaker, text] => g.episodes.push(GoldEpisode {
                    episode: Episode {
                        id: id.to_string(),
                        text: text.to_string(),
                        ts: g.episodes.len() as i64,
                        speaker: (!speaker.is_empty()).then(|| speaker.to_string()),
                    },
                    positive: vec![],
                    negative: vec![],
                }),
                [sign @ ("+" | "-"), s, p, o] => {
                    let ep = g.episodes.last_mut().ok_or_else(err)?;
                    let t = (s.to_string(), p.to_string(), o.to_string());
                    if *sign == "+" { ep.positive.push(t) } else { ep.negative.push(t) }
                }
                _ => return Err(err()),
            }
        }
        Ok(g)
    }

    /// The gold entity table as a deterministic linker: linking is then
    /// perfect for known aliases, so a score isolates NER, relations and
    /// negation from entity-linking errors.
    pub fn table_linker(&self) -> TableLinker {
        self.entities.iter().fold(TableLinker::new(), |t, e| {
            let aliases: Vec<&str> = e.aliases.iter().map(String::as_str).collect();
            t.entity(&e.id, None, &aliases)
        })
    }
}

/// One wrong outcome, for error analysis.
#[derive(Debug, Clone, PartialEq)]
pub enum Miss {
    /// A `+` triple that was not extracted.
    Missed(String, (String, String, String)),
    /// A `-` triple that was extracted.
    Asserted(String, (String, String, String)),
    /// An extracted triple with no gold line.
    Spurious(String, (String, String, String)),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Confusion {
    pub tp: usize,
    pub tn: usize,
    pub fp: usize,
    pub fn_: usize,
    pub misses: Vec<Miss>,
}

impl Confusion {
    /// Matthews correlation coefficient; 0 when a margin is empty.
    pub fn mcc(&self) -> f64 {
        let (tp, tn, fp, fn_) = (self.tp as f64, self.tn as f64, self.fp as f64, self.fn_ as f64);
        let d = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
        if d == 0.0 { 0.0 } else { (tp * tn - fp * fn_) / d }
    }

    pub fn precision(&self) -> f64 {
        self.tp as f64 / (self.tp + self.fp).max(1) as f64
    }

    pub fn recall(&self) -> f64 {
        self.tp as f64 / (self.tp + self.fn_).max(1) as f64
    }
}

/// Run `x` over every gold episode and score its facts.
pub fn score<E: Extractor>(x: &mut E, gold: &Gold) -> Confusion {
    let mut c = Confusion::default();
    for g in &gold.episodes {
        let id = g.episode.id.clone();
        let got: BTreeSet<(String, String, String)> =
            x.extract(&g.episode).into_iter().map(|f| (f.subj, f.pred, f.obj)).collect();
        for t in &g.positive {
            if got.contains(t) { c.tp += 1 } else { c.fn_ += 1; c.misses.push(Miss::Missed(id.clone(), t.clone())) }
        }
        for t in &g.negative {
            if got.contains(t) { c.fp += 1; c.misses.push(Miss::Asserted(id.clone(), t.clone())) } else { c.tn += 1 }
        }
        for t in got.iter().filter(|t| !g.positive.contains(t) && !g.negative.contains(t)) {
            c.fp += 1;
            c.misses.push(Miss::Spurious(id.clone(), t.clone()));
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::{FixtureBackend, GlinerExtractor};

    const GOLD: &str = "@\tperson:bob\tperson\tBob\n@\torg:acme\torganization\tAcme\n@\torg:initech\torganization\tInitech\n\
        ep\te1\t\tBob works at Acme.\n+\tperson:bob\tworks_at\torg:acme\n\
        ep\te2\t\tBob doesn't work at Initech.\n-\tperson:bob\tworks_at\torg:initech\n";

    #[test]
    fn perfect_extractor_scores_mcc_one() {
        let g = Gold::parse(GOLD).unwrap();
        assert_eq!(g.episodes.len(), 2);
        let backend = FixtureBackend::new()
            .relation("Bob", "person", "works at", "Acme", "organization", 0.9)
            .relation("Bob", "person", "works at", "Initech", "organization", 0.9);
        let mut x = GlinerExtractor::new(backend, g.table_linker());
        let c = score(&mut x, &g);
        assert_eq!((c.tp, c.tn, c.fp, c.fn_), (1, 1, 0, 0));
        assert!((c.mcc() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn negation_off_is_penalised() {
        let g = Gold::parse(GOLD).unwrap();
        let backend = FixtureBackend::new().relation("Bob", "person", "works at", "Initech", "organization", 0.9);
        let mut x = GlinerExtractor::new(backend, g.table_linker());
        x.negex = crate::extract::NegEx::default().with_window(0);
        let c = score(&mut x, &g);
        assert_eq!((c.tp, c.tn, c.fp, c.fn_), (0, 0, 1, 1));
        assert!(c.mcc() < 0.0);
    }

    #[test]
    fn bad_lines_are_errors() {
        assert!(Gold::parse("+\ta\tb\tc\n").is_err()); // triple before any episode
        assert!(Gold::parse("ep\tonly-two\n").is_err());
    }
}
