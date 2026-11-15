//! NegEx: rule-based negation detection (Chapman et al., 2001), adapted to
//! relation facts in agent memory.
//!
//! A relation `subject --pred--> object` found in a sentence is *negated*
//! when a negation trigger governs the object mention:
//!
//! * a **pre-negation** trigger ("not", "doesn't", "no longer", "never",
//!   "quit", ...) that starts at or after the subject and ends at most
//!   `window` tokens before the object, or
//! * a **post-negation** trigger ("anymore", "no more", "any longer", ...)
//!   that starts at most `window` tokens after the object,
//!
//! with no **termination** term ("but", "however", ".", ";", ...) between the
//! trigger and the object. **Pseudo-triggers** ("not only", "no doubt",
//! "can't wait", ...) contain trigger words but do not negate, so they are
//! matched first and skipped.
//!
//! Everything here is deterministic: the same text and spans always give the
//! same answer, and nothing calls a model.

/// One word or punctuation token with its byte range in the sentence.
#[derive(Debug, Clone, PartialEq)]
pub struct Tok {
    pub lower: String,
    pub start: usize,
    pub end: usize,
}

/// Split into lower-cased word tokens (letters, digits, apostrophes, `-`, `_`)
/// and single punctuation tokens. Curly apostrophes are normalised to `'`.
pub fn tokenize(text: &str) -> Vec<Tok> {
    let mut out = Vec::new();
    let mut cur: Option<(usize, String)> = None;
    for (i, c) in text.char_indices() {
        let c = if c == '\u{2019}' { '\'' } else { c };
        let wordy = c.is_alphanumeric() || c == '\'' || c == '-' || c == '_';
        if wordy {
            match cur.as_mut() {
                Some((_, s)) => s.extend(c.to_lowercase()),
                None => cur = Some((i, c.to_lowercase().collect())),
            }
            continue;
        }
        if let Some((s, w)) = cur.take() {
            out.push(Tok { lower: w, start: s, end: i });
        }
        if !c.is_whitespace() {
            out.push(Tok { lower: c.to_string(), start: i, end: i + c.len_utf8() });
        }
    }
    if let Some((s, w)) = cur.take() {
        out.push(Tok { lower: w, start: s, end: text.len() });
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Pseudo,
    Pre,
    Post,
    Term,
}

/// A matched trigger: token index range `[first, last]` and its kind.
#[derive(Debug, Clone, Copy)]
struct Hit {
    kind: Kind,
    first: usize,
    last: usize,
}

/// The NegEx rule set and scope window.
#[derive(Debug, Clone)]
pub struct NegEx {
    rules: Vec<(Vec<String>, Kind)>,
    /// Maximum number of tokens between a trigger and the object it negates.
    pub window: usize,
}

const PSEUDO: &[&str] = &[
    "not only", "not just", "no doubt", "not to mention", "nothing but", "never mind",
    "can't wait", "cannot wait", "no problem", "not bad", "no wonder", "not necessarily",
];
const PRE: &[&str] = &[
    "not", "no", "never", "no longer", "neither", "nor", "without", "cannot", "can't",
    "don't", "doesn't", "didn't", "isn't", "wasn't", "aren't", "weren't", "won't",
    "haven't", "hasn't", "hadn't", "wouldn't", "shouldn't", "couldn't", "ain't",
    "denies", "denied", "deny", "quit", "quits", "stopped", "ceased", "refused",
    "declined",
];
const POST: &[&str] = &[
    "anymore", "any more", "no more", "any longer", "is false", "was false", "not true",
    "is untrue",
];
const TERM: &[&str] = &[
    "but", "however", "although", "though", "yet", "except", "whereas", "instead",
    "now", "still", ".", ";", "!", "?", ":",
];

impl Default for NegEx {
    fn default() -> Self {
        let mut rules = Vec::new();
        let mut add = |list: &[&str], k: Kind| {
            for p in list {
                rules.push((p.split_whitespace().map(str::to_string).collect::<Vec<_>>(), k));
            }
        };
        add(PSEUDO, Kind::Pseudo);
        add(POST, Kind::Post);
        add(PRE, Kind::Pre);
        add(TERM, Kind::Term);
        // longest phrase first, so "no longer" beats "no" and "not only" beats "not"
        rules.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        NegEx { rules, window: 6 }
    }
}

impl NegEx {
    pub fn with_window(mut self, window: usize) -> Self {
        self.window = window;
        self
    }

    fn hits(&self, toks: &[Tok]) -> Vec<Hit> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < toks.len() {
            let found = self.rules.iter().find(|(words, _)| {
                i + words.len() <= toks.len()
                    && words.iter().zip(&toks[i..]).all(|(w, t)| *w == t.lower)
            });
            match found {
                Some((words, kind)) => {
                    out.push(Hit { kind: *kind, first: i, last: i + words.len() - 1 });
                    i += words.len();
                }
                None => i += 1,
            }
        }
        out
    }

    /// Is the object span `[obj.0, obj.1)` (byte offsets into `sentence`)
    /// negated? `anchor` is the byte offset where the relation's subject
    /// starts: pre-negation triggers before it do not govern this relation.
    pub fn negated(&self, sentence: &str, obj: (usize, usize), anchor: usize) -> bool {
        let toks = tokenize(sentence);
        let Some(o_first) = toks.iter().position(|t| t.end > obj.0) else {
            return false;
        };
        let o_last = toks.iter().rposition(|t| t.start < obj.1).unwrap_or(o_first);
        let hits = self.hits(&toks);
        let terminated = |a: usize, b: usize| {
            hits.iter().any(|h| h.kind == Kind::Term && h.first > a && h.last < b)
        };
        hits.iter().any(|h| match h.kind {
            Kind::Pre => {
                toks[h.first].start >= anchor
                    && h.last < o_first
                    && o_first - h.last - 1 <= self.window
                    && !terminated(h.last, o_first)
            }
            Kind::Post => {
                h.first > o_last
                    && h.first - o_last - 1 <= self.window
                    && !terminated(o_last, h.first)
            }
            _ => false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn neg(s: &str, obj: &str, subj: &str) -> bool {
        let o = s.find(obj).unwrap();
        NegEx::default().negated(s, (o, o + obj.len()), s.find(subj).unwrap())
    }

    #[test]
    fn basic_triggers() {
        assert!(neg("Bob doesn't work at Acme.", "Acme", "Bob"));
        assert!(neg("Bob no longer works at Acme", "Acme", "Bob"));
        assert!(neg("Bob works at Acme no more", "Acme", "Bob"));
        assert!(!neg("Bob works at Acme", "Acme", "Bob"));
    }

    #[test]
    fn pseudo_and_termination() {
        assert!(!neg("Bob not only works at Acme, he runs it", "Acme", "Bob"));
        assert!(!neg("Bob left Initech but works at Acme", "Acme", "Bob"));
        assert!(!neg("Bob doesn't like tea; he works at Acme", "Acme", "Bob"));
    }

    #[test]
    fn trigger_before_subject_does_not_govern() {
        assert!(!neg("Not much happened: Bob works at Acme", "Acme", "Bob"));
    }
}
