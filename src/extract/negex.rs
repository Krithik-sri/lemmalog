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
//! trigger and the object, no other entity mention between them (unless it is
//! coordinated with the object), and the trigger not inside a comma-enclosed
//! aside after the subject. **Cessation** words ("left", "resigned",
//! "former", ...) count as pre-triggers; **sentential** triggers ("it's not
//! true that ...") may precede the subject. **Pseudo-triggers** ("not only", "no doubt",
//! "can't wait", "doesn't mind", "never stopped", "hasn't left", ...) contain
//! trigger words but do not negate, so they are matched first and skipped.
//!
//! A **subject-side** trigger ("neither", "nor", "not") directly before the
//! subject negates the relation too: "Neither Bob nor Dana works at Acme",
//! "Not Bob but Dana works at Acme" (for Bob).
//!
//! Scope details: an adverbial termination term ("however", "now", ...)
//! inside a comma-enclosed aside (", however,") or directly after a trigger
//! ("does not now work") does not end the scope; a relative pronoun
//! ("which", "who") ends a post-trigger's scope ("works at Acme, which is
//! not true of Dana"). "left" followed by a determiner and a noun that is not
//! a job or an organisation ("left his laptop at Acme") is not cessation.
//!
//! [`dislikes_between`] is the relation-specific check for `likes`: a
//! dislike verb ("dislikes", "hates", "can't stand") between subject and
//! object contradicts the relation.
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
    /// Sentential negation ("it's not true that ..."): may precede the
    /// subject and governs everything up to a termination term.
    Sent,
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

/// Phrases that contain a trigger word but do not negate the relation.
/// Matched before triggers (longest phrase first), so their words are
/// consumed and cannot fire. Includes negated-positive idioms and double
/// negatives ("never stopped", "hasn't left").
const PSEUDO: &[&str] = &[
    "not only", "not just", "no doubt", "not to mention", "nothing but", "never mind",
    "can't wait", "cannot wait", "no problem", "not bad", "no wonder", "not necessarily",
    "without doubt", "without a doubt", "doesn't only", "don't only", "didn't only",
    "isn't only", "isn't just", "aren't just", "wasn't just", "doesn't just", "don't just",
    "didn't just", "not unhappy", "not unlike",
    "doesn't mind", "don't mind", "didn't mind", "wouldn't mind", "didn't hesitate",
    "doesn't hesitate", "never hesitates", "never hesitated", "couldn't be happier",
    "can't stop", "cannot stop", "couldn't stop", "can't get enough", "cannot get enough",
    "never misses", "never miss", "never missed", "never fails", "never fail", "never failed",
    "never stopped", "never stops", "never quit", "never quits", "never left", "never leaves",
    "hasn't left", "haven't left", "hadn't left", "didn't leave", "doesn't leave",
    "didn't quit", "hasn't quit", "haven't quit", "won't leave", "won't quit",
    "left for", "leaves for", "leaving for", "quit for", "no doubts",
];
/// Triggers that negate the next entity in their clause.
const PRE: &[&str] = &[
    "not", "no", "never", "no longer", "neither", "nor", "without", "cannot", "can't",
    "don't", "doesn't", "didn't", "isn't", "wasn't", "aren't", "weren't", "won't",
    "haven't", "hasn't", "hadn't", "wouldn't", "shouldn't", "couldn't", "ain't",
    // the same contractions typed without the apostrophe
    "dont", "doesnt", "didnt", "isnt", "wasnt", "arent", "werent", "havent", "hasnt",
    "hadnt", "wouldnt", "shouldnt", "couldnt", "cant",
    "denies", "denied", "deny", "quit", "quits", "stopped", "ceased", "refused",
    "declined",
    // cessation: the relation held and has ended
    "left", "leaves", "leaving", "resigned", "retired", "fired", "laid off", "let go",
    "former", "formerly", "ex", "ex-employee", "ex-member", "moved out of",
    "moved away from", "departed", "dropped out of",
];
const POST: &[&str] = &[
    "anymore", "any more", "no more", "any longer", "is false", "was false", "not true",
    "is untrue",
];
const SENT: &[&str] = &[
    "not true that", "isn't true that", "is not true that", "wasn't true that",
    "untrue that", "false that", "not the case that", "isn't the case that",
];
const TERM: &[&str] = &[
    "but", "however", "although", "though", "yet", "except", "whereas", "instead",
    "now", "still", ".", ";", "!", "?", ":",
];
/// Termination terms that are adverbs: inside a comma-enclosed aside
/// (", however,") or straight after a trigger ("does not now work") they do
/// not end a scope.
const ADVERB_TERM: &[&str] = &["however", "though", "yet", "instead", "now", "still"];
/// Relative pronouns end a post-trigger's scope ("works at Acme, which is
/// not true of Dana"): the trigger belongs to the relative clause.
const POST_TERM: &[&str] = &["which", "who", "whom", "whose"];
/// Triggers that negate the subject they stand directly before ("Neither
/// Bob nor Dana works at Acme").
const SUBJ: &[&str] = &["neither", "nor", "not"];
/// "left" is cessation before these nouns ("left his job at Acme") and an
/// ordinary verb before any other determiner + noun ("left his laptop").
const LEFT: &[&str] = &["left", "leaves", "leaving"];
const DETERMINERS: &[&str] = &["his", "her", "my", "their", "our", "your", "its", "the", "a", "an"];
const MEMBERSHIP_NOUNS: &[&str] = &[
    "job", "jobs", "post", "position", "role", "team", "company", "firm", "employer", "organisation",
    "organization", "group", "board", "department", "club", "band", "school", "university", "college",
    "startup", "business", "lab", "project", "party", "union", "church", "practice", "agency", "studio",
    "employment", "career", "staff", "workforce",
];
/// Verbs that contradict `likes` ("Bob dislikes sushi").
const DISLIKE: &[&str] = &[
    "dislike", "dislikes", "disliked", "disliking", "hate", "hates", "hated", "hating", "detest",
    "detests", "detested", "loathe", "loathes", "loathed", "despise", "despises", "despised",
    "can't stand", "cannot stand", "cant stand", "couldn't stand", "can't bear", "cannot bear",
];

/// Is there a dislike verb ("dislikes", "hates", "can't stand") between the
/// two byte spans (in either order)? Used to reject `likes` relations that
/// the sentence contradicts.
pub fn dislikes_between(sentence: &str, a: (usize, usize), b: (usize, usize)) -> bool {
    let (lo, hi) = if a.0 <= b.0 { (a.1, b.0) } else { (b.1, a.0) };
    let words: Vec<String> =
        tokenize(sentence).into_iter().filter(|t| t.start >= lo && t.end <= hi).map(|t| t.lower).collect();
    DISLIKE.iter().any(|d| {
        let d: Vec<&str> = d.split_whitespace().collect();
        words.windows(d.len()).any(|w| w.iter().zip(&d).all(|(x, y)| x == y))
    })
}

/// Tokens allowed between two coordinated entities ("Acme, Initech or Globex").
const COORD: &[&str] = &[",", "or", "and", "nor", "/", "&", "either"];

impl Default for NegEx {
    fn default() -> Self {
        let mut rules = Vec::new();
        let mut add = |list: &[&str], k: Kind| {
            for p in list {
                rules.push((p.split_whitespace().map(str::to_string).collect::<Vec<_>>(), k));
            }
        };
        add(PSEUDO, Kind::Pseudo);
        add(SENT, Kind::Sent);
        add(POST, Kind::Post);
        add(PRE, Kind::Pre);
        add(TERM, Kind::Term);
        // longest phrase first, so "no longer" beats "no" and "not only" beats
        // "not"; the sort is stable, so equal lengths keep list order
        // (pseudo before trigger)
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
                    let ordinary_left = *kind == Kind::Pre
                        && words.len() == 1
                        && LEFT.contains(&words[0].as_str())
                        && toks.get(i + 1).is_some_and(|t| DETERMINERS.contains(&t.lower.as_str()))
                        && toks.get(i + 2).is_some_and(|t| {
                            t.lower.chars().all(|c| c.is_alphanumeric())
                                && !MEMBERSHIP_NOUNS.contains(&t.lower.as_str())
                        });
                    let kind = if ordinary_left { Kind::Pseudo } else { *kind };
                    out.push(Hit { kind, first: i, last: i + words.len() - 1 });
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
        self.negated_with(sentence, obj, anchor, &[])
    }

    /// [`negated`](Self::negated), also given the byte spans of the
    /// sentence's other entity mentions. A trigger governs the first entity
    /// after it (before it, for post-triggers): another mention between the
    /// trigger and the object blocks the scope ("quit Initech to join Acme"),
    /// unless it is coordinated with the object ("not Acme or Initech").
    pub fn negated_with(
        &self,
        sentence: &str,
        obj: (usize, usize),
        anchor: usize,
        others: &[(usize, usize)],
    ) -> bool {
        let toks = tokenize(sentence);
        let Some(o_first) = toks.iter().position(|t| t.end > obj.0) else {
            return false;
        };
        let o_last = toks.iter().rposition(|t| t.start < obj.1).unwrap_or(o_first);
        let hits = self.hits(&toks);
        let tok = |i: usize| toks.get(i).map_or("", |t| t.lower.as_str());
        // the last token of each trigger, for "does not now work"
        let trigger_end = |i: usize| hits.iter().any(|h| matches!(h.kind, Kind::Pre | Kind::Sent) && h.last == i);
        let ends_scope = |h: &Hit| {
            if h.kind != Kind::Term {
                return false;
            }
            if h.first == h.last && ADVERB_TERM.contains(&tok(h.first)) {
                let aside = h.first > 0 && tok(h.first - 1) == "," && tok(h.last + 1) == ",";
                // straight after a trigger, or after a trigger and an aside
                // ("does not, as far as I know, still work")
                let mut j = h.first.checked_sub(1);
                if let Some(c) = j.filter(|&c| tok(c) == ",") {
                    j = (0..c).rev().find(|&k| tok(k) == ",").and_then(|k| k.checked_sub(1));
                }
                let after_trigger = j.is_some_and(trigger_end);
                return !aside && !after_trigger;
            }
            true
        };
        let terminated = |a: usize, b: usize| hits.iter().any(|h| h.first > a && h.last < b && ends_scope(h));
        let rel_clause = |a: usize, b: usize| (a + 1..b).any(|i| POST_TERM.contains(&tok(i)));
        let comma = |a: usize, b: usize| (a + 1..b).any(|i| toks[i].lower == ",");
        // token ranges of the other mentions, ignoring any overlapping the object
        let other: Vec<(usize, usize)> = others
            .iter()
            .filter(|(s, e)| *e <= obj.0 || *s >= obj.1)
            .filter_map(|&(s, e)| {
                let f = toks.iter().position(|t| t.end > s)?;
                let l = toks.iter().rposition(|t| t.start < e)?;
                (f <= l).then_some((f, l))
            })
            .collect();
        let covered = |i: usize| other.iter().any(|&(f, l)| f <= i && i <= l);
        let filler = |i: usize| covered(i) || COORD.contains(&toks[i].lower.as_str());
        // only coordination tokens and other mentions between a and b
        let coordinated = |a: usize, b: usize| (a + 1..b).all(filler);
        // tokens between a and b, not counting coordinated list members
        let dist = |a: usize, b: usize| (a + 1..b).filter(|&i| !filler(i)).count();
        // a mention between the trigger and the object that is not
        // coordinated with the object takes the trigger's scope
        let blocked_pre = |a: usize| {
            other.iter().any(|&(f, l)| f > a && l < o_first && !coordinated(l, o_first))
        };
        let blocked_post = |b: usize| {
            other.iter().any(|&(f, l)| f > o_last && l < b && !coordinated(o_last, f))
        };
        let a_tok = toks.iter().position(|t| t.start >= anchor).unwrap_or(0);
        hits.iter().any(|h| match h.kind {
            // "Neither Bob nor Dana ...", "Not Bob but Dana ...": a trigger
            // directly before the subject negates the subject's relation
            Kind::Pre if h.first == h.last && h.last + 1 == a_tok && a_tok < o_first && SUBJ.contains(&tok(h.first)) => true,
            Kind::Pre => {
                toks[h.first].start >= anchor
                    && h.last < o_first
                    && dist(h.last, o_first) <= self.window
                    && !terminated(h.last, o_first)
                    && !blocked_pre(h.last)
                    // "Bob, who has no car, works at Acme": the trigger sits in
                    // a comma-enclosed aside after the subject
                    && !(comma(a_tok, h.first) && comma(h.last, o_first))
            }
            Kind::Post => {
                h.first > o_last
                    && dist(o_last, h.first) <= self.window
                    && !terminated(o_last, h.first)
                    && !rel_clause(o_last, h.first)
                    && !blocked_post(h.first)
            }
            Kind::Sent => h.last < o_first && !terminated(h.last, o_first),
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
