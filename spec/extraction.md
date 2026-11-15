# 7. Fact extraction without an LLM

Lemmalog turns each conversation episode into candidate facts
`subject --predicate--> object` through the `Extractor` trait. The LLM
extractors (`LlmExtractor`, `LlmClientExtractor`) ask a chat model for the
facts. `GlinerExtractor` (`src/extract/`) replaces that call with three
deterministic stages:

1. **gline-rs** runs a GLiNER multitask model (ONNX, token mode) for named
   entities and relations. Feature `gliner`.
2. **NegEx** drops relations whose object sits inside a negation scope.
3. **glinker-rs**, the Rust port of GLinker, links each mention to a canonical
   entity id in-process. It runs `Executor::link_with_mentions`, which skips L1
   because gline-rs already found the spans. Feature `glinker`. Tests use the
   deterministic `TableLinker`.

The three obligations below are what this change promises. Each one has
tagged Rust tests in `tests/extraction_test.rs`. `obligations.lock` on `main`
pins the obligations and the tests, so a later PR cannot weaken them unnoticed.

## 7.1 Obligations

```yaml
obligations:
  - id: SPEC-7.1-O1
    says: >
      A relation whose object is governed by a negation trigger produces no
      fact. The trigger is either a pre-negation trigger that starts at or
      after the subject and ends within the window before the object, or a
      post-negation trigger within the window after the object, with no
      termination term in between. The relation is reported in rejected()
      with reason Negated. Pseudo-triggers ("not only", "no doubt") and
      triggers cut off by a termination term ("but", ";", ".") do not negate.
    kind: property
    risk: high
    scope:
      - src/extract/negex.rs::negated
      - src/extract/pipeline.rs::run
  - id: SPEC-7.1-O2
    says: >
      Every emitted fact's subject and object are canonical ids returned by the
      EntityLinker for that mention, never the raw mention text. A relation
      with a mention that does not link produces no fact and is reported with
      reason Unlinked. First-person mentions (I, me, my) are linked as the
      episode's speaker.
    kind: contract
    risk: high
    scope:
      - src/extract/pipeline.rs::run
      - src/extract/link.rs
  - id: SPEC-7.1-O3
    says: >
      Extraction is deterministic and memoized, and calls no LLM. The backend
      runs at most once per episode id, and extracting the same episode again
      returns identical facts. Output holds one fact per (subject, predicate,
      object), with the highest confidence, sorted by that triple. A backend
      error yields no facts and counts as a failure in stats().
    kind: property
    risk: medium
    scope:
      - src/extract/pipeline.rs::extract
      - src/extract/mod.rs
      - src/extract/gline.rs
```

## 7.2 Out of scope

- Coreference beyond first-person pronouns ("she", "the company").
- Temporal negation ("used to work at"). This is a supersession, so it belongs
  to the bi-temporal layer rather than to extraction.
- Accuracy targets for the ONNX models and the entity database. Measure those
  with MCC on a labelled set (glinker-rs `glinker eval` reports MCC for
  linking). They are not pass/fail checks for this gate.
