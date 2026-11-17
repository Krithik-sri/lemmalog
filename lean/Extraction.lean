/-
  Formal guarantees for the extraction fixes (core Lean 4, no Mathlib).
  Check with `lean Extraction.lean`.

  Rust ↔ Lean correspondence
  * `gline::map_relations` + `RELATION_PROMPT_OFFSET`  ↔ `promptSlice`, theorem `prompt_offset_roundtrip`
  * `pipeline::run` span guard (BadSpan)               ↔ `validSpan`, theorem `valid_span_slice_exact`
  * `DEFAULT_RELATIONS` `inverse` + swap in `run`      ↔ `orient`, theorems `manager_of_is_reports_to`, `orient_involutive`
  * `pipeline::resolve_relations` pair search          ↔ `pairs`, theorem `resolved_subject_ne_object`
  * `eval::score` counting                             ↔ `countTp`/`countFn`, theorem `tp_add_fn_eq_positives`

  Strings are modelled as lists (of bytes or chars); a Rust slice `s[a..b]` is
  `(s.drop a).take (b - a)`.
-/

/-- `s[a..b]`. -/
def slice {α} (s : List α) (a b : Nat) : List α := (s.drop a).take (b - a)

/-! ### 1. gline-rs prompt offsets map back to the sentence

gline-rs runs relation extraction on `prompt = prefix ++ s` (prefix includes the
separating space) and reports object spans in prompt coordinates. Subtracting
`prefix.length` gives the sentence span covering the same text. -/

theorem prompt_offset_roundtrip {α} (pfx s : List α) (a b : Nat) :
    slice (pfx ++ s) (a + pfx.length) (b + pfx.length) = slice s a b := by
  unfold slice
  have hd : (pfx ++ s).drop (a + pfx.length) = s.drop a := by
    rw [Nat.add_comm, ← List.drop_drop, List.drop_left]
  rw [hd, Nat.add_sub_add_right]

/-- The pre-fix code used the prompt span directly as a sentence span: past the
end of the sentence it covers nothing, so the object was never found. -/
theorem unmapped_span_past_end_is_empty {α} (s : List α) (a b : Nat) (h : s.length ≤ a) :
    slice s a b = [] := by
  unfold slice
  rw [List.drop_eq_nil_of_le h, List.take_nil]

/-! ### 2. Validated spans slice exactly

`run` rejects (BadSpan) any mention with `¬ (start ≤ end ∧ end ≤ len)`; every span
that reaches NegEx or the linker therefore slices to exactly `end - start`
elements (the Rust slice cannot panic; char boundaries are checked separately). -/

def validSpan {α} (s : List α) (a b : Nat) : Prop := a ≤ b ∧ b ≤ s.length

theorem valid_span_slice_exact {α} (s : List α) (a b : Nat) (h : validSpan s a b) :
    (slice s a b).length = b - a := by
  unfold slice
  rw [List.length_take, List.length_drop]
  exact Nat.min_eq_left (Nat.sub_le_sub_right h.2 a)

theorem valid_span_context_total {α} (s : List α) (a b : Nat) (h : validSpan s a b) :
    s.take a ++ slice s a b ++ s.drop b = s := by
  unfold slice
  rw [List.append_assoc]
  conv => rhs; rw [← List.take_append_drop a s]
  congr 1
  conv => rhs; rw [← List.take_append_drop (b - a) (s.drop a)]
  congr 1
  rw [List.drop_drop, Nat.add_sub_cancel' h.1]

/-! ### 3. Relation direction

lemmalog reads `X --manager--> Y` as "X reports to Y". `orient` is the swap
`run` applies for relations marked `inverse`. -/

structure RelDef where
  label : String
  pred : String
  inverse : Bool

def reportsTo : RelDef := ⟨"reports to", "manager", false⟩
def managerOf : RelDef := ⟨"manager of", "manager", true⟩

def orient (r : RelDef) (subj obj : String) : String × String × String :=
  if r.inverse then (obj, r.pred, subj) else (subj, r.pred, obj)

/-- "Dana is the manager of Bob" and "Bob reports to Dana" give the same fact. -/
theorem manager_of_is_reports_to (x y : String) :
    orient managerOf x y = orient reportsTo y x := by
  simp [orient, managerOf, reportsTo]

theorem orient_involutive (r : RelDef) (x y : String) :
    orient r x y = orient { r with inverse := !r.inverse } y x := by
  cases h : r.inverse <;> simp [orient, h]

/-! ### 4. The resolver never makes a mention its own subject

`resolve_relations` picks the best pair among `(subject, object)` candidate
spans with `subject ≠ object`; whatever it picks satisfies that. -/

def pairs {β} [DecidableEq β] (subs objs : List β) : List (β × β) :=
  (subs.flatMap fun s => objs.map fun o => (s, o)).filter fun p => p.1 ≠ p.2

theorem resolved_subject_ne_object {β} [DecidableEq β] (subs objs : List β) (p : β × β)
    (h : p ∈ pairs subs objs) : p.1 ≠ p.2 := by
  unfold pairs at h
  have := (List.mem_filter.mp h).2
  simpa using this

/-! ### 5. Scoring counts every labelled positive exactly once

`eval::score` adds each `+` triple to TP if extracted, else FN. -/

def countTp {τ} (got : τ → Bool) (pos : List τ) : Nat := (pos.filter got).length
def countFn {τ} (got : τ → Bool) (pos : List τ) : Nat := (pos.filter fun t => !got t).length

theorem tp_add_fn_eq_positives {τ} (got : τ → Bool) (pos : List τ) :
    countTp got pos + countFn got pos = pos.length := by
  induction pos with
  | nil => rfl
  | cons t ts ih =>
    unfold countTp countFn at *
    cases h : got t <;> simp [List.filter_cons, h] <;> omega
