# Changelog

## 0.1.0 — 2026-04-29

### Drop `colors` field — algorithmic extraction belongs outside the VLM

`colors` is no longer a `SceneAnalysis` field. The five remaining
detection-array fields (`subjects` / `objects` / `actions` / `mood` /
`lighting`) keep their semantics; the public field count on
`SceneAnalysis` drops from ten to nine.

**Why:** dominant-color extraction is a closed-form image-processing
problem — k-means or histogram clustering on pixel data, plus a
perceptual-distance lookup against a named-color dataset (xkcd's
crowd-named survey is the standard, but NBS-ISCC, Pantone, or any
custom vocabulary works the same way). Asking a 3B-parameter VLM to
emit color labels is strictly worse on every axis a search index
cares about:

- **Accuracy:** pixel statistics are ground truth; the VLM
  approximates from a language head with vocabulary drift.
- **Determinism:** xkcd-NN over LAB returns the same label every
  call; VLM color labels drift across runs even under greedy decoding
  (constrained JSON-schema decoding bounds the *shape* but not the
  *vocabulary*).
- **Speed:** lookup is microseconds per frame; VLM inference is
  tens to hundreds of milliseconds per token.
- **Granularity:** the lookup table is swappable per-consumer
  without retraining; the VLM is locked to whatever color
  vocabulary it generalized at training time.

Round-10 H1 already showed the prompt couldn't enumerate color
examples without biasing the model away from them in deterministic
mode (`presence_penalty` over `seq.get_toks()` covers prompt +
generated tokens, so listing `"warm tones"` / `"neon colors"` /
`"blue and white"` as examples gave those exact labels a
`-presence_penalty` logit shift before generation began). That
constraint already made `colors` the weakest field in the schema —
the model couldn't even be steered with examples. Round-13 H1 then
showed the field couldn't clear the indexable-content threshold on
its own: a payload with only `colors: ["blue"]` was treated as a
regression and rejected. Two consecutive review rounds had already
quarantined `colors` as low-signal; this round acknowledges it
shouldn't be a VLM output at all.

`lighting` stays. Pixel statistics handle the *low-level* lighting
vocabulary (brightness, contrast, color temperature) trivially, but
the *semantic* terms callers actually search for ("backlit",
"spotlight", "golden hour", "low-key", "dramatic") need scene-level
visual reasoning — subject-vs-background luminance segmentation,
focused-peak detection, directional-gradient analysis combined with
color temperature. Replacing the field with histogram heuristics
would lose information the VLM genuinely contributes; replacing it
with a small dedicated lighting classifier is feasible but not
packaged off-the-shelf at silero-vad's level (academic CNNs exist;
a drop-in equivalent does not). The algorithmic ROI for `colors`
doesn't carry over.

The algorithmic color pipeline (k-means dominants + LAB-NN naming +
palette descriptors like "warm tones" / "earth tones" computed from
the dominants' hue/saturation/lightness statistics) belongs in
whatever orchestrates keyframes → final indexed record, not in this
crate. The crate's positioning — *"Qwen3-VL structured-output
engine"* — explicitly scopes it to VLM machinery; non-VLM signal
extraction would dilute that contract and pull in image-processing
dependencies the VLM path doesn't need.

API changes (breaking):

- **Removed:** `SceneAnalysis::colors`, `with_colors`,
  `set_colors`. Consumers that need a colors list compute it from
  the keyframes themselves and merge with this crate's output.
- **Schema change:** `SCENE_PROMPT` no longer requests `colors`;
  `REQUIRED_FIELDS` drops `"colors"`; the JSON schema drops the
  `colors` property; `QwenScenePayload` drops the `colors` field.
  A model response that emits `colors` now fails parsing
  (`#[serde(deny_unknown_fields)]`), which is the desired
  behavior — surfaces drift loudly instead of silently dropping
  the value.
- **Doc rewrites:** module-level docstring (records *why* `colors`
  is excluded and *why* `lighting` stays, so the contract is
  self-documenting), `SceneAnalysis` docstring, `with_accept_empty`
  doc, `parse()` comment block, `lacks_indexable_content` doc, and
  `DetectionLabels` doc all updated to reflect the five-bucket
  detection shape.

Test surface:

- **Deleted:** `reject_colors_only_payload_by_default` — the field
  no longer exists, so the regression case is structurally
  impossible.
- **Updated:** every multi-field JSON fixture in the parser tests
  drops `"colors": []` / `"colors": [...]` (15 fixtures via
  bulk replace_all + 6 single-line fixtures by hand). The
  `accept_all_required_fields_empty_payload_when_opted_in` test
  drops the corresponding `result.colors().is_empty()` assertion.
- **Updated:** `array_elements_are_not_comma_split` retains its
  round-7 comma-bearing-label coverage by moving the comma-bearing
  label onto the `lighting` bucket
  (`"natural, dramatic backlight"`) — `lighting` is still a
  `DetectionLabels` field with the same string-fallback contract,
  so the test invariant (detection arrays don't comma-split inside
  string elements) is unchanged.
- **Updated:** `reject_attribute_only_payload_by_default` now pins
  that the two remaining attribute buckets (`mood` + `lighting`)
  together fail the threshold; the test docstring rewrites "three
  single-attribute reject tests" → "two".
- **Updated:** `scene_prompt_does_not_enumerate_value_tokens`
  drops the three colors-vocabulary banned tokens (`"warm tones"`,
  `"blue and white"`, `"neon colors"`). The prompt no longer has
  a colors field for them to leak into, so the guards are
  vacuous; removing them keeps the banned-token list aligned with
  fields that actually exist.
- **Updated:** `tests/integration_scene.rs` — the comment on the
  idempotency assertion drops the stale "seven detection lists
  with their (label, confidence) pairs" enumeration (already
  inaccurate after round-14's `Vec<SmolStr>` migration; rewriting
  the line for the field-count change was the prompt to fix it).
- Lib test count: 37 → 36 (-1: `reject_colors_only_payload_by_default`
  deleted; round-14's deletion of `parse_with_custom_default_confidence`
  took the previous total from 38 → 37 even though that round's
  CHANGELOG entry didn't restate the absolute count).

Verification:
- `cargo fmt --check` → clean.
- `cargo clippy --lib --tests --examples --locked -- -D warnings`
  → clean.
- `cargo test --lib --locked` → 36 pass (28 in `scene::tests`,
  8 in other modules).
- Real-model integration test not re-run for this change — the
  schema removal is parser-side only and the model can still emit
  any field it wants without affecting non-`colors` paths; the
  deterministic-idempotency contract is structurally unchanged.

### Round-14: drop per-detection confidence, simplify to `Vec<SmolStr>`

The previous design wrapped every emitted detection in
`Detection { label: SmolStr, confidence: f32 }` and stamped a
hardcoded `default_confidence = 0.8` on each entry, with a `SceneTask`
knob to override the default. Round 14 removes the `Detection` type
entirely. Six detection-array fields on `SceneAnalysis` —
`subjects` / `objects` / `actions` / `mood` / `lighting` / `colors` —
become `Vec<SmolStr>`. The `classifications` field (previously a
single-element `Vec<Detection>` derived from `scene` to mirror
`findit-proto`'s wire shape) is gone too.

**Why:** the `confidence: 0.8` value was a placeholder, not a feature.
It didn't enter Gemma-style embeddings at index time (text → vector,
no confidence channel), didn't enter ANN similarity at search time,
and didn't drive any re-ranker. Two real use cases for variable
confidence — UX score ("92% match") and LanceDB re-ranking weight —
both need calibrated numbers, but VLM self-reported confidence is
poorly calibrated, and a flat 0.8 is a no-op for both. The honest
move is to expose label lists without a fake-precision number, and
defer real per-detection scoring to later sources where it can
actually be calibrated:

- search-time embedding similarity (per-query, not per-scene),
- frame-frequency aggregation (count keyframes that emit each label
  across a single scene's keyframe set),
- constrained-decoder logprob extraction (if mistralrs exposes it).

If a downstream consumer eventually maps `qwen::scene::SceneAnalysis`
to `findit-proto::database::SceneVlmResult`, the `confidence` /
`dominance` fields on the proto side can be filled with one of the
above sources or a fixed sentinel — the qwen crate just shouldn't
manufacture the number.

API changes (breaking):

- **Removed:** `qwen::scene::Detection` (the entire struct + impl).
- **Removed:** `SceneTask::default_confidence`, `with_default_confidence`,
  `set_default_confidence`. The `f32` parameter on the internal
  `into_scene_analysis()` is gone too.
- **Removed:** `SceneAnalysis::classifications`,
  `with_classifications`, `set_classifications`. Consumers that need
  to map `scene` into a singleton classification list do so on their
  side now.
- **Type change:** `subjects()` / `objects()` / `actions()` /
  `mood()` / `lighting()` / `colors()` accessors now return
  `&[SmolStr]` (was `&[Detection]`); the corresponding `with_*` /
  `set_*` setters take `Vec<SmolStr>`.

Test surface:

- `parse_with_custom_default_confidence` deleted (the only test that
  exercised the removed knob).
- All `result.subjects()[0].label()` / `.objects()[0].label()` etc.
  callsites in the parser tests collapse to direct `result.subjects()[0]`
  comparisons against `&str` (SmolStr derefs).
- The `assert!(result.classifications().is_empty())` assertion in
  `accept_all_required_fields_empty_payload_when_opted_in` is dropped.
- Lib test count: 38 → 37 (one deletion, no additions; the H1 from
  round 13 covered the predicate paths and those are unchanged).

Documentation:

- Module-level rustdoc on `src/scene.rs` rewritten to drop `Detection`
  references and explain the round-14 rationale.
- `SceneAnalysis` struct doc rewritten.
- `lib.rs` rustdoc updated to point at the round-14 entry.
- `README.md`: the "Status" and "Architecture" sections drop
  `Detection` mentions; the per-category collapse paragraph
  (round-1 design rationale) is gone since there's nothing to
  collapse anymore.

Verification:
- `cargo fmt --check` → clean
- `cargo clippy --lib --tests --examples --locked -- -D warnings` → clean
- `cargo test --lib --locked` → 37 pass (was 38; -1 deleted test)
- `cargo check --all-targets --features integration --locked` → clean
- Real-model integration: both tests pass against `qwen3-vl-2b`
  (`scene_task_against_real_model`,
  `deterministic_run_is_idempotent`; 97.33s) — `SceneAnalysis`
  derives `PartialEq` over `Vec<SmolStr>` fields, so the
  `assert_eq!(result_a, result_b)` idempotency check works
  unchanged from the prior `Vec<Detection>` shape.

### Round-13 fixes

- **H1: substantive-detection path narrowed to `subjects` /
  `objects` / `actions` only.** Codex round 13 flagged that the
  round-12 detection-rich path treated all six detection buckets
  equally (`subjects`, `objects`, `actions`, `mood`, `lighting`,
  `colors`), so a regression that emitted only `colors: ["blue"]`
  or `mood: ["calm"]` (everything else empty) bypassed the
  no-usable-fields gate and got written to the search index as a
  single-attribute stub. The fix categorically separates
  **substantive** detection buckets (`subjects` / `objects` /
  `actions` — who/what/where) from **style/attribute** buckets
  (`mood` / `lighting` / `colors` — search filter axes). The
  substantive-detection path now requires at least one of the
  three substantive buckets to be non-empty; style-attribute-only
  payloads reject by default.
  - `lacks_indexable_content` body updated; doc-comments on the
    method, `accept_empty` accessors, the `parse()` comment block,
    and `SceneTask::new` all rewritten to describe the narrowed
    substantive-detection path and the categorical separation.
  - Test renames + additions:
    - `accept_single_detection_bucket_payload` →
      `accept_subjects_only_payload` (clearer name).
    - New `accept_objects_only_payload`,
      `accept_actions_only_payload` — each substantive bucket alone
      clears the threshold.
    - New `reject_mood_only_payload_by_default`,
      `reject_lighting_only_payload_by_default`,
      `reject_colors_only_payload_by_default` — Codex's specific
      single-attribute regressions.
    - New `reject_attribute_only_payload_by_default` — pins that
      stacking all three style-attribute buckets together still
      rejects (the categorical separation matters).
  - Lib test count: 32 → 38 (+6 round-13: 2 new `accept` tests
    for objects/actions; 3 new `reject` tests for
    mood/lighting/colors-only; 1 new `reject` for combined
    attribute-only; the `accept_subjects_only_payload` rename is
    not a count change).

- **H2 (declined again, same reasoning as rounds 9–12):**
  `mistralrs = "0.8"` stays unpinned. The recurring concern about
  patch drift is real but the disposition is unchanged: user
  direction has been consistent across rounds 4, 5, 7, 8, 9, 10,
  11, 12, 13 to keep `"0.8"` so downstream workspaces can pick up
  patch fixes; the committed `Cargo.lock` pins this repo's CI; the
  manual `CONTRIBUTING.md` "Bumping mistralrs" checklist is the
  gate. Severity escalation in this round (medium → high) doesn't
  change the user-aligned policy.

Verification:
- `cargo fmt --check` → clean
- `cargo clippy --lib --tests --examples --locked -- -D warnings` → clean
- `cargo test --lib --locked` → 38 pass (was 32; +6 round-13 tests)
- `cargo check --all-targets --features integration --locked` → clean
- Real-model integration: both tests pass against `qwen3-vl-2b`
  (`scene_task_against_real_model`,
  `deterministic_run_is_idempotent`; 94.51s) — round-13 is strictly
  more restrictive than round-12 (rejects style-attribute-only
  payloads), but the real model on airport keyframes produces
  description + tags + substantive detections together, so neither
  the prose+keyword path nor the substantive-detection path
  changes for the typical case.

### Round-12 fixes

- **H1: composite indexable-content predicate
  (round 9 ↔ round 12 reconciliation).** Codex round 12 flipped
  round 9's H1 framing: the round-9 predicate (`description` AND
  `tags` both populated) was rejecting payloads that had populated
  detection buckets (`subjects` / `objects` / `actions` / `mood` /
  `lighting` / `colors`) but empty `description` and empty `tags`.
  Those payloads carry real structured search metadata — the
  per-category fields are the whole reason this crate exposes them
  — so rejecting them as `NoUsableFields` was discarding
  otherwise-indexable content on partial decoder/model misses.
  The fix is a composite predicate that accepts when **either**
  path holds:
  - **Prose + keyword path:** `description` AND `tags` both
    populated (the round-9 threshold; matches the integration
    test's smoke pass criterion).
  - **Detection-rich path:** at least one of `subjects` /
    `objects` / `actions` / `mood` / `lighting` / `colors` is
    non-empty (round 12's case).

  Tags-only, scene-only, description-only, shot_type-only, and
  fully-empty payloads still fail both paths and remain rejected.
  `scene` and `shot_type` are intentionally excluded from the
  detection-rich path — they're single-label fields, and the
  round-9 intent to flag those-as-only as regressions is preserved.
  - `lacks_indexable_content` body updated; doc-comments on the
    method, `accept_empty` accessors, the `parse()` comment block,
    and `SceneTask::new` all rewritten to describe the composite
    predicate.
  - Two new tests:
    - `accept_detection_rich_payload_with_empty_description_and_tags`
      pins the round-12 case (subjects + objects + actions
      populated, description+tags empty → accept).
    - `accept_single_detection_bucket_payload` pins that any
      single non-empty detection bucket clears the threshold, so
      a future refactor can't accidentally narrow the predicate
      to "all detection buckets non-empty".
  - Lib test count: 30 → 32. All five round-9 reject tests
    (`reject_all_required_fields_empty_payload_by_default`,
    `reject_tags_only_payload_by_default`,
    `reject_scene_only_payload_by_default`,
    `reject_description_only_payload_by_default`, plus the
    null-required tests) continue to pass.

- **M2: CI now compiles the integration test.** Codex flagged
  that the previous `cargo check --all-targets --locked` step
  ran without `--features integration`, while
  `tests/integration_scene.rs` is gated behind a crate-level
  `#![cfg(feature = "integration")]`. That meant the integration
  test was effectively empty in CI — accessor renames or
  signature drift could rot the test until a manual
  `QWEN_MODEL_PATH` run. New CI step:
  `cargo check --all-targets --features integration --locked`.
  Compiles the integration test body without executing it
  (model + GPU still required for the real run, gated on
  `QWEN_MODEL_PATH`).

Verification:
- `cargo fmt --check` → clean
- `cargo clippy --lib --tests --examples --locked -- -D warnings` → clean
- `cargo test --lib --locked` → 32 pass (was 30; +2 new round-12 tests)
- `cargo check --all-targets --locked` → clean
- `cargo check --all-targets --features integration --locked` → clean
- Real-model integration: both tests pass against `qwen3-vl-2b`
  (`scene_task_against_real_model`,
  `deterministic_run_is_idempotent`; 93.05s) — the composite
  predicate is strictly more accepting than the round-9 predicate
  for any given payload, so on the typical real-model output
  (description + tags both populated) the test sees identical
  SceneAnalysis values to round 9, plus the new detection-rich
  partial-output path is now reachable without a parse error.

### Round-11 fixes

- **H1 (algorithm fix already in place; regression test added):** Codex
  round 11 re-flagged that "Branch HEAD's `SCENE_PROMPT` enumerates
  canonical output values" alongside the deterministic-mode
  `presence_penalty`. The actual algorithm fix is already applied —
  round 10 H1 scrubbed every `e.g. "..."` enumeration from the
  `SCENE_PROMPT` constant. Codex read the source file directly (so
  it had access to the working-tree version), but the cited line
  range (`src/scene.rs:378-386`) is the round-10 explanatory `//`
  Rust comment block above the constant — that comment *names* the
  banned tokens to explain why they were removed, and Codex
  conflated the comment text with prompt text. `//` comments are
  stripped at compile time; only the contents of the `SCENE_PROMPT`
  string literal reach the model. The string-literal contents were
  audited again here and remain free of value-token enumerations
  (single-shot label scenes, descriptive constraints only).
- **H1 follow-up: regression test added.** Codex's recommendation
  to add a test that prevents value tokens from being re-introduced
  is sensible regardless of the false positive. New test
  `scene_prompt_does_not_enumerate_value_tokens` scans the
  `SCENE_PROMPT` constant (not the surrounding source) for a curated
  list of distinctive value-tokens that appeared in the
  pre-round-10 prompt's `e.g.` enumerations: "stage performance",
  "middle-aged man", "golden retriever", "birthday cake", "vintage
  red sports car", "cutting cake", "taking photos", "wide shot",
  "close-up", "medium shot", "over-the-shoulder", "celebratory",
  "warm tones", "blue and white", "neon colors", "natural light",
  "low light", "backlit". The test is a guard against accidental
  regression (a future edit copy-pasting an `e.g. "..."` line
  from somewhere); it is not a defense against deliberate edits
  (a determined reverter can also remove tokens from the test
  list). Lib test count: 29 → 30.

- **M1 (declined again, same reasoning as rounds 9 + 10):**
  `mistralrs = "0.8"` stays unpinned. User direction unchanged; the
  manual `CONTRIBUTING.md` "Bumping mistralrs" checklist remains
  the gate. Documented in CHANGELOG so the position is auditable
  across review rounds.

Verification:
- `cargo fmt --check` → clean
- `cargo clippy --lib --tests --examples --locked -- -D warnings` → clean
- `cargo test --lib --locked` → 30 pass (was 29; +1 regression test)

### Round-10 fixes

- **H1: scrub example values from `SCENE_PROMPT`.** Codex round 10
  flagged that the deterministic-mode default applies
  `presence_penalty 1.5` over `seq.get_toks()` (prompt + generated
  tokens) — and the prior `SCENE_PROMPT` enumerated 30+ value
  examples (`"office"`, `"street"`, `"kitchen"`, `"stage performance"`,
  `"birthday cake with candles"`, `"vintage red sports car"`,
  `"cutting cake"`, `"taking photos"`, `"celebratory"`, `"tense"`,
  `"calm"`, `"wide shot"`, `"close-up"`, `"natural light"`,
  `"warm tones"`, `"blue and white"`, `"golden retriever"`, etc.).
  In greedy decoding (no sampling spread to dilute the bias), every
  one of those tokens got a flat `-1.5` logit shift before the model
  emitted anything. Scenes whose correct labels appeared in the
  enumeration were systematically pushed away from those terms,
  producing stable-but-worse search metadata across runs. The
  integration test caught only non-empty + idempotent contracts —
  not semantic-recall — so the bias hid behind a green CI signal.
  - Fix: rewrote the per-field instructions to use
    descriptive constraints instead of enumerated value examples.
    Example before:
    `scene: short scene category in English (e.g. "office", "street", "kitchen", "stage performance")`.
    After:
    `scene: a single short scene-category label in lowercase English, 1-3 words, no full sentence.`
  - The format guidance (word counts, lowercase) anchors the output
    shape without listing the value vocabulary the deterministic
    sampler is penalized against. Residual prompt-vocabulary bias
    only hits scaffolding/instruction tokens (e.g. "scene",
    "describing", "lowercase") which the model is unlikely to emit
    as values, and the JSON schema constraint preserves field names
    and structure regardless.
  - Added an in-source comment block above `SCENE_PROMPT` documenting
    the prompt-hygiene constraint so future edits don't silently
    re-introduce the bias.
  - Updated `RequestOptions::deterministic()` doc-comment: removed
    the now-stale `"office"` / `"stage performance"` /
    `"birthday cake with candles"` example list, replaced with a
    pointer to `SceneTask`'s value-token-free prompt and a note
    telling authors of custom `Task` implementations to follow the
    same prompt hygiene.
  - Pre-existing parser tests still pass — the test JSON fixtures
    are parser-side inputs (the parser doesn't care that "office"
    or "wide shot" are no longer in the prompt; it just parses
    what callers pass it). The real-model integration test was
    re-run against `qwen3-vl-2b` on the airport keyframes:
    `scene_task_against_real_model` (model still produces
    `description` and `tags` both populated, clearing the round-9
    indexable-content threshold) and `deterministic_run_is_idempotent`
    (two runs against identical inputs still produce bit-identical
    `SceneAnalysis`) both pass in 94.46s. Round-trip /
    idempotency / minimum-content invariants confirmed; quantifying
    the quality DELTA against the prior prompt still needs a
    labelled-fixture eval suite (future work, tracked under "Known
    limitations").

- **M1 (declined again, same reasoning as round 9): `mistralrs = "0.8"`
  stays unpinned.** Codex re-flagged the recurring concern that
  unpinned `0.8.x` lets downstream workspaces silently resolve a
  patch that breaks the deterministic-mode assumptions. User
  direction has been consistent across rounds 4, 5, 7, 8, 9: keep
  `"0.8"` unpinned so downstream picks up patch-level fixes; the
  committed `Cargo.lock` pins this repo's CI; `CONTRIBUTING.md`
  has the manual "Bumping mistralrs" checklist as the gate.
  Automated CI gating isn't viable (no GPU + 4 GB model on
  GitHub-hosted runners). Decision unchanged.

Verification:
- `cargo fmt --check` → clean
- `cargo clippy --lib --tests --examples --locked -- -D warnings` → clean
- `cargo test --lib --locked` → 29 pass (no test count change)
- `cargo check --all-targets --features integration --locked` → clean
- Real-model integration: both tests pass against `qwen3-vl-2b`
  (`scene_task_against_real_model`,
  `deterministic_run_is_idempotent`; 94.46s)

### Round-9 fixes

- **H1: parser predicate tightened from all-empty to "lacks
  indexable content".** Codex round 9 flagged that the round-8
  `payload.is_empty()` gate (rejecting only when **every** field
  was empty) let through partial-empty regressions: a model that
  emitted only `tags` or only `scene` (everything else blank) parsed
  successfully and produced a sparse `SceneAnalysis` that the
  indexing pipeline would silently use to overwrite previously-rich
  search records. The fix tightens the predicate to require
  `description` AND `tags` both populated by default — the same
  threshold the real-model integration test
  (`scene_task_against_real_model`) already pins as the smoke pass
  criterion against `qwen3-vl-2b` on airport keyframes.
  - Internal method renamed: `QwenScenePayload::is_empty()` →
    `lacks_indexable_content()`. The new body: `description.is_none()
    || tags.0.is_empty()` (description's deserializer collapses
    empty/whitespace strings to `None`, so `is_none()` covers both
    the JSON-`null` and JSON-`""` cases).
  - The `accept_empty` field name is unchanged (its semantic broadens
    to "accept payloads regardless of content threshold"); the
    `with_accept_empty` and `accept_empty()` accessor doc-comments
    were rewritten to describe the new behavior, including the
    rationale for the `description AND tags` threshold.
  - Four new parser tests pin the contract:
    - `reject_tags_only_payload_by_default` (Codex's specific case)
    - `reject_scene_only_payload_by_default` (Codex's other case)
    - `reject_description_only_payload_by_default` (symmetric)
    - `accept_minimal_indexable_payload` (lower bound: description +
      tags, everything else empty → accept; pins that the parser
      doesn't over-reject sparse-but-valid scenes)
  - Existing tests cover the no-change cases:
    `reject_all_required_fields_empty_payload_by_default` (still
    rejects), `accept_all_required_fields_empty_payload_when_opted_in`
    (opt-in path bypasses the predicate, accepts everything), and
    every "happy path" test (which already had both `description` and
    `tags` populated, so they continue to pass).
  - Lib test count: 25 → 29.
  - Trade-off: this is stricter than round 8. A genuine
    low-information frame (blank, fade-to-black, plain color) where
    the model legitimately complies with SCENE_PROMPT's "use empty
    arrays or empty strings when unknown" instruction is now
    rejected by default. Callers that distinguish "low-information
    scene" from "regression" elsewhere — e.g., scenesdetect's
    keyframe scoring — opt into pass-through with
    `SceneTask::with_accept_empty(true)`. The trade-off is the same
    one Codex flagged in round 8 (fail-strict prevents silent index
    drift on regressions); round 9 just extends the predicate to
    cover sparse-but-not-fully-empty regressions too.

- **M1 (declined): `mistralrs = "0.8"` stays unpinned.** Codex
  re-flagged the recurring concern that downstream workspaces
  resolving a later 0.8.x patch could change the engine's
  deterministic-mode assumptions (presence-penalty context,
  schema-constrained output, the missing seed/repetition APIs)
  without this repo's `--locked` CI gate exercising it. User
  direction across rounds 4, 5, 7, 8 has been to keep the loose
  `"0.8"` pin so downstream workspaces can pick up patch-level
  fixes; that direction stands. The committed `Cargo.lock` pins
  the exact resolved version (currently 0.8.1) for this repo's
  reproducible builds, and `CONTRIBUTING.md` already has an
  explicit "Bumping mistralrs" checklist (re-verify sampler
  context, schema constraint wiring, `add_image_message`
  signature; run `--features integration` against the local
  model). An automated CI gate isn't viable: GitHub-hosted
  runners don't have GPU + the ~4 GB model, so the real-model
  integration test can't run there. The manual checklist remains
  the gate.

Verification:
- `cargo fmt --check` → clean
- `cargo clippy --lib --tests --examples --locked -- -D warnings` → clean
- `cargo test --lib --locked` → 29 pass (was 25; +4 new H1 tests)
- `cargo check --all-targets --features integration --locked` → clean

### Sampler config collapsed into `EngineOptions::request: RequestOptions`

The `EngineOptions::deterministic: bool` flag and the matching
deterministic-vs-stochastic branch in `Engine::run_with` are gone.
What used to be two configuration concepts — a boolean flag plus four
sampler knobs — is now one: the engine carries a `RequestOptions`
default and `Engine::run_with` applies it (or the caller's per-call
override) uniformly. The "deterministic" choice is encoded in the
sampler values themselves, not in a side flag.

API changes:

- New `RequestOptions::deterministic()` constructor — greedy values
  (`temperature 0.0`, `top_p 1.0`, `top_k 1`, `presence_penalty 1.5`).
  Bit-stable output for identical inputs. Owns the long
  greedy-vs-presence-penalty trade-off documentation that previously
  lived on `EngineOptions::deterministic`.
- `EngineOptions::request: RequestOptions` replaces `deterministic: bool`.
  `EngineOptions::new` now embeds `RequestOptions::deterministic()` by
  default, preserving the indexing-safe behavior callers got from the
  old `deterministic = true` default.
- New accessors `EngineOptions::request()` / `with_request` / `set_request`
  (scenesdetect-style getter / `with_*` / `set_*` triple). The old
  `deterministic` / `with_deterministic` / `set_deterministic` are
  removed. Migration: `with_deterministic(false)` becomes
  `with_request(RequestOptions::new())`; `with_deterministic(true)` was
  the default — drop it.
- `Engine::deterministic()` accessor removed; new `Engine::request()`
  accessor returns the engine-level default profile.
- `Engine::run` now passes `self.options.request()` to `Engine::run_with`
  (was: `&RequestOptions::default()`). Existing callers that used the
  old `EngineOptions::new` defaults see no behavior change — the
  default profile was deterministic before and is still deterministic
  now, just expressed as a `RequestOptions` preset.
- `Engine::run_with` collapses to a single sampler-config code path:
  all four fields from `opts` are applied uniformly. The
  ~50-line deterministic-branch comment block is gone; that documentation
  now lives on `RequestOptions::deterministic()` where the preset is
  defined. The `tracing::instrument` field list drops `deterministic`
  and gains `temperature` (a single-value diagnostic for greedy vs
  stochastic in trace output).

Test surface:

- `engine_options_defaults_indexing_safe` →
  `engine_options_defaults_to_deterministic_request`. Asserts the four
  greedy values plus path / quantization / max_tokens.
- `engine_options_with_chains` / `engine_options_set_chains` updated
  to chain `with_request(RequestOptions::new())` (was
  `with_deterministic(false)`).
- New `request_options_deterministic_preset` pins the greedy values.
- Lib test count: 24 → 25.

README's "Deterministic mode (the default)" subsection became
"Indexing-safe sampler default"; the per-call overrides subsection
drops the deterministic-mode-ignores-temperature caveat (no longer
true). Integration test comments updated to reference the new opt-in
path.

Verification:
- `cargo test --lib --locked` → 25 pass
- `cargo clippy --lib --tests --examples --locked -- -D warnings` → clean
- `cargo check --all-targets --features integration --locked` → clean

### Decoupled from findit-proto

- The crate now owns its `SceneAnalysis` and `Detection` types in
  `qwen::scene`. Both follow the scenesdetect-style accessor pattern
  (private fields; `pub fn` getter, `with_*`, `set_*` triple per
  field; `#[cfg_attr(not(tarpaulin), inline(always))]` everywhere).
  No public fields anywhere in the crate's public API. `SceneAnalysis`
  derives `PartialEq` so the integration test's idempotency check
  collapses to a single `assert_eq!(result_a, result_b)`.
- Previously the parser produced
  `findit_proto::database::SceneVlmResult` directly and the crate had a
  path dependency on `../indexer/findit-proto`. That coupled qwen to
  findit-studio's wire shape and made the crate unusable by any other
  consumer.
- New shape: `SceneAnalysis` mirrors `SceneVlmResult` field-by-field
  (so a downstream service can map without re-running inference), but
  collapses the per-category detection types
  (`SubjectDetection` / `ObjectDetection` / `ColorDetection` / …)
  into a single `Detection { label, confidence }`. Justification: the
  model never emits bounding boxes or per-color dominance — the
  legacy code stamped `BoundingBox::default()` and reused the same
  float across `confidence`/`dominance`. Re-projecting the per-
  category shape is cleaner inside the downstream consumer.
- Cargo.toml drops `findit-proto = { path = "../indexer/findit-proto" }`.
  The crate now depends only on mistralrs, image, serde, serde_json,
  smol_str, thiserror, tracing.
- All 17 lib tests pass against the new types. Full real-model
  integration suite still depends on a pending re-run after the type
  rename.

### Per-call sampler overrides via `RequestOptions`

`Engine::run` previously used four hardcoded `const` sampler values
(temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5). Any
caller who needed a different sampler shape — colder for OCR-heavy
scenes, higher `top_k` for creative tagging, etc. — had to fork the
crate or accept the defaults.

API additions:

- New public `RequestOptions` struct with the four sampler knobs as
  private fields plus the standard scenesdetect-style accessor
  surface (`pub const fn` getter / `with_*` / `set_*` per field;
  `#[cfg_attr(not(tarpaulin), inline(always))]` on every accessor).
  Defaults via `RequestOptions::new()` / `RequestOptions::default()`
  match the Qwen3-VL Instruct model card and equal the previous
  hardcoded constants.
- New `Engine::run_with(task, images, opts: &RequestOptions)`. Same
  contract as `Engine::run` but uses the caller's sampler values.
- `Engine::run(task, images)` is now sugar over
  `Engine::run_with(task, images, &RequestOptions::default())`. No
  behavior change for existing callers; they still get the model-card
  profile.
- `RequestOptions` is re-exported from the crate root.

Deterministic-mode interaction (documented on
`EngineOptions::deterministic` and `RequestOptions`): in deterministic
mode, `temperature` / `top_p` / `top_k` from `RequestOptions` are
ignored (forced to `0.0` / `1.0` / `1` respectively). Only
`presence_penalty` is honored, in BOTH modes. Callers that genuinely
want greedy with no presence penalty can opt in via
`RequestOptions::default().with_presence_penalty(0.0)` and accept the
known repetition-loop hazard (documented at length in
`Engine::run_with`).

`repetition_penalty` and a sampler seed are still NOT exposed:
mistralrs 0.8 has no `set_sampler_repetition_penalty` and no
`set_sampler_seed`. The previous `repetition_penalty=1.0`-as-baseline
disclaimer (and the explicit warning against substituting
`set_sampler_frequency_penalty`) is preserved on `RequestOptions`.

The four `const SAMPLER_*` values that used to live in `engine.rs`
are gone — folded into `RequestOptions::new()`. New tests
(`request_options_defaults_match_model_card`,
`request_options_default_eq_new`, `request_options_with_chains`,
`request_options_set_chains`) pin both the values and the accessor
contract; lib test count: 20 → 24.

README has a new "Per-call sampler overrides" subsection with a
`run_with` example; the existing "Sampling defaults" table is
unchanged because the defaults didn't move.

Verification:
- `cargo test --lib --locked` → 24 pass
- `cargo clippy --lib --tests --examples --locked -- -D warnings`
  → clean

### API simplification: empty-string-as-absence on SceneAnalysis

`SceneAnalysis` previously wrapped `scene`, `description`, and
`shot_type` in `Option<SmolStr>`. Per user direction, those three
fields are now bare `SmolStr` with the empty string representing
absence. Rationale: `SCENE_PROMPT` already instructs the model to
"Use empty arrays or empty strings when a field is unknown", so
empty-as-absence is the domain semantic anyway, and the
`Option<&str>` accessor return type wasn't earning its complexity.

API changes:

- `scene()` / `description()` / `shot_type()` now return `&str`
  (was `Option<&str>`). Check `.is_empty()` to test for absence.
- `with_scene` / `with_description` / `with_shot_type` (and the
  `set_*` siblings) now take `impl Into<SmolStr>` (matches the
  `Detection::with_label` convention) instead of `Option<SmolStr>`.
  Pass an empty string (e.g. `""` or `SmolStr::default()`) to clear.
- The internal `QwenScenePayload` keeps `Option<String>` for the
  three fields — the deserializer helpers
  (`deserialize_optional_trimmed_string`,
  `deserialize_optional_single_label`) work in `Option<String>`
  terms — and `into_scene_analysis` collapses `None` to
  `SmolStr::default()` at the boundary.
- `is_empty()` on `QwenScenePayload` (used by the default-reject
  path of `accept_empty`) is unchanged in behavior.

Test updates:

- `parse_valid_json`: `result.scene()` / `.description()` are now
  `&str`, asserted against bare `"beach"` / `"Sunset over the
  ocean"` instead of `Some(...)`.
- `parse_shot_type_list_form`: `result.shot_type()` asserted
  against `"wide shot"` (bare).
- `accept_all_required_fields_empty_payload_when_opted_in`:
  `result.scene().is_none()` → `result.scene().is_empty()`, same
  for `description` and `shot_type`.

Integration test (`scene_task_against_real_model`): the loose
populated-content assertion changes from
`result.description().is_some()` to
`!result.description().is_empty()`.

Verification:
- `cargo test --lib --locked` → 20 pass
- `cargo clippy --lib --tests --examples --locked -- -D warnings`
  → clean
- Real-model integration: both tests pass against `qwen3-vl-2b`
  (`scene_task_against_real_model`,
  `deterministic_run_is_idempotent`; 113.89s)

### Round-8 fixes

- **F1 (revisited): opt-in `accept_empty` instead of always-accept.**
  Round 7 changed the default to accept all-empty payloads (rationale:
  prompt instructs the model to use empty values for unknown fields).
  Codex challenged that in round 8 with the symmetric counter: a
  decoder/model regression also produces all-empty, and silently
  overwriting good search metadata with an empty result is worse than
  failing. Both arguments have merit.
  - The fix splits the difference: `SceneTask::new` defaults to
    **reject** (`accept_empty = false`), so `parse` returns
    `ParseError::NoUsableFields` for any all-empty payload — surfacing
    regressions instead of masking them. Callers who legitimately
    want blank-frame pass-through opt in via
    `SceneTask::with_accept_empty(true)`.
  - `ParseError::NoUsableFields` is constructed again. The variant
    was already public; behavior is reverted, opt-in is additive.
  - Test rename: `accept_all_required_fields_empty_payload` →
    `reject_all_required_fields_empty_payload_by_default`. New test
    `accept_all_required_fields_empty_payload_when_opted_in` pins the
    opt-in path.
  - Lib test count: 19 → 20.

- **F2 (per user direction): `mistralrs` stays at `"0.8"`, comments
  go 0.8-generic.** Codex re-flagged the unpinned `"0.8"` for the Nth
  time. User direction is to keep the unpin and align comments with
  the actual Cargo.toml shape. Sweep:
  - `src/engine.rs` — `Engine::deterministic` doc and the `Engine::run`
    sampler-branch comment now refer to "mistralrs 0.8" generically
    with "re-verify when upgrading" rather than asserting "verified at
    0.8.1".
  - `.github/workflows/ci.yml` — header now says `mistralrs = "0.8"`
    and points at `CONTRIBUTING.md` for the upgrade workflow.
  - `CONTRIBUTING.md` — "Bumping mistralrs" section reworded to drop
    the 0.8.1-as-anchor framing.
  - `Cargo.lock` continues to pin the exact resolved version for
    this repo's CI builds; downstream workspaces still get whatever
    their own resolver picks.

Verification:
- `cargo test --lib --locked` → 20 pass (was 19; +1 new test, 1 rename)
- `cargo clippy --lib --tests --examples --locked -- -D warnings` → clean
- Real-model integration: both tests pass against `qwen3-vl-2b`
  (`scene_task_against_real_model`,
  `deterministic_run_is_idempotent`; 116.56s)

### Round-7 fixes

- **F1: all-empty payload now accepted as a valid empty
  `SceneAnalysis`.** SCENE_PROMPT explicitly tells the model "Use empty
  arrays or empty strings when a field is unknown". A blank frame /
  fade-to-black scene that complies returns exactly that shape, but
  the parser was rejecting it as `ParseError::NoUsableFields` —
  turning legitimate low-information output into a retry/indexing
  failure. The `if payload.is_empty()` rejection in
  `SceneTask::parse` is gone; an all-empty payload now produces a
  default `SceneAnalysis` and the caller decides whether to skip it
  (e.g., via `result.scene().is_none() && result.tags().is_empty()`).
  `ParseError::NoUsableFields` stays in the public API for forward
  compatibility but is no longer constructed.
  Test: `accept_all_required_fields_empty_payload`.

- **F2: detection arrays no longer comma-split scalar string
  fallback.** The legacy `StringList` deserializer treated a scalar
  string as a comma/semicolon/newline-separated list for *every* array
  field. That was the right resilience for the historically common
  tag-list-as-string drift, but wrong for detection-array fields where
  labels can naturally contain commas (e.g.
  `"red, white, and blue flag"`). If the constrained decoder drifts
  to a scalar string for `subjects`/`objects`/`actions`/`mood`/
  `lighting`/`colors`, the previous code would silently shred one
  comma-bearing label into several bogus detections.
  - Refactor: split `StringList` into two types:
    - `TagList` — used only for `tags`. Keeps the comma-split
      fallback (legacy resilience preserved for the field that
      historically needed it).
    - `DetectionLabels` — used for the six detection-array fields.
      Treats a scalar string as a single-element list (no splitting),
      preserving comma-bearing labels verbatim. Array form unchanged
      (each element trimmed and deduped).
  - Test contract change: `parse_mixed_shape_subjects` was renamed
    to `parse_array_form_subjects` and a sibling test
    `subjects_string_form_treated_as_single_label` verifies the new
    behavior. The original test's "comma-string yields N items"
    assertion was the symptom of the bug.

- **F3 (mistralrs pin): pushback per user direction.** Codex
  re-flagged the pin for the third time. User direction
  ("we do not want to pin 0.8.1") overrides; `Cargo.toml` stays at
  `"0.8"`. Mitigations remain in place: committed `Cargo.lock` pins
  the exact resolved version for this repo's reproducible builds,
  the in-source comments explicitly say "verified at 0.8.1; later
  0.8.x patches may change this — re-verify when upgrading", and
  `CONTRIBUTING.md` has an explicit "Bumping mistralrs" checklist.

Verification:
- `cargo test --lib --locked` → 19 pass (was 17; +2 new tests)
- `cargo clippy --lib --tests --examples --locked -- -D warnings` → clean
- `cargo check --all-targets --locked` → clean
- Real-model integration: both tests pass against `qwen3-vl-2b`
  (`scene_task_against_real_model`,
  `deterministic_run_is_idempotent`; 119.03s)

### Round-6 fixes

- **B1 (clippy 1.94 break): `doc_lazy_continuation` errors fixed.**
  The round-3 commit shipped with a doc-comment bullet list whose
  continuation lines used 3-space indent. Clippy 1.94 introduced a
  stricter `doc_lazy_continuation` lint that rejected this. Hand-fixed
  by removing a stray markdown-bullet ambiguity (a leading `+` token
  on a continuation line was being parsed as a sub-bullet, which made
  the indent-validation lint incoherent for that whole list item).
  CI didn't catch this earlier because the workflow only fires on
  pushes to `main` and on PRs, neither of which had been opened from
  the v0.1.0 branch.
- **B2 (lib.rs rustdoc lied about architecture): rewritten.** The
  crate-root doc comment still described a "two-layer" architecture
  with the scene preset producing `findit_proto::database::SceneVlmResult`
  directly. After the round-2 decoupling, the scene preset produces
  `qwen::scene::SceneAnalysis` and the crate has no `findit-proto`
  dependency. The lib.rs intro now reflects that, points readers at
  `SceneAnalysis`, and re-states the cancellation contract.
- **C1 (NaN hazard): documented.** `with_default_confidence` /
  `set_default_confidence` doc comments now explicitly warn that
  passing `f32::NAN` would propagate into every emitted detection's
  `confidence` field and break `PartialEq`-based equality (as used
  by `deterministic_run_is_idempotent`). Production path is safe
  because `SceneTask::new` hardcodes `0.8`; this is a hazard only
  for callers that override.
- **C2 (warmup caveat): documented.** `Engine::warmup` doc now
  notes that Metal kernel JIT specializes per tensor shape, so a
  1×1 warmup is not guaranteed to compile the kernels needed for
  720×1280-class keyframes. Treated as a "best-effort, swap for
  real-shape fixture if first-request latency still shows JIT
  cost" follow-up.
- **C3 (quality claim was anecdotal): hedge added.** The "Known
  limitations" entry on the deterministic-mode bias previously
  said "current output quality is reasonable" based on one
  real-model run against three airport keyframes. CHANGELOG now
  explicitly calls that data point what it is — anecdotal, not
  benchmarked — and labels any deterministic-mode quality claim
  as provisional until a labelled fixture set lands.
- **C4 (stale `SceneVlmResult` references in release notes):
  fixed.** The "Initial release / Scene preset" subsection
  incorrectly described v0.1.0 as producing
  `findit_proto::database::SceneVlmResult`. After the round-2
  decoupling, v0.1.0 actually ships `qwen::scene::SceneAnalysis`.
  The release-status subsection is now accurate; the historical
  fix-by-fix log preserves the old type names where they were
  true at the time.
- **C5 (clippy didn't lint examples/tests): scope expanded.**
  CI's clippy step now runs `--lib --tests --examples` instead of
  `--lib`. Catches lint regressions in the smoke binary or the
  integration test that the previous lib-only scope missed.
- **C7 (mistralrs upgrade discipline): CONTRIBUTING.md added.**
  New CONTRIBUTING.md includes an explicit checklist for
  bumping mistralrs: re-verify the sampler context behavior, the
  `add_image_message` signature, and the `Constraint::JsonSchema`
  wiring; then re-run `--features integration` against the local
  model. Closes the loop on the "verified at 0.8.1; later 0.8.x
  may differ" disclosure.

### Round-5 fixes

- **F3 (broken example): smoke binary updated for accessor surface.**
  When `SceneAnalysis` fields became private, `examples/smoke.rs` was
  not updated and stopped compiling. Codex flagged this. Fixed by
  switching every `result.scene` / `.description` / `.subjects` / etc.
  reference to the corresponding accessor method
  (`result.scene()` / `.description()` / ...). The README quick example
  had the same bug — also fixed.

- **CI now typechecks all targets.** Added `cargo check --all-targets
  --locked` to `.github/workflows/ci.yml`. Catches the regression
  class where `cargo build` (lib only) succeeds but `examples/` and
  `tests/` reference symbols that have changed since their last
  edit. Does not run the example or the real-model integration test
  — those require `~4 GiB qwen3-vl-2b` and `QWEN_MODEL_PATH`, which
  CI runners don't have.

- **F2 (mistralrs version coupling): comments now scoped generically.**
  Codex flagged that comments referenced `mistralrs 0.8.1` specifically
  while the manifest accepts any `0.8.x`. Per user direction the pin
  stays at `"0.8"` (Cargo.lock keeps this repo's builds reproducible),
  but the in-source comments now explicitly say "verified at the 0.8.1
  release; later 0.8.x patches may change this — re-verify when
  upgrading". Codex's recommendation to re-pin to `=0.8.1` is
  intentionally not applied; the user's earlier directive on this
  point overrides Codex.

### Known limitations / future work

- **Greedy + presence_penalty is biased against prompt vocabulary
  (Codex round-4/round-5 F1, kept after empirical verification).** In
  deterministic mode (the default), `presence_penalty=1.5` shifts
  every prompt token's logit by -1.5, including value examples in
  `SCENE_PROMPT` like "office", "stage performance", and
  "birthday cake with candles". Because temperature=0 has no sampling
  spread, this bias is not diluted. Codex recommended either removing
  the penalty or gating the default on a labelled-fixture quality
  test.

  **Quality evidence is anecdotal, not benchmarked.** The only data
  point we have is one real-model run against three airport keyframes
  extracted from `01_airport.mp4`: the model produced a sensible
  scene label, specific OCR ("Entrée interdite!"), and 16 search-
  relevant tags. That's not a labelled benchmark; it's one scene from
  one video. Whether the prompt-vocabulary bias measurably degrades
  output on a wider distribution of scenes is unknown. Until
  follow-up (a) below lands, treat any claim about "deterministic
  mode produces good labels" as provisional.

  Removing the penalty was tested and broke generation entirely
  (repetition loops blow past `max_tokens` mid-string). mistralrs 0.8
  has no generated-only repetition mechanism. **Future work:**
  (a) add a labelled fixture set that checks semantic label/tag
  quality, not just parseability and idempotency, and gate releases
  on it; (b) revisit if mistralrs ships a generated-only penalty in
  a future minor release. Both are tracked here rather than as design
  changes because the alternative (no rep control in greedy mode) is
  empirically worse than the
  current state.

### Adversarial-review fixes (committed during the v0.1.0 cycle)

- **Round 4 — F1 (kept with documentation): presence_penalty in
  deterministic mode is a documented trade-off, not a bug.** Codex
  flagged that mistralrs 0.8.1's `presence_penalty` is applied over
  `seq.get_toks()` (prompt + generated tokens, not just generated),
  verified at `mistralrs-core/src/sampler.rs::apply_freq_pres_rep_penalty`
  + `mistralrs-core/src/pipeline/sampling.rs`. With temperature=0
  there is no sampling spread to dilute the bias, so every token in
  `SCENE_PROMPT` (including example value vocabulary like "office",
  "stage performance") gets a flat -1.5 logit shift. Codex's
  recommendation was to remove the penalty in deterministic mode —
  but this was tested and broke generation: greedy without any
  repetition control falls into token loops that exhaust max_tokens
  mid-string, surfacing as `Error::Parse(Json("EOF while parsing a
  string"))`. mistralrs 0.8 has no generated-only repetition
  mechanism (`frequency_penalty`/`repetition_penalty` also operate
  over `seq.get_toks()`), and no `set_sampler_seed`. Among the
  available options, biased-but-parseable beats unbiased-but-broken,
  so the penalty stays. The trade-off is now documented prominently
  in `Engine::run`, `EngineOptions::deterministic`, and the README
  so future readers don't quietly remove it again.

- **Round 4 — F2: minimal CI workflow restored.** Task 1's scaffold
  cleanup deleted the template-rs `.github/` workflows and never
  replaced them. Adds `.github/workflows/ci.yml` running `cargo fmt
  --check`, `cargo clippy --lib -- -D warnings`, and `cargo test
  --lib` on macOS for every push to main and every PR. The
  `--features integration` real-model tests stay out of CI (they
  require the ~4 GiB qwen3-vl-2b model and `QWEN_MODEL_PATH`); run
  them locally before merging changes that touch the engine or scene
  preset.

- **mistralrs version pin loosened.** Cargo.toml is back to
  `mistralrs = "0.8"` (any 0.8.x). The committed `Cargo.lock` still
  pins the exact resolved version (0.8.1) for reproducible builds,
  so `cargo build` always uses the audited version on a fresh
  checkout. The exact `=0.8.1` pin in Cargo.toml was over-strict —
  it blocked even `cargo update -p mistralrs` from picking up
  patch-level fixes like 0.8.2 if/when they ship.



- **Round 3 — F1 (final): default constructor is indexing-safe.**
  Round 2 added `EngineOptions::for_indexing(path)` as a separate
  constructor, but Codex flagged that `EngineOptions::new` was still
  the obvious construction path documented in the migration plan and
  followed by every example. Callers who used `::new()` (the
  conventional Rust starter) silently got stochastic sampling and
  search-index drift on retries. The fix flips `EngineOptions::new`'s
  internal `deterministic` default to `true` so the conventional path
  IS the safe path. The redundant `for_indexing` constructor is
  removed. Stochastic sampling is now an explicit opt-out:
  `EngineOptions::new(path).with_deterministic(false)`. README,
  integration test, and unit tests updated to match. The
  `engine_options_defaults_indexing_safe` test guards against silent
  reversion of this default.

- **Round 2 — F1: indexing-safe constructor (superseded by round 3).**
  Originally added `EngineOptions::for_indexing(path)` as a separate
  constructor with `deterministic = true`. Round 3 made the default
  itself indexing-safe and removed `for_indexing`. (Kept here for
  history.)

- **Round 2 — F2: reject null required fields.** The parser checked
  only that required JSON keys were *present* in the top-level object,
  then deserialized into `QwenScenePayload`. Each `StringList` field
  uses `Option::<Repr>::deserialize`, which silently maps `null` to an
  empty `Vec`; the optional-string helpers do the same for the two
  string-typed required fields. Combined with `is_empty()` only firing
  when *every* field was empty, a model response like
  `{"subjects": null, "tags": ["x"], ...}` parsed successfully with
  `subjects` silently dropped — hiding constrained-decoder drift.
  `missing_required_fields` now flags both absent and null values; the
  error message is updated to "required fields missing or null". Adds
  three regression tests
  (`reject_null_required_array`, `reject_null_required_string`,
  `reject_multiple_null_required_fields`).



- **F1 — deterministic mode added for indexing idempotency.**
  `EngineOptions::with_deterministic(true)` now switches the engine to
  greedy decoding (`temperature 0.0`, `top_p 1.0`, `top_k 1`, keeping
  `presence_penalty 1.5` to prevent repetition loops). With identical
  inputs, two `Engine::run` calls in deterministic mode return
  bit-identical `SceneVlmResult` values. This is the right setting
  for indexing pipelines where retries (timeout, backfill, reprocess)
  must not cause search-index drift. The default stays at
  `deterministic = false` (model-card sampler) for quality-prioritised
  single-shot use. mistralrs 0.8 has no `set_sampler_seed`; greedy is
  the only deterministic mode available. New gated integration test
  `deterministic_run_is_idempotent` verifies the contract end-to-end.

- **F2 — mistralrs version pinned, Cargo.lock committed.**
  The manifest previously accepted any `0.8.x` mistralrs release,
  while design docs and verification were against a specific resolved
  patch. The pin is now `mistralrs = "=0.8.1"` and the lockfile is in
  git, so a fresh checkout cannot silently pull a different decoder /
  tokenizer / constrained-output implementation. Upgrade workflow
  documented in the commit message.

- **F3 — StringList no longer over-splits array elements.** The
  resilient list parser previously sent every JSON-array element
  through a comma/semicolon/newline splitter. A valid response like
  `["red, white, and blue flag"]` was being corrupted into three
  detections. The fix isolates splitting to the string-fallback
  branch (where the model returned a flattened comma-separated
  string instead of an array — real production drift). Array form
  now trims and dedupes verbatim. New regression test
  `array_elements_are_not_comma_split` covers comma-containing
  subjects, objects, colors, and tags.



Initial release. Standalone `qwen` crate for Qwen3-VL-2B structured-output
inference, built on `mistralrs 0.8` (Metal). Two layers:

### Generic engine

- `qwen::Engine` + `qwen::Task` trait. `Engine` is `Send + Sync + Clone`
  via `Arc<mistralrs::Model>`; concurrent `Engine::run` calls are
  continuous-batched by mistralrs's scheduler (not parallel decode).
- Wires the Qwen3-VL Instruct sampler profile per call —
  `temperature 0.7`, `top_p 0.8`, `top_k 20`, `presence_penalty 1.5`,
  configurable `max_tokens` (default 512), thinking off — plus
  `Constraint::JsonSchema(task.schema().clone())` for constrained decoding.
- `EngineOptions` and `SceneTask` follow the scenesdetect-style accessor
  pattern (getter / `with_*` / `set_*`) with `const fn` on `Copy` fields
  and `#[cfg_attr(not(tarpaulin), inline(always))]` on every accessor.
- `LoadError` is split out from `Error` because model load is a one-shot
  lifecycle event with different recovery semantics. `Error::Empty` and
  `Error::Parse` are kept distinct (rather than collapsed into a single
  "vlm_failed" variant) for cleaner operational logs.
- Cancellation contract: dropping the returned `Future` is a fast wakeup,
  not GPU cancellation. mistralrs has no abort handle in 0.8; in-flight
  scheduler steps run to completion and the response is discarded on send.

### Scene preset

- `qwen::scene::SceneTask` produces `qwen::scene::SceneAnalysis` — the
  crate's own output type (no `findit-proto` dependency). `SceneAnalysis`
  mirrors `findit-proto::database::SceneVlmResult`'s shape so a downstream
  consumer can map field-by-field; per-category detection types collapse
  into a single `qwen::scene::Detection` because Qwen3-VL never emits
  bounding boxes or per-color dominance.
- Ports the resilient parser from the legacy `findit-qwen` service
  (`StringList` accepts list-or-comma-string;
  `deserialize_optional_single_label` handles `"wide"` or `["wide"]`;
  `deny_unknown_fields` rejects unexpected top-level keys; empty-payload
  and null-required-field rejection).
- `default_confidence` (default `0.8`) is stamped on every emitted
  detection (replacing the seven hard-coded `0.8` values from the legacy
  `into_scene_vlm_result`, which the new internal helper
  `into_scene_analysis` supersedes).

### Tests + tooling

- Unit tests in `src/scene.rs::tests` — 11 parser tests (7 verbatim ports
  from `findit-qwen/src/lib.rs:1068-1124` plus 4 new tests covering
  custom-`default_confidence` stamping, mixed-shape `subjects`,
  `shot_type` list-form acceptance, and the F3 regression for
  comma-containing array elements).
- 3 accessor tests in `src/engine.rs::tests`.
- Phase-zero smoke binary at `examples/smoke.rs` (manual run; takes ~15 s
  on Apple Silicon Metal).
- Gated integration test at `tests/integration_scene.rs` behind
  `--features integration`; reads `QWEN_MODEL_PATH` env var or skips.
  Three keyframe fixtures shipped at `tests/fixtures/airport_0{1,2,3}.jpg`
  (extracted from `01_airport.mp4` at 2 s / 15 s / 30 s with ffmpeg);
  the test asserts that with real fixtures, the model produces both a
  populated `description` and at least one `tag`.

### Verification

End-to-end smoke verification against the local `qwen3-vl-2b` model with
the three airport keyframe fixtures: `Constraint::JsonSchema` round-trips
through the multimodal pipeline; the parser produces a populated
`SceneAnalysis` (`scene = "airport arrivals hall"`, ten-field structured
output, sixteen search tags). This covers a contract that mistralrs 0.8
itself has no end-to-end test for.

### Cargo features

- `integration`: enables `tests/integration_scene.rs`.
- `trace-output`: enables raw model-output logging at `tracing::trace`
  level.

### Notes

- Indexer-side integration with `services/findit-qwen` is deferred and
  tracked in `docs/superpowers/specs/2026-04-28-qwen-engine-design.md`.
  Today, this crate is standalone with `findit-proto` referenced via path
  dependency at `../indexer/findit-proto`.
- `mistralrs 0.8`'s `RequestBuilder` has no
  `set_sampler_repetition_penalty` and no `set_sampler_seed` — the
  Qwen3-VL model card's `repetition_penalty: 1.0` is the implicit
  no-penalty baseline, and `EngineOptions` exposes no seed knob. Do not
  substitute `set_sampler_frequency_penalty` for the missing
  `repetition_penalty`: the math is different (additive vs
  multiplicative).
