# `qwen` — Qwen3-VL Structured-Output Engine

**Date:** 2026-04-28
**Status:** design approved (rev 4, polish pass), ready to drive implementation plan
**Crate location:** `/Users/user/Develop/findit-studio/qwen` (currently the unchanged `template-rs` scaffold)

## Goal

Replace the in-service mistralrs wrapper that lives in `indexer/services/findit-qwen/src/lib.rs` with a focused, reusable Rust crate (`qwen`) that drives Qwen3-VL-2B-Instruct for **structured-output** tasks over keyframe images.

The new crate is the *runtime + prompt + parser* layer. The existing service crate (`findit-qwen`) keeps owning *threading, queues, health, lifecycle*.

## Context

Pipeline: `video → ffmpeg → packets → scenesdetect (scene boundaries + keyframe selection) → JPEG thumbnails → qwen3-vl → SceneVlmResult`.

Today's `findit-qwen` (Cargo: `mistralrs = "0.7"`, ISQ Q4K, Metal) does this in one file. Pain points motivating the split:

- The interesting code (prompt + JSON schema + resilient parser, lines 380-1126 of `findit-qwen/src/lib.rs`) is mixed with service-orchestration code (queues, health, shutdown). Both concerns are hard to evolve in isolation.
- mistralrs 0.7 predates the Qwen3-VL family fixes shipped in 0.8.0 (2026-04-02): "Fixes for Qwen 3 VL family", "correct Qwen VL multi-turn image processing and thinking model token decoding".
- The current code only sets `max_tokens` and `enable_thinking(false)` on the sampler — Qwen3-VL Instruct's recommended sampling profile (`temp 0.7 / top_p 0.8 / top_k 20 / presence_penalty 1.5`) is unused, plausibly costing JSON validity in tail cases.

mistralrs 0.8 implements Qwen3-VL faithfully (verified by source inspection at the `v0.8.0` tag of `mistralrs-core/src/vision_models/qwen3_vl/{config,vision}.rs` and `apply_interleaved_mrope` in `mistralrs-core/src/layers.rs` — DeepStack `deepstack_visual_indexes` are iterated and Interleaved-MRoPE is real, not a Qwen2.5-VL fallback). Constrained JSON-schema decoding is intact via `Constraint::JsonSchema(..)` in `mistralrs-core/src/pipeline/llg.rs`.

**Conclusion:** no from-scratch model implementation is needed. `qwen` is a thin focused wrapper around mistralrs 0.8.

## Scope (Approach 1 + B)

Two-layer crate:

1. **Generic engine** (`qwen::Engine` + `qwen::Task` trait) — Qwen3-VL with constrained-JSON output for arbitrary structured-output tasks. No `findit-proto` coupling.
2. **Scene-analysis preset** (`qwen::scene::SceneTask`) — produces `findit_proto::database::SceneVlmResult` directly. *This* layer depends on `findit-proto`.

Out of scope:

- Threading, queues, health, lifecycle, shutdown — those stay in `findit-qwen`.
- `findit-service` / `findit-proto` `Id` propagation — caller correlates request↔reply.
- Multi-turn conversations.
- Streaming token output.
- Native video-tensor input. (Today's pipeline feeds keyframe images one-by-one; if we ever want native video, that's a v0.2 expansion.)
- HuggingFace remote model download — model path is local-only.
- Per-task sampler overrides (deferred until a second preset needs them).
- A `MockEngine` / mistralrs mock — defer until a downstream test demands it.
- A user-supplied seed for sampling — mistralrs 0.8's `RequestBuilder` has no `set_sampler_seed` (verified via repo-wide search at `v0.8.0`), so `EngineOptions` has no seed knob.

## Module layout

```
qwen/
├── Cargo.toml             # standalone crate (sibling of scenesdetect, mediatime, etc.)
└── src/
    ├── lib.rs             # crate docs + re-exports (Engine, EngineOptions, Task, ParseError, Error, LoadError, scene::*, image::DynamicImage)
    ├── engine.rs          # Engine, EngineOptions, sampler defaults, mistralrs glue
    ├── task.rs            # Task trait, ParseError                              [no findit-proto]
    ├── error.rs           # Error, LoadError                                    [no findit-proto]
    └── scene.rs           # SceneTask preset, SCENE_PROMPT, schema, parser      [findit-proto]
```

Strip the `template-rs` `no_std` / `alloc` feature gates from `lib.rs` and `Cargo.toml` — they don't fit a GPU-runtime crate.

### Scaffold cleanup (template-rs artifacts to remove)

The `template-rs` scaffold left several files in `qwen/` that the migration must dispose of explicitly:

| Path | Action | Reason |
|---|---|---|
| `qwen/examples/foo.rs` | **Delete** | Placeholder; collides with the proposed `examples/smoke.rs` (phase-zero smoke test). |
| `qwen/tests/foo.rs` | **Delete** | Placeholder; collides with the proposed `tests/integration_scene.rs`. |
| `qwen/benches/foo.rs` | **Delete** | No benches in v0; removing this also lets the `[[bench]]` Cargo.toml stanza go. |
| `qwen/CHANGELOG.md` | **Replace** | Currently template-rs boilerplate; reset to a minimal v0 changelog. |
| `qwen/README.md` | **Replace** | Currently template-rs boilerplate (with shields.io badges and "template for creating Rust open-source repo" copy); rewrite as a short README for the `qwen` crate. |
| `qwen/README-zh_CN.md` | **Delete** | Template-rs Chinese translation; not relevant to this crate. |
| `qwen/.codecov.yml` | **Delete** | Template-rs coverage config; we don't run codecov here. |
| `qwen/.github/` | **Review and trim** | Template-rs CI workflows reference `template-rs` and run miri/tarpaulin sanitizers on a generic crate; replace with a minimal CI config (or delete and add later if needed). |
| `qwen/ci/miri_sb.sh`, `miri_tb.sh`, `sanitizer.sh` | **Delete** | Miri / sanitizers don't apply to a Metal GPU runtime crate (mistralrs uses unsafe FFI into Metal kernels — Miri can't model that). |
| `qwen/rustfmt.toml` | **Keep** | Standard rustfmt config; harmless. |
| `qwen/LICENSE-APACHE`, `LICENSE-MIT` | **Keep** | Real licenses. |
| `qwen/build.rs` | **Keep** | Implements `cfg(tarpaulin)` detection (verified byte-for-byte identical to `scenesdetect/build.rs`). The chosen accessor style — `#[cfg_attr(not(tarpaulin), inline(always))]` — depends on this `cfg` being defined when running under tarpaulin. |
| `qwen/.gitignore` | **Keep** | Standard. |

## Workspace integration (decided)

`qwen` is a **standalone Cargo project** at `/Users/user/Develop/findit-studio/qwen/`, with its own `.git` repository (`github.com/Findit-AI/qwen`). It is the **first cross-repo path dependency consumed by the `indexer/` workspace** — and therefore precedent-setting. Sibling crates like `scenesdetect/`, `mediatime/`, `silero/`, `hwdecode/` exist alongside `indexer` with the same structural shape, but none of them are currently consumed by the workspace via local path: `mediatime`, `silero`, `soundevents` show up only as crates.io version dependencies (`mediatime = "0.1"`, etc.), and `scenesdetect`/`hwdecode`/`locat`/`colconv`/`whispery`/`textclap` aren't referenced from the workspace at all yet (verified by grep across `indexer/Cargo.toml`, `findit-indexer/Cargo.toml`, and every `services/*/Cargo.toml`).

The choice between (a) standalone-with-path-dep and (b) move-into-indexer-workspace-as-member is a real one. We pick (a) — keep `qwen` standalone, register it as a workspace dependency from `indexer/Cargo.toml`'s `[workspace.dependencies]` — for these reasons:

- `qwen` already has its own `.git` history; absorbing it as a workspace member would either break that history or require a subtree merge.
- The standalone Cargo project shape (own `Cargo.toml`, own `[lints.rust]`, own `build.rs`) matches every other sibling crate in `findit-studio/`. Consistency wins over the (arguably cleaner) workspace-member option.
- (a) is reversible — promoting `qwen` to a workspace member later is a one-line edit if v0 reveals a real cost.

Concretely:

- `qwen/Cargo.toml`: `edition = "2024"`, `rust-version = "1.85.0"` (matches `scenesdetect/Cargo.toml`). Inline `[lints.rust]` (no workspace inheritance — qwen has no parent workspace).
- `findit-proto` is referenced as a path dependency: `findit-proto = { path = "../indexer/findit-proto" }`.
- `indexer/Cargo.toml`'s `[workspace.dependencies]` gains `qwen = { path = "../qwen" }`, so `findit-qwen/Cargo.toml` references it via `qwen = { workspace = true }` for path consistency. (See the explicit edit step in the migration section below.)
- `qwen` is **not** added to `indexer/Cargo.toml`'s `[workspace] members` list.

## Coding style — accessors

All public types follow the **scenesdetect accessor style**:

- All fields are private.
- Each field exposes three accessors:
  - `name(&self)` — getter.
  - `with_name(mut self, val) -> Self` — builder-style consuming setter.
  - `set_name(&mut self, val) -> &mut Self` — chainable in-place setter.
- `const fn` is preferred wherever the type allows; fall back to plain `pub fn` only for non-`Copy` payloads (`PathBuf`, `serde_json::Value`, `SmolStr`).
- For non-`Copy` setter parameters, use `impl Into<...>` (signature shape matches findit-proto's `set_label(&mut self, label: impl Into<SmolStr>)`).
- `#[cfg_attr(not(tarpaulin), inline(always))]` on every accessor (matches scenesdetect — keeps coverage tools happy while still inlining in release). Note: findit-proto uses plain `#[inline]`; we follow scenesdetect's convention since it pairs with the `cfg(tarpaulin)` lint declaration that the template-rs scaffold already has.

Reference: scenesdetect's `Components` / `Detector` impls in `scenesdetect/src/content.rs`.

## Public API

### `Engine` and `EngineOptions`

```rust
// engine.rs
pub struct EngineOptions {
    model_path: PathBuf,
    quantization: IsqType,                   // default IsqType::Q4K
    max_tokens: usize,                       // default 512
}

impl EngineOptions {
    pub fn new(model_path: impl Into<PathBuf>) -> Self;

    // Non-Copy field — plain pub fn, impl Into<PathBuf>
    pub fn model_path(&self) -> &Path;
    pub fn with_model_path(mut self, val: impl Into<PathBuf>) -> Self;
    pub fn set_model_path(&mut self, val: impl Into<PathBuf>) -> &mut Self;

    // Copy fields — const fn
    pub const fn quantization(&self) -> IsqType;
    pub const fn with_quantization(mut self, val: IsqType) -> Self;
    pub const fn set_quantization(&mut self, val: IsqType) -> &mut Self;

    pub const fn max_tokens(&self) -> usize;
    pub const fn with_max_tokens(mut self, val: usize) -> Self;
    pub const fn set_max_tokens(&mut self, val: usize) -> &mut Self;
}

pub struct Engine { /* private: mistralrs::Model + EngineOptions snapshot */ }

impl Engine {
    /// Load the model. Blocks for ~13s on Apple Silicon Metal at first call.
    /// Holds GPU memory until dropped.
    pub async fn load(opts: EngineOptions) -> Result<Self, LoadError>;

    /// Optional: run a tiny throwaway inference to JIT-compile Metal kernels
    /// before serving real requests. Logs duration at `debug`.
    pub async fn warmup(&self) -> Result<(), Error>;

    /// Single-turn, multi-image structured run. Consumes `images` because
    /// mistralrs's `MultimodalMessages::add_image_message` takes
    /// `Vec<DynamicImage>` by value — borrowing here would force a silent
    /// `.to_vec()` clone of potentially hundreds of MB of decoded image data.
    pub async fn run<T: Task>(&self, task: &T, images: Vec<DynamicImage>)
        -> Result<T::Output, Error>;

    // Read-only introspection
    pub fn model_path(&self) -> &Path;
    pub const fn quantization(&self) -> IsqType;
    pub const fn max_tokens(&self) -> usize;
}

impl Clone for Engine { /* cheap; mistralrs::Model is Arc<MistralRs> */ }
```

**Concurrency contract.** `Engine: Send + Sync + Clone`. mistralrs's `Model` is defined as `pub struct Model { pub(crate) runner: Arc<MistralRs> }` (verified at `mistralrs/src/model.rs` in `v0.8.0`), which is `Send + Sync` without any internal mutex at the `qwen` layer.

However, **concurrent `run()` calls do not execute in parallel inside the runtime.** mistralrs's engine loop holds the pipeline behind `pipeline: Arc<Mutex<dyn Pipeline>>` (verified at `mistralrs-core/src/engine/mod.rs` in `v0.8.0`); concurrent requests are **continuous-batched** into the same scheduler step, which is more efficient than strict sequential execution but is not parallel decode. The service layer can drop today's single-worker-thread serialization for throughput gains via batching, but should not expect parallelism-style latency improvements. Decisions in `findit-qwen` should be made on the batching property, not on imagined parallelism.

`EngineOptions::new(model_path)` requires a local path. No HuggingFace remote download — production already uses local weights, and removing the network branch removes failure modes.

### `Task` trait

```rust
// task.rs
pub trait Task: Send + Sync {
    type Output: Send;

    fn prompt(&self) -> &str;
    fn schema(&self) -> &serde_json::Value;        // borrowed; cached, not rebuilt per call
    fn parse(&self, raw: &str) -> Result<Self::Output, ParseError>;
}

#[derive(thiserror::Error, Debug)]
pub enum ParseError {
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("schema violation: missing fields {0:?}")]
    MissingFields(Vec<&'static str>),
    #[error("structured response had no usable fields")]
    NoUsableFields,
}
```

`Task: Send + Sync` and `Output: Send` are made explicit at the trait level (rather than as ad-hoc bounds on `Engine::run`) so trait objects (`dyn Task<Output = ...>`) and concurrent call sites work without extra work downstream.

Trait method choice rationale: a trait + custom `parse` keeps the existing parser resilience (`StringList` accepting `["a","b"]` *or* `"a, b, c"`, `deserialize_optional_single_label` handling `"wide"` *or* `["wide"]`, `normalize_string` for whitespace trimming, empty-payload rejection). Pure-serde with `schemars` would be sleek but would lose those tricks, which absorbed real-world model output drift in production.

### Top-level errors

```rust
// error.rs
#[derive(thiserror::Error, Debug)]
pub enum LoadError {
    #[error("model path not found: {0}")] NotFound(PathBuf),
    #[error("mistralrs build failed: {0}")] Build(String),
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("at least one image required")] NoImages,
    #[error("vision message build failed: {0}")] BuildMessage(String),
    #[error("inference failed: {0}")] Inference(String),
    #[error("model returned empty content")] Empty,
    #[error("parse failed: {0}")] Parse(#[from] ParseError),
}
```

`LoadError` is split out because it's one-shot and has different recovery semantics in the service layer (today: `health.failed(0)` and drain pending requests). Mixing it into `Error` would leave half the variants unreachable post-load.

`Error::Empty` and `Error::Parse` are kept distinct (rather than folded into a single `Inference`/`vlm_failed` variant): today's code at `findit-qwen/src/lib.rs:641` and `:649` collapses both into `vlm_failed`, making logs ambiguous. Splitting costs nothing and clarifies operational debugging.

mistralrs's `anyhow::Error` is stringified at the boundary — its deep context is noisy to surface, and a typed wrap isn't worth it for service-level use.

### Scene preset

```rust
// scene.rs
use findit_proto::{
    common::{
        ActionDetection, BoundingBox, ClassificationDetection, ColorDetection,
        LightingDetection, MoodDetection, ObjectDetection, SubjectDetection,
    },
    database::SceneVlmResult,
};

pub struct SceneTask {
    schema: serde_json::Value,
    default_confidence: f32,                       // default 0.8 — stamped on every emitted detection
}

impl SceneTask {
    pub fn new() -> Self;                          // builds schema, default_confidence = 0.8

    pub const fn default_confidence(&self) -> f32;
    pub const fn with_default_confidence(mut self, val: f32) -> Self;
    pub const fn set_default_confidence(&mut self, val: f32) -> &mut Self;
}

impl Task for SceneTask {
    type Output = SceneVlmResult;
    fn prompt(&self) -> &str { SCENE_PROMPT }
    fn schema(&self) -> &serde_json::Value { &self.schema }
    fn parse(&self, raw: &str) -> Result<SceneVlmResult, ParseError>;
}

const SCENE_PROMPT: &str = /* verbatim port of findit-qwen/src/lib.rs:382-401 */;
```

`SCENE_PROMPT` and the schema's required-field set port verbatim from `findit-qwen/src/lib.rs:36-47, 382-401, 821-860`.

The parser ports verbatim from `parse_vlm_output` (lines 873-895), `QwenScenePayload` (897-988), `StringList` (990-1018), `push_string_list_items` (1020-1027), `deserialize_optional_trimmed_string` (1029-1034), `deserialize_optional_single_label` (1036-1061), `normalize_string` (1063-1066), and `missing_required_fields` (862-870). The single substitution: replace the seven hard-coded `0.8` confidence values in `into_scene_vlm_result()` (lines 936-987) with `self.default_confidence`.

Field name `default_confidence` (rather than just `confidence`) signals the value is applied uniformly across detection categories; per-category overrides become a non-breaking addition later.

## Behavior

### Image input

`run<T: Task>(&self, task: &T, images: Vec<DynamicImage>)` consumes already-decoded images.

- **Why owned `Vec`:** mistralrs's `MultimodalMessages::add_image_message(role, text, images: Vec<DynamicImage>)` takes the vec by value (verified at `mistralrs/src/messages.rs` in `v0.8.0`). Borrowing in our API would force a `.to_vec()` clone of decoded image data inside `qwen::Engine::run`. The current `findit-qwen` caller at `lib.rs:791-810` already produces an owned `Vec<DynamicImage>` from `decode_scene_images` and never reuses it — passing by value is zero-cost.
- Callers that *do* need to keep the images can `images.clone()` at the call site explicitly; the cost becomes visible in the diff.

The existing `decode_scene_images` helper in `findit-qwen/src/lib.rs:792` keeps living there (caller's job).

**Multi-image semantics:** all images go into one `add_image_message` call, presented as a chronological keyframe sequence in a single user turn. This is what Interleaved-MRoPE expects.

**Empty input:** `engine.run(task, Vec::new())` returns `Err(Error::NoImages)`. The caller is responsible for filtering decoded-fail keyframes (today's code already does this).

### Sampling defaults

Wired through mistralrs 0.8's `RequestBuilder` (verified at `mistralrs/src/messages.rs` in `v0.8.0`) with the **exact** method names and parameter types as they appear in the source — note that `topp` / `topk` have no underscore, and the penalty methods are `f32` while `temperature` / `topp` are `f64`:

| Field | Value | mistralrs 0.8 method | Param type |
|---|---|---|---|
| `temperature` | `0.7` | `set_sampler_temperature` | `f64` |
| `top_p` | `0.8` | `set_sampler_topp` | `f64` |
| `top_k` | `20` | `set_sampler_topk` | `usize` |
| `presence_penalty` | `1.5` | `set_sampler_presence_penalty` | `f32` |
| `max_tokens` | `512` (configurable) | `set_sampler_max_len` | `usize` |
| thinking | off | `enable_thinking(false)` | `bool` |
| constraint | per-call | `set_constraint(Constraint::JsonSchema(task.schema().clone()))` | — |

**Two values from the model card cannot be set:**

- `repetition_penalty: 1.0` — mistralrs 0.8's `RequestBuilder` exposes no `set_sampler_repetition_penalty`; only `set_sampler_frequency_penalty` (f32) and `set_sampler_presence_penalty` (f32). `1.0` is the implicit "no penalty" baseline, so omitting it is faithful to the model card.

  **Do not substitute `frequency_penalty` for the missing `repetition_penalty`.** The two are distinct mechanisms that operate on different math — `repetition_penalty` is *multiplicative* on logits with baseline `1.0` (per HuggingFace `transformers`'s `RepetitionPenaltyLogitsProcessor`), while `frequency_penalty` is *additive* on logits with baseline `0.0` (per OpenAI's spec, which mistralrs follows). A future implementer might be tempted to "fill in" the model card's value as `set_sampler_frequency_penalty(1.0)` — that would silently diverge from the model card by doing additive penalty math against a multiplicative-baseline value. The only safe substitutions are: leave it absent (which we do), or call `set_sampler_frequency_penalty(0.0)` (a no-op that adds nothing).
- A user-supplied seed — no `set_sampler_seed` exists. `EngineOptions::seed` is therefore not in the design.

`max_tokens` is exposed via `EngineOptions::max_tokens`; the rest are private constants in `engine.rs`. Per-task overrides (`Task::sampler() -> Option<SamplerOverrides>`) are deferred until a second preset needs them.

### Cancellation contract

`qwen` exposes plain `Future`s — `engine.warmup()` and `engine.run(...)`. Callers wrap with `tokio::time::timeout(...)` or `tokio::select! { _ = shutdown => ... }`.

**Important:** dropping the future returned by `engine.run(...)` is a fast wakeup, **not** GPU cancellation. mistralrs 0.8's request path uses an mpsc `Sender<Response>` per sequence with no `AbortHandle` or cancellation token (verified at `mistralrs-core/src/sequence.rs` in `v0.8.0`); when the receiver is dropped, the in-flight scheduler step completes in the background and the response is silently discarded on send. **This matches today's behavior** (`findit-qwen/src/lib.rs:716-758` already inherits this property via its `tokio::select!` against shutdown polling), so the new design preserves it unchanged. Callers that need true mid-step abort would have to extend mistralrs upstream.

`qwen` does not take a `ShutdownToken` and knows nothing about service shutdown — this drops the `findit-service` dependency from `qwen` entirely. The service layer keeps its existing `run_with_timeout_and_shutdown` helper (`findit-qwen/src/lib.rs:717-759`); see migration section M5 below for why this helper must not be deleted (it owns the heartbeat-during-inference contract that `qwen` does not provide).

### Observability

- `qwen::run` span per `Engine::run()` call: fields `task_kind = std::any::type_name::<T>()`, `image_count`, `max_tokens`.
- `qwen::load` span around `Engine::load()`: logs `model_path`, `quantization`, load duration at `info`.
- `engine.warmup()`: duration at `debug`.
- Inference duration: `debug`.
- Raw model output: `trace`, gated behind a `trace-output` Cargo feature so it's not a default cost.

No metric emission, no health reporting — those are service-layer concerns.

## Testing

1. **Unit (parser) tests** in `scene.rs::tests`. Pure CPU, no GPU.
   - Verbatim ports of the seven existing tests in `findit-qwen/src/lib.rs:1068-1124` (`parse_valid_json`, `reject_json_with_wrapper_text`, `reject_plain_text_output`, `parse_comma_separated_tag_string`, `reject_empty_json_payload`, `reject_unknown_json_fields`, `reject_missing_required_fields`).
   - New: `parse_with_custom_default_confidence` (verifies `with_default_confidence(0.5)` is stamped on every emitted detection across all detection types), `parse_mixed_shape_subjects` (`["a", "b"]` and `"a, b"` each yield 2 subjects after parsing).
2. **Phase-zero smoke test** (run *before* deeper migration work; no permanent test artifact).
   - Build a 30-line standalone binary in `qwen/examples/smoke.rs`: load the model, run `SceneTask` against one fixture JPEG, print the result.
   - Purpose: confirm that `Constraint::JsonSchema` plus the multimodal pipeline actually produce schema-compliant output on 0.8 before the migration patches `findit-qwen`. mistralrs 0.8 has no end-to-end test that covers this combination; we're not betting the migration on it sight-unseen.
3. **Integration test** in `tests/integration_scene.rs`, gated behind `--features integration`.
   - Reads model path from `QWEN_MODEL_PATH` env var; if unset, the test skips.
   - Loads `Engine`, runs `SceneTask` against 3 fixture JPEGs in `tests/fixtures/`.
   - Asserts: parse succeeds, `description` non-empty, `tags.len() >= 1`.
   - Not run in default CI.

No mistralrs mocking, no `MockEngine`. Deferred until demanded.

## Cargo.toml

```toml
[package]
name = "qwen"
version = "0.1.0"
edition = "2024"
rust-version = "1.85.0"
description = "Qwen3-VL structured-output engine for findit-studio"
license = "MIT OR Apache-2.0"

[features]
default = []
integration  = []                                # enables tests/integration_scene.rs
trace-output = []                                # enables raw-output tracing at trace level

[dependencies]
mistralrs    = { version = "0.8", features = ["metal"] }
findit-proto = { path = "../indexer/findit-proto" }
image        = { version = "0.25", default-features = false, features = ["jpeg"] }
serde        = { version = "1", features = ["derive"] }
serde_json   = "1"
smol_str     = "0.3"
thiserror    = "2"
tracing      = "0.1"

[dev-dependencies]
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }

[lints.rust]
rust_2018_idioms = "warn"
single_use_lifetimes = "warn"
unexpected_cfgs = { level = "warn", check-cfg = [
  'cfg(all_tests)',
  'cfg(tarpaulin)',
] }
```

The `image` dependency is kept solely because `mistralrs::MultimodalMessages::add_image_message` takes `Vec<DynamicImage>`; `qwen` re-exports `image::DynamicImage` from its crate root for caller convenience.

Drop from the existing `template-rs` scaffold: the `[[bench]]` table, the `alloc/std` features, the `[profile.bench]` section (lives at the workspace root for workspace members; for a standalone crate we accept the cargo defaults), and the `criterion` / `tempfile` dev-deps.

## `findit-qwen` migration

### Delete (≈ 480 lines)

- `REQUIRED_QWEN_FIELDS` (lines 36-47)
- `ANALYSIS_PROMPT` const (lines 380-401)
- `qwen_output_schema` (821-860)
- `build_qwen_request` (814-819)
- `parse_vlm_output` (873-895)
- `QwenScenePayload` and `StringList` (897-1027)
- `deserialize_optional_trimmed_string`, `deserialize_optional_single_label`, `normalize_string`, `missing_required_fields` (862-870, 1029-1066)
- All seven parser tests (1068-1124)

### Keep

- `Service`, `Request`, `Reply`, `SpawnError`, `Callback` types.
- `decode_scene_images` (still needs JPEG → `DynamicImage`).
- Threading / runtime / channels / shutdown / health / lifecycle.
- `OperationError`, `run_with_timeout_and_shutdown`, `drain_with_error`. **Critical:** `run_with_timeout_and_shutdown` ticks `health.heartbeat(0)` at `heartbeat_interval` while the inference future is pending (`findit-qwen/src/lib.rs:728-755`). `qwen::Engine::run` does not surface a heartbeat hook; if the migration replaces this helper with a naked `tokio::time::timeout(..)` the worker loses its watchdog and gets marked unhealthy after roughly one heartbeat. Preserve the helper and keep wrapping `engine.run(...)` with it.
- `select_biased!` worker loop, heartbeat tick.
- `ProviderIdentifier` / `ProviderThreadService` / `ThreadService` impls.

### Modify `run_qwen_worker` (lines 503-704)

Replace this load path:

```rust
VisionModelBuilder::new(opts.model_id())
    .with_isq(opts.quantization())
    .build()
```

with the **mistralrs 0.8 API names** (`VisionModelBuilder` no longer exists in 0.8 — verified by repo-wide search at the `v0.8.0` tag):

```rust
qwen::Engine::load(
    qwen::EngineOptions::new(opts.engine().model_path())
        .with_quantization(opts.engine().quantization())
        .with_max_tokens(opts.engine().max_tokens()),
)
```

After load, build `let scene_task = qwen::scene::SceneTask::new().with_default_confidence(opts.default_confidence());` once, hold for worker lifetime.

Per request, replace the `VisionMessages::new().add_image_message(role, prompt, images, &model) → build_qwen_request(...) → model.send_chat_request(...) → response.choices[0]... → parse_vlm_output(text)` chain (note: `VisionMessages` and the `&model` argument are also gone in 0.8 — `MultimodalMessages::add_image_message(role, text, images)` is the replacement) with a single:

```rust
match engine.run(&scene_task, images).await {
    Ok(result) => (result, None),
    Err(qwen::Error::NoImages)        => (SceneVlmResult::default(), vlm_err("no decodable keyframes provided".into())),
    Err(qwen::Error::Parse(e))        => (SceneVlmResult::default(), vlm_err(format!("invalid Qwen JSON output: {e}"))),
    Err(qwen::Error::Empty)           => (SceneVlmResult::default(), vlm_err("empty model response".into())),
    Err(qwen::Error::BuildMessage(e)) => (SceneVlmResult::default(), vlm_err(e)),
    Err(qwen::Error::Inference(e))    => (SceneVlmResult::default(), vlm_err(e)),
}
```

Note the call site passes `images` by value (the `Vec<DynamicImage>` produced by `decode_scene_images`) — there is no `&images`.

The `SceneVlmResult` is produced directly — no `parse_vlm_output` step, no `into_scene_vlm_result` conversion.

The timeout/shutdown wrapping continues via `run_with_timeout_and_shutdown(..., engine.run(&scene_task, images))`.

### `ServiceOptions` reshape

```rust
pub struct ServiceOptions {
    engine: qwen::EngineOptions,        // model_path, quantization, max_tokens
    default_confidence: f32,            // default 0.8 — stamped on detections
    queue_capacity: usize,
    model_load_timeout: Duration,
    inference_timeout: Duration,
}
```

The existing `local_or_remote(models_dir)` constructor stays as a convenience but produces a `PathBuf` (no HF fallback — the existing fallback wasn't actually exercising HF download in production runs). **Follow-up:** the migration patch should include either a `scripts/download-qwen.sh` helper or documented manual instructions, so a fresh dev environment isn't broken when the HF fallback is removed. Tracked as the only item in §"Follow-ups" below.

### `findit-qwen/Cargo.toml`

- **Drop:** `mistralrs`, `serde_json`, `smol_str`. Likely also `serde` (review post-migration).
- **Keep:** `bytes`, `image`, `findit-proto`, `findit-service`, `crossbeam-channel`, `tokio`, `tracing`.
- **Add:** `qwen = { workspace = true }`.

### `indexer/Cargo.toml` (cross-repo seam — one-line edit)

Add to `[workspace.dependencies]` (alphabetical position next to the other path deps):

```toml
qwen = { path = "../qwen" }
```

This is the only change in `indexer/Cargo.toml`. It is the seam between the two repositories and is the precedent-setting cross-repo path dependency described in the Workspace integration section.

Net: `findit-qwen/src/lib.rs` shrinks from 1126 → ~640 lines, focused entirely on service orchestration.

## Decisions made (not deferred)

The expert review correctly flagged that "open questions in a design doc that the implementer is meant to resolve later are a smell." All previously deferred items are now resolved here:

1. **mistralrs API names.** `MultimodalModelBuilder` (not `VisionModelBuilder`); `MultimodalMessages` (not `VisionMessages`); `add_image_message(role, text, images: Vec<DynamicImage>)` (no `&model` arg). All verified by repo-wide `gh search code` queries against the `v0.8.0` tag — both old names returned zero hits and the new names appear throughout examples, docs, and source.
2. **`Model` `Sync`-ness.** `Model { runner: Arc<MistralRs> }` is `Send + Sync` natively. No internal `Mutex` is added at the `qwen` layer. `Engine: Send + Sync + Clone`.
3. **Concurrent `run()`.** Continuous-batched, not parallel decode; `mistralrs-core/src/engine/mod.rs` holds `Arc<Mutex<dyn Pipeline>>` across each `step()` call. Safe at the API level; `findit-qwen` may drop its single-worker invariant for batching throughput, but should not expect parallelism-style latency improvements.
4. **Quantization default.** `IsqType::Q4K` for v0 (matches today's behavior, reduces variables in the migration). MXFP4 evaluation (new in 0.8) is a post-migration A/B against the same fixture set; not blocking v0.
5. **`local_or_remote` constructor.** Keep with the path-only output (no HF fallback). A model-fetch script is the migration patch's responsibility (see Follow-ups).
6. **Workspace integration.** Standalone crate (own `.git`), consumed by `indexer/` as the **first cross-repo path dependency** in its `[workspace.dependencies]` (`qwen = { path = "../qwen" }`). `findit-proto` referenced via path. Not a member of the `indexer/` workspace.

## Non-goals / explicitly deferred

- Per-task sampler overrides — add `Task::sampler() -> Option<SamplerOverrides>` if a second preset needs them.
- Native video-tensor input via Qwen3-VL's video processor — today the pipeline feeds discrete keyframe images and that's well-supported.
- `MockEngine` for downstream testing — defer until a downstream test demands it.
- Streaming output — JSON is small (~100-300 tokens), latency wins are negligible.
- Prompt customization on `SceneTask` — fork the preset if a second prompt is needed.
- HuggingFace remote model download — production uses local weights.
- `no_std` support — incompatible with mistralrs's std/tokio/GPU runtime.
- True mid-step inference cancellation — would require extending mistralrs upstream.

## Follow-ups (post-migration, separate work)

- **Model fetch.** Add `scripts/download-qwen.sh` or document manual `huggingface-cli` steps so dev environments work without the HF fallback.
- **MXFP4 vs Q4K bake-off.** Run `SceneTask` against the existing fixture set with both quants, compare JSON validity rate and inference latency. Promote the winner to the default in a follow-up patch.
- **Drop the single-worker invariant in `findit-qwen`** if batching throughput proves the win; otherwise leave alone.

## References

- Old service: `indexer/services/findit-qwen/src/lib.rs` (the file being shrunk).
- Proto types: `indexer/findit-proto/src/database/scene_vlm.rs`, `indexer/findit-proto/src/common/{subject,object,action,mood,lighting,color,classification}_detection.rs`.
- Style reference: `scenesdetect/src/content.rs` (accessor pattern, `#[cfg_attr(not(tarpaulin), inline(always))]`).
- Model: `indexer/models/qwen3-vl-2b/{config,preprocessor_config,video_preprocessor_config,generation_config}.json`, `indexer/models/qwen3-vl-2b/README.md` (recommended sampler).
- mistralrs Qwen3-VL implementation: `EricLBuehler/mistral.rs` repository at `mistralrs-core/src/vision_models/qwen3_vl/`, `mistralrs-core/src/layers.rs::apply_interleaved_mrope`, `mistralrs-core/src/pipeline/llg.rs` (`Constraint::JsonSchema`).
- mistralrs 0.8 builder: `mistralrs/src/multimodal_model.rs::MultimodalModelBuilder` (verified at `v0.8.0`).
- mistralrs 0.8 messages + sampler: `mistralrs/src/messages.rs::{MultimodalMessages, RequestBuilder}` — `add_image_message(role, text: impl ToString, images: Vec<DynamicImage>)`, `set_sampler_temperature(f64)`, `set_sampler_topp(f64)`, `set_sampler_topk(usize)`, `set_sampler_presence_penalty(f32)`, `set_sampler_max_len(usize)`, `enable_thinking(bool)`, `set_constraint(Constraint)`.
- mistralrs 0.8 model: `mistralrs/src/model.rs::Model { runner: Arc<MistralRs> }`.
- mistralrs 0.8 engine: `mistralrs-core/src/engine/mod.rs` — `pipeline: Arc<Mutex<dyn Pipeline>>`, single async loop, continuous batching scheduler.
- mistralrs 0.8 cancellation: `mistralrs-core/src/sequence.rs` — `responder: Sender<Response>`, no `AbortHandle`.
- mistralrs Qwen3-VL guide: `EricLBuehler/mistral.rs/blob/master/docs/QWEN3VL.md`.
- mistralrs v0.8.0 release notes (2026-04-02): "Fixes for Qwen 3 VL family", "correct Qwen VL multi-turn image processing and thinking model token decoding".

## Changelog

- **2026-04-28 rev 4:** Polish pass after rev-3 audit.
  - P1: replaced stale "sibling-pattern" wording in Decisions #6 with "first cross-repo path dependency" framing (consistent with the Workspace integration section reworded in rev 3).
  - P2: replaced "currently the empty template-rs scaffold" with "currently the unchanged template-rs scaffold" in the header — accurate now that the rev-3 cleanup table enumerates 12 distinct on-disk artifacts.
  - P3 (rejected, not applied): rev-3 audit claimed `mediatime = "0.1"` was an unverified example. Verified against `indexer/Cargo.toml:49` (workspace crates.io version dep), `findit-proto/Cargo.toml:36` (`mediatime.workspace = true`), and three call sites in `findit-proto/src/{database/scene.rs, common/time_range.rs, common/timebase.rs}`. The example stands.
- **2026-04-28 rev 3:** Applied second-audit fixes R1, R2, R3, and the M6 sub-audit footnote:
  - R1: added explicit "Scaffold cleanup" sub-section under Module layout, listing every template-rs artifact (`examples/foo.rs`, `tests/foo.rs`, `benches/foo.rs`, `CHANGELOG.md`, `README.md`, `README-zh_CN.md`, `.codecov.yml`, `.github/`, `ci/*.sh`) with a per-file action. `build.rs` flagged as **keep** with rationale (verified byte-for-byte identical to scenesdetect's tarpaulin-detection script).
  - R2: reworded Workspace integration section. Removed the "mirroring the existing pattern" framing that was misleading (verified by grep: no sibling crate is currently consumed by `indexer/` via local path; `mediatime`, `silero`, `soundevents` are crates.io version deps, and `scenesdetect`/`hwdecode`/`locat`/`colconv`/`whispery`/`textclap` aren't referenced from the workspace at all). Now framed honestly: standalone-crate is precedent-setting for cross-repo path deps in `indexer/`; rationale is consistency with the sibling-crate *shape*, reversibility, and avoiding `.git` history disruption.
  - R3: added an explicit one-line edit step under "findit-qwen migration" calling out the `indexer/Cargo.toml` `[workspace.dependencies]` addition (`qwen = { path = "../qwen" }`).
  - M6 sub-audit: added a paragraph distinguishing `repetition_penalty` (multiplicative on logits, baseline 1.0) from `frequency_penalty` (additive on logits, baseline 0.0) and warning future implementers not to substitute one for the other.
- **2026-04-28 rev 2:** Applied expert-review fixes M1-M6 and N2-N6. Concrete changes:
  - M1: replaced `VisionModelBuilder` / `VisionMessages` references with `MultimodalModelBuilder` / `MultimodalMessages`; dropped the `&model` argument on `add_image_message`.
  - M2: `Engine::run` now consumes `images: Vec<DynamicImage>` (was `&[DynamicImage]`).
  - M3: clarified that concurrent `run()` is continuous-batched, not parallel.
  - M4: locked workspace integration as standalone-crate / sibling-pattern with path-dep on findit-proto.
  - M5: explicit cancellation contract documented; flagged the heartbeat-preservation requirement on `run_with_timeout_and_shutdown`.
  - M6: sampler defaults table now lists exact mistralrs 0.8 method names and parameter widths; dropped `repetition_penalty` (no API) and `EngineOptions::seed` (no API).
  - N2: renamed `confidence` → `default_confidence` on `SceneTask`.
  - N3: kept `Error::Empty` and `Error::Parse` distinct (already in design); doc reasoning added.
  - N4: added `Send + Sync` supertrait on `Task`, `Send` bound on `Output`.
  - N5: added phase-zero smoke test (`examples/smoke.rs`) before migration.
  - N6: promoted all "open questions" to decisions.
- **2026-04-28 rev 1:** Initial design committed.
