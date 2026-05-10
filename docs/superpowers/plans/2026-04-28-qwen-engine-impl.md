# `qwen` Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the in-service mistralrs wrapper at `indexer/services/findit-qwen/src/lib.rs` with a focused `qwen` crate (a thin Qwen3-VL structured-output engine) plus a `qwen::scene` preset that produces `findit_proto::SceneVlmResult` directly.

**Architecture:** Two-layer crate. `qwen::Engine` + `qwen::Task` trait owns model loading, sampler defaults, and constrained-JSON inference (no `findit-proto` coupling). `qwen::scene::SceneTask` is the scene-analysis preset that ports the prompt + JSON schema + resilient parser verbatim from today's `findit-qwen` and produces `findit_proto::database::SceneVlmResult`. `findit-qwen` shrinks to service-orchestration only (threading, queues, health, lifecycle).

**Tech Stack:** Rust 2024 edition (`rust-version = "1.85.0"`), `mistralrs = "0.8"` with Metal feature, `findit-proto` via path dep, `image = "0.25"` (jpeg-only), `serde` + `serde_json`, `smol_str`, `thiserror`, `tracing`. Standalone Cargo project at `/Users/user/Develop/findit-studio/qwen/`; consumed by `indexer/` workspace via `qwen = { path = "../qwen" }` registered in `indexer/Cargo.toml`'s `[workspace.dependencies]`.

**Spec:** `docs/superpowers/specs/2026-04-28-qwen-engine-design.md` (rev 4).

**Working repos:**
- `/Users/user/Develop/findit-studio/qwen/` — own git, branch `0.1.0`. Tasks 1–15 commit here.
- `/Users/user/Develop/findit-studio/indexer/` — own git, branch `feat/lifecycle`. Tasks 16–21 commit here.

**Build-time note:** the first `cargo check`/`cargo build` after adding `mistralrs = "0.8"` will take 5–15 minutes on a clean build. Subsequent incremental builds are fast. Use `cargo check` (no codegen) wherever possible; only run `cargo build` / `cargo test` when needed for verification. Where a step says "Expected output", treat the salient line as the assertion — long compile noise above it is normal.

---

## File structure

After the plan completes, the repo state is:

```
qwen/                                                       [own git, branch 0.1.0]
├── Cargo.toml                                              ← rewritten
├── README.md                                               ← rewritten
├── CHANGELOG.md                                            ← rewritten
├── build.rs                                                ← kept verbatim
├── rustfmt.toml                                            ← kept
├── LICENSE-APACHE, LICENSE-MIT                             ← kept
├── .gitignore                                              ← kept
├── docs/
│   └── superpowers/
│       ├── specs/2026-04-28-qwen-engine-design.md          ← already exists
│       └── plans/2026-04-28-qwen-engine-impl.md            ← this file
├── examples/
│   └── smoke.rs                                            ← new (replaces foo.rs)
├── tests/
│   └── integration_scene.rs                                ← new (replaces foo.rs)
└── src/
    ├── lib.rs                                              ← rewritten (re-exports)
    ├── error.rs                                            ← new
    ├── task.rs                                             ← new
    ├── engine.rs                                           ← new
    └── scene.rs                                            ← new

indexer/                                                    [own git, branch feat/lifecycle]
├── Cargo.toml                                              ← +1 line in [workspace.dependencies]
└── services/findit-qwen/
    ├── Cargo.toml                                          ← deps swap
    └── src/lib.rs                                          ← shrunk by ~480 lines
```

Files responsible for:

| File | Responsibility |
|---|---|
| `qwen/src/error.rs` | `LoadError` (one-shot model-load failures) and `Error` (per-call inference + parse failures), both `thiserror::Error`. No `findit-proto`. |
| `qwen/src/task.rs` | `Task` trait (`Send + Sync` supertrait, `Output: Send`), `ParseError` enum. No `findit-proto`. |
| `qwen/src/engine.rs` | `EngineOptions` (model_path, quantization, max_tokens) with full scenesdetect-style accessors; `Engine` (wraps mistralrs `Model`, exposes `load`, `warmup`, `run`); private sampler-default constants; mistralrs glue. No `findit-proto`. |
| `qwen/src/scene.rs` | `SceneTask` preset producing `SceneVlmResult`; `SCENE_PROMPT` constant; JSON schema; resilient parser (port of `findit-qwen`'s `QwenScenePayload` + `StringList` + helpers). Depends on `findit-proto`. |
| `qwen/src/lib.rs` | Crate docs + re-exports (`Engine`, `EngineOptions`, `Task`, `ParseError`, `Error`, `LoadError`, `scene::*`, `image::DynamicImage`). |
| `qwen/examples/smoke.rs` | Phase-zero smoke test: load the model, run `SceneTask` against one fixture, print the result. Validates the mistralrs 0.8 `Constraint::JsonSchema` + multimodal contract before deeper migration. |
| `qwen/tests/integration_scene.rs` | Gated (`--features integration`) integration test; reads `QWEN_MODEL_PATH` env var, runs `SceneTask` against fixture JPEGs. |
| `indexer/Cargo.toml` | Adds `qwen = { path = "../qwen" }` to `[workspace.dependencies]`. |
| `indexer/services/findit-qwen/Cargo.toml` | Drops `mistralrs`, `serde_json`, `smol_str`; adds `qwen = { workspace = true }`. |
| `indexer/services/findit-qwen/src/lib.rs` | Threading, queues, health, lifecycle (kept). The 480 lines of prompt/schema/parser code go away; `run_qwen_worker` calls `qwen::Engine` instead of mistralrs directly. |

---

## Phase 0 — Scaffold cleanup and Cargo.toml

### Task 1: Remove template-rs scaffold artifacts

**Files:**
- Delete: `qwen/examples/foo.rs`
- Delete: `qwen/tests/foo.rs`
- Delete: `qwen/benches/foo.rs`
- Delete: `qwen/benches/` (the directory; empty after the file is gone)
- Delete: `qwen/ci/miri_sb.sh`
- Delete: `qwen/ci/miri_tb.sh`
- Delete: `qwen/ci/sanitizer.sh`
- Delete: `qwen/ci/` (the directory; empty after the files are gone)
- Delete: `qwen/.codecov.yml`
- Delete: `qwen/README-zh_CN.md`
- Delete: `qwen/.github/` (template-rs CI workflows referencing `template-rs`)

**Keep:** `qwen/build.rs` (byte-for-byte identical to `scenesdetect/build.rs`; provides `cfg(tarpaulin)` detection that the `#[cfg_attr(not(tarpaulin), inline(always))]` accessor pattern depends on), `qwen/rustfmt.toml`, `qwen/.gitignore`, `qwen/LICENSE-APACHE`, `qwen/LICENSE-MIT`.

- [ ] **Step 1: Delete the placeholder files and directories**

```bash
cd /Users/user/Develop/findit-studio/qwen
rm -f examples/foo.rs tests/foo.rs benches/foo.rs
rmdir benches
rm -f ci/miri_sb.sh ci/miri_tb.sh ci/sanitizer.sh
rmdir ci
rm -f .codecov.yml README-zh_CN.md
rm -rf .github
```

- [ ] **Step 2: Verify the kept files survived**

Run:
```bash
ls -1 build.rs rustfmt.toml .gitignore LICENSE-APACHE LICENSE-MIT
```

Expected output (one per line, no errors):
```
build.rs
rustfmt.toml
.gitignore
LICENSE-APACHE
LICENSE-MIT
```

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove template-rs scaffold artifacts

Drops examples/foo.rs, tests/foo.rs, benches/foo.rs (and the
benches/ dir), ci/miri_*.sh + sanitizer.sh, .codecov.yml,
README-zh_CN.md, and the template-rs .github/ workflows that
reference template-rs.

build.rs is kept (byte-for-byte identical to scenesdetect/build.rs;
implements cfg(tarpaulin) detection that the chosen accessor
style depends on)."
```

---

### Task 2: Rewrite Cargo.toml

**Files:**
- Modify: `qwen/Cargo.toml` (full rewrite)

- [ ] **Step 1: Replace `qwen/Cargo.toml` with the production version**

Write the file with this exact content:

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
# Enables tests/integration_scene.rs (requires QWEN_MODEL_PATH env var).
integration = []
# Enables raw-output tracing at trace level (off by default — heavyweight).
trace-output = []

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

- [ ] **Step 2: Verify Cargo.toml parses**

Run:
```bash
cargo metadata --no-deps --format-version 1 --manifest-path Cargo.toml > /dev/null
```

Expected: exit 0, no error output.

- [ ] **Step 3: Commit**

```bash
git add Cargo.toml
git commit -m "chore: rewrite Cargo.toml for the qwen crate

Drops template-rs alloc/std features, [[bench]] entry, criterion
and tempfile dev-deps. Adds the production deps: mistralrs 0.8
(metal), findit-proto via path dep, image, serde, serde_json,
smol_str, thiserror, tracing. Edition 2024 / rust-version 1.85.0
matches scenesdetect."
```

---

### Task 3: Replace lib.rs with a working skeleton

**Files:**
- Modify: `qwen/src/lib.rs` (full rewrite)

- [ ] **Step 1: Replace `qwen/src/lib.rs`**

Write the file with this exact content:

```rust
//! Qwen3-VL structured-output engine for findit-studio.
//!
//! See `docs/superpowers/specs/2026-04-28-qwen-engine-design.md` for the design
//! rationale. The crate is two layers:
//!
//! - [`Engine`] / [`Task`]: a generic Qwen3-VL constrained-JSON inference engine
//!   with no `findit-proto` coupling.
//! - [`scene`]: the scene-analysis preset that produces
//!   `findit_proto::database::SceneVlmResult` directly.
//!
//! The crate is async-only and exposes plain `Future`s — callers wrap them in
//! `tokio::time::timeout(..)` or `tokio::select!` for shutdown observation.
//! Dropping a future is a fast wakeup, not GPU cancellation; mistralrs's
//! engine loop runs the in-flight step to completion in the background.

#![deny(missing_docs)]

pub mod engine;
pub mod error;
pub mod scene;
pub mod task;

pub use crate::{
  engine::{Engine, EngineOptions},
  error::{Error, LoadError},
  task::{ParseError, Task},
};

/// Re-exported from the [`image`] crate for caller convenience —
/// [`Engine::run`] consumes `Vec<DynamicImage>`.
pub use image::DynamicImage;
```

- [ ] **Step 2: Create the empty module files so `lib.rs` compiles**

```bash
echo "//! See module docs in lib.rs." > src/error.rs
echo "//! See module docs in lib.rs." > src/task.rs
echo "//! See module docs in lib.rs." > src/engine.rs
echo "//! See module docs in lib.rs." > src/scene.rs
```

- [ ] **Step 3: Verify the crate compiles (this is the slow first compile — 5-15 min on clean)**

Run:
```bash
cargo check --lib 2>&1 | tail -20
```

Expected: ends with `error[E0432]` complaining about unresolved imports `Engine`, `EngineOptions`, etc. — that's fine, we'll fill them in next. **Crucially, the unrelated mistralrs / findit-proto / image deps must have built.** If any of those fail (e.g., findit-proto path doesn't resolve), stop and fix.

- [ ] **Step 4: Commit**

```bash
git add src/
git commit -m "feat: scaffold lib.rs + empty module files

lib.rs declares the four submodules (engine, error, scene, task)
and the re-export surface (Engine, EngineOptions, Task, ParseError,
Error, LoadError, image::DynamicImage). Empty module files added
so the next tasks can fill them in without each touching lib.rs."
```

---

## Phase 1 — Generic engine layer (no findit-proto)

### Task 4: Implement `error.rs` (LoadError + Error)

**Files:**
- Modify: `qwen/src/error.rs`

- [ ] **Step 1: Write the file**

Replace `qwen/src/error.rs` with:

```rust
//! Top-level error types for the `qwen` crate.
//!
//! `LoadError` is split out from `Error` because it has different recovery
//! semantics in service layers (one-shot at startup; failure typically
//! aborts the worker), while `Error` covers per-call failures that the
//! caller may want to swallow into a default response.

use std::path::PathBuf;

/// Errors returned by [`crate::Engine::load`].
#[derive(thiserror::Error, Debug)]
pub enum LoadError {
  /// The model directory does not exist on disk.
  #[error("model path not found: {0}")]
  NotFound(PathBuf),
  /// mistralrs's builder returned an error during model load.
  #[error("mistralrs build failed: {0}")]
  Build(String),
}

/// Errors returned by [`crate::Engine::run`] and [`crate::Engine::warmup`].
#[derive(thiserror::Error, Debug)]
pub enum Error {
  /// Caller passed an empty image list.
  #[error("at least one image required")]
  NoImages,
  /// mistralrs's `MultimodalMessages` builder rejected the message.
  #[error("vision message build failed: {0}")]
  BuildMessage(String),
  /// mistralrs returned an inference error.
  #[error("inference failed: {0}")]
  Inference(String),
  /// The model returned empty content (after trimming).
  #[error("model returned empty content")]
  Empty,
  /// The model's output failed schema/parse validation.
  #[error("parse failed: {0}")]
  Parse(#[from] crate::task::ParseError),
}
```

- [ ] **Step 2: Verify it compiles**

Run:
```bash
cargo check --lib 2>&1 | tail -10
```

Expected: still has unresolved `Engine`/`EngineOptions`/`Task` errors (those live in other modules), but `error.rs` itself produces no errors.

- [ ] **Step 3: Commit**

```bash
git add src/error.rs
git commit -m "feat(error): add LoadError and Error enums

LoadError is split out because model-load is a one-shot lifecycle
event with different recovery semantics than per-call failures.
Error::Parse wraps task::ParseError via #[from] for ergonomic
'?' propagation from Task::parse implementations."
```

---

### Task 5: Implement `task.rs` (Task trait + ParseError)

**Files:**
- Modify: `qwen/src/task.rs`

- [ ] **Step 1: Write the file**

Replace `qwen/src/task.rs` with:

```rust
//! The [`Task`] trait that callers implement to drive a structured-output
//! request, and the [`ParseError`] type its `parse` method may return.

/// A structured-output task description.
///
/// Implementations supply the prompt, the JSON schema for constrained
/// decoding, and a parser that turns the model's raw text into a typed
/// `Output`. The trait is `Send + Sync` and `Output: Send` so trait
/// objects (`dyn Task<Output = ...>`) and concurrent call sites work
/// without extra bounds at the call site.
///
/// Implementations should cache their schema (build it once in `new`)
/// rather than rebuilding it per call — `schema` returns a borrow.
pub trait Task: Send + Sync {
  /// The typed result of a successful run.
  type Output: Send;

  /// The user-message prompt sent alongside the images.
  fn prompt(&self) -> &str;

  /// JSON schema used for constrained decoding.
  fn schema(&self) -> &serde_json::Value;

  /// Parse the model's raw text output into a typed `Output`.
  fn parse(&self, raw: &str) -> Result<Self::Output, ParseError>;
}

/// Errors returned by [`Task::parse`].
#[derive(thiserror::Error, Debug)]
pub enum ParseError {
  /// `serde_json` failed to parse the response as valid JSON.
  #[error("invalid JSON: {0}")]
  Json(#[from] serde_json::Error),
  /// JSON parsed but did not contain all required schema fields.
  #[error("schema violation: missing fields {0:?}")]
  MissingFields(Vec<&'static str>),
  /// JSON parsed and had no missing fields, but every value was empty.
  #[error("structured response had no usable fields")]
  NoUsableFields,
}
```

- [ ] **Step 2: Verify it compiles**

Run:
```bash
cargo check --lib 2>&1 | tail -10
```

Expected: still has unresolved `Engine`/`EngineOptions` errors. `task.rs` and `error.rs` produce no errors.

- [ ] **Step 3: Commit**

```bash
git add src/task.rs
git commit -m "feat(task): add Task trait and ParseError

Task is Send + Sync (so trait objects and concurrent call sites
work) with Output: Send. The schema is borrowed so implementations
can cache it once at construction rather than rebuilding per call."
```

---

### Task 6: Implement `EngineOptions` accessors (TDD)

**Files:**
- Modify: `qwen/src/engine.rs`

- [ ] **Step 1: Write the failing accessor test**

Replace `qwen/src/engine.rs` with:

```rust
//! The [`Engine`] and [`EngineOptions`] types.

use std::path::{Path, PathBuf};

use mistralrs::IsqType;

/// Configuration for [`Engine::load`].
#[derive(Debug, Clone)]
pub struct EngineOptions {
  model_path: PathBuf,
  quantization: IsqType,
  max_tokens: usize,
}

impl EngineOptions {
  /// Construct with the given model path and the default quantization
  /// (`IsqType::Q4K`) and `max_tokens` (`512`).
  pub fn new(model_path: impl Into<PathBuf>) -> Self {
    Self {
      model_path: model_path.into(),
      quantization: IsqType::Q4K,
      max_tokens: 512,
    }
  }

  // --- model_path (non-Copy: plain pub fn, impl Into<PathBuf>) ---

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn model_path(&self) -> &Path {
    &self.model_path
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_model_path(mut self, val: impl Into<PathBuf>) -> Self {
    self.model_path = val.into();
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_model_path(&mut self, val: impl Into<PathBuf>) -> &mut Self {
    self.model_path = val.into();
    self
  }

  // --- quantization (Copy: const fn) ---

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn quantization(&self) -> IsqType {
    self.quantization
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_quantization(mut self, val: IsqType) -> Self {
    self.quantization = val;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_quantization(&mut self, val: IsqType) -> &mut Self {
    self.quantization = val;
    self
  }

  // --- max_tokens (Copy: const fn) ---

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn max_tokens(&self) -> usize {
    self.max_tokens
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_max_tokens(mut self, val: usize) -> Self {
    self.max_tokens = val;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_max_tokens(&mut self, val: usize) -> &mut Self {
    self.max_tokens = val;
    self
  }
}

/// Stub — filled in by the next task.
pub struct Engine;

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn engine_options_defaults() {
    let opts = EngineOptions::new("/tmp/model");
    assert_eq!(opts.model_path(), Path::new("/tmp/model"));
    assert!(matches!(opts.quantization(), IsqType::Q4K));
    assert_eq!(opts.max_tokens(), 512);
  }

  #[test]
  fn engine_options_with_chains() {
    let opts = EngineOptions::new("/tmp/a")
      .with_model_path("/tmp/b")
      .with_quantization(IsqType::Q8_0)
      .with_max_tokens(1024);
    assert_eq!(opts.model_path(), Path::new("/tmp/b"));
    assert!(matches!(opts.quantization(), IsqType::Q8_0));
    assert_eq!(opts.max_tokens(), 1024);
  }

  #[test]
  fn engine_options_set_chains() {
    let mut opts = EngineOptions::new("/tmp/a");
    opts
      .set_model_path("/tmp/b")
      .set_quantization(IsqType::Q8_0)
      .set_max_tokens(1024);
    assert_eq!(opts.model_path(), Path::new("/tmp/b"));
    assert!(matches!(opts.quantization(), IsqType::Q8_0));
    assert_eq!(opts.max_tokens(), 1024);
  }
}
```

(The `lib.rs` `pub use crate::engine::{Engine, EngineOptions};` line at the top works against the stub `pub struct Engine;`.)

- [ ] **Step 2: Run the tests**

Run:
```bash
cargo test --lib engine_options_ 2>&1 | tail -15
```

Expected: 3 tests pass (`engine_options_defaults`, `engine_options_with_chains`, `engine_options_set_chains`).

- [ ] **Step 3: Commit**

```bash
git add src/engine.rs
git commit -m "feat(engine): add EngineOptions with full accessor surface

Three accessors per field (getter / with_* / set_*) following the
scenesdetect style. const fn where the type allows; impl Into for
non-Copy setter params; #[cfg_attr(not(tarpaulin), inline(always))]
on every accessor. Engine itself is a stub for now — filled in
in the next task."
```

---

### Task 7: Implement `Engine::load` (loads the mistralrs model)

**Files:**
- Modify: `qwen/src/engine.rs`

- [ ] **Step 1: Replace the `Engine` stub with the real type and `load` method**

Find the stub `pub struct Engine;` line and replace it with:

```rust
use std::sync::Arc;

use mistralrs::{Constraint, MultimodalMessages, MultimodalModelBuilder, RequestBuilder, TextMessageRole};
use tracing::{debug, info, instrument};

use crate::{
  error::{Error, LoadError},
  task::Task,
};

// Sampler defaults from the Qwen3-VL Instruct (non-thinking) model card.
// See indexer/models/qwen3-vl-2b/README.md §"Generation Hyperparameters → VL".
const SAMPLER_TEMPERATURE: f64 = 0.7;
const SAMPLER_TOPP: f64 = 0.8;
const SAMPLER_TOPK: usize = 20;
const SAMPLER_PRESENCE_PENALTY: f32 = 1.5;
// repetition_penalty is intentionally not set — mistralrs 0.8 has no
// set_sampler_repetition_penalty, and 1.0 is the implicit no-penalty
// baseline. Do NOT substitute set_sampler_frequency_penalty(1.0): the
// math is different (additive vs multiplicative). See spec §"Sampling
// defaults".

/// A Qwen3-VL structured-output inference engine.
///
/// Construct via [`Engine::load`]. `Engine` is `Send + Sync + Clone` —
/// `mistralrs::Model` is `Arc<MistralRs>` internally, so cloning is cheap.
/// Concurrent `run()` calls from multiple tasks are safe and are
/// continuous-batched by mistralrs's scheduler (not parallel decode).
#[derive(Clone)]
pub struct Engine {
  model: Arc<mistralrs::Model>,
  options: EngineOptions,
}

impl Engine {
  /// Load the Qwen3-VL model at `opts.model_path()` with the given
  /// quantization. Blocks for ~13s on Apple Silicon Metal at first call.
  /// Holds GPU memory until the last clone is dropped.
  #[instrument(name = "qwen::load", skip(opts), fields(model_path = %opts.model_path().display(), quantization = ?opts.quantization()))]
  pub async fn load(opts: EngineOptions) -> Result<Self, LoadError> {
    if !opts.model_path().exists() {
      return Err(LoadError::NotFound(opts.model_path().to_path_buf()));
    }
    let started = std::time::Instant::now();
    info!("loading Qwen3-VL model");
    let model_id = opts.model_path().to_string_lossy().into_owned();
    let model = MultimodalModelBuilder::new(model_id)
      .with_isq(opts.quantization())
      .build()
      .await
      .map_err(|e| LoadError::Build(e.to_string()))?;
    info!(elapsed_ms = started.elapsed().as_millis() as u64, "model loaded");
    Ok(Self {
      model: Arc::new(model),
      options: opts,
    })
  }

  /// Returns the local model directory the engine was loaded from.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn model_path(&self) -> &Path {
    self.options.model_path()
  }

  /// Returns the quantization the engine was loaded with.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn quantization(&self) -> IsqType {
    self.options.quantization()
  }

  /// Returns the configured `max_tokens` ceiling for [`Engine::run`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn max_tokens(&self) -> usize {
    self.options.max_tokens()
  }
}
```

- [ ] **Step 2: Verify it compiles (no test for `load` — needs the real model)**

Run:
```bash
cargo check --lib 2>&1 | tail -10
```

Expected: zero errors. The unused `MultimodalMessages` / `RequestBuilder` / `Constraint` / `TextMessageRole` / `Task` / `Error` / `debug` imports may produce `unused_imports` warnings — that's fine, the next task uses them.

- [ ] **Step 3: Commit**

```bash
git add src/engine.rs
git commit -m "feat(engine): add Engine::load wrapping mistralrs 0.8 builder

Engine wraps Arc<mistralrs::Model>; Send + Sync + Clone are derived
from mistralrs's own Arc<MistralRs> internals. Sampler defaults are
private constants matching the Qwen3-VL Instruct model card.
Note: repetition_penalty is intentionally not set — mistralrs 0.8
has no set_sampler_repetition_penalty, and 1.0 is the no-penalty
baseline. The comment explicitly warns against substituting
frequency_penalty (different math)."
```

---

### Task 8: Implement `Engine::run` and `Engine::warmup`

**Files:**
- Modify: `qwen/src/engine.rs`

- [ ] **Step 1: Add `run` and `warmup` to the `impl Engine` block**

Insert these methods inside `impl Engine { ... }`, after `max_tokens`:

```rust
  /// Optional pre-warm: runs one tiny inference against a 1×1 black image
  /// to JIT-compile Metal kernels before serving real requests. Logs
  /// duration at `debug`. Errors are propagated to the caller — typically
  /// you ignore them in production (warmup is best-effort).
  #[instrument(name = "qwen::warmup", skip(self))]
  pub async fn warmup(&self) -> Result<(), Error> {
    use image::{DynamicImage, RgbImage};

    let started = std::time::Instant::now();
    let blank = DynamicImage::ImageRgb8(RgbImage::new(1, 1));
    let messages = MultimodalMessages::new()
      .add_image_message(TextMessageRole::User, "Reply with: ok", vec![blank]);
    let request = RequestBuilder::from(messages)
      .set_sampler_max_len(4)
      .enable_thinking(false);
    let _ = self
      .model
      .send_chat_request(request)
      .await
      .map_err(|e| Error::Inference(e.to_string()))?;
    debug!(elapsed_ms = started.elapsed().as_millis() as u64, "warmup complete");
    Ok(())
  }

  /// Single-turn, multi-image structured run.
  ///
  /// Consumes `images` because mistralrs's `MultimodalMessages::add_image_message`
  /// takes `Vec<DynamicImage>` by value — borrowing here would force a silent
  /// `.to_vec()` clone of decoded image data. Returns `Error::NoImages`
  /// for an empty input.
  ///
  /// Dropping the returned future is a fast wakeup, not GPU cancellation:
  /// mistralrs's engine loop completes the in-flight scheduler step in the
  /// background; the response is silently discarded on send. Wrap in
  /// `tokio::time::timeout(..)` for a deadline.
  #[instrument(
    name = "qwen::run",
    skip(self, task, images),
    fields(
      task_kind = std::any::type_name::<T>(),
      image_count = images.len(),
      max_tokens = self.options.max_tokens(),
    ),
  )]
  pub async fn run<T: Task>(
    &self,
    task: &T,
    images: Vec<image::DynamicImage>,
  ) -> Result<T::Output, Error> {
    if images.is_empty() {
      return Err(Error::NoImages);
    }

    let messages = MultimodalMessages::new().add_image_message(
      TextMessageRole::User,
      task.prompt(),
      images,
    );

    let request = RequestBuilder::from(messages)
      .set_sampler_temperature(SAMPLER_TEMPERATURE)
      .set_sampler_topp(SAMPLER_TOPP)
      .set_sampler_topk(SAMPLER_TOPK)
      .set_sampler_presence_penalty(SAMPLER_PRESENCE_PENALTY)
      .set_sampler_max_len(self.options.max_tokens().max(1))
      .enable_thinking(false)
      .set_constraint(Constraint::JsonSchema(task.schema().clone()));

    let started = std::time::Instant::now();
    let response = self
      .model
      .send_chat_request(request)
      .await
      .map_err(|e| Error::Inference(e.to_string()))?;
    debug!(elapsed_ms = started.elapsed().as_millis() as u64, "inference complete");

    let text = response
      .choices
      .first()
      .and_then(|c| c.message.content.clone())
      .filter(|s| !s.trim().is_empty())
      .ok_or(Error::Empty)?;

    #[cfg(feature = "trace-output")]
    tracing::trace!(raw = %text, "model output");

    Ok(task.parse(&text)?)
  }
```

- [ ] **Step 2: Verify the crate compiles**

Run:
```bash
cargo check --lib 2>&1 | tail -10
```

Expected: zero errors, zero warnings about unused imports.

- [ ] **Step 3: Run the existing accessor tests to confirm nothing regressed**

Run:
```bash
cargo test --lib engine_options_ 2>&1 | tail -10
```

Expected: 3 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/engine.rs
git commit -m "feat(engine): add Engine::run and Engine::warmup

run() consumes Vec<DynamicImage> (mistralrs takes the vec by value;
borrowing would force a silent .to_vec() of decoded image data).
The full Qwen3-VL Instruct sampler profile is wired through:
temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5, plus
the per-call JSON-schema constraint. Output is consumed via
response.choices.first().message.content; empty content maps to
Error::Empty (distinct from Error::Parse for cleaner logs).

warmup() runs a 4-token request against a 1×1 black image to JIT
Metal kernels — best-effort.

#[instrument] spans on both methods log task_kind / image_count /
max_tokens (per spec §Observability). trace-output feature gates
raw-output logging at trace level."
```

---

## Phase 2 — Scene preset (uses findit-proto)

### Task 9: Write `scene.rs` parser tests (TDD red commit)

**Files:**
- Modify: `qwen/src/scene.rs`

This task ports the seven existing parser tests from `findit-qwen/src/lib.rs:1068-1124` and adds two new ones, before any of the parser code is implemented. The test module won't compile until Task 10 adds the types — that's intentional. We commit a "test scaffold" state where `cargo test` errors with "cannot find type SceneTask"; Task 10 turns it green.

- [ ] **Step 1: Write the test scaffold + first stubs**

Replace `qwen/src/scene.rs` with:

```rust
//! The scene-analysis preset: [`SceneTask`] produces
//! `findit_proto::database::SceneVlmResult` directly.

use serde_json::Value;

use crate::task::{ParseError, Task};

/// The scene-analysis prompt — verbatim port from `findit-qwen/src/lib.rs:382-401`.
const SCENE_PROMPT: &str = "TODO: filled in next task";

/// Stub that the next task replaces. The tests in this file already
/// reference `SceneTask::new` and `SceneTask::with_default_confidence`,
/// so the stub exposes those signatures (with no-op bodies) so
/// `cargo check` passes before the real impl lands.
pub struct SceneTask;

impl SceneTask {
  /// Stub.
  pub fn new() -> Self {
    Self
  }
  /// Stub — replaced in Task 10. The unit struct doesn't actually carry
  /// a value here; the real `SceneTask` does.
  pub const fn with_default_confidence(self, _val: f32) -> Self {
    self
  }
}

impl Task for SceneTask {
  type Output = findit_proto::database::SceneVlmResult;
  fn prompt(&self) -> &str {
    SCENE_PROMPT
  }
  fn schema(&self) -> &Value {
    unimplemented!("stub — replaced in Task 10")
  }
  fn parse(&self, _raw: &str) -> Result<Self::Output, ParseError> {
    unimplemented!("stub — replaced in Task 10")
  }
}

#[cfg(test)]
mod tests {
  use smol_str::SmolStr;

  use super::*;

  // --- 7 ports verbatim from findit-qwen/src/lib.rs:1068-1124 ---

  #[test]
  fn parse_valid_json() {
    let json = r#"{"scene":"beach","description":"Sunset over the ocean","subjects":["person"],"objects":["sun"],"actions":["watching"],"mood":["calm"],"shot_type":"wide shot","lighting":["golden hour"],"colors":["orange and blue"],"tags":["sunset","ocean"]}"#;
    let task = SceneTask::new();
    let result = task.parse(json).expect("parse should succeed");
    assert_eq!(result.scene.as_deref(), Some("beach"));
    assert_eq!(result.description.as_deref(), Some("Sunset over the ocean"));
    assert_eq!(result.mood.len(), 1);
    assert_eq!(result.subjects.len(), 1);
  }

  #[test]
  fn reject_json_with_wrapper_text() {
    let text =
      "Here is the analysis:\n{\"scene\":\"office\",\"description\":\"People working\"}\nDone.";
    let task = SceneTask::new();
    assert!(task.parse(text).is_err());
  }

  #[test]
  fn reject_plain_text_output() {
    let text = "A beautiful sunset over the ocean.";
    let task = SceneTask::new();
    assert!(task.parse(text).is_err());
  }

  #[test]
  fn parse_comma_separated_tag_string() {
    let json = r#"{"scene":"stage performance","description":"A singer on stage","subjects":[],"objects":["microphone"],"actions":["singing"],"mood":["energetic"],"shot_type":"medium shot","lighting":["spotlight"],"colors":["blue"],"tags":"concert, live music, spotlight"}"#;
    let task = SceneTask::new();
    let result = task.parse(json).expect("parse should succeed");
    let tags: Vec<SmolStr> = result.tags.iter().cloned().collect();
    assert_eq!(
      tags,
      vec![
        SmolStr::from("concert"),
        SmolStr::from("live music"),
        SmolStr::from("spotlight"),
      ]
    );
  }

  #[test]
  fn reject_empty_json_payload() {
    let task = SceneTask::new();
    assert!(task.parse("{}").is_err());
  }

  #[test]
  fn reject_unknown_json_fields() {
    let json = r#"{"description":"A singer on stage","extra":"unexpected"}"#;
    let task = SceneTask::new();
    assert!(task.parse(json).is_err());
  }

  #[test]
  fn reject_missing_required_fields() {
    let json = r#"{"description":"A singer on stage","tags":["concert"]}"#;
    let task = SceneTask::new();
    assert!(task.parse(json).is_err());
  }

  // --- 2 new tests (per spec §Testing) ---

  #[test]
  fn parse_with_custom_default_confidence() {
    let json = r#"{"scene":"beach","description":"Sunset","subjects":["person"],"objects":["sun"],"actions":["watching"],"mood":["calm"],"shot_type":"wide shot","lighting":["golden hour"],"colors":["orange"],"tags":["sunset"]}"#;
    let task = SceneTask::new().with_default_confidence(0.5);
    let result = task.parse(json).expect("parse should succeed");
    // Confidence stamped on every detection emitted from the parser.
    assert_eq!(result.subjects[0].confidence(), 0.5);
    assert_eq!(result.objects[0].confidence(), 0.5);
    assert_eq!(result.actions[0].confidence(), 0.5);
    assert_eq!(result.mood[0].confidence(), 0.5);
    assert_eq!(result.lighting[0].confidence(), 0.5);
    assert_eq!(result.colors[0].confidence(), 0.5);
    assert_eq!(result.classifications[0].confidence(), 0.5);
  }

  #[test]
  fn parse_mixed_shape_subjects() {
    // List form yields N items.
    let json_list = r#"{"scene":"x","description":"y","subjects":["a","b"],"objects":[],"actions":[],"mood":[],"shot_type":"x","lighting":[],"colors":[],"tags":["t"]}"#;
    let task = SceneTask::new();
    let result = task.parse(json_list).expect("list-form parse");
    assert_eq!(result.subjects.len(), 2);
    assert_eq!(result.subjects[0].label(), "a");
    assert_eq!(result.subjects[1].label(), "b");

    // Comma-string form also yields N items via StringList resilience.
    let json_string = r#"{"scene":"x","description":"y","subjects":"a, b","objects":[],"actions":[],"mood":[],"shot_type":"x","lighting":[],"colors":[],"tags":["t"]}"#;
    let result = task.parse(json_string).expect("string-form parse");
    assert_eq!(result.subjects.len(), 2);
    assert_eq!(result.subjects[0].label(), "a");
    assert_eq!(result.subjects[1].label(), "b");
  }
}
```

- [ ] **Step 2: Verify `cargo check` is clean (the test module compiles, even if `parse` is unimplemented)**

Run:
```bash
cargo check --lib --tests 2>&1 | tail -10
```

Expected: zero errors. (Tests reference `SceneTask::new`, `with_default_confidence`, and `Task::parse` — all wired up to stubs that compile.)

- [ ] **Step 3: Run the tests — they should fail at runtime via `unimplemented!()`**

Run:
```bash
cargo test --lib scene::tests 2>&1 | tail -30
```

Expected: 9 tests fail. The failure messages mention "stub — replaced in Task 10" or `not implemented`. Some tests will fail before the panic if `with_default_confidence` is missing — that's **expected**, the next task adds it.

Note: `parse_with_custom_default_confidence` will fail with a "method not found" compile error. **That's part of the red state — fix it next task.** If `cargo test` exits with a build error (not a test failure), that's still red — we just need the next task to make it green.

- [ ] **Step 4: Commit (red — TDD scaffold)**

```bash
git add src/scene.rs
git commit -m "test(scene): port the 7 existing parser tests + 2 new

Verbatim ports of parse_valid_json, reject_json_with_wrapper_text,
reject_plain_text_output, parse_comma_separated_tag_string,
reject_empty_json_payload, reject_unknown_json_fields, and
reject_missing_required_fields from findit-qwen/src/lib.rs:1068-1124.

Plus two new tests:
- parse_with_custom_default_confidence: every detection emitted by
  the parser carries the configured default confidence (0.5 here).
- parse_mixed_shape_subjects: subjects accepts list-form OR comma-
  separated string form (StringList resilience that absorbed real
  model output drift in production).

SceneTask is a stub for now (unimplemented! in parse/schema); the
next task replaces it with the real impl. This is the red commit
in TDD order."
```

---

### Task 10: Implement the real `SceneTask` + parser

**Files:**
- Modify: `qwen/src/scene.rs` (replace the stubs and add helpers)

- [ ] **Step 1: Replace the entire file with the production implementation**

Write `qwen/src/scene.rs` with this content. The test module from Task 9 stays at the bottom; everything above it is the production code:

```rust
//! The scene-analysis preset: [`SceneTask`] produces
//! `findit_proto::database::SceneVlmResult` directly.

use findit_proto::{
  common::{
    ActionDetection, BoundingBox, ClassificationDetection, ColorDetection, LightingDetection,
    MoodDetection, ObjectDetection, SubjectDetection,
  },
  database::SceneVlmResult,
};
use serde::Deserialize;
use serde_json::{Value, json};
use smol_str::SmolStr;

use crate::task::{ParseError, Task};

/// The scene-analysis prompt — verbatim port from `findit-qwen/src/lib.rs:382-401`.
const SCENE_PROMPT: &str = r#"Analyze the following video keyframes (in chronological order) from a single scene.

Return ONLY a valid JSON object with exactly these fields:
scene: short scene category in English (e.g. "office", "street", "kitchen", "stage performance")
description: 1-2 concise sentences in English describing the stable visual facts across the scene. Cover who is present, what they are doing, the setting, and the overall mood or visual style. If readable on-screen text appears, quote that text first, then continue the description.
subjects: array of distinct people or animals as short noun phrases with visible distinguishing features (e.g. ["middle-aged man in red jacket", "golden retriever"])
objects: array of notable, search-relevant objects as short noun phrases (e.g. ["birthday cake with candles", "vintage red sports car"])
actions: array of visible actions as short verb phrases (e.g. ["cutting cake", "taking photos", "walking"])
mood: array of scene-level mood terms (e.g. ["celebratory", "tense", "calm"])
shot_type: one short camera-shot label in English (e.g. "wide shot", "close-up", "medium shot", "aerial", "POV", "over-the-shoulder")
lighting: array of lighting terms (e.g. ["natural light", "low light", "backlit"])
colors: array of dominant color or palette terms (e.g. ["warm tones", "blue and white", "neon colors"])
tags: array of 8-12 short English search tags in lowercase. Prefer high-confidence search terms, complementary synonyms, style words, and culture-specific terms only when visually supported.

Rules:
- Use only information supported by the keyframes.
- Prefer concrete visual facts over speculation.
- Keep arrays deduplicated.
- Use empty arrays or empty strings when a field is unknown.
- Do not return markdown or any text outside the JSON object."#;

const REQUIRED_FIELDS: &[&str] = &[
  "scene",
  "description",
  "subjects",
  "objects",
  "actions",
  "mood",
  "shot_type",
  "lighting",
  "colors",
  "tags",
];

/// The scene-analysis task. Construct via [`SceneTask::new`].
pub struct SceneTask {
  schema: Value,
  default_confidence: f32,
}

impl SceneTask {
  /// Construct with `default_confidence = 0.8` (matches the value
  /// hard-coded in the legacy `findit-qwen` parser).
  pub fn new() -> Self {
    Self {
      schema: build_schema(),
      default_confidence: 0.8,
    }
  }

  // --- default_confidence (Copy: const fn) ---

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn default_confidence(&self) -> f32 {
    self.default_confidence
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_default_confidence(mut self, val: f32) -> Self {
    self.default_confidence = val;
    self
  }

  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_default_confidence(&mut self, val: f32) -> &mut Self {
    self.default_confidence = val;
    self
  }
}

impl Default for SceneTask {
  fn default() -> Self {
    Self::new()
  }
}

impl Task for SceneTask {
  type Output = SceneVlmResult;

  fn prompt(&self) -> &str {
    SCENE_PROMPT
  }

  fn schema(&self) -> &Value {
    &self.schema
  }

  fn parse(&self, raw: &str) -> Result<Self::Output, ParseError> {
    let value: Value = serde_json::from_str(raw.trim())?;
    let object = value
      .as_object()
      .ok_or_else(|| ParseError::Json(serde::de::Error::custom("expected top-level object")))?;
    let missing = missing_required_fields(object);
    if !missing.is_empty() {
      return Err(ParseError::MissingFields(missing));
    }
    let payload: QwenScenePayload = serde_json::from_value(value)?;
    if payload.is_empty() {
      return Err(ParseError::NoUsableFields);
    }
    Ok(payload.into_scene_vlm_result(self.default_confidence))
  }
}

fn build_schema() -> Value {
  json!({
    "type": "object",
    "properties": {
      "scene": { "type": "string" },
      "description": { "type": "string" },
      "subjects": { "type": "array", "items": { "type": "string" } },
      "objects": { "type": "array", "items": { "type": "string" } },
      "actions": { "type": "array", "items": { "type": "string" } },
      "mood": { "type": "array", "items": { "type": "string" } },
      "shot_type": { "type": "string" },
      "lighting": { "type": "array", "items": { "type": "string" } },
      "colors": { "type": "array", "items": { "type": "string" } },
      "tags": { "type": "array", "items": { "type": "string" } }
    },
    "required": REQUIRED_FIELDS,
    "additionalProperties": false
  })
}

fn missing_required_fields(object: &serde_json::Map<String, Value>) -> Vec<&'static str> {
  REQUIRED_FIELDS
    .iter()
    .copied()
    .filter(|field| !object.contains_key(*field))
    .collect()
}

// --- Internal payload (port from findit-qwen/src/lib.rs:897-988) ---

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct QwenScenePayload {
  #[serde(default, deserialize_with = "deserialize_optional_trimmed_string")]
  scene: Option<String>,
  #[serde(default, deserialize_with = "deserialize_optional_trimmed_string")]
  description: Option<String>,
  #[serde(default)]
  subjects: StringList,
  #[serde(default)]
  objects: StringList,
  #[serde(default)]
  actions: StringList,
  #[serde(default)]
  mood: StringList,
  #[serde(default, deserialize_with = "deserialize_optional_single_label")]
  shot_type: Option<String>,
  #[serde(default)]
  lighting: StringList,
  #[serde(default)]
  colors: StringList,
  #[serde(default)]
  tags: StringList,
}

impl QwenScenePayload {
  fn is_empty(&self) -> bool {
    self.scene.is_none()
      && self.description.is_none()
      && self.subjects.0.is_empty()
      && self.objects.0.is_empty()
      && self.actions.0.is_empty()
      && self.mood.0.is_empty()
      && self.shot_type.is_none()
      && self.lighting.0.is_empty()
      && self.colors.0.is_empty()
      && self.tags.0.is_empty()
  }

  fn into_scene_vlm_result(self, confidence: f32) -> SceneVlmResult {
    let scene = self.scene.map(SmolStr::from);
    let classifications = scene
      .iter()
      .cloned()
      .map(|label| ClassificationDetection::new(label, confidence))
      .collect();
    SceneVlmResult {
      scene,
      description: self.description.map(SmolStr::from),
      subjects: self
        .subjects
        .0
        .into_iter()
        .map(|label| SubjectDetection::new(label, confidence, BoundingBox::default()))
        .collect(),
      objects: self
        .objects
        .0
        .into_iter()
        .map(|label| ObjectDetection::new(label, confidence))
        .collect(),
      actions: self
        .actions
        .0
        .into_iter()
        .map(|label| ActionDetection::new(label, confidence))
        .collect(),
      mood: self
        .mood
        .0
        .into_iter()
        .map(|label| MoodDetection::new(label, confidence))
        .collect(),
      shot_type: self.shot_type.map(SmolStr::from),
      lighting: self
        .lighting
        .0
        .into_iter()
        .map(|label| LightingDetection::new(label, confidence))
        .collect(),
      colors: self
        .colors
        .0
        .into_iter()
        .map(|label| ColorDetection::new(label, confidence))
        .collect(),
      tags: self.tags.0.into_iter().map(SmolStr::from).collect(),
      classifications,
    }
  }
}

#[derive(Debug, Default)]
struct StringList(Vec<String>);

impl<'de> Deserialize<'de> for StringList {
  fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
  where
    D: serde::Deserializer<'de>,
  {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Repr {
      String(String),
      List(Vec<String>),
    }

    let raw = Option::<Repr>::deserialize(deserializer)?;
    let mut values = Vec::new();
    match raw {
      Some(Repr::String(value)) => push_string_list_items(&mut values, &value),
      Some(Repr::List(items)) => {
        for item in items {
          push_string_list_items(&mut values, &item);
        }
      }
      None => {}
    }
    Ok(Self(values))
  }
}

fn push_string_list_items(values: &mut Vec<String>, raw: &str) {
  for part in raw.split([',', ';', '\n']) {
    let part = part.trim();
    if !part.is_empty() && !values.iter().any(|existing| existing == part) {
      values.push(part.to_owned());
    }
  }
}

fn deserialize_optional_trimmed_string<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
  D: serde::Deserializer<'de>,
{
  Ok(Option::<String>::deserialize(deserializer)?.and_then(normalize_string))
}

fn deserialize_optional_single_label<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
  D: serde::Deserializer<'de>,
{
  #[derive(Deserialize)]
  #[serde(untagged)]
  enum Repr {
    String(String),
    List(Vec<String>),
  }

  match Option::<Repr>::deserialize(deserializer)? {
    Some(Repr::String(value)) => Ok(normalize_string(value)),
    Some(Repr::List(values)) => {
      let mut normalized = values.into_iter().filter_map(normalize_string);
      let first = normalized.next();
      if normalized.next().is_some() {
        return Err(serde::de::Error::custom(
          "expected a single shot_type label, got multiple values",
        ));
      }
      Ok(first)
    }
    None => Ok(None),
  }
}

fn normalize_string(value: String) -> Option<String> {
  let trimmed = value.trim();
  (!trimmed.is_empty()).then(|| trimmed.to_owned())
}

#[cfg(test)]
mod tests {
  // (test module unchanged from Task 9 — copy verbatim)
}
```

**Important:** keep the `#[cfg(test)] mod tests { ... }` block from Task 9 verbatim at the bottom of the file. The replacement above adds production code above the test module; the tests themselves do not change.

- [ ] **Step 2: Verify the crate compiles**

Run:
```bash
cargo check --lib 2>&1 | tail -10
```

Expected: zero errors, zero warnings.

- [ ] **Step 3: Run all parser tests — should be 9/9 green**

Run:
```bash
cargo test --lib scene:: 2>&1 | tail -20
```

Expected:
```
running 9 tests
test scene::tests::parse_valid_json ... ok
test scene::tests::reject_json_with_wrapper_text ... ok
test scene::tests::reject_plain_text_output ... ok
test scene::tests::parse_comma_separated_tag_string ... ok
test scene::tests::reject_empty_json_payload ... ok
test scene::tests::reject_unknown_json_fields ... ok
test scene::tests::reject_missing_required_fields ... ok
test scene::tests::parse_with_custom_default_confidence ... ok
test scene::tests::parse_mixed_shape_subjects ... ok

test result: ok. 9 passed; 0 failed; ...
```

- [ ] **Step 4: Commit (green)**

```bash
git add src/scene.rs
git commit -m "feat(scene): implement SceneTask and resilient parser

Ports SCENE_PROMPT, the JSON schema, REQUIRED_FIELDS, and the full
QwenScenePayload + StringList + helper machinery verbatim from
findit-qwen/src/lib.rs:380-1066. The single substitution: the seven
hard-coded 0.8 confidence values in into_scene_vlm_result() are
replaced by the SceneTask's default_confidence field.

The parser preserves the field-shape resilience that absorbed real
model output drift: StringList accepts list-or-comma-string,
deserialize_optional_single_label accepts \"wide\" or [\"wide\"],
and deny_unknown_fields rejects any unexpected top-level keys.

All 9 tests now pass (7 ports + 2 new)."
```

---

### Task 11: Run the full test suite to confirm no module is broken

**Files:** none (verification step).

- [ ] **Step 1: Run all tests**

Run:
```bash
cargo test --lib 2>&1 | tail -20
```

Expected: 12 tests pass (3 EngineOptions + 9 scene parser).

- [ ] **Step 2: Run clippy in dry mode to catch silly issues**

Run:
```bash
cargo clippy --lib -- -D warnings 2>&1 | tail -10
```

Expected: zero warnings, exit 0. If clippy flags anything, fix inline before continuing.

- [ ] **Step 3: No commit needed — this is a checkpoint.**

---

## Phase 3 — Phase-zero smoke test

### Task 12: Write `examples/smoke.rs`

**Files:**
- Create: `qwen/examples/smoke.rs`

This is the **phase-zero smoke test** mandated by the spec (§Testing item 2). It is run *manually* before deeper migration work to verify mistralrs 0.8's `Constraint::JsonSchema` actually produces schema-compliant output on the multimodal pipeline. mistralrs has no end-to-end test that covers this combination.

- [ ] **Step 1: Create `qwen/examples/smoke.rs`**

```rust
//! Phase-zero smoke test for the qwen crate.
//!
//! Run manually with the model on disk:
//! ```sh
//! cargo run --release --example smoke -- /path/to/qwen3-vl-2b /path/to/keyframe.jpg
//! ```
//!
//! Validates that mistralrs 0.8's Constraint::JsonSchema applied through
//! the multimodal pipeline produces well-formed JSON for the scene-analysis
//! schema. Intentionally not a `#[test]` — runs against the real model and
//! takes ~15s on Apple Silicon Metal. See spec §Testing item 2.

use std::{path::PathBuf, process::ExitCode};

use qwen::{Engine, EngineOptions, scene::SceneTask};

#[tokio::main]
async fn main() -> ExitCode {
  tracing_subscriber::fmt::init();

  let mut args = std::env::args().skip(1);
  let model_path = match args.next() {
    Some(p) => PathBuf::from(p),
    None => {
      eprintln!("usage: smoke <model_path> <image_path> [<image_path> ...]");
      return ExitCode::from(2);
    }
  };
  let image_paths: Vec<PathBuf> = args.map(PathBuf::from).collect();
  if image_paths.is_empty() {
    eprintln!("usage: smoke <model_path> <image_path> [<image_path> ...]");
    return ExitCode::from(2);
  }

  let images: Vec<image::DynamicImage> = match image_paths
    .iter()
    .map(|p| image::open(p).map_err(|e| format!("{}: {}", p.display(), e)))
    .collect::<Result<Vec<_>, _>>()
  {
    Ok(v) => v,
    Err(e) => {
      eprintln!("failed to load image: {e}");
      return ExitCode::from(1);
    }
  };

  let opts = EngineOptions::new(&model_path);
  let engine = match Engine::load(opts).await {
    Ok(e) => e,
    Err(e) => {
      eprintln!("model load failed: {e}");
      return ExitCode::from(1);
    }
  };

  let task = SceneTask::new();
  let result = match engine.run(&task, images).await {
    Ok(r) => r,
    Err(e) => {
      eprintln!("inference failed: {e}");
      return ExitCode::from(1);
    }
  };

  println!("scene:        {:?}", result.scene.as_deref());
  println!("description:  {:?}", result.description.as_deref());
  println!("subjects:     {} items", result.subjects.len());
  println!("objects:      {} items", result.objects.len());
  println!("actions:      {} items", result.actions.len());
  println!("tags:         {:?}", result.tags);
  ExitCode::SUCCESS
}
```

- [ ] **Step 2: Add `tracing-subscriber` to dev-deps so the smoke binary builds**

Modify `qwen/Cargo.toml`. In the `[dev-dependencies]` block, add the second line:

```toml
[dev-dependencies]
tokio              = { version = "1", features = ["macros", "rt-multi-thread"] }
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

- [ ] **Step 3: Verify the example compiles (no execution)**

Run:
```bash
cargo build --release --example smoke 2>&1 | tail -10
```

Expected: zero errors. (Slow build because of `--release`; if iteration speed matters use `cargo build --example smoke` first.)

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml examples/smoke.rs
git commit -m "test(examples): add phase-zero smoke binary

Runs the SceneTask preset against the real model + a fixture image,
prints the parsed result. Intentionally not a #[test] — takes ~15s
on Metal and requires the model on disk. Validates mistralrs 0.8's
Constraint::JsonSchema applied through the multimodal pipeline,
which has no end-to-end coverage in mistralrs itself.

Adds tracing-subscriber as a dev-dep so the example binary can
emit the qwen::load and qwen::run spans for diagnostic value."
```

- [ ] **Step 5 (manual, gated by model availability): run the smoke test**

If `/Users/user/Develop/findit-studio/indexer/models/qwen3-vl-2b/` exists locally and you have at least one keyframe JPEG, run:

```bash
cargo run --release --example smoke -- \
    /Users/user/Develop/findit-studio/indexer/models/qwen3-vl-2b \
    /path/to/some/keyframe.jpg
```

Expected: ~15s of model load + ~5s of inference, then output like:

```
scene:        Some("office")
description:  Some("...")
subjects:     ...
objects:      ...
actions:      ...
tags:         [...]
```

**This is the phase-zero gate.** If the smoke test fails (model load error, parse error, schema violation), pause and diagnose before proceeding to the migration tasks. If it succeeds, proceed.

If you don't have the model locally, mark this step as deferred — the migration tasks can still be implemented and tested in isolation, and the smoke test can run later.

---

## Phase 4 — Integration test

### Task 13: Write `tests/integration_scene.rs`

**Files:**
- Create: `qwen/tests/integration_scene.rs`

- [ ] **Step 1: Create the file**

```rust
//! Integration test for SceneTask, gated behind `--features integration`.
//!
//! Reads model path from `QWEN_MODEL_PATH` env var; if unset, the test
//! skips (returns success without invoking the model). Run with:
//!
//! ```sh
//! QWEN_MODEL_PATH=/path/to/qwen3-vl-2b \
//!   cargo test --features integration --test integration_scene
//! ```

#![cfg(feature = "integration")]

use std::path::PathBuf;

use qwen::{Engine, EngineOptions, scene::SceneTask};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scene_task_against_real_model() {
  let Ok(model_path) = std::env::var("QWEN_MODEL_PATH") else {
    eprintln!("skipping: QWEN_MODEL_PATH not set");
    return;
  };

  // Fixture images: ship 3 small JPEGs in tests/fixtures/. If they don't
  // exist, fall back to a single 1×1 black image so the test still validates
  // the inference path (the parse may fail for a black image — that's fine,
  // we only care that the engine + schema path round-trips).
  let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
  let mut images: Vec<image::DynamicImage> = Vec::new();
  if fixture_dir.exists() {
    for entry in std::fs::read_dir(&fixture_dir).expect("fixtures readable") {
      let entry = entry.expect("dir entry");
      if entry
        .path()
        .extension()
        .map(|e| e == "jpg" || e == "jpeg")
        .unwrap_or(false)
      {
        let img = image::open(entry.path()).expect("decode fixture");
        images.push(img);
      }
    }
  }
  if images.is_empty() {
    images.push(image::DynamicImage::ImageRgb8(image::RgbImage::new(1, 1)));
  }

  let engine = Engine::load(EngineOptions::new(model_path))
    .await
    .expect("model loads");

  let task = SceneTask::new();
  let result = engine.run(&task, images).await.expect("inference");

  // Loose assertions — the model decides what's in the scene; we only
  // care that the structured output round-trips.
  assert!(
    result.description.is_some() || !result.tags.is_empty(),
    "expected at least one populated field"
  );
}
```

- [ ] **Step 2: Verify it compiles under the feature flag (no execution)**

Run:
```bash
cargo check --tests --features integration 2>&1 | tail -10
```

Expected: zero errors.

- [ ] **Step 3: Verify the default build is unaffected (test is feature-gated)**

Run:
```bash
cargo check --tests 2>&1 | tail -10
```

Expected: zero errors. The integration test is `#![cfg(feature = "integration")]` so it's silently absent without the feature.

- [ ] **Step 4: Commit**

```bash
git add tests/integration_scene.rs
git commit -m "test(integration): add gated integration_scene test

Behind --features integration. Reads QWEN_MODEL_PATH env var; if
unset, the test silently returns success (no model required for
default CI). Loads tests/fixtures/*.jpg if present, else falls
back to a 1×1 black image for inference-path coverage. The
assertion is loose: at least one of description/tags must be
populated — the model owns content decisions, the test owns the
'structured output round-trips' contract."
```

---

## Phase 5 — `findit-qwen` migration

The remaining tasks happen in **`/Users/user/Develop/findit-studio/indexer/`** (a different git repo, currently on branch `feat/lifecycle`). The qwen crate at `/Users/user/Develop/findit-studio/qwen/` does **not** change in these tasks.

### Task 14: Register `qwen` in the indexer workspace

**Files:**
- Modify: `indexer/Cargo.toml`

- [ ] **Step 1: Add the workspace dependency**

In `/Users/user/Develop/findit-studio/indexer/Cargo.toml`, locate the `[workspace.dependencies]` block (currently lines 21-32). Add `qwen = { path = "../qwen" }` in alphabetical order — between `findit-storage` and `findit-silero-vad`:

Before:
```toml
[workspace.dependencies]
findit-service = { path = "findit-service" }
findit-proto = { path = "findit-proto" }
findit-storage = { path = "findit-storage" }
findit-silero-vad = { path = "services/findit-silero-vad" }
```

After:
```toml
[workspace.dependencies]
findit-service = { path = "findit-service" }
findit-proto = { path = "findit-proto" }
findit-storage = { path = "findit-storage" }
qwen = { path = "../qwen" }
findit-silero-vad = { path = "services/findit-silero-vad" }
```

- [ ] **Step 2: Verify the indexer workspace still resolves**

Run:
```bash
cd /Users/user/Develop/findit-studio/indexer
cargo metadata --no-deps --format-version 1 > /dev/null
```

Expected: exit 0, no error.

- [ ] **Step 3: Commit (in indexer repo)**

```bash
git add Cargo.toml
git commit -m "chore: register qwen as workspace dependency

Adds qwen = { path = \"../qwen\" } to [workspace.dependencies].
This is the first cross-repo path dependency consumed by the
indexer/ workspace; sibling crates (scenesdetect, mediatime,
silero, etc.) exist alongside but are not currently consumed
via local path. See qwen/docs/superpowers/specs/2026-04-28-
qwen-engine-design.md §Workspace integration for rationale."
```

---

### Task 15: Swap `findit-qwen` deps

**Files:**
- Modify: `indexer/services/findit-qwen/Cargo.toml`

- [ ] **Step 1: Replace deps**

In `/Users/user/Develop/findit-studio/indexer/services/findit-qwen/Cargo.toml`, change:

Before:
```toml
[dependencies]
bytes.workspace = true
crossbeam-channel = "0.5"
findit-proto = { workspace = true, features = ["tokio"] }
findit-service.workspace = true
image = { version = "0.25", default-features = false, features = ["jpeg"] }
mistralrs = { version = "0.7", features = ["metal"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
smol_str = "0.3"
tokio.workspace = true
tracing.workspace = true
```

After:
```toml
[dependencies]
bytes.workspace = true
crossbeam-channel = "0.5"
findit-proto = { workspace = true, features = ["tokio"] }
findit-service.workspace = true
image = { version = "0.25", default-features = false, features = ["jpeg"] }
qwen.workspace = true
tokio.workspace = true
tracing.workspace = true
```

(Drops `mistralrs`, `serde`, `serde_json`, `smol_str` — these moved into the `qwen` crate. Adds `qwen.workspace = true`.)

- [ ] **Step 2: Do **not** build yet** — `findit-qwen/src/lib.rs` still uses `mistralrs::*` types and won't compile until Task 17. Skip ahead.

- [ ] **Step 3: No commit yet** — keep the Cargo.toml change unstaged until lib.rs is done in Task 17, then commit them together.

---

### Task 16: Reshape `ServiceOptions` and run worker (in lib.rs)

**Files:**
- Modify: `indexer/services/findit-qwen/src/lib.rs` (full rewrite)

This is the biggest task. The full new file is provided below. Read the spec §"`findit-qwen` migration" for the rationale: 480 lines of parser/schema/prompt code go away; the worker calls `qwen::Engine` and the service layer no longer touches mistralrs directly.

- [ ] **Step 1: Replace `findit-qwen/src/lib.rs` entirely**

Write `/Users/user/Develop/findit-studio/indexer/services/findit-qwen/src/lib.rs` with this content:

```rust
//! Long-running Qwen3-VL vision-language model service thread.
//!
//! Single worker thread — GPU inference is continuous-batched by mistralrs
//! internally but the engine loop holds the pipeline under a mutex, so we
//! keep one worker for now. The thread loads the model once at startup
//! (~13s on Apple Silicon Metal) and processes scene batches sequentially.
//!
//! Input: `Request` via crossbeam bounded channel.
//! Output: `Reply` via callback back to the processor-local coordinator.

use std::{future::Future, sync::Arc, time::Duration};

use bytes::Bytes;
use crossbeam_channel::{self as channel, TryRecvError, select_biased};
use findit_proto::{
  Id,
  common::{ErrorCode, ErrorInfo},
  database::SceneVlmResult,
};
use findit_service::{
  Lifecycle, ProviderIdentifier, ProviderKey, ProviderKind, ProviderThreadService,
  ServiceHealthConfig, ServiceHealthReporter, ShutdownToken, ThreadHandles, ThreadService,
  ThreadServiceContext, ThreadServiceHealthSpec, VideoLifecycle,
};
use image::DynamicImage;
use tracing::{debug, info, warn};

const SERVICE_NAME: &str = "qwen-vlm";
const SHUTDOWN_POLL_INTERVAL: Duration = Duration::from_millis(250);

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// Options for the Qwen VLM service.
#[derive(Debug, Clone)]
pub struct ServiceOptions {
  engine: qwen::EngineOptions,
  default_confidence: f32,
  queue_capacity: usize,
  model_load_timeout: Duration,
  inference_timeout: Duration,
}

impl Default for ServiceOptions {
  fn default() -> Self {
    Self {
      engine: qwen::EngineOptions::new("/dev/null"),
      default_confidence: 0.8,
      queue_capacity: 8,
      model_load_timeout: Duration::from_secs(5 * 60),
      inference_timeout: Duration::from_secs(45),
    }
  }
}

impl ServiceOptions {
  pub fn new(model_path: impl Into<std::path::PathBuf>) -> Self {
    Self {
      engine: qwen::EngineOptions::new(model_path),
      ..Default::default()
    }
  }

  /// Construct from a models directory containing a `qwen3-vl-2b` subfolder.
  pub fn local(models_dir: impl AsRef<std::path::Path>) -> Self {
    let local = models_dir.as_ref().join("qwen3-vl-2b");
    Self::new(local)
  }

  // --- engine (non-Copy: plain pub fn) ---

  #[inline]
  pub fn engine(&self) -> &qwen::EngineOptions {
    &self.engine
  }

  #[inline]
  pub fn with_engine(mut self, val: qwen::EngineOptions) -> Self {
    self.engine = val;
    self
  }

  #[inline]
  pub fn set_engine(&mut self, val: qwen::EngineOptions) -> &mut Self {
    self.engine = val;
    self
  }

  // --- default_confidence (Copy: const fn) ---

  #[inline]
  pub const fn default_confidence(&self) -> f32 {
    self.default_confidence
  }

  #[inline]
  pub const fn with_default_confidence(mut self, val: f32) -> Self {
    self.default_confidence = val;
    self
  }

  #[inline]
  pub const fn set_default_confidence(&mut self, val: f32) -> &mut Self {
    self.default_confidence = val;
    self
  }

  // --- queue_capacity / timeouts (Copy: const fn) ---

  #[inline]
  pub const fn queue_capacity(&self) -> usize {
    self.queue_capacity
  }

  #[inline]
  pub const fn with_queue_capacity(mut self, val: usize) -> Self {
    self.queue_capacity = if val == 0 { 1 } else { val };
    self
  }

  #[inline]
  pub const fn set_queue_capacity(&mut self, val: usize) -> &mut Self {
    self.queue_capacity = if val == 0 { 1 } else { val };
    self
  }

  #[inline]
  pub const fn model_load_timeout(&self) -> Duration {
    self.model_load_timeout
  }

  #[inline]
  pub const fn with_model_load_timeout(mut self, val: Duration) -> Self {
    self.model_load_timeout = val;
    self
  }

  #[inline]
  pub const fn set_model_load_timeout(&mut self, val: Duration) -> &mut Self {
    self.model_load_timeout = val;
    self
  }

  #[inline]
  pub const fn inference_timeout(&self) -> Duration {
    self.inference_timeout
  }

  #[inline]
  pub const fn with_inference_timeout(mut self, val: Duration) -> Self {
    self.inference_timeout = val;
    self
  }

  #[inline]
  pub const fn set_inference_timeout(&mut self, val: Duration) -> &mut Self {
    self.inference_timeout = val;
    self
  }
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

/// Messages sent from processor tasks to the Qwen VLM service.
pub struct Request {
  video_id: Id,
  scene_id: Id,
  /// (keyframe_id, jpeg_data) pairs — sent as a sequence to the model.
  keyframes: Arc<[(Id, Bytes)]>,
  reply: Callback,
}

impl Request {
  #[inline]
  pub fn new(video_id: Id, scene_id: Id, keyframes: Arc<[(Id, Bytes)]>, reply: Callback) -> Self {
    Self {
      video_id,
      scene_id,
      keyframes,
      reply,
    }
  }
  #[inline]
  pub const fn video_id(&self) -> Id {
    self.video_id
  }
  #[inline]
  pub const fn scene_id(&self) -> Id {
    self.scene_id
  }
  #[inline]
  pub fn keyframes(&self) -> &[(Id, Bytes)] {
    &self.keyframes
  }
  #[inline]
  pub fn reply(&self) -> &Callback {
    &self.reply
  }
  #[allow(clippy::type_complexity)]
  #[inline]
  pub fn into_parts(self) -> (Id, Id, Arc<[(Id, Bytes)]>, Callback) {
    (self.video_id, self.scene_id, self.keyframes, self.reply)
  }
}

pub struct Reply {
  scene_id: Id,
  result: SceneVlmResult,
  error: Option<ErrorInfo>,
}

impl Reply {
  #[inline]
  pub fn new(scene_id: Id, result: SceneVlmResult, error: Option<ErrorInfo>) -> Self {
    Self {
      scene_id,
      result,
      error,
    }
  }
  #[inline]
  pub const fn scene_id(&self) -> Id {
    self.scene_id
  }
  #[inline]
  pub const fn result(&self) -> &SceneVlmResult {
    &self.result
  }
  #[inline]
  pub const fn error(&self) -> Option<&ErrorInfo> {
    self.error.as_ref()
  }
  #[inline]
  pub fn into_parts(self) -> (Id, SceneVlmResult, Option<ErrorInfo>) {
    (self.scene_id, self.result, self.error)
  }
}

pub type Callback = Box<dyn FnOnce(Reply) + Send + 'static>;

// ---------------------------------------------------------------------------
// Service plumbing (unchanged from the legacy version)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SpawnError {
  error: ErrorInfo,
}

impl SpawnError {
  fn io(context: &'static str, error: std::io::Error) -> Self {
    Self {
      error: ErrorInfo::new(
        ErrorCode::service_unavailable(),
        format!("{context}: {error}"),
      ),
    }
  }
}

impl std::fmt::Display for SpawnError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}", self.error)
  }
}

impl std::error::Error for SpawnError {}

impl From<SpawnError> for ErrorInfo {
  fn from(error: SpawnError) -> Self {
    error.error
  }
}

#[derive(Debug, Clone, Copy)]
pub struct Service(());

impl ThreadService for Service {
  type Input = Request;
  type Options = ServiceOptions;
  type SpawnError = SpawnError;
  type Handle = ThreadHandles<Self::Input>;

  #[inline]
  fn name() -> &'static str {
    SERVICE_NAME
  }

  fn health_spec(_options: &Self::Options) -> ThreadServiceHealthSpec {
    ThreadServiceHealthSpec::new(1, ServiceHealthConfig::default())
  }

  fn spawn(
    options: Self::Options,
    ctx: ThreadServiceContext,
  ) -> Result<Self::Handle, Self::SpawnError> {
    let (tx, rx) = channel::bounded::<Self::Input>(options.queue_capacity().max(1));
    let (shutdown, health_reporter, health_handle, health_config) = ctx.into_parts();
    let handle = std::thread::Builder::new()
      .name(format!("{}-0", Self::name()))
      .spawn(move || {
        run_qwen_worker(
          options,
          shutdown,
          rx,
          health_reporter,
          health_config.heartbeat_interval(),
        )
      })
      .map_err(|error| SpawnError::io("failed to spawn worker thread", error))?;
    info!(service = SERVICE_NAME, "Qwen VLM service started");
    Ok(ThreadHandles::with_named_service_health(
      Self::name(),
      tx,
      vec![handle],
      Some(health_handle),
    ))
  }
}

impl ProviderIdentifier for Service {
  const KEY: ProviderKey = ProviderKey::internal_after(
    Lifecycle::Video(VideoLifecycle::KeyframeExtract),
    Lifecycle::Video(VideoLifecycle::VisionAnalysis),
    SERVICE_NAME,
  );
  const IMPLEMENTATION_HASH: u64 = 0;
}

impl ProviderThreadService for Service {
  const KIND: ProviderKind = ProviderKind::Standard;

  type LifecycleInput = Request;
  type LifecycleOutput = Reply;
}

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

fn run_qwen_worker(
  opts: ServiceOptions,
  shutdown: ShutdownToken,
  rx: channel::Receiver<Request>,
  health: ServiceHealthReporter,
  heartbeat_interval: Duration,
) {
  health.starting(0);

  let rt = match tokio::runtime::Builder::new_current_thread()
    .enable_all()
    .build()
  {
    Ok(rt) => rt,
    Err(e) => {
      let msg = format!("runtime init failed: {e}");
      health.failed(0, ErrorInfo::new(ErrorCode::vlm_model_error(), msg.clone()));
      warn!(service = SERVICE_NAME, err = %e, "failed to create Qwen service runtime");
      drain_with_error(&rx, &shutdown, ErrorCode::vlm_model_error(), &msg);
      return;
    }
  };

  info!(service = SERVICE_NAME, model_path = %opts.engine().model_path().display(), "loading Qwen VLM model");
  let engine = match rt.block_on(run_with_timeout_and_shutdown(
    shutdown.clone(),
    health.clone(),
    0,
    heartbeat_interval,
    opts.model_load_timeout(),
    qwen::Engine::load(opts.engine().clone()),
  )) {
    Ok(m) => {
      health.ready(0);
      info!(service = SERVICE_NAME, "Qwen VLM model loaded");
      m
    }
    Err(OperationError::Shutdown) => {
      health.stopped(0);
      info!(service = SERVICE_NAME, "shutdown received while loading Qwen VLM model");
      return;
    }
    Err(OperationError::Timeout(timeout)) => {
      let msg = format!("model load timed out after {timeout:?}");
      health.failed(0, ErrorInfo::new(ErrorCode::timeout(), msg.clone()));
      warn!(service = SERVICE_NAME, timeout = ?timeout, "Qwen VLM model load timed out");
      drain_with_error(&rx, &shutdown, ErrorCode::timeout(), &msg);
      return;
    }
    Err(OperationError::Inner(e)) => {
      let e = e.to_string();
      health.failed(
        0,
        ErrorInfo::new(ErrorCode::vlm_model_error(), format!("model load failed: {e}")),
      );
      warn!(service = SERVICE_NAME, err = %e, "failed to load Qwen VLM model");
      let msg = format!("model load failed: {e}");
      drain_with_error(&rx, &shutdown, ErrorCode::vlm_model_error(), &msg);
      return;
    }
  };

  let scene_task = qwen::scene::SceneTask::new().with_default_confidence(opts.default_confidence());
  let mut processed = 0u64;
  let heartbeat = channel::tick(heartbeat_interval);

  loop {
    select_biased! {
      recv(shutdown.receiver()) -> _ => break,
      recv(heartbeat) -> _ => {
        health.heartbeat(0);
      }
      recv(rx) -> msg => match msg {
        Ok(msg) => {
          let (video_id, scene_id, keyframes, reply) = msg.into_parts();
          health.heartbeat(0);

          let vlm_err =
            |msg: String| Some(ErrorInfo::new(ErrorCode::vlm_failed(), msg));
          let timeout_err =
            |msg: String| Some(ErrorInfo::new(ErrorCode::timeout(), msg));
          let cancelled_err =
            |msg: String| Some(ErrorInfo::new(ErrorCode::cancelled(), msg));

          let (images, dropped_keyframes) = decode_scene_images(&keyframes);
          if !dropped_keyframes.is_empty() {
            warn!(
              service = SERVICE_NAME,
              video_id = %video_id,
              scene_id = %scene_id,
              dropped = dropped_keyframes.len(),
              dropped_keyframe_ids = ?dropped_keyframes,
              "skipping invalid scene keyframes for Qwen analysis"
            );
          }

          let mut should_exit = false;
          let (result, error) = if images.is_empty() {
            (
              SceneVlmResult::default(),
              vlm_err("no decodable keyframes provided".into()),
            )
          } else {
            let count = images.len();
            match rt.block_on(run_with_timeout_and_shutdown(
              shutdown.clone(),
              health.clone(),
              0,
              heartbeat_interval,
              opts.inference_timeout(),
              engine.run(&scene_task, images),
            )) {
              Ok(result) => {
                processed += 1;
                debug!(service = SERVICE_NAME, processed, images = count, video_id = %video_id, "Qwen inference complete");
                (result, None)
              }
              Err(OperationError::Inner(qwen::Error::NoImages)) => (
                SceneVlmResult::default(),
                vlm_err("no decodable keyframes provided".into()),
              ),
              Err(OperationError::Inner(qwen::Error::Empty)) => {
                warn!(service = SERVICE_NAME, video_id = %video_id, "Qwen returned empty content");
                (SceneVlmResult::default(), vlm_err("empty model response".into()))
              }
              Err(OperationError::Inner(qwen::Error::Parse(e))) => {
                warn!(service = SERVICE_NAME, video_id = %video_id, err = %e, "Qwen returned invalid structured output");
                (SceneVlmResult::default(), vlm_err(format!("invalid Qwen JSON output: {e}")))
              }
              Err(OperationError::Inner(qwen::Error::BuildMessage(e))) => {
                warn!(service = SERVICE_NAME, video_id = %video_id, err = %e, "Qwen message build failed");
                (SceneVlmResult::default(), vlm_err(e))
              }
              Err(OperationError::Inner(qwen::Error::Inference(e))) => {
                warn!(service = SERVICE_NAME, video_id = %video_id, err = %e, "Qwen inference failed");
                (SceneVlmResult::default(), vlm_err(e))
              }
              Err(OperationError::Timeout(timeout)) => {
                let msg = format!("Qwen inference timed out after {timeout:?}");
                warn!(
                  service = SERVICE_NAME,
                  video_id = %video_id,
                  scene_id = %scene_id,
                  timeout = ?timeout,
                  "Qwen inference timed out"
                );
                (SceneVlmResult::default(), timeout_err(msg))
              }
              Err(OperationError::Shutdown) => {
                should_exit = true;
                (
                  SceneVlmResult::default(),
                  cancelled_err("Qwen service shutting down".into()),
                )
              }
            }
          };

          reply(Reply::new(scene_id, result, error));
          health.heartbeat(0);
          if should_exit {
            break;
          }
        }
        Err(_) => break,
      }
    }
  }
  health.stopped(0);

  info!(service = SERVICE_NAME, processed, "Qwen VLM service worker exited");
}

// ---------------------------------------------------------------------------
// Helpers (unchanged from legacy)
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum OperationError<E> {
  Inner(E),
  Timeout(Duration),
  Shutdown,
}

async fn run_with_timeout_and_shutdown<F, T, E>(
  shutdown: ShutdownToken,
  health: ServiceHealthReporter,
  worker_id: usize,
  heartbeat_interval: Duration,
  timeout: Duration,
  future: F,
) -> Result<T, OperationError<E>>
where
  F: Future<Output = Result<T, E>>,
{
  let heartbeat_interval = if heartbeat_interval.is_zero() {
    SHUTDOWN_POLL_INTERVAL
  } else {
    heartbeat_interval
  };
  let mut heartbeat = tokio::time::interval(heartbeat_interval);
  let mut shutdown_poll = tokio::time::interval(SHUTDOWN_POLL_INTERVAL);
  let timeout = timeout.max(SHUTDOWN_POLL_INTERVAL);
  let timeout_sleep = tokio::time::sleep(timeout);

  tokio::pin!(future);
  tokio::pin!(timeout_sleep);

  loop {
    tokio::select! {
      biased;
      _ = &mut timeout_sleep => return Err(OperationError::Timeout(timeout)),
      _ = shutdown_poll.tick() => {
        if shutdown_requested(&shutdown) {
          return Err(OperationError::Shutdown);
        }
      }
      _ = heartbeat.tick() => {
        health.heartbeat(worker_id);
        if shutdown_requested(&shutdown) {
          return Err(OperationError::Shutdown);
        }
      }
      result = &mut future => return result.map_err(OperationError::Inner),
    }
  }
}

fn shutdown_requested(shutdown: &ShutdownToken) -> bool {
  matches!(
    shutdown.receiver().try_recv(),
    Ok(_) | Err(TryRecvError::Disconnected)
  )
}

fn drain_with_error(
  rx: &channel::Receiver<Request>,
  shutdown: &ShutdownToken,
  code: ErrorCode,
  reason: &str,
) {
  loop {
    select_biased! {
      recv(shutdown.receiver()) -> _ => return,
      recv(rx) -> msg => match msg {
        Ok(msg) => {
          let (_, scene_id, _, reply) = msg.into_parts();
          reply(Reply::new(
            scene_id,
            SceneVlmResult::default(),
            Some(ErrorInfo::new(code.clone(), reason)),
          ));
        }
        Err(_) => return,
      }
    }
  }
}

fn decode_scene_images(keyframes: &[(Id, Bytes)]) -> (Vec<DynamicImage>, Vec<Id>) {
  let mut images = Vec::with_capacity(keyframes.len());
  let mut dropped_keyframes = Vec::new();

  for (keyframe_id, bytes) in keyframes {
    match image::load_from_memory(bytes) {
      Ok(image) => images.push(image),
      Err(err) => {
        warn!(
          service = SERVICE_NAME,
          keyframe_id = %keyframe_id,
          err = %err,
          "failed to decode Qwen keyframe JPEG"
        );
        dropped_keyframes.push(*keyframe_id);
      }
    }
  }

  (images, dropped_keyframes)
}
```

- [ ] **Step 2: Verify the crate compiles**

Run:
```bash
cd /Users/user/Develop/findit-studio/indexer
cargo check -p findit-qwen 2>&1 | tail -15
```

Expected: zero errors, zero warnings. (Slow first compile pulling in qwen + mistralrs.)

- [ ] **Step 3: Run any existing tests for `findit-qwen`** (the seven parser tests are gone — verify their absence isn't a build break)

Run:
```bash
cargo test -p findit-qwen --no-run 2>&1 | tail -10
```

Expected: zero errors. There are no tests left in `findit-qwen` (the parser tests moved to qwen); the test binary compiles empty.

- [ ] **Step 4: Verify the rest of the indexer workspace still builds**

Run:
```bash
cargo check --workspace 2>&1 | tail -15
```

Expected: zero errors. (`findit-indexer/src/processor/router.rs:25` still imports `findit_qwen::Service` — which is unchanged — so the consumer is unaffected.)

- [ ] **Step 5: Commit (in indexer repo) — single commit containing both Cargo.toml and lib.rs changes**

```bash
git add services/findit-qwen/Cargo.toml services/findit-qwen/src/lib.rs
git commit -m "refactor(findit-qwen): delegate to qwen crate

The 480 lines of prompt + JSON schema + resilient parser code that
used to live in services/findit-qwen/src/lib.rs (\`ANALYSIS_PROMPT\`,
\`qwen_output_schema\`, \`build_qwen_request\`, \`parse_vlm_output\`,
\`QwenScenePayload\`, \`StringList\`, \`deserialize_optional_*\`,
\`normalize_string\`, \`missing_required_fields\`, plus seven parser
tests) all moved to the qwen crate. The service now calls
qwen::Engine::run(&scene_task, images) and gets back a
SceneVlmResult directly — no in-service conversion.

ServiceOptions also reshapes: model_id (String) → engine
(qwen::EngineOptions); confidence becomes default_confidence and
flows through to qwen::scene::SceneTask. local_or_remote is
replaced by local — no more HuggingFace fallback (production
already uses local weights, and removing the network branch
removes failure modes).

Drops mistralrs / serde / serde_json / smol_str from the
Cargo.toml; adds qwen.workspace = true. Threading, queues,
health, lifecycle, and run_with_timeout_and_shutdown
(critical: preserves heartbeat-during-inference) are unchanged."
```

---

### Task 17: Verify the workspace at large

**Files:** none (verification step).

- [ ] **Step 1: Full workspace check**

Run:
```bash
cd /Users/user/Develop/findit-studio/indexer
cargo check --workspace --all-targets 2>&1 | tail -15
```

Expected: zero errors. Warnings about pre-existing issues unrelated to qwen are acceptable; new warnings from qwen-introduced code are not.

- [ ] **Step 2: Workspace test compilation (no execution — model not required)**

Run:
```bash
cargo test --workspace --no-run 2>&1 | tail -15
```

Expected: zero errors.

- [ ] **Step 3: Run `findit-qwen`'s test suite (scoped to avoid kicking off unrelated workspace integration tests)**

Run:
```bash
cargo test -p findit-qwen 2>&1 | tail -10
```

Expected: zero tests run, exit 0 (`findit-qwen` no longer has any tests — the seven parser tests moved to qwen). If the count is non-zero, double-check that Task 16 successfully removed the legacy `#[cfg(test)] mod tests` block.

- [ ] **Step 4: No commit** — this is a checkpoint. If anything fails, fix it inline before continuing.

---

## Phase 6 — Final verification

### Task 18: Final smoke test against the real model (manual)

**Files:** none (verification).

If the model is available locally, run the smoke binary one more time against the full pipeline to catch any regression introduced by Tasks 14-17.

- [ ] **Step 1: Run the smoke test (in qwen repo)**

```bash
cd /Users/user/Develop/findit-studio/qwen
cargo run --release --example smoke -- \
    /Users/user/Develop/findit-studio/indexer/models/qwen3-vl-2b \
    /path/to/keyframe1.jpg \
    /path/to/keyframe2.jpg
```

Expected: ~15s load + ~5s inference, then printed output with non-empty `description`, `subjects`, and `tags`.

- [ ] **Step 2: If smoke passes, the migration is complete.** No commit needed.

If smoke fails, treat it as a bug. The most likely failure modes:

| Symptom | Likely cause |
|---|---|
| `model load failed: ...` | Wrong path, wrong quantization, or mistralrs 0.8 changed something between when the spec was written and now. Check the actual error. |
| `inference failed: ...` | mistralrs 0.8 inference path issue. Check the error string. |
| `parse failed: invalid JSON` | The model emitted text that doesn't parse — check whether `Constraint::JsonSchema` is actually being applied (re-read `engine.rs:run` and grep for `set_constraint`). |
| `parse failed: structured response had no usable fields` | The schema let through an empty `{}` — investigate whether the model is following the prompt at all. Try a simpler prompt as a sanity check. |

Diagnose, fix in qwen, recommit, re-run smoke.

---

## Done checklist (entire plan)

- [ ] Task 1: scaffold cleanup
- [ ] Task 2: rewrite Cargo.toml
- [ ] Task 3: lib.rs skeleton
- [ ] Task 4: error.rs (LoadError + Error)
- [ ] Task 5: task.rs (Task trait + ParseError)
- [ ] Task 6: EngineOptions accessors (TDD, 3 tests)
- [ ] Task 7: Engine::load
- [ ] Task 8: Engine::run + warmup
- [ ] Task 9: scene.rs parser tests (red commit, 9 tests stubbed)
- [ ] Task 10: scene.rs production impl (green: 9 tests pass)
- [ ] Task 11: full test suite + clippy checkpoint
- [ ] Task 12: examples/smoke.rs (+ smoke test run if model available)
- [ ] Task 13: tests/integration_scene.rs (gated, +features integration)
- [ ] Task 14: register qwen in indexer workspace
- [ ] Task 15: swap findit-qwen Cargo.toml deps
- [ ] Task 16: rewrite findit-qwen/src/lib.rs
- [ ] Task 17: workspace check
- [ ] Task 18: final smoke test (if model available)

After all tasks complete, the qwen repo is on branch `0.1.0` with ~15 commits ahead of the initial commit, and the indexer repo is on `feat/lifecycle` with 2 new commits. `findit-qwen/src/lib.rs` has shrunk from 1126 → ~640 lines; the parser/prompt/schema work lives in `qwen/src/scene.rs` where it belongs.
