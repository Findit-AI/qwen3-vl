//! Top-level error types for the `qwen3-vl` crate.
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
  /// `RequestOptions` carried a value outside its valid range
  /// (e.g. negative temperature, top_p > 1.0, top_k = 0). Issue #1
  /// H-002 — sampler parameters were previously accepted without
  /// validation; passing an out-of-range value to mistralrs's
  /// sampler produces undefined behavior in most LLM engines.
  #[error("invalid RequestOptions: {0}")]
  InvalidRequest(&'static str),
  /// Inference exceeded the configured timeout. Issue #1 H-001 —
  /// `send_chat_request` was previously awaited without a deadline;
  /// a stuck model (Metal JIT stall, GPU memory exhaustion) would
  /// block the caller indefinitely. Returned by [`Engine::run`] /
  /// [`Engine::run_with`] when the inference duration exceeds
  /// `EngineOptions::inference_timeout`.
  #[error("inference timed out after {0:?}")]
  InferenceTimeout(std::time::Duration),
  /// mistralrs's `MultimodalMessages` builder rejected the message.
  ///
  /// **Reserved variant.** mistralrs 0.8's
  /// `MultimodalMessages::add_image_message` is infallible, so no current
  /// code path constructs this. It exists for forward compatibility with
  /// future mistralrs versions that may surface builder-validation errors,
  /// and to keep the migration arms in
  /// `docs/superpowers/specs/2026-04-28-qwen-engine-design.md`
  /// §"`findit-qwen` migration" exhaustive.
  #[error("vision message build failed: {0}")]
  BuildMessage(String),
  /// mistralrs returned an inference error.
  #[error("inference failed: {0}")]
  Inference(String),
  /// The model returned empty content (after trimming).
  #[error("model returned empty content")]
  Empty,
  /// The model hit `max_tokens` before producing a natural stop
  /// (mistralrs surfaces this via `Choice::finish_reason = "length"`).
  /// The raw text is included so callers can decide whether to
  /// retry with a higher `EngineOptions::max_tokens` or accept
  /// the partial output. finding: the engine
  /// previously parsed length-truncated JSON as success, which
  /// can persist incomplete metadata to a search index.
  #[error(
    "generation truncated by max_tokens (finish_reason={finish_reason:?}); raw output {raw_len} bytes"
  )]
  Truncated {
    /// The non-`stop` finish_reason mistralrs reported (e.g.,
    /// `"length"`, `"model_length"`).
    finish_reason: String,
    /// Length of the raw output in bytes (not the full text — that
    /// would inflate error logs without aiding diagnosis).
    raw_len: usize,
  },
  /// The model's output failed Task::parse — boxed because the
  /// generic `Task::ParseError` type varies per Task. JSON tasks
  /// surface as `Parse(Box::new(JsonParseError::...))`; custom
  /// Tasks surface as `Parse(Box::new(MyParseError))`. A concrete
  /// `From<JsonParseError>` bound at the engine call site would
  /// compile-time block any Task that uses a different ParseError
  /// type — including ones whose only purpose is to receive
  /// `UnsupportedGrammar` for routing to a different engine.
  #[error("parse failed: {0}")]
  Parse(Box<dyn core::error::Error + Send + Sync + 'static>),
  /// The supplied [`llmtask::Task`] returned a [`llmtask::Grammar`]
  /// variant qwen3-vl cannot route to mistralrs. mistralrs 0.8 only
  /// accepts JSON Schema; Lark / Regex tasks must run on an
  /// llguidance-backed engine (e.g., the `lfm` crate).
  #[error("{0}")]
  UnsupportedGrammar(#[from] llmtask::UnsupportedGrammar),
}
