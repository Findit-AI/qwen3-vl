//! Qwen3-VL structured-output engine for findit-studio.
//!
//! Imported as `qwen3_vl` (the package name is `qwen3-vl` on
//! crates.io; Cargo derives the lib name by replacing hyphens
//! with underscores).
//!
//! Standalone crate with no `findit-proto` dependency: the
//! image-analysis preset produces a typed
//! [`image_analysis::ImageAnalysis`] (re-exported from the
//! `llmtask` sibling crate, which hosts the canonical shared
//! type), whose shape mirrors `findit-proto::database::SceneVlmResult`
//! so a downstream consumer can map field-by-field without
//! re-running inference.
//!
//! The crate has two layers:
//!
//! - [`Engine`] + [`Task`] — a generic Qwen3-VL constrained-JSON inference
//!   engine. Wraps `mistralrs 0.8`; backend selection (metal / cuda /
//!   …) is up to the consumer (see the README). Async-only; callers
//!   wrap returned `Future`s with `tokio::time::timeout(..)` or
//!   `tokio::select!` for shutdown observation.
//! - [`image_analysis`] — the only preset that ships today. Owns the
//!   prompt, the constrained-JSON schema, the resilient parser, and
//!   the [`image_analysis::ImageAnalysis`] output type. Detection-array
//!   fields (`subjects`, `objects`, etc.) are flat `Vec<SmolStr>`; see
//!   the `image_analysis` module doc and `CHANGELOG.md` for
//!   the rationale on dropping the previous
//!   `Detection { label, confidence }` wrapper.
//!
//! Cancellation contract: dropping the future returned by [`Engine::run`]
//! is a fast wakeup, **not** GPU cancellation. mistralrs's engine loop
//! runs the in-flight scheduler step to completion in the background; the
//! response is silently discarded on send. Use `tokio::time::timeout(..)`
//! for a deadline.

#![deny(missing_docs)]

pub mod engine;
pub mod error;
pub mod image_analysis;

pub use crate::{
  engine::{Engine, EngineOptions, RequestOptions},
  error::{Error, LoadError},
  image_analysis::ImageAnalysisTask,
};
pub use llmtask::{ImageAnalysis, JsonParseError, Task};

/// Re-exported from the [`image`] crate for caller convenience —
/// [`Engine::run`] consumes `Vec<DynamicImage>`.
pub use image::DynamicImage;
