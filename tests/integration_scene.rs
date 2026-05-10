//! Integration test for ImageAnalysisTask, gated behind `--features integration`.
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

use qwen3_vl::{Engine, EngineOptions, image_analysis::ImageAnalysisTask};

/// Loads JPEG fixtures from `tests/fixtures/` (sorted by file name for
/// stable input ordering across runs), or falls back to a single 1×1
/// black image if the directory is empty. Stable ordering matters for
/// the deterministic-retry test below; without it, two `read_dir`
/// iterations could enumerate fixtures in different orders, which would
/// change the prompt token sequence and break the equality comparison
/// even with greedy decoding.
fn load_fixtures() -> Vec<image::DynamicImage> {
  let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
  let mut paths: Vec<PathBuf> = Vec::new();
  if fixture_dir.exists() {
    for entry in std::fs::read_dir(&fixture_dir).expect("fixtures readable") {
      let entry = entry.expect("dir entry");
      if entry
        .path()
        .extension()
        .map(|e| e == "jpg" || e == "jpeg")
        .unwrap_or(false)
      {
        paths.push(entry.path());
      }
    }
  }
  paths.sort();
  let mut images: Vec<image::DynamicImage> = paths
    .into_iter()
    .map(|p| image::open(p).expect("decode fixture"))
    .collect();
  if images.is_empty() {
    images.push(image::DynamicImage::ImageRgb8(image::RgbImage::new(1, 1)));
  }
  images
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scene_task_against_real_model() {
  let Ok(model_path) = std::env::var("QWEN_MODEL_PATH") else {
    eprintln!("skipping: QWEN_MODEL_PATH not set");
    return;
  };

  let images = load_fixtures();

  let engine = Engine::load(EngineOptions::new(model_path))
    .await
    .expect("model loads");

  let task = ImageAnalysisTask::new();
  let result = engine.run(&task, images).await.expect("inference");

  // With real fixtures, the model reliably produces both a description
  // and tags. The contract: structured output round-trips AND populates
  // both fields. The 1×1 black-image fallback path will fail this
  // assertion — that's intentional, fixtures must be present.
  assert!(
    !result.description().is_empty() && !result.tags().is_empty(),
    "expected description and at least one tag (got description={:?}, tags.len={})",
    result.description(),
    result.tags().len(),
  );
}

/// Codex adversarial-review F1: with stochastic sampling, the same
/// keyframes reprocessed (timeout retry, backfill, reprocess) can
/// produce different `ImageAnalysis` values, causing search-index
/// drift. The fix is `EngineOptions::new`'s indexing-safe default
/// sampler (`RequestOptions::deterministic`), which switches to greedy
/// decoding. This test runs the engine twice with default options
/// against the same fixtures and asserts the full `ImageAnalysis` is
/// identical between runs.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deterministic_run_is_idempotent() {
  let Ok(model_path) = std::env::var("QWEN_MODEL_PATH") else {
    eprintln!("skipping: QWEN_MODEL_PATH not set");
    return;
  };

  // EngineOptions::new embeds RequestOptions::deterministic() as the
  // engine-level default sampler, so the obvious construction path is
  // automatically idempotent. Stochastic sampling requires an explicit
  // .with_request(RequestOptions::new()) opt-in.
  let engine = Engine::load(EngineOptions::new(model_path))
    .await
    .expect("model loads");

  let task = ImageAnalysisTask::new();

  // Two independent runs against identical inputs.
  let result_a = engine.run(&task, load_fixtures()).await.expect("run a");
  let result_b = engine.run(&task, load_fixtures()).await.expect("run b");

  // ImageAnalysis derives PartialEq, so the whole-struct equality is
  // the contract: every field must match bit-for-bit.
  assert_eq!(
    result_a, result_b,
    "deterministic mode produced different ImageAnalysis values across runs"
  );
}
