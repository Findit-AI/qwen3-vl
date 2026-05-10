//! Phase-zero smoke test for the qwen3-vl crate.
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

use qwen3_vl::{Engine, EngineOptions, image_analysis::ImageAnalysisTask};

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

  let task = ImageAnalysisTask::new();
  let result = match engine.run(&task, images).await {
    Ok(r) => r,
    Err(e) => {
      eprintln!("inference failed: {e}");
      return ExitCode::from(1);
    }
  };

  println!("scene:        {:?}", result.scene());
  println!("description:  {:?}", result.description());
  println!("subjects:     {} items", result.subjects().len());
  println!("objects:      {} items", result.objects().len());
  println!("actions:      {} items", result.actions().len());
  println!("tags:         {:?}", result.tags());
  ExitCode::SUCCESS
}
