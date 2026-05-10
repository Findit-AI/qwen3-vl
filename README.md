<div align="center">
<h1>qwen3-vl</h1>
</div>
<div align="center">

Qwen3-VL-2B structured-output inference engine — async, mistralrs-backed, JSON-Schema-constrained. Implements the engine-agnostic [`llmtask::Task`] contract so the same prompt + schema + parser runs on any [`llmtask`]-compatible backend (`lfm`, `qwen3-vl`, …) without translation.

[<img alt="github" src="https://img.shields.io/badge/github-findit--ai/qwen-8da0cb?style=for-the-badge&logo=Github" height="22">][Github-url]
<img alt="LoC" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fal8n%2F327b2a8aef9003246e45c6e47fe63937%2Fraw%2Fqwen3-vl" height="22">
[<img alt="Build" src="https://img.shields.io/github/actions/workflow/status/findit-ai/qwen/ci.yml?logo=Github-Actions&style=for-the-badge" height="22">][CI-url]
[<img alt="codecov" src="https://img.shields.io/codecov/c/gh/findit-ai/qwen?style=for-the-badge&token=REPLACE_WITH_CODECOV_TOKEN&logo=codecov" height="22">][codecov-url]

[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-qwen3--vl-66c2a5?style=for-the-badge&labelColor=555555&logo=data:image/svg+xml;base64,PHN2ZyByb2xlPSJpbWciIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDUxMiA1MTIiPjxwYXRoIGZpbGw9IiNmNWY1ZjUiIGQ9Ik00ODguNiAyNTAuMkwzOTIgMjE0VjEwNS41YzAtMTUtOS4zLTI4LjQtMjMuNC0zMy43bC0xMDAtMzcuNWMtOC4xLTMuMS0xNy4xLTMuMS0yNS4zIDBsLTEwMCAzNy41Yy0xNC4xIDUuMy0yMy40IDE4LjctMjMuNCAzMy43VjIxNGwtOTYuNiAzNi4yQzkuMyAyNTUuNSAwIDI2OC45IDAgMjgzLjlWMzk0YzAgMTMuNiA3LjcgMjYuMSAxOS45IDMyLjJsMTAwIDUwYzEwLjEgNS4xIDIyLjEgNS4xIDMyLjIgMGwxMDMuOS01MiAxMDMuOSA1MmMxMC4xIDUuMSAyMi4xIDUuMSAzMi4yIDBsMTAwLTUwYzEyLjItNi4xIDE5LjktMTguNiAxOS45LTMyLjJWMjgzLjljMC0xNS05LjMtMjguNC0yMy40LTMzLjd6TTM1OCAyMTQuOGwtODUgMzEuOXYtNjguMmw4NS0zN3Y3My4zek0xNTQgMTA0LjFsMTAyLTM4LjIgMTAyIDM4LjJ2LjZsLTEwMiA0MS40LTEwMi00MS40di0uNnptODQgMjkxLjFsLTg1IDQyLjV2LTc5LjFsODUtMzguOHY3NS40em0wLTExMmwtMTAyIDQxLjQtMTAyLTQxLjR2LS42bDEwMi0zOC4yIDEwMiAzOC4ydi42em0yNDAgMTEybC04NSA0Mi41di03OS4xbDg1LTM4Ljh2NzUuNHptMC0xMTJsLTEwMiA0MS40LTEwMi00MS40di0uNmwxMDItMzguMiAxMDIgMzguMnYuNnoiPjwvcGF0aD48L3N2Zz4K" height="20">][doc-url]
[<img alt="crates.io" src="https://img.shields.io/crates/v/qwen3-vl?style=for-the-badge&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iaXNvLTg4NTktMSI/Pg0KPCEtLSBHZW5lcmF0b3I6IEFkb2JlIElsbHVzdHJhdG9yIDE5LjAuMCwgU1ZHIEV4cG9ydCBQbHVnLUluIC4gU1ZHIFZlcnNpb246IDYuMDAgQnVpbGQgMCkgIC0tPg0KPHN2ZyB2ZXJzaW9uPSIxLjEiIGlkPSJMYXllcl8xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB4PSIwcHgiIHk9IjBweCINCgkgdmlld0JveD0iMCAwIDUxMiA1MTIiIHhtbDpzcGFjZT0icHJlc2VydmUiPg0KPGc+DQoJPGc+DQoJCTxwYXRoIGQ9Ik0yNTYsMEwzMS41MjgsMTEyLjIzNnYyODcuNTI4TDI1Niw1MTJsMjI0LjQ3Mi0xMTIuMjM2VjExMi4yMzZMMjU2LDB6IE0yMzQuMjc3LDQ1Mi41NjRMNzQuOTc0LDM3Mi45MTNWMTYwLjgxDQoJCQlsMTU5LjMwMyw3OS42NTFWNDUyLjU2NHogTTEwMS44MjYsMTI1LjY2MkwyNTYsNDguNTc2bDE1NC4xNzQsNzcuMDg3TDI1NiwyMDIuNzQ5TDEwMS44MjYsMTI1LjY2MnogTTQzNy4wMjYsMzcyLjkxMw0KCQkJbC0xNTkuMzAzLDc5LjY1MVYyNDAuNDYxbDE1OS4zMDMtNzkuNjUxVjM3Mi45MTN6IiBmaWxsPSIjRkZGIi8+DQoJPC9nPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8L3N2Zz4NCg==" height="22">][crates-url]
[<img alt="crates.io" src="https://img.shields.io/crates/d/qwen3-vl?color=critical&logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBzdGFuZGFsb25lPSJubyI/PjwhRE9DVFlQRSBzdmcgUFVCTElDICItLy9XM0MvL0RURCBTVkcgMS4xLy9FTiIgImh0dHA6Ly93d3cudzMub3JnL0dyYXBoaWNzL1NWRy8xLjEvRFREL3N2ZzExLmR0ZCI+PHN2ZyB0PSIxNjQ1MTE3MzMyOTU5IiBjbGFzcz0iaWNvbiIgdmlld0JveD0iMCAwIDEwMjQgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9IjM0MjEiIGRhdGEtc3BtLWFuY2hvci1pZD0iYTMxM3guNzc4MTA2OS4wLmkzIiB3aWR0aD0iNDgiIGhlaWdodD0iNDgiIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIj48ZGVmcz48c3R5bGUgdHlwZT0idGV4dC9jc3MiPjwvc3R5bGU+PC9kZWZzPjxwYXRoIGQ9Ik00NjkuMzEyIDU3MC4yNHYtMjU2aDg1LjM3NnYyNTZoMTI4TDUxMiA3NTYuMjg4IDM0MS4zMTIgNTcwLjI0aDEyOHpNMTAyNCA2NDAuMTI4QzEwMjQgNzgyLjkxMiA5MTkuODcyIDg5NiA3ODcuNjQ4IDg5NmgtNTEyQzEyMy45MDQgODk2IDAgNzYxLjYgMCA1OTcuNTA0IDAgNDUxLjk2OCA5NC42NTYgMzMxLjUyIDIyNi40MzIgMzAyLjk3NiAyODQuMTYgMTk1LjQ1NiAzOTEuODA4IDEyOCA1MTIgMTI4YzE1Mi4zMiAwIDI4Mi4xMTIgMTA4LjQxNiAzMjMuMzkyIDI2MS4xMkM5NDEuODg4IDQxMy40NCAxMDI0IDUxOS4wNCAxMDI0IDY0MC4xOTJ6IG0tMjU5LjItMjA1LjMxMmMtMjQuNDQ4LTEyOS4wMjQtMTI4Ljg5Ni0yMjIuNzItMjUyLjgtMjIyLjcyLTk3LjI4IDAtMTgzLjA0IDU3LjM0NC0yMjQuNjQgMTQ3LjQ1NmwtOS4yOCAyMC4yMjQtMjAuOTI4IDIuOTQ0Yy0xMDMuMzYgMTQuNC0xNzguMzY4IDEwNC4zMi0xNzguMzY4IDIxNC43MiAwIDExNy45NTIgODguODMyIDIxNC40IDE5Ni45MjggMjE0LjRoNTEyYzg4LjMyIDAgMTU3LjUwNC03NS4xMzYgMTU3LjUwNC0xNzEuNzEyIDAtODguMDY0LTY1LjkyLTE2NC45MjgtMTQ0Ljk2LTE3MS43NzZsLTI5LjUwNC0yLjU2LTUuODg4LTMwLjk3NnoiIGZpbGw9IiNmZmZmZmYiIHAtaWQ9IjM0MjIiIGRhdGEtc3BtLWFuY2hvci1pZD0iYTMxM3guNzc4MTA2OS4wLmkwIiBjbGFzcz0iIj48L3BhdGg+PC9zdmc+&style=for-the-badge" height="22">][crates-url]
<img alt="license" src="https://img.shields.io/badge/License-Apache%202.0/MIT-blue.svg?style=for-the-badge&fontColor=white&logoColor=f5c076&logo=data:image/svg+xml;base64,PCFET0NUWVBFIHN2ZyBQVUJMSUMgIi0vL1czQy8vRFREIFNWRyAxLjEvL0VOIiAiaHR0cDovL3d3dy53My5vcmcvR3JhcGhpY3MvU1ZHLzEuMS9EVEQvc3ZnMTEuZHRkIj4KDTwhLS0gVXBsb2FkZWQgdG86IFNWRyBSZXBvLCB3d3cuc3ZncmVwby5jb20sIFRyYW5zZm9ybWVkIGJ5OiBTVkcgUmVwbyBNaXhlciBUb29scyAtLT4KPHN2ZyBmaWxsPSIjZmZmZmZmIiBoZWlnaHQ9IjgwMHB4IiB3aWR0aD0iODAwcHgiIHZlcnNpb249IjEuMSIgaWQ9IkNhcGFfMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgdmlld0JveD0iMCAwIDI3Ni43MTUgMjc2LjcxNSIgeG1sOnNwYWNlPSJwcmVzZXJ2ZSIgc3Ryb2tlPSIjZmZmZmZmIj4KDTxnIGlkPSJTVkdSZXBvX2JnQ2FycmllciIgc3Ryb2tlLXdpZHRoPSIwIi8+Cg08ZyBpZD0iU1ZHUmVwb190cmFjZXJDYXJyaWVyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KDTxnIGlkPSJTVkdSZXBvX2ljb25DYXJyaWVyIj4gPGc+IDxwYXRoIGQ9Ik0xMzguMzU3LDBDNjIuMDY2LDAsMCw2Mi4wNjYsMCwxMzguMzU3czYyLjA2NiwxMzguMzU3LDEzOC4zNTcsMTM4LjM1N3MxMzguMzU3LTYyLjA2NiwxMzguMzU3LTEzOC4zNTcgUzIxNC42NDgsMCwxMzguMzU3LDB6IE0xMzguMzU3LDI1OC43MTVDNzEuOTkyLDI1OC43MTUsMTgsMjA0LjcyMywxOCwxMzguMzU3UzcxLjk5MiwxOCwxMzguMzU3LDE4IHMxMjAuMzU3LDUzLjk5MiwxMjAuMzU3LDEyMC4zNTdTMjA0LjcyMywyNTguNzE1LDEzOC4zNTcsMjU4LjcxNXoiLz4gPHBhdGggZD0iTTE5NC43OTgsMTYwLjkwM2MtNC4xODgtMi42NzctOS43NTMtMS40NTQtMTIuNDMyLDIuNzMyYy04LjY5NCAxMy41OTMtMjMuNTAzLDIxLjcwOC0zOS42MTQsMjEuNzA4IGMtMjUuOTA4LDAtNDYuOTg1LTIxLjA3OC00Ni45ODUtNDYuOTg2czIxLjA3Ny00Ni45ODYsNDYuOTg1LTQ2Ljk4NmMxNS42MzMsMCwzMC4yLDcuNzQ3LDM4Ljk2OCwyMC43MjMgYzIuNzgyLDQuMTE3LDguMzc1LDUuMjAxLDEyLjQ5NiwyLjQxOGM0LjExOC0yLjc4Miw1LjIwMS04LjM3NywyLjQxOC0xMi40OTYgYy0xMi4xMTgtMTcuOTM3LTMyLjI2Mi0yOC42NDUtNTMuODgyLTI4LjY0NSBjLTM1LjgzMywwLTY0Ljk4NSwyOS4xNTItNjQuOTg1LDY0Ljk4NnMyOS4xNTIsNjQuOTg2LDY0Ljk4NSw2NC45ODZjMjIuMjgxLDAsNDIuNzU5LTExLjIxOCw1NC43NzgtMzAuMDA5IEMyMDAuMjA4LDE2OS4xNDcsMTk4Ljk4NSwxNjMuNTgyLDE5NC43OTgsMTYwLjkwM3oiLz4gPC9nPiA8L2c+Cg08L3N2Zz4=" height="22">

</div>

## Overview

`qwen3-vl` runs the [Qwen3-VL-2B-Instruct][qwen3vl-card] vision-language model through [mistralrs][mistralrs] with JSON-Schema-constrained sampling. It implements the engine-agnostic [`llmtask::Task`] contract — so any `Task` written against `llmtask` runs through `qwen3-vl` unchanged, and the public API stays backend-pluggable.

- **[`Engine`]** — async, mistralrs-backed Qwen3-VL inference. `Engine::run<T: Task<Value = serde_json::Value>>` accepts any JSON-schema task; the result is decoded by the task's `parse` impl.
- **[`ImageAnalysisTask`]** — built-in image-analysis preset (single-image VLM scene description). Owns the prompt, the JSON schema, and the resilient parser ported from the legacy `findit-qwen` service. Produces the canonical [`llmtask::ImageAnalysis`] output type.
- **CPU by default, opt-in GPU** — `qwen3-vl` does not re-export mistralrs's hardware-backend features. Consumers depend on `mistralrs` directly with the desired backend (`metal` / `cuda` / `cudnn` / …); Cargo unifies feature sets and `qwen3-vl` picks up the selection.

[`Engine`]: https://docs.rs/qwen3-vl/latest/qwen3_vl/engine/struct.Engine.html
[`ImageAnalysisTask`]: https://docs.rs/qwen3-vl/latest/qwen3_vl/image_analysis/struct.ImageAnalysisTask.html
[`llmtask::Task`]: https://docs.rs/llmtask/latest/llmtask/task/trait.Task.html
[`llmtask::ImageAnalysis`]: https://docs.rs/llmtask/latest/llmtask/image_analysis/struct.ImageAnalysis.html
[`llmtask`]: https://docs.rs/llmtask

## Why an `llmtask`-driven engine?

A bespoke `qwen3_vl::Task` would force every prompt + schema + parser to be rewritten against the next inference engine. Implementing [`llmtask::Task`] instead means the same `Task` code targets `qwen3-vl` (mistralrs), [`lfm`] (llguidance), or any future `llmtask`-compatible backend without modification — only the hardware backend selection differs.

```text
                                ┌──────────────────────────┐
   YourTask: impl Task   ──▶    │   llmtask::Task contract │   ──▶  qwen3-vl / lfm / …
                                │     prompt + Grammar     │
                                │     parse → Output       │
                                └──────────────────────────┘
```

[`lfm`]: https://docs.rs/lfm

## Features

- **Async, single-engine inference** — `Engine::run(&task, images).await`. No built-in cancellation token; wrap with `tokio::time::timeout` or `tokio::select!`.
- **Bounded inference timeout** — every `Engine::run` is wrapped in `tokio::time::timeout(EngineOptions::inference_timeout)` (default 300 s). A stuck model (Metal JIT stall, GPU memory exhaustion) surfaces as `Error::InferenceTimeout` instead of blocking the caller indefinitely.
- **`finish_reason` discipline** — mistralrs's `Choice::finish_reason != "stop"` (e.g. `"length"`, `"model_length"`) is surfaced as `Error::Truncated` BEFORE the parser runs, so partial JSON can never silently land in a downstream search index.
- **Sampler-options validation** — `RequestOptions::validate` rejects out-of-range values (negative temperature, `top_p > 1.0`, `top_k = 0`) at the engine boundary instead of hitting undefined behavior inside mistralrs's sampler.
- **Resilient JSON parser (`ImageAnalysisTask`)** — `TagList` / `DetectionLabels` accept list-or-string forms; `#[serde(deny_unknown_fields)]` on the schema struct; required arrays set to `null` are rejected (not coerced to empty); an indexable-content gate surfaces decoder/model regressions as `JsonParseError::NoUsableFields` by default.
- **Indexing-safe greedy default** — `EngineOptions::new` embeds `RequestOptions::deterministic()` (greedy, `temperature = 0.0`) so retries / timeouts / backfills produce bit-stable `ImageAnalysis` across runs. Swap to the model-card stochastic sampler with `.with_request(RequestOptions::new())` or `Engine::run_with`.

## Example

```rust,no_run
use qwen3_vl::{Engine, EngineOptions, image_analysis::ImageAnalysisTask};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Engine-level default sampler is greedy (deterministic) so retries
    // and backfills produce bit-stable ImageAnalysis values. Swap to the
    // Qwen3-VL model card stochastic profile via
    // `.with_request(RequestOptions::new())`, or override per-call with
    // `Engine::run_with`.
    let engine = Engine::load(EngineOptions::new("/path/to/qwen3-vl-2b")).await?;
    let task = ImageAnalysisTask::new();

    let images = vec![
        image::open("scene_keyframe_1.jpg")?,
        image::open("scene_keyframe_2.jpg")?,
    ];

    let result = engine.run(&task, images).await?;
    println!("scene: {:?}, tags: {:?}", result.scene(), result.tags());
    Ok(())
}
```

`Engine::run` consumes `Vec<DynamicImage>` because mistralrs 0.8's `MultimodalMessages::add_image_message` takes the vec by value — borrowing would force a silent `.to_vec()` clone of decoded image data.

### Per-call sampler override

```rust,no_run
# use qwen3_vl::{Engine, RequestOptions, image_analysis::ImageAnalysisTask};
# async fn x(engine: Engine, task: ImageAnalysisTask, images: Vec<image::DynamicImage>)
#   -> Result<(), Box<dyn std::error::Error>> {
let opts = RequestOptions::new()
    .with_temperature(0.3)
    .with_top_k(50);
let result = engine.run_with(&task, images, &opts).await?;
# Ok(()) }
```

## Installation

```toml
[dependencies]
qwen3-vl = "0.1"
```

```rust,ignore
use qwen3_vl::{Engine, EngineOptions};
```

### Hardware backend selection

Default features are CPU-only — `qwen3-vl` builds out of the box on every host mistralrs supports. To enable a hardware backend (Metal, CUDA, etc.), depend on `mistralrs` directly and select its feature; Cargo unifies feature sets across all references to the same crate, so `qwen3-vl` automatically picks up your selection:

```toml
[dependencies]
qwen3-vl  = "0.1"
# Pick at most one primary GPU backend; accelerated BLAS / cuDNN /
# NCCL / flash-attn options layer on top.
mistralrs = { version = "0.8", features = ["metal"] }    # Apple Metal
# mistralrs = { version = "0.8", features = ["cuda"] }   # NVIDIA CUDA
# mistralrs = { version = "0.8", features = ["accelerate"] }  # Apple Accelerate BLAS (CPU)
```

The full backend matrix mistralrs supports: `metal`, `cuda`, `cudnn`, `flash-attn`, `accelerate`, `mkl`, `nccl`, `ring`. Each may require an external toolchain (Xcode Command Line Tools for `metal` / `accelerate`, the CUDA toolkit for `cuda`, etc.) — see the [mistralrs README][mistralrs] for prerequisites.

### Cargo features

| Feature         | Default | What it adds                                                                            |
| --------------- | ------- | --------------------------------------------------------------------------------------- |
| `integration`   | no      | Enables `tests/integration_scene.rs` (needs `QWEN_MODEL_PATH` and ~4 GB of weights)     |
| `trace-output`  | no      | Logs raw model output at `tracing::trace` level — heavyweight; debugging only           |

## MSRV

Rust 1.95.

## License

`qwen3-vl` is dual-licensed under the [MIT license](LICENSE-MIT) and the [Apache License, Version 2.0](LICENSE-APACHE).

The Qwen3-VL model weights this crate runs are governed by their own license — see the [model card][qwen3vl-card] for terms.

Copyright (c) 2026 FinDIT Studio authors.

[mistralrs]: https://github.com/EricLBuehler/mistral.rs
[qwen3vl-card]: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct

[Github-url]: https://github.com/findit-ai/qwen/
[CI-url]: https://github.com/findit-ai/qwen/actions/workflows/ci.yml
[doc-url]: https://docs.rs/qwen3-vl
[crates-url]: https://crates.io/crates/qwen3-vl
[codecov-url]: https://app.codecov.io/gh/findit-ai/qwen/
