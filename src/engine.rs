//! The [`Engine`] and [`EngineOptions`] types.

use std::{
  path::{Path, PathBuf},
  sync::Arc,
  time::Duration,
};

use mistralrs::{
  Constraint, IsqType, MultimodalMessages, MultimodalModelBuilder, RequestBuilder, TextMessageRole,
};
use tracing::{debug, info, instrument};

use crate::error::{Error, LoadError};
use llmtask::Task;

/// Default per-call inference timeout (issue #1 H-001). Five
/// minutes covers cold-cache Metal JIT specialization on real
/// keyframes and pathological prompts; a stuck model (kernel
/// deadlock, GPU OOM) trips this rather than blocking the caller
/// forever. Override per-engine via
/// [`EngineOptions::with_inference_timeout`].
pub const DEFAULT_INFERENCE_TIMEOUT: Duration = Duration::from_secs(300);

/// Configuration for [`Engine::load`].
#[derive(Debug, Clone)]
pub struct EngineOptions {
  model_path: PathBuf,
  quantization: IsqType,
  max_tokens: usize,
  request: RequestOptions,
  inference_timeout: Duration,
}

impl EngineOptions {
  /// Construct with the given model path, default quantization
  /// (`IsqType::Q4K`), default `max_tokens` (`1024`), and an
  /// indexing-safe default sampler profile
  /// ([`RequestOptions::deterministic`]).
  ///
  /// The default request is deterministic because this crate's primary
  /// use case is producing structured output that gets persisted to a
  /// search index (see [`crate::image_analysis::ImageAnalysisTask`]).
  /// Stochastic sampling means the same keyframes reprocessed after
  /// a timeout, retry, or backfill can produce different
  /// `ImageAnalysis` values,
  /// silently drifting the index. Greedy decoding closes that hole at
  /// the cost of diverging from the Qwen3-VL Instruct model card's
  /// recommended sampler — see [`RequestOptions::deterministic`] for
  /// the full trade-off.
  ///
  /// For one-shot, quality-prioritised use where reproducibility
  /// doesn't matter, swap the engine default for the model-card
  /// stochastic profile via
  /// `EngineOptions::new(path).with_request(RequestOptions::new())`,
  /// or override per-call via [`Engine::run_with`].
  pub fn new(model_path: impl Into<PathBuf>) -> Self {
    Self {
      model_path: model_path.into(),
      quantization: IsqType::Q4K,
      // Issue #1 M-003: bumped from 512 → 1024. The 512 default
      // truncated complex scenes mid-JSON (many subjects/objects/
      // actions), surfacing as
      // ParseError::Json(EOF while parsing a string). 1024 covers
      // the long tail observed empirically without inflating
      // worst-case latency materially under greedy decoding.
      max_tokens: 1024,
      request: RequestOptions::deterministic(),
      inference_timeout: DEFAULT_INFERENCE_TIMEOUT,
    }
  }

  /// Returns the configured model path.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn model_path(&self) -> &Path {
    &self.model_path
  }

  /// Builder-style setter for `model_path`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_model_path(mut self, val: impl Into<PathBuf>) -> Self {
    self.model_path = val.into();
    self
  }

  /// In-place setter for `model_path`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_model_path(&mut self, val: impl Into<PathBuf>) -> &mut Self {
    self.model_path = val.into();
    self
  }

  /// Returns the configured quantization (default `IsqType::Q4K`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn quantization(&self) -> IsqType {
    self.quantization
  }

  /// Builder-style setter for `quantization`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_quantization(mut self, val: IsqType) -> Self {
    self.quantization = val;
    self
  }

  /// In-place setter for `quantization`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_quantization(&mut self, val: IsqType) -> &mut Self {
    self.quantization = val;
    self
  }

  /// Returns the configured `max_tokens` ceiling (default `1024`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn max_tokens(&self) -> usize {
    self.max_tokens
  }

  /// Builder-style setter for `max_tokens`. Any value is accepted at
  /// the type level (no setter-side validation); a value of `0` is
  /// clamped up to `1` at request time inside [`Engine::run_with`]
  /// before being passed to mistralrs's `set_sampler_max_len`, so a
  /// zero here means "let the model emit at least one token", not
  /// "skip generation entirely".
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_max_tokens(mut self, val: usize) -> Self {
    self.max_tokens = val;
    self
  }

  /// In-place setter for `max_tokens`. See [`Self::with_max_tokens`]
  /// for the runtime `0 → 1` clamp note.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_max_tokens(&mut self, val: usize) -> &mut Self {
    self.max_tokens = val;
    self
  }

  /// Returns the engine-level default [`RequestOptions`]. This is the
  /// sampler profile used by [`Engine::run`]; per-call overrides go
  /// through [`Engine::run_with`].
  ///
  /// Default ([`EngineOptions::new`]): [`RequestOptions::deterministic`]
  /// — see that constructor for the indexing-vs-quality trade-off.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn request(&self) -> &RequestOptions {
    &self.request
  }

  /// Builder-style setter for `request`. Replaces the engine-level
  /// default sampler profile wholesale.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn with_request(mut self, val: RequestOptions) -> Self {
    self.request = val;
    self
  }

  /// In-place setter for `request`. Replaces the engine-level default
  /// sampler profile wholesale.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub fn set_request(&mut self, val: RequestOptions) -> &mut Self {
    self.request = val;
    self
  }

  /// Returns the per-call inference timeout (default
  /// [`DEFAULT_INFERENCE_TIMEOUT`] = 5 min). Issue #1 H-001.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn inference_timeout(&self) -> Duration {
    self.inference_timeout
  }

  /// Builder-style setter for `inference_timeout`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_inference_timeout(mut self, val: Duration) -> Self {
    self.inference_timeout = val;
    self
  }

  /// In-place setter for `inference_timeout`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_inference_timeout(&mut self, val: Duration) -> &mut Self {
    self.inference_timeout = val;
    self
  }
}

/// Sampler configuration applied per call by [`Engine::run`] /
/// [`Engine::run_with`].
///
/// Two named presets ship out of the box:
///
/// - [`RequestOptions::new`] / [`RequestOptions::default`] — the
///   Qwen3-VL Instruct (non-thinking) model card sampler
///   (`temperature 0.7`, `top_p 0.8`, `top_k 20`,
///   `presence_penalty 1.5`). Best output quality; the same input
///   can produce different outputs across runs.
/// - [`RequestOptions::deterministic`] — greedy decoding
///   (`temperature 0.0`, `top_p 1.0`, `top_k 1`) with
///   `presence_penalty 1.5` retained (greedy without it falls into
///   token loops). Bit-stable output for identical inputs; the
///   right choice for indexing pipelines that must avoid silent
///   drift on retries / backfills. This is the profile
///   [`EngineOptions::new`] embeds by default.
///
/// `Engine::run_with` applies all four fields to the underlying
/// mistralrs `RequestBuilder` uniformly — there is no separate
/// deterministic branch; the preset itself encodes the choice.
///
/// `repetition_penalty` and a sampler seed are intentionally absent —
/// mistralrs 0.8 has no `set_sampler_repetition_penalty` and no
/// `set_sampler_seed`. Do NOT substitute
/// `set_sampler_frequency_penalty(1.0)` for the missing
/// `repetition_penalty`: the math is different (additive vs
/// multiplicative).
#[derive(Debug, Clone)]
pub struct RequestOptions {
  temperature: f64,
  top_p: f64,
  top_k: usize,
  presence_penalty: f32,
}

impl RequestOptions {
  /// Construct with the Qwen3-VL Instruct (non-thinking) model card
  /// defaults: `temperature 0.7`, `top_p 0.8`, `top_k 20`,
  /// `presence_penalty 1.5`. Best output quality; not bit-stable
  /// across runs. Pair with [`EngineOptions::with_request`] (or
  /// [`Engine::run_with`]) when reproducibility doesn't matter and
  /// quality does.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn new() -> Self {
    Self {
      temperature: 0.7,
      top_p: 0.8,
      top_k: 20,
      presence_penalty: 1.5,
    }
  }

  /// Indexing-safe greedy decoding: `temperature 0.0`, `top_p 1.0`,
  /// `top_k 1`, `presence_penalty 1.5`. Output is bit-stable for
  /// identical inputs — the right choice for pipelines that persist
  /// VLM output to a search index, where retries / timeouts /
  /// backfills must not silently drift the index. This is the preset
  /// [`EngineOptions::new`] embeds by default.
  ///
  /// Greedy decoding is the only deterministic mode mistralrs 0.8
  /// supports (no `set_sampler_seed`).
  ///
  /// **The retained `presence_penalty 1.5` is a documented
  /// trade-off:**
  ///
  /// - In mistralrs 0.8 (re-verify when upgrading; future patches
  ///   could change this and this preset's assumption would no
  ///   longer hold), `presence_penalty` is applied over the full
  ///   `seq.get_toks()` (prompt tokens plus generated tokens,
  ///   verified in
  ///   `mistralrs-core/src/sampler.rs::apply_freq_pres_rep_penalty`
  ///   and `mistralrs-core/src/pipeline/sampling.rs`). With
  ///   `temperature 0` there is no sampling spread, so every token
  ///   appearing in the task prompt gets a flat `-1.5` logit shift
  ///   even before the model emits anything. To minimize the
  ///   collateral on legitimate value tokens,
  ///   [`crate::image_analysis::ImageAnalysisTask`]'s
  ///   `IMAGE_ANALYSIS_PROMPT` is intentionally written without enumerated
  ///   value examples (H1) — format guidance is descriptive
  ///   (word counts, lowercase) rather than enumerative. Residual
  ///   bias only hits scaffolding/instruction tokens (e.g. "scene",
  ///   "describing", "lowercase"), which the model is unlikely to
  ///   want to emit as values, and the JSON schema constraint
  ///   preserves field names and structure regardless.
  /// - Removing the penalty was tested and broke generation: greedy
  ///   without any repetition control falls into token loops that
  ///   exhaust `max_tokens` mid-string and surface as
  ///   `Error::Parse(Json(EOF while parsing a string))`. mistralrs
  ///   0.8 has no generated-only repetition mechanism
  ///   (`frequency_penalty` and `repetition_penalty` also operate
  ///   over `seq.get_toks()`), so biased-but-parseable beats
  ///   unbiased-but-broken.
  ///
  /// Callers that genuinely want greedy with no repetition control
  /// can chain `.with_presence_penalty(0.0)` and accept the
  /// repetition-loop hazard themselves.
  ///
  /// Custom `Task` implementations should follow the same prompt
  /// hygiene: avoid enumerating expected value tokens in the prompt,
  /// because they will be penalized in this preset.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn deterministic() -> Self {
    Self {
      temperature: 0.0,
      top_p: 1.0,
      top_k: 1,
      presence_penalty: 1.5,
    }
  }

  /// Returns the configured sampling temperature.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn temperature(&self) -> f64 {
    self.temperature
  }

  /// Builder-style setter for `temperature`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_temperature(mut self, val: f64) -> Self {
    self.temperature = val;
    self
  }

  /// In-place setter for `temperature`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_temperature(&mut self, val: f64) -> &mut Self {
    self.temperature = val;
    self
  }

  /// Returns the configured `top_p`. Note: mistralrs 0.8's builder
  /// method is `set_sampler_topp` (no underscore between `top` and
  /// `p`).
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn top_p(&self) -> f64 {
    self.top_p
  }

  /// Builder-style setter for `top_p`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_top_p(mut self, val: f64) -> Self {
    self.top_p = val;
    self
  }

  /// In-place setter for `top_p`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_top_p(&mut self, val: f64) -> &mut Self {
    self.top_p = val;
    self
  }

  /// Returns the configured `top_k`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn top_k(&self) -> usize {
    self.top_k
  }

  /// Builder-style setter for `top_k`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_top_k(mut self, val: usize) -> Self {
    self.top_k = val;
    self
  }

  /// In-place setter for `top_k`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_top_k(&mut self, val: usize) -> &mut Self {
    self.top_k = val;
    self
  }

  /// Returns the configured `presence_penalty`. With the
  /// [`RequestOptions::deterministic`] preset this is the only
  /// repetition control mistralrs 0.8 supports — greedy without it
  /// falls into token loops; see that constructor for the trade-off.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn presence_penalty(&self) -> f32 {
    self.presence_penalty
  }

  /// Builder-style setter for `presence_penalty`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn with_presence_penalty(mut self, val: f32) -> Self {
    self.presence_penalty = val;
    self
  }

  /// In-place setter for `presence_penalty`.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn set_presence_penalty(&mut self, val: f32) -> &mut Self {
    self.presence_penalty = val;
    self
  }

  /// Validate sampler parameters before they reach mistralrs (issue
  /// #1 H-002).
  ///
  /// - `temperature` must be finite and ≥ 0 (negative values invert
  ///   the softmax sign and produce nonsensical distributions).
  /// - `top_p` must be finite and in `(0, 1]` (0 selects nothing;
  ///   > 1 is meaningless; NaN poisons the sampler).
  /// - `top_k` must be ≥ 1 (0 selects nothing).
  /// - `presence_penalty` must be finite (NaN/Inf would poison the
  ///   logit shift).
  ///
  /// Called automatically by [`Engine::run_with`]; callers can
  /// invoke it themselves to fail fast on a bad preset.
  pub const fn validate(&self) -> Result<(), Error> {
    if !self.temperature.is_finite() || self.temperature < 0.0 {
      return Err(Error::InvalidRequest(
        "temperature must be finite and >= 0.0",
      ));
    }
    if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
      return Err(Error::InvalidRequest(
        "top_p must be finite and in (0.0, 1.0]",
      ));
    }
    if self.top_k == 0 {
      return Err(Error::InvalidRequest("top_k must be >= 1"));
    }
    if !self.presence_penalty.is_finite() {
      return Err(Error::InvalidRequest("presence_penalty must be finite"));
    }
    Ok(())
  }
}

impl Default for RequestOptions {
  fn default() -> Self {
    Self::new()
  }
}

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
  ///
  /// **Recommended model:** `Qwen/Qwen3-VL-2B-Instruct` (BF16
  /// safetensors) — the engine then applies in-situ quantization
  /// at load time per `EngineOptions::quantization` (default
  /// `IsqType::Q4K`), which gives Q4_K-class RAM compression
  /// without re-encoding the weights on disk.
  ///
  /// **Auto-detect of pre-quantized weights.** If
  /// `<model_path>/config.json` declares a `quantization_config`
  /// block (true for `Qwen/Qwen3-VL-2B-Instruct-FP8`, AWQ / GPTQ
  /// checkpoints, and any other pre-quantized variants),
  /// in-situ quantization is skipped automatically — the
  /// configured [`EngineOptions::quantization`] would otherwise
  /// re-quantize already-quantized weights.
  ///
  /// > **Apple Silicon caveat (mistralrs 0.8.x):** the
  /// > `Qwen3-VL-2B-Instruct-FP8` checkpoint loads cleanly but
  /// > inference fails on the first prompt step inside mistralrs's
  /// > Metal `BlockwiseFP8` dequant kernel (`Command buffer …
  /// > Ignored (kIOGPUCommandBufferCallbackErrorSubmissionsIgnored)`).
  /// > This is upstream and out of scope here. Use the BF16
  /// > checkpoint on Apple Silicon; CUDA hosts should be unaffected.
  #[instrument(name = "qwen3_vl::load", skip(opts), fields(model_path = %opts.model_path().display(), quantization = ?opts.quantization()))]
  pub async fn load(opts: EngineOptions) -> Result<Self, LoadError> {
    if !opts.model_path().exists() {
      return Err(LoadError::NotFound(opts.model_path().to_path_buf()));
    }
    let started = std::time::Instant::now();
    let pre_quantized = is_pre_quantized(opts.model_path());
    if pre_quantized {
      info!("loading Qwen3-VL model (pre-quantized weights detected — skipping ISQ)");
    } else {
      info!("loading Qwen3-VL model");
    }
    let model_id = opts.model_path().to_string_lossy().into_owned();
    let mut builder = MultimodalModelBuilder::new(model_id);
    if !pre_quantized {
      builder = builder.with_isq(opts.quantization());
    }
    let model = builder
      .build()
      .await
      .map_err(|e| LoadError::Build(e.to_string()))?;
    info!(
      elapsed_ms = started.elapsed().as_millis() as u64,
      "model loaded"
    );
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

  /// Returns the engine-level default sampler profile. See
  /// [`EngineOptions::request`].
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn request(&self) -> &RequestOptions {
    self.options.request()
  }

  /// Returns the per-call inference timeout. Issue #1 H-001.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub const fn inference_timeout(&self) -> Duration {
    self.options.inference_timeout()
  }

  /// Optional pre-warm: runs one tiny inference against a 1×1 black image
  /// to JIT-compile Metal kernels before serving real requests. Logs
  /// duration at `debug`. Errors are propagated to the caller — typically
  /// you ignore them in production (warmup is best-effort).
  ///
  /// **Caveat:** Metal's kernel JIT specializes per tensor shape, so
  /// the kernels compiled for a 1×1 image are not guaranteed to match
  /// the kernels needed for production-sized keyframes (e.g.,
  /// 720×1280). For shape-matched warmup, use
  /// [`Self::warmup_with_image`] (issue #1 M-002).
  #[instrument(name = "qwen3_vl::warmup", skip(self))]
  pub async fn warmup(&self) -> Result<(), Error> {
    use image::{DynamicImage, RgbImage};
    let blank = DynamicImage::ImageRgb8(RgbImage::new(1, 1));
    self.warmup_with_image(blank).await
  }

  /// Pre-warm with a caller-supplied image (issue #1 M-002). Use a
  /// representative production-sized keyframe (e.g., 720×1280 black
  /// frame, or a real fixture) so Metal's per-shape kernel JIT
  /// specializes for the shapes the production path will hit. The
  /// 1×1 [`Self::warmup`] only exercises the load → encode → decode
  /// pipeline structurally; first real-keyframe inference can still
  /// incur JIT cost without this.
  #[instrument(name = "qwen3_vl::warmup_with_image", skip(self, image))]
  pub async fn warmup_with_image(&self, image: image::DynamicImage) -> Result<(), Error> {
    let started = std::time::Instant::now();
    let messages = MultimodalMessages::new().add_image_message(
      TextMessageRole::User,
      "Reply with: ok",
      vec![image],
    );
    let request = RequestBuilder::from(messages)
      .set_sampler_max_len(4)
      .enable_thinking(false);
    // Same timeout as run_with: a stuck warmup shouldn't block
    // worker startup forever.
    let timeout = self.options.inference_timeout();
    let _ = tokio::time::timeout(timeout, self.model.send_chat_request(request))
      .await
      .map_err(|_| Error::InferenceTimeout(timeout))?
      .map_err(|e| Error::Inference(e.to_string()))?;
    debug!(
      elapsed_ms = started.elapsed().as_millis() as u64,
      "warmup complete"
    );
    Ok(())
  }

  /// Single-turn, multi-image structured run with the engine-level
  /// default sampler ([`EngineOptions::request`]). Equivalent to
  /// [`Self::run_with`] called with that profile.
  ///
  /// Consumes `images` because mistralrs's
  /// `MultimodalMessages::add_image_message` takes `Vec<DynamicImage>`
  /// by value — borrowing here would force a silent `.to_vec()` clone
  /// of decoded image data. Returns `Error::NoImages` for an empty
  /// input.
  ///
  /// Dropping the returned future is a fast wakeup, not GPU
  /// cancellation: mistralrs's engine loop completes the in-flight
  /// scheduler step in the background; the response is silently
  /// discarded on send. Wrap in `tokio::time::timeout(..)` for a
  /// deadline.
  #[cfg_attr(not(tarpaulin), inline(always))]
  pub async fn run<T: Task>(
    &self,
    task: &T,
    images: Vec<image::DynamicImage>,
  ) -> Result<T::Output, Error>
  where
    T::ParseError: Send + Sync + 'static,
  {
    self.run_with(task, images, self.options.request()).await
  }

  /// Same as [`Self::run`] but with a caller-supplied
  /// [`RequestOptions`] that replaces the engine-level default for
  /// this call. Use this when a specific call needs a sampler profile
  /// other than [`EngineOptions::request`].
  ///
  /// All four fields from `opts` are applied uniformly to the
  /// underlying mistralrs sampler — there is no separate deterministic
  /// branch; the preset itself encodes the choice between greedy
  /// ([`RequestOptions::deterministic`]) and stochastic
  /// ([`RequestOptions::new`] / `default`).
  #[instrument(
    name = "qwen3_vl::run_with",
    skip(self, task, images, opts),
    fields(
      task_kind = std::any::type_name::<T>(),
      image_count = images.len(),
      max_tokens = self.options.max_tokens(),
      temperature = opts.temperature(),
    ),
  )]
  pub async fn run_with<T: Task>(
    &self,
    task: &T,
    images: Vec<image::DynamicImage>,
    opts: &RequestOptions,
  ) -> Result<T::Output, Error>
  where
    // bound at the call site only.
    // `Send + Sync + 'static` lets us box the parse error into
    // `Error::Parse(Box<dyn Error + Send + Sync + 'static>)`,
    // which works for any Task — including ones whose only
    // purpose is to receive `UnsupportedGrammar` for routing.
    T::ParseError: Send + Sync + 'static,
  {
    if images.is_empty() {
      return Err(Error::NoImages);
    }
    // Issue #1 H-002: validate sampler parameters before mistralrs
    // sees them. Negative temperature, top_p > 1.0, top_k = 0, or
    // non-finite presence_penalty all produce undefined behavior in
    // mistralrs's sampler.
    opts.validate()?;

    // Pull the task's grammar and route to mistralrs's
    // Constraint::JsonSchema. mistralrs 0.8 only accepts JSON
    // Schema; non-JSON variants (Lark, Regex) are rejected via
    // UnsupportedGrammar so callers can route to an
    // llguidance-backed engine instead (e.g., the `lfm` crate).
    let grammar = task.grammar();
    let schema = grammar
      .as_json_schema()
      .ok_or_else(|| {
        Error::UnsupportedGrammar(llmtask::UnsupportedGrammar::new(
          grammar.kind(),
          "json_schema",
        ))
      })?
      .clone();

    let messages =
      MultimodalMessages::new().add_image_message(TextMessageRole::User, task.prompt(), images);

    let request = RequestBuilder::from(messages)
      .set_sampler_max_len(self.options.max_tokens().max(1))
      .enable_thinking(false)
      .set_constraint(Constraint::JsonSchema(schema))
      .set_sampler_temperature(opts.temperature())
      .set_sampler_topp(opts.top_p())
      .set_sampler_topk(opts.top_k())
      .set_sampler_presence_penalty(opts.presence_penalty());

    let started = std::time::Instant::now();
    // Issue #1 H-001: bound inference duration. A stuck model
    // (Metal JIT stall, GPU OOM, scheduler deadlock) would
    // otherwise block the caller indefinitely. Drop on timeout —
    // mistralrs will silently complete the in-flight scheduler
    // step in the background and discard the response.
    let timeout = self.options.inference_timeout();
    let response = tokio::time::timeout(timeout, self.model.send_chat_request(request))
      .await
      .map_err(|_| Error::InferenceTimeout(timeout))?
      .map_err(|e| Error::Inference(e.to_string()))?;
    debug!(
      elapsed_ms = started.elapsed().as_millis() as u64,
      "inference complete"
    );

    let choice = response.choices.first().ok_or(Error::Empty)?;
    // finding: reject length-truncated generations
    // before parsing. mistralrs `Display for StopReason` maps Eos
    // → "stop" and `Length`/`ModelLength` → "length"; "stop" is
    // the only outcome where the constrained decoder produced a
    // full natural completion. Anything else (length, error, etc.)
    // means the JSON could be syntactically valid but semantically
    // incomplete — persisting it to a search index would silently
    // truncate metadata.
    if choice.finish_reason != "stop" {
      let raw_len = choice
        .message
        .content
        .as_ref()
        .map(|s| s.len())
        .unwrap_or(0);
      return Err(Error::Truncated {
        finish_reason: choice.finish_reason.clone(),
        raw_len,
      });
    }
    let text = choice
      .message
      .content
      .clone()
      .filter(|s| !s.trim().is_empty())
      .ok_or(Error::Empty)?;

    #[cfg(feature = "trace-output")]
    tracing::trace!(raw = %text, "model output");

    task.parse(&text).map_err(|e| Error::Parse(Box::new(e)))
  }
}

/// Returns true when `<model_path>/config.json` declares a
/// top-level `quantization_config` block — the marker HuggingFace
/// uses for pre-quantized weights (Qwen FP8, AWQ, GPTQ, BitsandBytes,
/// etc.). Applying ISQ to weights that are already quantized would
/// either no-op or re-quantize (lossy), so `Engine::load` skips
/// `with_isq` in that case.
///
/// Failure modes are intentionally lenient: missing file, IO error,
/// invalid JSON, missing field — all return `false`, which falls
/// back to the unquantized-weights ISQ path. The model load itself
/// will surface a clear error if the directory is genuinely
/// unloadable; this helper only chooses between two valid load
/// paths, so a false negative just means "let mistralrs apply the
/// default ISQ on weights it'll then accept anyway".
fn is_pre_quantized(model_path: &Path) -> bool {
  let cfg = match std::fs::read_to_string(model_path.join("config.json")) {
    Ok(s) => s,
    Err(_) => return false,
  };
  match serde_json::from_str::<serde_json::Value>(&cfg) {
    Ok(v) => v.get("quantization_config").is_some(),
    Err(_) => false,
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Build a throwaway model-dir-style tmpdir under
  /// `target/tmp/<unique>/` (relative to cargo, cleaned up by
  /// `cargo clean`). Returns the path so tests can drop a
  /// `config.json` into it. Unique per process+counter so parallel
  /// tests don't collide.
  fn model_tmpdir(name: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
      "qwen3vl-test-{}-{}-{}",
      std::process::id(),
      N.fetch_add(1, Ordering::Relaxed),
      name
    ));
    std::fs::create_dir_all(&dir).expect("mk tmpdir");
    dir
  }

  #[test]
  fn is_pre_quantized_returns_false_for_missing_config() {
    let dir = model_tmpdir("missing-config");
    assert!(!is_pre_quantized(&dir));
    let _ = std::fs::remove_dir_all(&dir);
  }

  #[test]
  fn is_pre_quantized_returns_false_for_invalid_json() {
    let dir = model_tmpdir("invalid-json");
    std::fs::write(dir.join("config.json"), b"not-json{").unwrap();
    assert!(!is_pre_quantized(&dir));
    let _ = std::fs::remove_dir_all(&dir);
  }

  #[test]
  fn is_pre_quantized_returns_false_for_unquantized_safetensors_config() {
    // Mirrors the structure of `Qwen/Qwen3-VL-2B-Instruct/config.json`
    // — no `quantization_config` block. ISQ should still apply.
    let dir = model_tmpdir("unquantized");
    std::fs::write(
      dir.join("config.json"),
      br#"{"model_type":"qwen3_vl","architectures":["Qwen3VLForConditionalGeneration"]}"#,
    )
    .unwrap();
    assert!(!is_pre_quantized(&dir));
    let _ = std::fs::remove_dir_all(&dir);
  }

  #[test]
  fn is_pre_quantized_returns_true_for_fp8_config() {
    // Mirrors the structure of `Qwen/Qwen3-VL-2B-Instruct-FP8/config.json`
    // — top-level `quantization_config` block declaring fp8. ISQ
    // should be skipped.
    let dir = model_tmpdir("fp8");
    std::fs::write(
      dir.join("config.json"),
      br#"{"model_type":"qwen3_vl","quantization_config":{"quant_method":"fp8","weight_block_size":[128,128]}}"#,
    )
    .unwrap();
    assert!(is_pre_quantized(&dir));
    let _ = std::fs::remove_dir_all(&dir);
  }

  #[test]
  fn engine_options_defaults_to_deterministic_request() {
    // EngineOptions::new embeds RequestOptions::deterministic() as the
    // engine-level default sampler. This test guards against silent
    // reversion: if someone later flips the default back to the
    // stochastic model-card profile, every caller that uses the obvious
    // ::new() constructor would silently start drifting their search
    // index on retries/backfills. To get the model-card stochastic
    // sampler, callers must opt in explicitly via
    // .with_request(RequestOptions::new()).
    let opts = EngineOptions::new("/tmp/model");
    assert_eq!(opts.model_path(), Path::new("/tmp/model"));
    assert!(matches!(opts.quantization(), IsqType::Q4K));
    // Issue #1 M-003: default raised from 512 to 1024.
    assert_eq!(opts.max_tokens(), 1024);
    let req = opts.request();
    assert_eq!(req.temperature(), 0.0);
    assert_eq!(req.top_p(), 1.0);
    assert_eq!(req.top_k(), 1);
    assert_eq!(
      req.presence_penalty(),
      1.5,
      "deterministic preset must keep presence_penalty 1.5 — greedy \
       without it falls into token loops"
    );
  }

  #[test]
  fn engine_options_with_chains() {
    let opts = EngineOptions::new("/tmp/a")
      .with_model_path("/tmp/b")
      .with_quantization(IsqType::Q8_0)
      .with_max_tokens(1024)
      .with_request(RequestOptions::new());
    assert_eq!(opts.model_path(), Path::new("/tmp/b"));
    assert!(matches!(opts.quantization(), IsqType::Q8_0));
    assert_eq!(opts.max_tokens(), 1024);
    // Swapping in RequestOptions::new() flips the engine to the
    // model-card stochastic profile (temperature 0.7).
    assert_eq!(opts.request().temperature(), 0.7);
  }

  #[test]
  fn engine_options_set_chains() {
    let mut opts = EngineOptions::new("/tmp/a");
    opts
      .set_model_path("/tmp/b")
      .set_quantization(IsqType::Q8_0)
      .set_max_tokens(1024)
      .set_request(RequestOptions::new());
    assert_eq!(opts.model_path(), Path::new("/tmp/b"));
    assert!(matches!(opts.quantization(), IsqType::Q8_0));
    assert_eq!(opts.max_tokens(), 1024);
    assert_eq!(opts.request().temperature(), 0.7);
  }

  #[test]
  fn request_options_defaults_match_model_card() {
    // Hard-coded against the Qwen3-VL Instruct model card values to
    // catch silent drift if the defaults are ever edited without a
    // CHANGELOG note. See models/qwen3-vl-2b/README.md (or the
    // upstream HuggingFace model card if no local checkout exists).
    let opts = RequestOptions::new();
    assert_eq!(opts.temperature(), 0.7);
    assert_eq!(opts.top_p(), 0.8);
    assert_eq!(opts.top_k(), 20);
    assert_eq!(opts.presence_penalty(), 1.5);
  }

  #[test]
  fn request_options_default_eq_new() {
    let new_opts = RequestOptions::new();
    let default_opts = RequestOptions::default();
    assert_eq!(new_opts.temperature(), default_opts.temperature());
    assert_eq!(new_opts.top_p(), default_opts.top_p());
    assert_eq!(new_opts.top_k(), default_opts.top_k());
    assert_eq!(new_opts.presence_penalty(), default_opts.presence_penalty());
  }

  #[test]
  fn request_options_with_chains() {
    let opts = RequestOptions::new()
      .with_temperature(0.3)
      .with_top_p(0.95)
      .with_top_k(50)
      .with_presence_penalty(0.0);
    assert_eq!(opts.temperature(), 0.3);
    assert_eq!(opts.top_p(), 0.95);
    assert_eq!(opts.top_k(), 50);
    assert_eq!(opts.presence_penalty(), 0.0);
  }

  #[test]
  fn request_options_set_chains() {
    let mut opts = RequestOptions::new();
    opts
      .set_temperature(0.3)
      .set_top_p(0.95)
      .set_top_k(50)
      .set_presence_penalty(0.0);
    assert_eq!(opts.temperature(), 0.3);
    assert_eq!(opts.top_p(), 0.95);
    assert_eq!(opts.top_k(), 50);
    assert_eq!(opts.presence_penalty(), 0.0);
  }

  #[test]
  fn request_options_deterministic_preset() {
    // Hard-coded greedy values: temperature=0 + top_k=1 forces argmax,
    // top_p=1 disables nucleus filtering. presence_penalty 1.5 is kept
    // (greedy without it falls into token loops). See
    // RequestOptions::deterministic doc for the trade-off.
    let opts = RequestOptions::deterministic();
    assert_eq!(opts.temperature(), 0.0);
    assert_eq!(opts.top_p(), 1.0);
    assert_eq!(opts.top_k(), 1);
    assert_eq!(opts.presence_penalty(), 1.5);
  }

  // ===== Issue #1 H-002: RequestOptions::validate =====

  #[test]
  fn request_options_validate_accepts_presets() {
    // Both shipped presets must validate.
    assert!(RequestOptions::new().validate().is_ok());
    assert!(RequestOptions::deterministic().validate().is_ok());
  }

  #[test]
  fn request_options_validate_rejects_negative_temperature() {
    let opts = RequestOptions::new().with_temperature(-0.1);
    assert!(matches!(opts.validate(), Err(Error::InvalidRequest(_))));
  }

  #[test]
  fn request_options_validate_rejects_non_finite_temperature() {
    assert!(matches!(
      RequestOptions::new().with_temperature(f64::NAN).validate(),
      Err(Error::InvalidRequest(_))
    ));
    assert!(matches!(
      RequestOptions::new()
        .with_temperature(f64::INFINITY)
        .validate(),
      Err(Error::InvalidRequest(_))
    ));
  }

  #[test]
  fn request_options_validate_rejects_top_p_out_of_range() {
    assert!(matches!(
      RequestOptions::new().with_top_p(0.0).validate(),
      Err(Error::InvalidRequest(_))
    ));
    assert!(matches!(
      RequestOptions::new().with_top_p(1.5).validate(),
      Err(Error::InvalidRequest(_))
    ));
    assert!(matches!(
      RequestOptions::new().with_top_p(-0.1).validate(),
      Err(Error::InvalidRequest(_))
    ));
    assert!(matches!(
      RequestOptions::new().with_top_p(f64::NAN).validate(),
      Err(Error::InvalidRequest(_))
    ));
  }

  #[test]
  fn request_options_validate_accepts_top_p_one() {
    // top_p = 1.0 disables nucleus filtering — used by the
    // deterministic preset. Must pass.
    assert!(RequestOptions::new().with_top_p(1.0).validate().is_ok());
  }

  #[test]
  fn request_options_validate_rejects_top_k_zero() {
    let opts = RequestOptions::new().with_top_k(0);
    assert!(matches!(opts.validate(), Err(Error::InvalidRequest(_))));
  }

  #[test]
  fn request_options_validate_rejects_non_finite_presence_penalty() {
    assert!(matches!(
      RequestOptions::new()
        .with_presence_penalty(f32::NAN)
        .validate(),
      Err(Error::InvalidRequest(_))
    ));
    assert!(matches!(
      RequestOptions::new()
        .with_presence_penalty(f32::INFINITY)
        .validate(),
      Err(Error::InvalidRequest(_))
    ));
  }

  #[test]
  fn request_options_validate_accepts_negative_presence_penalty() {
    // mistralrs allows negative presence_penalty (encourages
    // repetition). Validate only checks finiteness.
    assert!(
      RequestOptions::new()
        .with_presence_penalty(-1.0)
        .validate()
        .is_ok()
    );
  }

  // ===== Issue #1 H-001 + M-003 =====

  #[test]
  fn engine_options_default_inference_timeout() {
    let opts = EngineOptions::new("/nonexistent");
    assert_eq!(opts.inference_timeout(), DEFAULT_INFERENCE_TIMEOUT);
    assert_eq!(opts.inference_timeout(), Duration::from_secs(300));
  }

  #[test]
  fn engine_options_with_inference_timeout() {
    let opts = EngineOptions::new("/nonexistent").with_inference_timeout(Duration::from_secs(10));
    assert_eq!(opts.inference_timeout(), Duration::from_secs(10));
  }

  #[test]
  fn engine_options_default_max_tokens_bumped_to_1024() {
    // Issue #1 M-003: default raised from 512 to 1024 to avoid
    // mid-JSON truncation on complex scenes.
    let opts = EngineOptions::new("/nonexistent");
    assert_eq!(opts.max_tokens(), 1024);
  }
}
