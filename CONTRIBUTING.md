# Contributing to qwen

Quick rules for the kinds of changes this crate sees most often.

## Default check before pushing

```sh
cargo fmt --check
cargo clippy --lib --tests --examples --locked -- -D warnings
cargo test --lib --locked
cargo check --all-targets --locked
```

The CI workflow at `.github/workflows/ci.yml` runs the same commands
on macOS for every PR.

## Real-model integration tests

`tests/integration_scene.rs` and `examples/smoke.rs` need the
`qwen3-vl-2b` model on disk and won't run in CI. Run them locally
before merging anything that touches:

- `src/engine.rs` (sampler, request shape, model loading)
- `src/scene.rs` (prompt, schema, parser)
- `Cargo.toml`'s mistralrs version

```sh
QWEN_MODEL_PATH=/path/to/qwen3-vl-2b \
    cargo test --release --features integration --test integration_scene -- --test-threads=1
```

The integration test is intentionally strict: it asserts that two
deterministic runs against the same fixtures produce a bit-identical
`SceneAnalysis`. A passing run takes ~120 s on Apple Silicon (one cold
model load plus two ~5 s inferences).

## Bumping mistralrs

`Cargo.toml` declares `mistralrs = "0.8"` (any 0.8.x); the committed
`Cargo.lock` pins the exact version this repo's CI tests against. The
crate's deterministic-mode rationale is verified against
`mistralrs-core/src/sampler.rs::apply_freq_pres_rep_penalty` and
`mistralrs-core/src/pipeline/sampling.rs` in **mistralrs 0.8**.

Future 0.8.x patches could change the sampler context behavior or the
multimodal message API. If you run `cargo update -p mistralrs` and
pick up a newer 0.8.x patch, **re-verify** these before merging:

1. The sampler still applies `presence_penalty` over `seq.get_toks()`
   (prompt + generated). If a future patch changes this — or, ideally,
   adds a generated-only penalty — the deterministic-mode trade-off
   in `Engine::run` should be revisited (see the long comment block
   there).
2. `MultimodalMessages::add_image_message(role, text, Vec<DynamicImage>)`
   signature is unchanged. If `add_image_message` gains a new
   parameter or changes its message ordering, the `Engine::run`
   call site must be updated.
3. `Constraint::JsonSchema(Value)` still wires through to the
   constrained decoder. The integration test catches this one — if
   the model produces unparseable output after a mistralrs bump, a
   schema-constraint regression is the most likely cause.

Then re-run the real-model integration test (above) and update
`CHANGELOG.md` to record the new audited version.

## Public API: accessor pattern

All public types use the **scenesdetect-style accessor surface**:
private fields, getter / `with_*` / `set_*` per field, `const fn`
where the type allows, `impl Into<...>` for non-`Copy` setter
parameters, `#[cfg_attr(not(tarpaulin), inline(always))]` on every
accessor. No public fields. See `EngineOptions`, `SceneTask`,
`SceneAnalysis`, `Detection` for the pattern.

When adding a new field, add the full triple
(`fn name() -> ...`, `fn with_name(...)`, `fn set_name(...)`)
and a doc comment on each (the crate sets `#![deny(missing_docs)]`).

## Errors

- `LoadError` is for `Engine::load` (one-shot; failure aborts the
  worker in service callers).
- `Error` is for `Engine::run` and `Engine::warmup` (per-call;
  callers may swallow into a default response).
- `ParseError` is for `Task::parse`. It's lifted into `Error::Parse`
  via `#[from]` so `?` propagation works.

If you add a new failure mode, prefer a new variant on the right
enum over a new `String`-stringified path. Kept-distinct variants
(`Error::Empty` vs `Error::Parse`) make operational logs cleaner.
