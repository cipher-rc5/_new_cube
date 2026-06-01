# new-cube — Usage Guide

This codebase is `new-cube`, a custom fork (maintained by @cipher-rc5) of the `Roblox/cube`
text-to-shape model, migrated to `uv` + Python 3.14.5
and made correct on Apple Silicon (Metal / MPS), with an optional MLX backend for the GPT
decode loop. PyTorch on MPS is the default and the source of truth; MLX is opt-in.

This document covers installation, environment setup, generation, the MLX path, environment
variables, tests, and troubleshooting. It assumes no prior familiarity with the repo.

---

## 1. Requirements

- macOS 14.0 or newer (required for MLX; also the sensible floor for current MPS).
- Apple Silicon (M-series) for the MPS and MLX paths. The model also runs on CUDA (Linux/NVIDIA)
  and CPU.
- `uv` for environment and dependency management. Install it with:

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

  or via Homebrew (`brew install uv`).

Confirm your interpreter is native arm64 (not x86 under Rosetta):

```bash
python -c "import platform; print(platform.processor())"   # should print: arm
```

---

## 2. Installation (uv)

This project uses `uv` exclusively. Do not use bare `pip install -e .`. From the repo root:

```bash
uv python install 3.14.5     # install the pinned interpreter
uv python pin 3.14.5         # writes .python-version (already committed)
uv venv                      # create .venv
uv sync --extra meshlab      # core dependencies + optional mesh simplification
```

On Apple Silicon, to also install the optional MLX backend:

```bash
uv sync --extra meshlab --extra mlx
```

Verify PyTorch sees the Metal backend:

```bash
uv run python -c "import torch; print(torch.__version__); print('mps', torch.backends.mps.is_available())"
```

Expect a non-`+cpu` torch build and `mps True`.

### Available extras

| Extra      | Installs                | When to use                                              |
|------------|-------------------------|----------------------------------------------------------|
| `meshlab`  | `pymeshlab`             | Optional mesh simplification. Best-effort on cp314.      |
| `cuda`     | `warp-lang` (Linux only)| CUDA-accelerated marching cubes. Never installs on macOS.|
| `mlx`      | `mlx` (arm64 macOS only)| Optional MLX GPT-decode backend.                         |
| `lint`     | `ruff`                  | Linting.                                                 |

### Python version note

`requires-python` is `==3.14.*`. If a native dependency (for example `pymeshlab`, or
`fpsample` in `cubepart`) has no cp314 wheel on your platform, fall back to Python 3.13
and document why. The `cubepart` package in particular may need to stay on 3.13 until
`fpsample` ships a 3.14 wheel.

---

## 3. Download model weights

Weights are hosted on Hugging Face. The `hf` CLI ships with the `huggingface_hub`
dependency (huggingface-hub 1.x; the legacy `huggingface-cli` command has been removed).
Download into a local directory, for example `model_weights/`:

```bash
uv run hf download Roblox/cube3d-v0.5 --local-dir ./model_weights
```

(Refer to the project README / model card for the exact repo id and file names if they differ.)

---

## 4. Generating a shape

The entry point is `cube3d.generate`. Always run it through `uv run` so it uses the project venv:

```bash
uv run python -m cube3d.generate \
  --gpt-ckpt-path model_weights/<gpt_checkpoint>.safetensors \
  --shape-ckpt-path model_weights/<shape_checkpoint>.safetensors \
  --prompt "broadsword" \
  --output-dir outputs/
```

This produces a `.obj` mesh in `outputs/`.

### All CLI flags

| Flag                       | Purpose                                                                 |
|----------------------------|-------------------------------------------------------------------------|
| `--config-path`            | Path to the model config (defaults to the packaged config).             |
| `--gpt-ckpt-path`          | Path to the GPT decoder safetensors checkpoint.                         |
| `--shape-ckpt-path`        | Path to the shape (VQ-VAE) safetensors checkpoint.                      |
| `--output-dir`             | Where to write the generated `.obj` (and `.gif` if rendering).          |
| `--prompt`                 | Text prompt describing the shape.                                       |
| `--backend`                | `torch` (default) or `mlx`. See section 6.                              |
| `--fast-inference`         | CUDA-only fast path. On Mac it is accepted but falls back (section 5).  |
| `--top-p`                  | Nucleus sampling cutoff. Omit for deterministic (argmax) decoding.      |
| `--bounding-box-xyz`       | Target bounding box dimensions.                                         |
| `--resolution-base`        | Grid resolution exponent. Cost grows as roughly `2^resolution_base` per axis. |
| `--render-gif`             | Render a turntable GIF (requires Blender on `PATH`).                    |
| `--disable-postprocessing` | Skip mesh postprocessing.                                               |

### Deterministic output

Omit `--top-p` to use argmax decoding. Runs are then reproducible across invocations on the
same machine. This is also the mode used by the MLX-vs-torch parity test.

---

## 5. Apple Silicon (Metal / MPS) notes

The device is selected automatically at runtime: `cuda` if available, else `mps`, else `cpu`.
On a Mac you will be on `mps` with no flags required.

### `--fast-inference` is CUDA-only

`--fast-inference` uses CUDA graphs via the `EngineFast` engine, which has no Metal equivalent.
On a Mac the flag is accepted but generation transparently falls back to the standard `Engine`,
printing a one-line notice. It does not crash.

### MPS operator fallback

A small number of PyTorch ops are not yet implemented on MPS. If you hit a
`NotImplementedError` for an MPS operator, enable CPU fallback for the missing op:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python -m cube3d.generate ...
```

This is a diagnostic, not a default — the CPU fallback can be slow.

### Marching cubes on Mac

Geometry extraction uses `skimage` marching cubes on MPS/CPU (the CUDA `warp-lang` path is
skipped automatically when CUDA is unavailable, with no per-call warning spam). `skimage`
marching cubes is single-threaded CPU and is the dominant cost of shape decode at high
`--resolution-base`. If decode feels slow on a Mac, lower `--resolution-base` first — cost
grows fast because the grid is cubic.

---

## 6. Optional MLX backend (`--backend mlx`)

MLX is Apple's native Metal array framework. The MLX backend runs **only the GPT decode loop**
on MLX; CLIP conditioning and shape decoding stay on PyTorch/MPS. The handoff between the two
frameworks is integer token IDs, so no live tensors are shared.

Requirements:

- Apple Silicon Mac.
- `uv sync --extra mlx` (installs `mlx`).

Usage:

```bash
uv run python -m cube3d.generate --backend mlx \
  --gpt-ckpt-path model_weights/<gpt_checkpoint>.safetensors \
  --shape-ckpt-path model_weights/<shape_checkpoint>.safetensors \
  --prompt "broadsword" \
  --output-dir outputs/
```

If `mlx` is not installed or you are not on an MPS device, the program exits with a clear
message. The default backend is `torch`; pass `--backend mlx` to opt in.

Known parity caveats (documented in `cube3d/inference/mlx_engine.py` and exercised by the
parity test):

- The torch path's `top_p` filtering uses a fixed cutoff; the MLX path mirrors that quirk.
  Under `top_p` sampling the two paths diverge by RNG. Use argmax (omit `--top-p`) for exact
  parity.
- The MLX decode currently recomputes the prefix each step (equivalent to torch decode with
  the KV cache disabled). A KV-cache optimization is a documented follow-up.

---

## 7. Environment variables

| Variable                     | Default     | Meaning                                                                 |
|------------------------------|-------------|-------------------------------------------------------------------------|
| `CUBE_MPS_AUTOCAST_DTYPE`    | `bfloat16`  | Autocast / KV-cache dtype on MPS. Allowed: `bfloat16`, `float16`, `float32`. Unknown values fall back to `bfloat16` with a warning. Switch to `float16`/`float32` if a bf16 op is missing or unstable on your macOS / PyTorch version. |
| `PYTORCH_ENABLE_MPS_FALLBACK`| unset       | Set to `1` to run unimplemented MPS ops on CPU instead of crashing. Diagnostic only; can be slow. |
| `CUBE_TEST_WEIGHTS_DIR`      | unset       | Points the parity test at a weights directory so it runs on a real Mac (see section 8). |

Example combining the autocast override:

```bash
CUBE_MPS_AUTOCAST_DTYPE=float32 uv run python -m cube3d.generate ...
```

---

## 8. Running the tests

The suite lives in `tests/` and runs on any machine. Tests that require `torch`, `mlx`, or
model weights skip cleanly when those are unavailable rather than failing.

```bash
uv run python -m pytest tests/ -q
```

Without torch/mlx/weights installed you should see import-guarded tests skip and the rest pass.

| Test file                     | What it checks                                                           |
|-------------------------------|--------------------------------------------------------------------------|
| `test_imports.py`             | `grid` imports without `warp-lang`; `mlx_engine` imports without `mlx`; `WARP_AVAILABLE`/`MLX_AVAILABLE` are bools; `marching_cubes_with_warp` raises `RuntimeError` when warp is absent. |
| `test_autocast_dtype.py`      | `select_autocast_dtype` returns the right dtype for cuda/mps/cpu and honors `CUBE_MPS_AUTOCAST_DTYPE`. |
| `test_backend_selection.py`   | `generate.py` exposes `--backend` with choices `["torch","mlx"]`, default `torch`, and the correct routing/guards. |
| `test_parity.py`              | (Mac + weights) MLX vs torch `run_gpt` produce identical `output_ids` under argmax; mesh vertex/face counts and bbox match within tolerance. |

To run the parity gate on a real Apple Silicon machine with weights:

```bash
uv sync --extra mlx
CUBE_TEST_WEIGHTS_DIR=./model_weights uv run python -m pytest tests/test_parity.py -q
```

---

## 9. cubepart (multi-part decomposition)

`cubepart/` is a separate package for multi-part shape decomposition via diffusion. It is a
second-phase migration target with heavier native dependencies (`diffusers`, `torchvision`,
`fpsample`, and a Linux-only `warp-lang`). It does not need to resolve on 3.14 for the primary
`cube3d` workflow to work, and may stay on Python 3.13 until `fpsample` ships a cp314 wheel.
Its packaging mirrors the `cube3d` style (`warp-lang` and `fpsample` are behind extras).

---

## 10. Rendering (optional)

`--render-gif` shells out to a `blender` binary. Install Blender and ensure `blender` is on
your `PATH`. This is unrelated to the compute backend.

---

## 11. Troubleshooting

- **`uv lock` fails to resolve a dependency on 3.14** — a native package may lack a cp314 wheel.
  Drop the optional extra (`meshlab`) if it is the cause, or fall back to Python 3.13
  (`uv python pin 3.13`) and document why.
- **`NotImplementedError` for an MPS op** — set `PYTORCH_ENABLE_MPS_FALLBACK=1`, or switch
  `CUBE_MPS_AUTOCAST_DTYPE` to `float16`/`float32`.
- **`--fast-inference` "falls back" message on Mac** — expected; fast inference is CUDA-only.
- **`mlx backend requires Apple Silicon with mlx installed`** — run `uv sync --extra mlx` on an
  Apple Silicon Mac, and confirm you are on the `mps` device.
- **Slow shape decode on Mac** — lower `--resolution-base`; the CPU `skimage` marching cubes is
  the bottleneck.
- **Import error mentioning `warp`** — on macOS `warp-lang` is intentionally not installed; the
  code guards its import. If you see a hard import failure, confirm you installed via `uv sync`
  (not a stale environment).

---

## 12. What changed in this migration (summary)

- `uv`-managed, `requires-python = "==3.14.*"`, `.python-version` pins `3.14.5`.
- `warp-lang` moved to a Linux-only `cuda` extra; `mlx` added as an Apple-Silicon-only extra;
  `safetensors` declared explicitly; broken packaging config fixed; legacy `setup.py` removed.
- MPS correctness: warp attempted only on CUDA (no warning spam), warp import guarded,
  `--fast-inference` falls back off-CUDA, env-driven autocast/KV-cache dtype helper, cubepart
  hardcoded-`cuda` autocast bug fixed.
- Optional `MlxEngine` GPT-decode backend behind `--backend mlx`.
- Test suite and this guide.

A full unified diff of the migration is included as `migration.patch` at the repo root.
