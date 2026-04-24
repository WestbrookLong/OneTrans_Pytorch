# AGENTS.md

## Scope

This file describes the working conventions for this repository. Follow these rules when editing code, adding experiments, or extending the training pipeline.

## Repository layout

- `main_pytorch.py`
  Contains the core OneTrans implementation used as the reference backbone:
  - `FFNLayer`
  - `CausalMaskAttention`
  - `OneTransBlock`
  - `MultiOneTransBlock`
- `models/`
  Contains model wrappers built on top of the backbone. The current training model is `models/taac_onetrans.py`.
- `utils/`
  Contains reusable non-model logic:
  - common helpers
  - metric computation
  - TAAC dataset loading and tensorization
- `scripts/`
  Contains runnable entrypoints. The main training entrypoint is `scripts/run_taac2026_sample.py`.
- `outputs/`
  Runtime artifacts such as metadata JSON and checkpoints.
- `Test.md`
  User-facing command examples for the current training script.

## Architecture boundaries

- Keep backbone logic in `main_pytorch.py`.
- Keep downstream wrappers or task-specific heads in `models/`.
- Keep dataset parsing, feature conversion, metrics, and generic helpers in `utils/`.
- Keep `scripts/` thin. A script should mainly do:
  - argument parsing
  - loader construction
  - model/optimizer/scaler setup
  - training loop
  - checkpoint and metadata save

Do not move task-specific training logic back into `utils/` unless it is clearly reusable across scripts.

## Current model behavior

- The project uses a OneTrans-style backbone with:
  - non-sequential features mapped into `ns_len` pseudo tokens
  - sequential features mapped into `seq_len` sequence tokens
  - token order `[seq_tokens, ns_tokens]`
- In attention and FFN:
  - the last `ns_len` tokens use token-specific parameter groups
  - the leading sequence tokens share the extra group at index `ns_len`
- Attention mask modes:
  - `paper_causal`: paper-aligned causal mask for the `[seq_tokens, ns_tokens]` layout
  - `origin`: original soft bias mask
  - `hard_mask`: original hard additive mask
  - `bimask_soft`: ns-token rows unmasked, seq-token rows use soft causal bias
  - `bimask_hard`: ns-token rows unmasked, seq-token rows use hard causal mask

The default downstream wrapper also inserts a learnable `[SEP]` token between sequence tokens and ns tokens.
Backbone attention runs through PyTorch SDPA, and activation checkpointing can be enabled during training.

When changing attention semantics, keep `main_pytorch.py` and any model wrapper usage aligned.

## Training script contract

The current training script is `scripts/run_taac2026_sample.py`.

Expected supported behaviors:

- train from Hugging Face dataset or local parquet
- AMP support with CUDA autocast and optional GradScaler
- configurable attention mask via `--mask_type`
- configurable linear pyramid depth via `--num-pyramid-layers` and `--pyramid-align`
- optional `[SEP]` token control via `--sep-token` and `--no-sep-token`
- optional activation checkpointing via `--activation-checkpoint`
- checkpoint save with timestamped filenames
- checkpoint resume via `--resume`
- save metadata to `run_metadata.json`

If new options are added, update:

- script argument parser
- `Test.md` usage examples
- this file if the workflow contract changes materially

## Checkpoint conventions

- Checkpoints must be saved with timestamped names, not fixed filenames.
- `--resume` should accept either:
  - a checkpoint filename under `output-dir`
  - an absolute checkpoint path
- Resume should restore, when present:
  - model weights
  - optimizer state
  - scaler state
  - last completed epoch
  - best validation metric tracked by the script
- Saving a checkpoint after resume must create a new file and must not overwrite the original checkpoint.

## Metrics and evaluation

- The current training target is classification trained with `CrossEntropyLoss`.
- The main validation selection metric is AUC.
- Keep metric code in `utils/metrics.py`.
- If metric definitions change, keep training logs and checkpoint metadata consistent with the new metric.

## Data handling

- TAAC dataset parsing logic lives in `utils/taac_data.py`.
- Support both:
  - the flattened dataset schema
  - the fallback/raw parquet-like schema when applicable
- Preserve schema autodetection unless there is a strong reason to remove it.
- When adding new feature engineering, keep the transformation explicit and document its effect on tensor shape.

## Coding expectations

- Prefer small, local changes over wide rewrites.
- Preserve current CLI behavior unless the change explicitly updates the interface.
- Keep shape semantics easy to trace.
- When changing tensor routing or parameter-sharing logic, verify with at least:
  - `python -m py_compile ...`
  - a minimal forward or training run

## Validation checklist

For model or script changes, run the smallest validation that proves the path works.

Typical checks:

```bash
python -m py_compile main_pytorch.py models/taac_onetrans.py scripts/run_taac2026_sample.py
python main_pytorch.py
python scripts/run_taac2026_sample.py --max-rows 32 --epochs 1 --batch-size 8
```

If the change affects mask behavior, checkpointing, or resume, validate that specific path explicitly.

## Documentation upkeep

When behavior changes, keep documentation aligned:

- `README.md` for high-level repo description
- `Test.md` for command-line usage
- note files outside the repo only if explicitly requested

Do not silently let docs drift away from the implemented CLI or model behavior.
