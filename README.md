# OneTrans PyTorch Port

This repository is a PyTorch reading-and-reimplementation project based on the original TensorFlow demo code from:

- Source repo: `https://github.com/KiraYeetar/OneTrans_Tensorlfow`
- Source file: `https://github.com/KiraYeetar/OneTrans_Tensorlfow/blob/main/main.py`

The original project is a compact single-file implementation of a OneTrans-style architecture that mixes:

- non-sequential features, projected into a small number of pseudo tokens
- sequential features, projected into ordinary sequence tokens

This repository keeps that backbone idea, ports it to PyTorch, and adds a runnable training pipeline around it.

## What This Repository Contains

There are two layers in this codebase:

1. The backbone port
   This is the PyTorch translation of the original OneTrans building blocks.
2. The training wrapper
   This adds dataset loading, feature tensorization, metrics, mixed precision, checkpointing, and CLI training support.

In other words, this repo is no longer just a shape demo. It is now a small but runnable training project built around the original OneTrans backbone.

## Code Origin

The architectural core comes from the TensorFlow demo linked above. The following ideas are inherited directly from that implementation:

- non-sequential features are converted into `ns_len` pseudo tokens
- sequential features are converted into `seq_len` sequence tokens
- token order is `[ns_tokens, seq_tokens]`
- the first `ns_len` tokens use token-specific parameter groups
- the remaining sequence tokens share one extra parameter group
- pyramid-style compression progressively reduces token length until only the `ns_tokens` remain

The PyTorch port in this repository preserves those high-level ideas, but the surrounding engineering has been extended substantially.

## Repository Structure

### `main_pytorch.py`

This file is the backbone reference implementation. It contains:

- `FFNLayer`
- `CausalMaskAttention`
- `OneTransBlock`
- `MultiOneTransBlock`

This is the closest file to the original single-file TensorFlow structure.

### `models/`

Task-level model wrappers live here.

Current file:

- `models/taac_onetrans.py`

This wraps the backbone into a trainable classifier:

- projects non-sequential features into `ns_tokens`
- projects sequence features into `seq_tokens`
- applies the base OneTrans block and pyramid stack
- pools the final `ns_token` outputs
- applies a classification head

### `utils/`

Reusable non-model logic lives here.

- `utils/common.py`
  General helpers such as seed setup and split generation.
- `utils/metrics.py`
  Accuracy and AUC computation.
- `utils/taac_data.py`
  Dataset loading, schema handling, feature conversion, and tensor construction.

The intent is to keep model code separate from data and training utilities.

### `scripts/`

Runnable entrypoints live here.

Current file:

- `scripts/run_taac2026_sample.py`

This script is the main training entrypoint. It handles:

- dataset download or local parquet reading
- feature tensorization
- model construction
- AMP setup
- training and validation loops
- checkpoint save and resume

## Backbone Architecture

At a high level, the model flow is:

1. Map non-sequential features into `ns_len` pseudo tokens.
2. Map sequential features into `seq_len` sequence tokens.
3. Concatenate them into one token sequence:
   `[ns_tokens, seq_tokens]`
4. Run a base `MultiOneTransBlock`.
5. Run a pyramid stack that shortens the query length step by step.
6. Stop when only `ns_len` tokens remain.
7. Pool the remaining `ns_tokens` for downstream prediction.

### Input Shapes

- non-sequential input:
  `[batch_size, non_seq_dim]`
- sequential input:
  `[batch_size, seq_len, seq_feature_dim]`

### Tokenization Stage

- `non_seq_tokenizer` maps
  `[B, non_seq_dim] -> [B, ns_len, d_model]`
- `seq_tokenizer` maps
  `[B, seq_len, seq_feature_dim] -> [B, seq_len, d_model]`

### Attention Parameter Sharing

OneTrans is not using a single shared Q/K/V projection for every token.

Instead:

- the first `ns_len` tokens each use their own projection group
- all remaining sequence tokens share one extra projection group

This same pattern is also used by the FFN inside `OneTransBlock`:

- the first `ns_len` tokens each use their own FFN
- the remaining sequence tokens share one extra FFN

This token-type-aware parameterization is one of the defining features of the implementation.

### Pyramid Compression

After the base block, later blocks do not always use the full token list as query.

Instead, the model shortens the query length step by step:

- full length
- full length minus 1
- full length minus 2
- ...
- until only `ns_len` tokens remain

The effect is that sequence information is gradually absorbed into the prefix `ns_tokens`, which are then used as the final summary representation.

## Attention Mask Modes

The backbone currently supports multiple attention mask modes through `mask_type`:

- `origin`
  Original soft bias mask behavior from the inherited implementation.
- `hard_mask`
  Original mask region, but applied as a hard additive mask.
- `bimask_soft`
  `ns_token` queries are fully open; `seq_token` queries use a soft causal-style bias.
- `bimask_hard`
  `ns_token` queries are fully open; `seq_token` queries use a hard strict causal mask.

These modes are exposed in the training script via `--mask_type`.

## Training Pipeline

The current training target is the Hugging Face dataset:

- `TAAC2026/data_sample_1000`

The training script supports:

- remote dataset loading
- local parquet loading
- automatic schema handling
- mixed precision training on CUDA
- checkpoint save with timestamp-based filenames
- checkpoint resume with optimizer and scaler state restoration

### Mixed Precision

AMP is enabled by default on CUDA.

Supported modes:

- default CUDA AMP
- `--amp-dtype fp16`
- `--amp-dtype bf16`
- `--no-amp`

### Checkpointing

Checkpoints are saved with timestamped filenames such as:

- `best_model_20260413_172711.pt`

Resume behavior:

- `--resume` accepts either a filename under `output-dir`
- or a full checkpoint path

The script restores:

- model weights
- optimizer state
- scaler state
- last completed epoch
- best validation AUC

## Recommended Entry Points

### Backbone sanity check

```bash
python main_pytorch.py
```

This runs the small demo flow and prints the shape transitions through the backbone.

### Training run

```bash
python scripts/run_taac2026_sample.py --epochs 5 --batch-size 32 --save-checkpoint
```

### Training with a specific mask mode

```bash
python scripts/run_taac2026_sample.py --epochs 5 --batch-size 32 --mask_type bimask_hard
```

### Resume training

```bash
python scripts/run_taac2026_sample.py --epochs 10 --batch-size 32 --resume best_model_20260413_172711.pt --save-checkpoint
```

## What This README Tries To Clarify

This repository is not a fresh architecture design from scratch. It is:

- a PyTorch port of an existing TensorFlow OneTrans demo
- plus a structured local training scaffold
- plus several implementation-level extensions such as:
  - multiple mask modes
  - mixed precision
  - dataset utilities
  - checkpoint save and resume

If you are trying to understand where a piece of code came from:

- start with `main_pytorch.py` for the original architectural core
- then read `models/taac_onetrans.py` for the downstream task wrapper
- then read `scripts/run_taac2026_sample.py` for the training workflow
