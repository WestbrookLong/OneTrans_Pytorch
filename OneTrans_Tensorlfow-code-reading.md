# OneTrans_Tensorlfow Code Reading

Source: `https://github.com/KiraYeetar/OneTrans_Tensorlfow/blob/main/main.py`

## What the original code does

The repository is a single-file TensorFlow demo of a custom transformer-style block that mixes two feature types:

- sequential features shaped like `[batch_size, seq_len, feat_dim]`
- non-sequential features reshaped into pseudo-tokens shaped like `[batch_size, ns_len, d_model]`

The model concatenates non-sequential tokens before sequential tokens, runs a custom attention block, and then repeatedly shortens the sequence with a pyramid-style stack until only the non-sequential token segment remains.

## Core modules

- `FFNLayer`: a two-layer feed-forward network with `swish` activations.
- `CausalMaskAttention`: multi-head attention with per-position projection layers for the first `ns_len` tokens and a shared projection for the rest.
- `OneTransBlock`: `LayerNorm -> attention -> residual -> LayerNorm -> FFN -> residual`.
- `MultiOneTransBlock`: runs several independent `OneTransBlock` modules and averages their outputs.

## Shape flow in the demo

1. Sequential features are projected from `feat_dim` to `d_model`.
2. Non-sequential features are projected and reshaped into `ns_len` tokens.
3. Both token groups are concatenated as `[non_seq_tokens, seq_tokens]`.
4. A base `MultiOneTransBlock` processes the full token list.
5. Three more blocks progressively reduce the query length from `seq_len + ns_len` down to `ns_len`.
6. The final token set is mean-pooled for downstream tasks.

## Notes carried into the PyTorch port

- The PyTorch version keeps the same module structure and demo pipeline.
- The original attention mask is preserved as written, even though it is not a strict `-inf` causal mask.
- The feed-forward sublayer keeps the same output size assumption as the TensorFlow example, where the second FFN layer returns `d_model`.

## Local port

The PyTorch implementation is in `main_pytorch.py` in the current workspace.
