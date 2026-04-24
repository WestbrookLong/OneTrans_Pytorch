# `run_taac2026_sample.py` 命令行用法

## 基本训练

```bash
python scripts/run_taac2026_sample.py --epochs 5 --batch-size 32
```

## 训练并保存 checkpoint

保存的文件会按时间戳命名，例如 `best_model_20260413_172711.pt`。

```bash
python scripts/run_taac2026_sample.py --epochs 5 --batch-size 32 --save-checkpoint
```

## 选择注意力 mask 类型

默认 attention 使用 `paper_causal`。  
可选值为：

- `paper_causal`
- `origin`
- `hard_mask`
- `bimask_soft`
- `bimask_hard`

示例：

```bash
python scripts/run_taac2026_sample.py --epochs 5 --batch-size 32 --mask_type paper_causal
python scripts/run_taac2026_sample.py --epochs 5 --batch-size 32 --mask_type origin
python scripts/run_taac2026_sample.py --epochs 5 --batch-size 32 --mask_type hard_mask
python scripts/run_taac2026_sample.py --epochs 5 --batch-size 32 --mask_type bimask_soft
python scripts/run_taac2026_sample.py --epochs 5 --batch-size 32 --mask_type bimask_hard
```

## 控制金字塔压缩层数

默认使用 `6` 层线性 schedule，并在大 token 长度配置下按 `32` 对齐。

```bash
python scripts/run_taac2026_sample.py --epochs 5 --batch-size 32 --num-pyramid-layers 6 --pyramid-align 32
python scripts/run_taac2026_sample.py --epochs 5 --batch-size 32 --num-pyramid-layers 4 --pyramid-align 1
```

## 开启或关闭 `[SEP]` token

默认插入一个可学习的 `[SEP]` token，位置在 sequence tokens 和 ns tokens 之间。

```bash
python scripts/run_taac2026_sample.py --epochs 5 --batch-size 32
python scripts/run_taac2026_sample.py --epochs 5 --batch-size 32 --no-sep-token
```

## 开启 activation checkpoint

默认关闭 activation checkpoint。打开后，训练时会在 OneTrans block 内重算激活以换取更低显存占用。

```bash
python scripts/run_taac2026_sample.py --epochs 5 --batch-size 32 --activation-checkpoint
python scripts/run_taac2026_sample.py --epochs 5 --batch-size 32 --activation-checkpoint --no-amp
```

## 开启/关闭混合精度

CUDA 下默认开启 AMP。

```bash
python scripts/run_taac2026_sample.py --epochs 5 --batch-size 32
python scripts/run_taac2026_sample.py --epochs 5 --batch-size 32 --amp-dtype bf16
python scripts/run_taac2026_sample.py --epochs 5 --batch-size 32 --no-amp
```

## 从已有 checkpoint 继续训练

`--resume` 可以填写：

- `output-dir` 下的 checkpoint 文件名
- 或 checkpoint 的绝对路径

继续训练时会自动恢复：

- model 参数
- optimizer 状态
- GradScaler 状态
- 已训练 epoch
- best validation AUC

示例：

```bash
python scripts/run_taac2026_sample.py --epochs 10 --batch-size 32 --resume best_model_20260413_172711.pt
python scripts/run_taac2026_sample.py --epochs 10 --batch-size 32 --resume best_model_20260413_172711.pt --save-checkpoint
python scripts/run_taac2026_sample.py --epochs 10 --batch-size 32 --resume D:\Users\WESTBROOK\PycharmProjects\RecAlgo\OneTrans_Pytorch\outputs\taac2026_sample\best_model_20260413_172711.pt
```

如果同时开启 `--save-checkpoint`，训练结束后会再保存一个新的时间戳 checkpoint，不会覆盖旧文件。

## 读取本地 parquet

```bash
python scripts/run_taac2026_sample.py --local-parquet D:\path\to\demo_1000.parquet --epochs 5 --batch-size 32
```

## 常用调参项

```bash
python scripts/run_taac2026_sample.py ^
  --epochs 5 ^
  --batch-size 32 ^
  --seq-len 16 ^
  --ns-len 4 ^
  --d-model 128 ^
  --num-heads 4 ^
  --ffn-hidden 256 ^
  --multi-num 4 ^
  --num-pyramid-layers 6 ^
  --pyramid-align 32 ^
  --lr 1e-3 ^
  --weight-decay 1e-4
```

## 查看完整参数

```bash
python scripts/run_taac2026_sample.py --help
```
