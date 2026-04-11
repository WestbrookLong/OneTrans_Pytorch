from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.taac_onetrans import TAACOneTransClassifier
from utils.common import json_ready_args, set_seed, split_indices, take_rows
from utils.metrics import accuracy_from_logits, multiclass_auc_from_logits
from utils.taac_data import build_tensors, load_train_split


def parse_device_type(device: str) -> str:
    return torch.device(device).type


def resolve_amp_dtype(amp_dtype: str) -> torch.dtype:
    if amp_dtype == "fp16":
        return torch.float16
    if amp_dtype == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported amp dtype: {amp_dtype}")


def should_enable_amp(device_type: str, amp: bool, amp_dtype: str) -> bool:
    if not amp:
        return False
    if device_type != "cuda":
        return False
    if amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        return False
    return True


def build_scaler(device_type: str, use_amp: bool, amp_dtype: torch.dtype) -> torch.amp.GradScaler:
    scaler_enabled = use_amp and device_type == "cuda" and amp_dtype == torch.float16
    return torch.amp.GradScaler(device=device_type, enabled=scaler_enabled)


def autocast_context(device_type: str, amp_dtype: torch.dtype, use_amp: bool) -> contextlib.AbstractContextManager[Any]:
    return torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download TAAC2026/data_sample_1000, convert it into OneTrans inputs, and run a small classifier."
    )
    parser.add_argument("--dataset-id", default="TAAC2026/data_sample_1000")
    parser.add_argument("--local-parquet", type=Path, default=None, help="Skip Hugging Face download and read a local parquet.")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / ".cache" / "taac2026")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit the number of rows loaded for quick checks.")
    parser.add_argument("--seq-len", type=int, default=16, help="Maximum sequence length kept per sample.")
    parser.add_argument("--ns-len", type=int, default=4, help="Number of non-sequence pseudo tokens.")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-hidden", type=int, default=256)
    parser.add_argument("--multi-num", type=int, default=4, help="How many OneTrans blocks to average in each stage.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", dest="amp", action="store_true", help="Enable automatic mixed precision.")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable automatic mixed precision.")
    parser.set_defaults(amp=torch.cuda.is_available())
    parser.add_argument(
        "--amp-dtype",
        choices=("fp16", "bf16"),
        default="fp16",
        help="Autocast dtype on CUDA. Use bf16 if fp16 overflows or is unstable.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "taac2026_sample")
    parser.add_argument("--save-checkpoint", action="store_true")
    return parser.parse_args()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    amp_dtype: torch.dtype,
    use_amp: bool,
    scaler: torch.amp.GradScaler,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float]:
    is_train = optimizer is not None
    device_type = parse_device_type(device)
    model.train(is_train)
    total_loss = 0.0
    total_acc = 0.0
    total_items = 0
    epoch_logits: list[torch.Tensor] = []
    epoch_labels: list[torch.Tensor] = []

    for non_seq_x, seq_x, labels in loader:
        non_seq_x = non_seq_x.to(device)
        seq_x = seq_x.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with autocast_context(device_type=device_type, amp_dtype=amp_dtype, use_amp=use_amp):
            logits = model(non_seq_x, seq_x)
            loss = criterion(logits, labels)

        if is_train:
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_items += batch_size
        total_loss += loss.item() * batch_size
        total_acc += accuracy_from_logits(logits.detach(), labels) * batch_size
        epoch_logits.append(logits.detach().cpu())
        epoch_labels.append(labels.detach().cpu())

    if total_items == 0:
        return 0.0, 0.0, float("nan")

    auc = multiclass_auc_from_logits(torch.cat(epoch_logits, dim=0), torch.cat(epoch_labels, dim=0))
    return total_loss / total_items, total_acc / total_items, auc


def build_loaders(
    non_seq_x: torch.Tensor,
    seq_x: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    train_idx, val_idx = split_indices(labels.size(0), val_ratio, seed)
    train_dataset = TensorDataset(non_seq_x[train_idx], seq_x[train_idx], labels[train_idx])
    val_dataset = TensorDataset(non_seq_x[val_idx], seq_x[val_idx], labels[val_idx])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def build_model(args: argparse.Namespace, non_seq_x: torch.Tensor, seq_x: torch.Tensor, labels: torch.Tensor) -> nn.Module:
    return TAACOneTransClassifier(
        non_seq_dim=non_seq_x.size(1),
        seq_feature_dim=seq_x.size(2),
        num_classes=max(int(labels.max().item()) + 1, 2),
        seq_len=args.seq_len,
        ns_len=args.ns_len,
        d_model=args.d_model,
        num_heads=args.num_heads,
        ffn_hidden=args.ffn_hidden,
        multi_num=args.multi_num,
    ).to(args.device)


def save_run_artifacts(
    output_dir: Path,
    metadata: dict[str, Any],
    args_payload: dict[str, Any],
    best_state: dict[str, Any] | None,
    save_checkpoint: bool,
) -> None:
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps({**metadata, "args": args_payload}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[run] metadata saved to {metadata_path}")

    if save_checkpoint and best_state is not None:
        checkpoint_path = output_dir / "best_model.pt"
        torch.save(best_state, checkpoint_path)
        print(f"[run] checkpoint saved to {checkpoint_path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device_type = parse_device_type(args.device)
    amp_dtype = resolve_amp_dtype(args.amp_dtype)
    use_amp = should_enable_amp(device_type=device_type, amp=args.amp, amp_dtype=args.amp_dtype)
    scaler = build_scaler(device_type=device_type, use_amp=use_amp, amp_dtype=amp_dtype)

    dataset = load_train_split(args.dataset_id, args.cache_dir, args.local_parquet)
    rows = take_rows(dataset, args.max_rows)
    non_seq_x, seq_x, labels, metadata = build_tensors(rows, args.seq_len)
    train_loader, val_loader = build_loaders(
        non_seq_x=non_seq_x,
        seq_x=seq_x,
        labels=labels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    model = build_model(args, non_seq_x, seq_x, labels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    print("[run] metadata")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"[run] device={args.device} samples={labels.size(0)} train={len(train_loader.dataset)} val={len(val_loader.dataset)}")
    print(f"[run] non_seq={tuple(non_seq_x.shape)} seq={tuple(seq_x.shape)} classes={int(labels.max().item()) + 1}")
    print(
        f"[run] amp={use_amp} amp_dtype={args.amp_dtype} "
        f"grad_scaler={scaler.is_enabled()} device_type={device_type}"
    )

    best_val_auc = float("-inf")
    best_state: dict[str, Any] | None = None
    args_payload = json_ready_args(args)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_auc = run_epoch(
            model,
            train_loader,
            criterion,
            args.device,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
            scaler=scaler,
            optimizer=optimizer,
        )
        val_loss, val_acc, val_auc = run_epoch(
            model,
            val_loader,
            criterion,
            args.device,
            amp_dtype=amp_dtype,
            use_amp=use_amp,
            scaler=scaler,
        )
        print(
            f"[epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} train_auc={train_auc:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_auc={val_auc:.4f} val_acc={val_acc:.4f}"
        )
        if not math.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {
                "model": model.state_dict(),
                "metadata": metadata,
                "args": args_payload,
                "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            }

    save_run_artifacts(args.output_dir, metadata, args_payload, best_state, args.save_checkpoint)


if __name__ == "__main__":
    main()
