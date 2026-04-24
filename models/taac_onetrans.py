from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main_pytorch import MultiOneTransBlock


def linear_pyramid_schedule(total_tokens: int, ns_len: int, num_layers: int, align_to: int = 32) -> list[int]:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if total_tokens < ns_len:
        raise ValueError("total_tokens must be greater than or equal to ns_len")
    if align_to <= 0:
        raise ValueError("align_to must be positive")

    if num_layers == 1:
        return [ns_len]

    schedule = [total_tokens]
    for layer_idx in range(1, num_layers - 1):
        raw = total_tokens + (ns_len - total_tokens) * layer_idx / (num_layers - 1)
        target_len = int(round(raw))
        if align_to > 1 and total_tokens > align_to:
            target_len = int(round(target_len / align_to) * align_to)
        target_len = max(ns_len, min(schedule[-1], target_len))
        schedule.append(target_len)
    schedule.append(ns_len)
    return schedule


class TAACOneTransClassifier(nn.Module):
    def __init__(
        self,
        non_seq_dim: int,
        seq_feature_dim: int,
        num_classes: int,
        seq_len: int,
        ns_len: int,
        d_model: int,
        num_heads: int,
        ffn_hidden: int,
        multi_num: int,
        mask_type: str = "paper_causal",
        num_pyramid_layers: int = 6,
        pyramid_align: int = 32,
        use_sep_token: bool = True,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.ns_len = ns_len
        self.seq_len = seq_len
        self.use_sep_token = use_sep_token
        self.non_seq_tokenizer = nn.Linear(non_seq_dim, ns_len * d_model)
        self.seq_tokenizer = nn.Linear(seq_feature_dim, d_model)
        if use_sep_token:
            self.sep_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.base_block = MultiOneTransBlock(
            ns_len=ns_len,
            d_model=d_model,
            num_heads=num_heads,
            ffn_units=(ffn_hidden, d_model),
            n=multi_num,
            mask_type=mask_type,
            use_checkpoint=use_checkpoint,
        )
        total_tokens = ns_len + seq_len + (1 if use_sep_token else 0)
        schedule = linear_pyramid_schedule(
            total_tokens=total_tokens,
            ns_len=ns_len,
            num_layers=num_pyramid_layers,
            align_to=pyramid_align,
        )
        self.stack_blocks = nn.ModuleList(
            [
                MultiOneTransBlock(
                    ns_len=ns_len,
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_units=(ffn_hidden, d_model),
                    n=multi_num,
                    pyramid_stack_len=target_len,
                    mask_type=mask_type,
                    use_checkpoint=use_checkpoint,
                )
                for target_len in schedule
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, non_seq_x: torch.Tensor, seq_x: torch.Tensor) -> torch.Tensor:
        batch_size = non_seq_x.size(0)
        ns_tokens = self.non_seq_tokenizer(non_seq_x).view(batch_size, self.ns_len, -1)
        seq_tokens = self.seq_tokenizer(seq_x)
        if self.use_sep_token:
            sep_token = self.sep_token.expand(batch_size, -1, -1)
            x = torch.cat([seq_tokens, sep_token, ns_tokens], dim=1)
        else:
            x = torch.cat([seq_tokens, ns_tokens], dim=1)
        x = self.base_block(x)
        for block in self.stack_blocks:
            x = block(x)
        pooled = x.mean(dim=1)
        return self.head(pooled)
