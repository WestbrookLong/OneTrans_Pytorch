from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main_pytorch import MultiOneTransBlock


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
    ) -> None:
        super().__init__()
        self.ns_len = ns_len
        self.seq_len = seq_len
        self.non_seq_tokenizer = nn.Linear(non_seq_dim, ns_len * d_model)
        self.seq_tokenizer = nn.Linear(seq_feature_dim, d_model)
        self.base_block = MultiOneTransBlock(
            ns_len=ns_len,
            d_model=d_model,
            num_heads=num_heads,
            ffn_units=(ffn_hidden, d_model),
            n=multi_num,
        )
        total_tokens = ns_len + seq_len
        self.stack_blocks = nn.ModuleList(
            [
                MultiOneTransBlock(
                    ns_len=ns_len,
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_units=(ffn_hidden, d_model),
                    n=multi_num,
                    pyramid_stack_len=target_len,
                )
                for target_len in range(total_tokens - 1, ns_len - 1, -1)
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
        x = torch.cat([ns_tokens, seq_tokens], dim=1)
        x = self.base_block(x)
        for block in self.stack_blocks:
            x = block(x)
        pooled = x.mean(dim=1)
        return self.head(pooled)
