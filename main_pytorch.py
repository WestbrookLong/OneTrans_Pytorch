import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

VALID_MASK_TYPES = {"paper_causal", "origin", "hard_mask", "bimask_soft", "bimask_hard"}


class FFNLayer(nn.Module):
    def __init__(self, input_dim: int, unit_1: int = 256, unit_2: int = 128) -> None:
        super().__init__()
        self.proj_1 = nn.Linear(input_dim, unit_1)
        self.proj_2 = nn.Linear(unit_1, unit_2)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.proj_1(x))
        x = self.act(self.proj_2(x))
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_fp32 = x.float()
        rms = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x_fp32 * rms).to(input_dtype) * self.weight


class CausalMaskAttention(nn.Module):
    def __init__(
        self,
        ns_len: int,
        d_model: int = 128,
        num_heads: int = 4,
        if_mask: bool = True,
        mask_type: str = "paper_causal",
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if mask_type not in VALID_MASK_TYPES:
            raise ValueError(f"Unsupported mask_type: {mask_type}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.ns_len = ns_len
        self.if_mask = if_mask
        self.mask_type = mask_type
        self.dense = nn.Linear(d_model, d_model)
        self.kqv_list = nn.ModuleList(
            [nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)]) for _ in range(ns_len + 1)]
        )

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def create_attention_mask(self, query_len: int, key_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.mask_type == "paper_causal":
            row_idx = torch.arange(query_len, device=device).unsqueeze(1)
            col_idx = torch.arange(key_len, device=device).unsqueeze(0)
            q_abs = row_idx + (key_len - query_len)
            allowed = col_idx <= q_abs
            mask = torch.zeros(query_len, key_len, device=device, dtype=dtype)
            return mask.masked_fill(~allowed, torch.finfo(dtype).min)

        row_idx = torch.arange(query_len, device=device).unsqueeze(1)
        col_idx = torch.arange(key_len, device=device).unsqueeze(0)
        origin_allowed = (col_idx - row_idx) <= (self.ns_len - 1)
        if self.mask_type == "origin":
            return origin_allowed.to(dtype=dtype) + 1e-9
        if self.mask_type == "hard_mask":
            mask = torch.zeros(query_len, key_len, device=device, dtype=dtype)
            return mask.masked_fill(~origin_allowed, torch.finfo(dtype).min)

        ns_query_rows = row_idx < self.ns_len
        strict_causal_allowed = col_idx <= row_idx
        if self.mask_type == "bimask_soft":
            mask = torch.zeros(query_len, key_len, device=device, dtype=dtype)
            seq_allowed = (~ns_query_rows) & strict_causal_allowed
            return mask + seq_allowed.to(dtype=dtype)

        allowed = ns_query_rows | strict_causal_allowed
        mask = torch.zeros(query_len, key_len, device=device, dtype=dtype)
        return mask.masked_fill(~allowed, torch.finfo(dtype).min)

    def _cal_kqv(self, x: torch.Tensor, group_idx: int, proj_idx: int) -> torch.Tensor:
        return self.kqv_list[group_idx][proj_idx](x)

    def _project_one(self, x: torch.Tensor, proj_idx: int) -> torch.Tensor:
        seq_len = x.size(1) - self.ns_len
        shared_group_idx = self.ns_len
        res = []
        if seq_len > 0:
            res.append(self._cal_kqv(x[:, :seq_len, :], shared_group_idx, proj_idx))
        for i in range(self.ns_len):
            start = seq_len + i
            res.append(self._cal_kqv(x[:, start : start + 1, :], i, proj_idx))
        return torch.cat(res, dim=1)

    def cal_mix_param_kqv(self, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._project_one(x[0], 0), self._project_one(x[1], 1), self._project_one(x[2], 2)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        seq_len_k = x[0].size(1)
        seq_len_q = x[1].size(1)

        k, q, v = self.cal_mix_param_kqv(x)
        k = self.split_heads(k)
        q = self.split_heads(q)
        v = self.split_heads(v)

        attention_mask = None
        if self.if_mask:
            attention_mask = self.create_attention_mask(
                seq_len_q, seq_len_k, device=q.device, dtype=q.dtype
            )
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

        output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).contiguous()
        output = output.view(output.size(0), -1, self.d_model)
        return self.dense(output)


class OneTransBlock(nn.Module):
    def __init__(
        self,
        ns_len: int,
        d_model: int,
        num_heads: int = 4,
        ffn_units: tuple[int, int] | None = None,
        pyramid_stack_len: int | None = None,
        mask_type: str = "paper_causal",
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.ns_len = ns_len
        self.d_model = d_model
        self.pyramid_stack_len = pyramid_stack_len
        self.use_checkpoint = use_checkpoint
        if ffn_units is None:
            ffn_units = (256, d_model)
        self.rms_0 = RMSNorm(d_model)
        self.rms_1 = RMSNorm(d_model)
        self.cma = CausalMaskAttention(
            ns_len=ns_len,
            d_model=d_model,
            num_heads=num_heads,
            mask_type=mask_type,
        )
        self.ffn_list = nn.ModuleList(
            [FFNLayer(input_dim=d_model, unit_1=ffn_units[0], unit_2=ffn_units[1]) for _ in range(ns_len + 1)]
        )

    def cal_mix_param_ffn(self, x: torch.Tensor) -> torch.Tensor:
        res = []
        seq_len = x.size(1) - self.ns_len
        if seq_len > 0:
            res.append(self.ffn_list[self.ns_len](x[:, :seq_len, :]))
        for i in range(self.ns_len):
            start = seq_len + i
            res.append(self.ffn_list[i](x[:, start : start + 1, :]))
        return torch.cat(res, dim=1)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rms_0(x)
        k_x, q_x, v_x = x, x, x
        if self.pyramid_stack_len is not None and self.pyramid_stack_len >= self.ns_len:
            q_x = x[:, -self.pyramid_stack_len :, :]
        origin_x = q_x

        x = self.cma((k_x, q_x, v_x))
        x = origin_x + x
        origin_x = x

        x = self.rms_1(x)
        x = self.cal_mix_param_ffn(x)
        x = origin_x + x
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


class MultiOneTransBlock(nn.Module):
    def __init__(
        self,
        ns_len: int = 4,
        d_model: int = 128,
        num_heads: int = 4,
        ffn_units: tuple[int, int] | None = None,
        n: int = 4,
        pyramid_stack_len: int | None = None,
        mask_type: str = "paper_causal",
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.otb_list = nn.ModuleList(
            [
                OneTransBlock(
                    ns_len=ns_len,
                    d_model=d_model,
                    num_heads=num_heads,
                    ffn_units=ffn_units,
                    pyramid_stack_len=pyramid_stack_len,
                    mask_type=mask_type,
                    use_checkpoint=use_checkpoint,
                )
                for _ in range(n)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for otb in self.otb_list:
            x = otb(x)
        return x


def main() -> None:
    torch.manual_seed(0)

    batch_size = 4
    seq_len = 3
    feat_dim = 8
    d_model = 16

    seq_feature = torch.randn(batch_size, seq_len, feat_dim)
    seq_tokenizer = nn.Linear(feat_dim, d_model)
    s_feat = seq_tokenizer(seq_feature)
    print("Sequence feature [batch_size, seq_len, feat_dim]:", tuple(seq_feature.shape))
    print("Encoded sequence feature [batch_size, seq_len, d_model]:", tuple(s_feat.shape))

    n_seq_feature = torch.randn(batch_size, 128)
    ns_len = 2
    non_seq_tokenizer = nn.Linear(128, ns_len * d_model)
    ns_feat = non_seq_tokenizer(n_seq_feature).view(batch_size, ns_len, d_model)
    print("Non-sequence feature [batch_size, random_dim]:", tuple(n_seq_feature.shape))
    print("Encoded non-sequence feature [batch_size, ns_len, d_model]:", tuple(ns_feat.shape))
    print()

    num_head = 4
    multi_num = 8
    ffn_units = (64, d_model)

    base_block = MultiOneTransBlock(
        ns_len=ns_len,
        d_model=d_model,
        num_heads=num_head,
        ffn_units=ffn_units,
        n=multi_num,
    )
    base_embedding = base_block(torch.cat([s_feat, ns_feat], dim=1))
    print("After base OneTrans block [batch_size, seq_len + ns_len, d_model]:", tuple(base_embedding.shape))

    base_seq_len = base_embedding.size(1)

    stack_block_1 = MultiOneTransBlock(
        ns_len=ns_len,
        d_model=d_model,
        num_heads=num_head,
        ffn_units=ffn_units,
        n=multi_num,
        pyramid_stack_len=base_seq_len - 1,
    )
    stack_embedding = stack_block_1(base_embedding)
    print("After compression block 1 [batch_size, seq_len + ns_len - 1, d_model]:", tuple(stack_embedding.shape))

    stack_block_2 = MultiOneTransBlock(
        ns_len=ns_len,
        d_model=d_model,
        num_heads=num_head,
        ffn_units=ffn_units,
        n=multi_num,
        pyramid_stack_len=base_seq_len - 2,
    )
    stack_embedding = stack_block_2(stack_embedding)
    print("After compression block 2 [batch_size, seq_len + ns_len - 2, d_model]:", tuple(stack_embedding.shape))

    stack_block_3 = MultiOneTransBlock(
        ns_len=ns_len,
        d_model=d_model,
        num_heads=num_head,
        ffn_units=ffn_units,
        n=multi_num,
        pyramid_stack_len=base_seq_len - 3,
    )
    stack_embedding = stack_block_3(stack_embedding)
    print("After compression block 3 [batch_size, seq_len + ns_len - 3, d_model]:", tuple(stack_embedding.shape))
    print()

    print("Final compressed result [batch_size, ns_len, d_model]:", tuple(stack_embedding.shape))
    final_embedding = stack_embedding.mean(dim=1)
    print("Pooling result before downstream task:", tuple(final_embedding.shape))


if __name__ == "__main__":
    main()
