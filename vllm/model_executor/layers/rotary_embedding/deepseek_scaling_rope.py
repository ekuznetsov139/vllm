# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch
import triton
import triton.language as tl

from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

from .base import RotaryEmbeddingBase
from .common import (
    rotate_gptj,
    rotate_neox,
    yarn_find_correction_range,
    yarn_linear_ramp_mask,
)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


@triton.jit
def deepseek_scaling_rotary_emb_kernel_gptj(cos_sin, q, stride1: int,
                                            stride2: int, stride_cs: int,
                                            dim1: int, dim2: int, dim3: int,
                                            BLOCK_SIZE: tl.constexpr):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)
    offsets_cs = tl.arange(0, BLOCK_SIZE) + pid2 * BLOCK_SIZE
    offsets_q = tl.arange(0, BLOCK_SIZE * 2) + pid2 * BLOCK_SIZE * 2

    offsets = pid0 * stride1 + pid1 * stride2 + offsets_q
    mask = offsets_cs < dim3
    mask2 = offsets_q < dim3 * 2

    v_cos = tl.load(cos_sin + pid0 * stride_cs + offsets_cs, mask=mask)
    v_cos2 = tl.interleave(v_cos, v_cos)
    v_sin = tl.load(cos_sin + pid0 * stride_cs + dim3 + offsets_cs, mask=mask)
    v_sin2 = tl.interleave(v_sin, v_sin)
    x12 = tl.load(q + offsets, mask=mask2)
    x1, x2 = tl.split(x12.reshape([BLOCK_SIZE, 2]))
    # we are both reading and writing 'q'; make sure all warps are in sync
    tl.debug_barrier()
    x12_ = tl.ravel(tl.join(-x2, x1))
    x12 = x12 * v_cos2 + x12_ * v_sin2
    tl.store(q + offsets, x12, mask=mask2)


class DeepseekScalingRotaryEmbedding(RotaryEmbeddingBase):
    """RotaryEmbedding extended with YaRN method.

    Credits to Peng et al. github.com/jquesnelle/yarn
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
        reference: bool = False,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.reference = reference
        # Get n-d magnitude scaling corrected for interpolation.
        self.mscale = float(
            yarn_get_mscale(self.scaling_factor, float(mscale))
            / yarn_get_mscale(self.scaling_factor, float(mscale_all_dim))
            * attn_factor
        )
        self.use_flashinfer = (
            self.enabled()
            and dtype in (torch.float16, torch.bfloat16)
            and current_platform.is_cuda()
            and has_flashinfer()
            and head_size in [64, 128, 256, 512]
        )
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        pos_freqs = self.base ** (
            torch.arange(
                0,
                self.rotary_dim,
                2,
                dtype=torch.float,
            )
            / self.rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
        )
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (
            1
            - yarn_linear_ramp_mask(low, high, self.rotary_dim // 2, dtype=torch.float)
        ) * self.extrapolation_factor
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = torch.arange(
            self.max_position_embeddings * self.scaling_factor,
            dtype=torch.float32,
        )
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * self.mscale
        sin = freqs.sin() * self.mscale
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """PyTorch-native implementation equivalent to forward()."""
        assert key is not None
        cos_sin_cache = self._match_cos_sin_cache_dtype(query)
        query_rot = query[..., : self.rotary_dim]
        key_rot = key[..., : self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim :]
            key_pass = key[..., self.rotary_dim :]

        cos_sin = cos_sin_cache[
            torch.add(positions, offsets) if offsets is not None else positions
        ]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            # NOTE(woosuk): Here we assume that the positions tensor has the
            # shape [batch_size, seq_len].
            cos = cos.repeat(1, 1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 1, 2).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        rotate_fn = rotate_neox if self.is_neox_style else rotate_gptj
        query_rot = query_rot * cos + rotate_fn(query_rot) * sin
        key_rot = key_rot * cos + rotate_fn(key_rot) * sin

        if self.rotary_dim < self.head_size:
            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
        else:
            query = query_rot
            key = key_rot
        return query, key

    def forward_xpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return torch.ops.vllm.xpu_ops_deepseek_scaling_rope(
            positions,
            query,
            key,
            offsets,
            self._match_cos_sin_cache_dtype(query),
            self.rotary_dim,
            self.is_neox_style,
        )

    def forward_hip(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert key is not None
        if not self.is_neox_style and not self.reference:
            assert len(query.shape) == 3
            cos_sin = self.cos_sin_cache[
                torch.add(positions, offsets) if offsets is not None else positions
            ]

            def call(q):
                BLOCK_SIZE = 64
                grid = (
                    q.shape[-3],
                    q.shape[-2],
                    triton.cdiv(self.rotary_dim // 2, BLOCK_SIZE),
                )
                deepseek_scaling_rotary_emb_kernel_gptj[grid](
                    cos_sin,
                    q,
                    stride1=q.stride()[-3],
                    stride2=q.stride()[-2],
                    stride_cs=cos_sin.stride()[-2],
                    dim1=q.shape[0],
                    dim2=q.shape[1],
                    dim3=self.rotary_dim // 2,
                    BLOCK_SIZE=BLOCK_SIZE,
                    num_warps=1)

            call(query)
            call(key)
            return query, key
        else:
            return self.forward_native(positions, query, key, offsets)

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.use_flashinfer:
            torch.ops.vllm.flashinfer_rotary_embedding(
                torch.add(positions, offsets) if offsets is not None else positions,
                query,
                key,
                self.head_size,
                self.cos_sin_cache,
                self.is_neox_style,
            )
            return query, key
        else:
            return self.forward_native(positions, query, key, offsets)
