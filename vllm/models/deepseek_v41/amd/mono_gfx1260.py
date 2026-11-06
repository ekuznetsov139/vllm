# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The gfx1260 (Arcadia) runner of ``mono_decode``: aiter's TP1 Gluon decode kernels
(``aiter.ops.triton._gluon_kernels.gfx1260.dsv41_mono``) behind the interface the
layer's ``__call__`` drives.

The kernels read MXFP8 dense weights and MXFP4 experts in the checkpoint's formats.
vLLM keeps the dense weights as bf16 on gfx1260, so each layer loads its own copy from
the checkpoint at its first mono step (about 7.5 GB a layer on the device).

``VLLM_DS41_MONO_MODE``: ``mono`` (default) is one launch a kernel, ``split`` one launch
a stage. The two are bit-identical; split needs no co-resident workgroups.
"""

import os
from pathlib import Path
from typing import Any, Callable

import torch


class Gfx1260MonoRunner:
    def __init__(self, ckpt: str, device: torch.device) -> None:
        from aiter.ops.triton._gluon_kernels.gfx1260.dsv41_mono.dims import Dims
        from aiter.ops.triton._gluon_kernels.gfx1260.dsv41_mono.runner import MonoState

        self.ckpt = Path(ckpt)
        self.device = device
        self.mode = os.environ.get("VLLM_DS41_MONO_MODE", "mono")
        self.d = Dims.from_config()
        self.state = MonoState(self.d, device=device)
        self._cos_sin: dict[int, torch.Tensor] = {}

    def first_pre_mix(self, rows: int) -> torch.Tensor:
        pre = torch.zeros(rows, self.d.hc, dtype=torch.float32, device=self.device)
        pre[:, 0] = 1.0
        return pre

    def layer_weights(self, layer_id: int):
        from aiter.ops.triton._gluon_kernels.gfx1260.dsv41_mono.weights import (
            kernel_weights,
            load_layer,
        )

        return kernel_weights(
            load_layer(layer_id, self.d, self.ckpt), self.d, device=self.device
        )

    def forward(
        self,
        weights,
        x: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        pre_mix: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        swa_cache: torch.Tensor,
        swa_indices: torch.Tensor,
        swa_lens: torch.Tensor,
        token_to_req: torch.Tensor,
        fold: bool = True,
        select: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
        topk_indices: torch.Tensor | None = None,
        comp_cache: torch.Tensor | None = None,
        comp_block_table: torch.Tensor | None = None,
        cos_sin: torch.Tensor | None = None,
        ratio: int = 0,
    ) -> tuple[torch.Tensor, ...]:
        from aiter.ops.triton._gluon_kernels.gfx1260.dsv41_mono.runner import (
            DeviceStep,
            k1_handouts,
            run_k1,
            run_k2,
        )

        rows, dev = x.shape[0], x.device
        assert cos_sin is not None
        if cos_sin.dtype != torch.float32:
            cos_sin = self._cos_sin.setdefault(id(cos_sin), cos_sin.float())
        dummy_i32 = torch.full((1, 1), -1, dtype=torch.int32, device=dev)
        args: dict[str, Any] = dict(
            topk=dummy_i32,
            comp_bt=dummy_i32,
            comp=torch.zeros(1, dtype=torch.uint8, device=dev),
            comp_page=0,
            comp_block=1,
        )
        if ratio:
            assert topk_indices is not None and comp_cache is not None
            assert comp_block_table is not None
            args = dict(
                topk=topk_indices,
                comp_bt=comp_block_table.to(torch.int32),
                comp=comp_cache,
                comp_page=comp_cache.stride(0),
                comp_block=comp_cache.shape[1],
            )
        step = DeviceStep(
            M=rows,
            ratio=ratio,
            positions=positions.to(torch.int64),
            slot_mapping=slot_mapping.to(torch.int64),
            token_to_req=token_to_req.to(torch.int32),
            swa_idx=swa_indices.to(torch.int32),
            swa_lens=swa_lens.to(torch.int32),
            swa=swa_cache,
            swa_page=swa_cache.stride(0),
            swa_block=swa_cache.shape[1],
            cos_sin=cos_sin,
            **args,
        )
        post = post_mix.squeeze(-1) if post_mix.dim() == 3 else post_mix
        index = select is not None
        run_k1(self.state, weights, step, x, residual, post, res_mix, pre_mix,
               fold=fold, index=index, mode=self.mode)
        if select is not None:
            select(*k1_handouts(self.state, rows))
        out, res, post, comb, pre = run_k2(
            self.state, weights, step, index=index, mode=self.mode
        )
        return out, res, post.unsqueeze(-1), comb, pre
