# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4-Flash decode monokernel on gfx1260 (Arcadia), TP1 (``--monokernel``).

A decode step's backbone layer runs as aiter's two Gluon launches
(``aiter.ops.triton._gluon_kernels.gfx1260.dsv4_mono``): K1, the attention mHC
seam and the attention front, then K2, the attention back, the FFN seam and the
MoE. vLLM's compressors and indexer run between them, on K1's normed attention
input and qr, as vLLM's own attention runs them: the main compressor on every
compressed layer (ratio 4 or 128), the indexer's compressor and the indexer on
a ratio-4 layer, which leave the top-k K2 reads.

The kernels read the checkpoint's MXFP8 dense weights and MXFP4 experts, while
vLLM keeps the dense weights as bf16 on gfx1260, so each layer loads its own
copy from the checkpoint at its first mono step.

A step takes the kernels when it is decode only and has at most ``MAX_ROWS``
rows; anything else runs the layer as before. Both paths read and write the same
tensors at the layer boundary, so they interleave freely: on the fused mHC path
the kernels take and return the owed (x, residual, post_mix, res_mix); on the
unfused path they take the settled streams and the hook settles their output.

``VLLM_DSV4_MONO_MODE``: ``mono`` (default) is one launch a kernel, ``split``
one launch a stage. The two are bit-identical; split needs no co-resident
workgroups.
"""

import dataclasses
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import CUDAGraphMode
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.models.deepseek_v4.amd.model import DeepseekV4DecoderLayer

logger = init_logger(__name__)

RECORD = 584  # an fp8_ds_mla KV record: 576 data bytes, 8 scale bytes
MAX_ROWS = 48  # the kernels' step rows at most (dsv4_mono.plan.MAX_ROWS)

_runner: "_Runner | None" = None


def _record_view(kv: torch.Tensor, block: int) -> torch.Tensor:
    """A layer's fp8_ds_mla cache as [blocks, block, RECORD] bytes (a block's
    data rows, then its scale words: the kernels index both from it)."""
    kv = kv if kv.dtype == torch.uint8 else kv.view(torch.uint8)
    assert kv.stride(0) >= block * RECORD, (tuple(kv.shape), kv.stride(), block)
    return torch.as_strided(kv, (kv.shape[0], block, RECORD), (kv.stride(0), RECORD, 1))


class _Runner:
    """The kernels' per-device state, shared by every layer, and each layer's
    weights, loaded from the checkpoint at its first mono step."""

    def __init__(self, ckpt: str, max_model_len: int, device: torch.device) -> None:
        from aiter.ops.triton._gluon_kernels.gfx1260.dsv4_mono.dims import Dims
        from aiter.ops.triton._gluon_kernels.gfx1260.dsv4_mono.runner import MonoState

        self.ckpt = ckpt
        self.device = device
        self.mode = os.environ.get("VLLM_DSV4_MONO_MODE", "mono")
        # The dense (ratio 128) key table is sized by the longest context served.
        self.d = dataclasses.replace(
            Dims.from_config(Path(ckpt) / "config.json"), max_model_len=max_model_len
        )
        self.state = MonoState(self.d, device=device)
        self._weights: dict[int, Any] = {}
        self._cos_sin: dict[int, torch.Tensor] = {}

    def weights(self, layer_id: int):
        w = self._weights.get(layer_id)
        if w is None:
            from aiter.ops.triton._gluon_kernels.gfx1260.dsv4_mono.weights import (
                kernel_weights,
                load_layer,
            )

            w = kernel_weights(
                load_layer(layer_id, self.d, self.ckpt), self.d, device=self.device
            )
            self._weights[layer_id] = w
        return w

    def cos_sin(self, table: torch.Tensor) -> torch.Tensor:
        if table.dtype == torch.float32:
            return table
        return self._cos_sin.setdefault(id(table), table.float())


def _get_runner(ckpt: str, max_model_len: int, device: torch.device) -> _Runner:
    global _runner
    if _runner is None:
        _runner = _Runner(ckpt, max_model_len, device)
    return _runner


class MonoDecodeLayer:
    """One decoder layer's mono path; ``create`` returns None when the layer is
    not eligible."""

    def __init__(self, layer_id: int, ckpt: str, max_model_len: int) -> None:
        self.layer_id = layer_id
        self.ckpt = ckpt
        self.max_model_len = max_model_len

    @staticmethod
    def create(
        layer: "DeepseekV4DecoderLayer", layer_id: int, vllm_config
    ) -> "MonoDecodeLayer | None":
        if not vllm_config.kernel_config.monokernel:
            return None
        from vllm.platforms.rocm import on_gfx1260

        if not on_gfx1260():
            raise ValueError("--monokernel for DeepSeek-V4 is only implemented on gfx1260.")
        if get_tensor_model_parallel_world_size() != 1:
            raise ValueError("--monokernel on gfx1260 needs tensor parallel size 1.")
        hf = vllm_config.model_config.hf_config
        attn = layer.attn
        why = None
        if layer_id >= hf.num_hidden_layers:
            why = "an MTP layer"
        elif attn.compress_ratio not in (1, 4, 128):
            why = f"compress ratio {attn.compress_ratio}"
        elif attn.kv_cache_dtype != "fp8_ds_mla":
            why = f"a {attn.kv_cache_dtype} KV cache"
        elif (attn.compress_ratio == 4) != (attn.indexer is not None):
            why = "an indexer on a layer that is not ratio 4"
        if why is not None:
            logger.info_once("DSv4 mono decode off for some layers: %s", why)
            return None
        logger.info_once("DSv4 mono decode: gfx1260 TP1 kernels")
        return MonoDecodeLayer(
            layer_id, vllm_config.model_config.model, vllm_config.model_config.max_model_len
        )

    @staticmethod
    def _skip(why: str) -> None:
        logger.debug_once("DSv4 mono decode skipped a step: %s", why)
        return None

    @staticmethod
    def _decode_step(layer: "DeepseekV4DecoderLayer", rows: int):
        """(why this step takes the original path, None for a decode-only step
        the kernels take; the layer's SWA metadata; its compressed cache's)."""
        if not 1 <= rows <= MAX_ROWS:
            return f"{rows} rows", None, None
        if not is_forward_context_available():
            return "no forward context", None, None
        fc = get_forward_context()
        md = fc.attn_metadata
        attn = layer.attn
        if fc.cudagraph_runtime_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL):
            return f"{fc.cudagraph_runtime_mode} step", None, None
        if not isinstance(md, dict):  # a profile / dummy run, or DBO ubatches
            return "no attention metadata", None, None
        swa = md.get(attn.swa_cache_layer.prefix)
        comp = md.get(attn.prefix) if attn.compress_ratio > 1 else None
        why = None
        if swa is None or (attn.compress_ratio > 1 and comp is None):
            why = "layer metadata missing"
        elif swa.num_prefills != 0 or swa.num_decodes == 0:
            why = "not a decode-only step"
        elif swa.num_decode_tokens != rows:
            why = "rows other than the decode tokens"
        elif (
            swa.decode_swa_indices is None
            or swa.decode_swa_lens is None
            or swa.token_to_req_indices is None
        ):
            why = "SWA decode metadata missing"
        return why, swa, comp

    @staticmethod
    def _between(
        layer: "DeepseekV4DecoderLayer",
        positions: torch.Tensor,
        normed: torch.Tensor,
        qr: torch.Tensor,
    ) -> None:
        """vLLM's compressors and indexer on K1's hand-outs, as
        ``DeepseekV4ROCMAiterMLAAttention._prepare_and_attn`` runs them."""
        attn = layer.attn
        comp, indexer = attn.compressor, attn.indexer
        if comp is not None:
            score = torch.mm(normed, comp.fused_wkv_wgate.weight.T, out_dtype=torch.float32)
            comp(score, positions, attn.rotary_emb)
        if indexer is None:
            return
        score = torch.mm(
            normed, indexer.compressor.fused_wkv_wgate.weight.T, out_dtype=torch.float32
        )
        indexer.compressor(score, positions, attn.indexer_rotary_emb)
        weights_in, _ = indexer.weights_proj(normed)
        q, q_scale, weights = indexer(
            normed,
            qr,
            None,
            weights_in,
            positions,
            attn.indexer_rotary_emb,
            None,
            skip_compressor=True,
        )
        # None: a short context, where the indexer filled the top-k itself
        if q is not None:
            assert weights is not None
            indexer.indexer_op(
                normed, (q, q_scale) if q_scale is not None else q, None, weights
            )

    def __call__(
        self,
        layer: "DeepseekV4DecoderLayer",
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None,
        res_mix: torch.Tensor | None,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...] | None:
        """The layer's outputs by the kernels, in the form the layer's own
        forward returns them, or None when this step takes the original path."""
        rows = x.shape[0]
        why, swa, comp = self._decode_step(layer, rows)
        settled = residual is None  # the first layer, or the unfused mHC path
        if why is None:
            if settled and x.dim() != 3:
                why = "settled streams not [rows, hc, hidden]"
            elif not settled and (post_mix is None or res_mix is None):
                why = "owed mixes missing"
        if why is not None:
            return self._skip(why)
        assert swa is not None
        attn = layer.attn
        runner = _get_runner(self.ckpt, self.max_model_len, x.device)
        d = runner.d
        kind = d.layer_kind(self.layer_id)
        if kind["hash"] and input_ids is None:
            return self._skip("hash routing without token ids")
        ratio = 0 if attn.compress_ratio == 1 else attn.compress_ratio
        assert kind["ratio"] == ratio, (self.layer_id, kind, attn.compress_ratio)
        logger.info_once(
            "DSv4 mono decode layer active (decode steps of <= %d rows)", MAX_ROWS
        )

        from aiter.ops.triton._gluon_kernels.gfx1260.dsv4_mono.runner import (
            DeviceStep,
            k1_handouts,
            run_k1,
            run_k2,
        )

        dev = x.device
        dummy_i32 = torch.full((1, 1), -1, dtype=torch.int32, device=dev)
        topk = dummy_i32
        comp_args: dict[str, Any] = dict(
            comp_bt=dummy_i32,
            comp=torch.zeros(1, dtype=torch.uint8, device=dev),
            comp_page=0,
            comp_block=1,
        )
        if ratio:
            assert comp is not None
            cache = _record_view(attn.kv_cache, comp.block_size // ratio)
            comp_args = dict(
                comp_bt=comp.block_table.to(torch.int32),
                comp=cache,
                comp_page=cache.stride(0),
                comp_block=cache.shape[1],
            )
        if kind["index"]:
            assert attn.topk_indices_buffer is not None
            topk = attn.topk_indices_buffer[:rows]
        swa_cache = _record_view(attn.swa_cache_layer.kv_cache, swa.block_size)
        token_ids = (
            input_ids[:rows].to(torch.int32)
            if input_ids is not None
            else torch.zeros(rows, dtype=torch.int32, device=dev)
        )
        step = DeviceStep(
            M=rows,
            ratio=ratio,
            index=kind["index"],
            positions=positions[:rows].to(torch.int64),
            slot_mapping=swa.slot_mapping[:rows].to(torch.int64),
            token_ids=token_ids,
            token_to_req=swa.token_to_req_indices[:rows].to(torch.int32),
            swa_idx=swa.decode_swa_indices.to(torch.int32),
            swa_lens=swa.decode_swa_lens.to(torch.int32),
            swa=swa_cache,
            swa_page=swa_cache.stride(0),
            swa_block=swa_cache.shape[1],
            topk=topk,
            cos_sin=runner.cos_sin(attn.rotary_emb.cos_sin_cache),
            **comp_args,
        )
        if settled:
            res = x.contiguous()
            x_in = res[:, 0].contiguous()  # unread without the fold
            post = torch.zeros(rows, d.hc, dtype=torch.float32, device=dev)
            comb = torch.zeros(rows, d.hc, d.hc, dtype=torch.float32, device=dev)
        else:
            assert residual is not None and post_mix is not None and res_mix is not None
            x_in, res = x.contiguous(), residual.contiguous()
            post = (post_mix.squeeze(-1) if post_mix.dim() == 3 else post_mix).float()
            comb = res_mix.float().contiguous()
        weights = runner.weights(self.layer_id)
        run_k1(
            runner.state,
            weights,
            step,
            x_in,
            res,
            post.contiguous(),
            comb,
            fold=not settled,
            index=kind["index"],
            dense=kind["dense"],
            mode=runner.mode,
        )
        if attn.compressor is not None or attn.indexer is not None:
            self._between(layer, positions[:rows], *k1_handouts(runner.state, rows))
        out, res2, post2, comb2 = run_k2(
            runner.state,
            weights,
            step,
            index=kind["index"],
            hash_=kind["hash"],
            mode=runner.mode,
        )
        # K2's outputs are the kernels' scratch, reused by the next layer
        out, res2 = out.clone(), res2.clone()
        post2, comb2 = post2.unsqueeze(-1).clone(), comb2.clone()
        if layer.use_fused_mhc:
            return out, res2, post2, comb2
        return layer.hc_post(out, res2, post2, comb2), None, None, None
