# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import importlib
from importlib.util import find_spec

import os
import torch

from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton

from vllm.logger import init_logger

logger = init_logger(__name__) 

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops


@triton.jit
def _indexer_k_quant_and_cache_kernel(
    k_ptr,  # [num_tokens, head_dim]
    kv_cache_ptr,  # [n_blks, blk_size//tile_block, head_dim // 16B, tile_block, 16B]
    # [n_blocks, blk_size, head_dim]
    kv_cache_scale_ptr,  # [n_blks, blk_size]
    slot_mapping_ptr,  # [num_tokens]
    kv_cache_scale_stride,
    kv_cache_value_stride,
    block_size,
    num_tokens,
    head_dim: tl.constexpr,
    LAYOUT: tl.constexpr,
    BLOCK_TILE_SIZE: tl.constexpr,
    HEAD_TILE_SIZE: tl.constexpr,
    IS_FNUZ: tl.constexpr,
    USE_UE8M0: tl.constexpr,
):
    tid = tl.program_id(0)
    offset = tl.arange(0, head_dim)
    if LAYOUT == "SHUFFLE":
        tile_offset = (
            offset // HEAD_TILE_SIZE * BLOCK_TILE_SIZE * HEAD_TILE_SIZE
            + offset % HEAD_TILE_SIZE
        )
    else:
        tile_offset = offset
    tile_store_offset = tile_offset
    # for idx in tl.range(tid, num_tokens, n_program):
    src_ptr = k_ptr + tid * head_dim
    slot_id = tl.load(slot_mapping_ptr + tid)
    if slot_id < 0:
        return
    block_id = slot_id // block_size
    block_offset = slot_id % block_size
    tile_block_id = block_offset // BLOCK_TILE_SIZE
    tile_block_offset = block_offset % BLOCK_TILE_SIZE
    val = tl.load(src_ptr + offset)
    amax = tl.max(val.abs(), axis=-1).to(tl.float32)
    if IS_FNUZ:
        scale = tl.maximum(1e-4, amax) / 224.0
    else:
        scale = tl.maximum(1e-4, amax) / 448.0

    if USE_UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))

    fp8_val = (val.to(tl.float32) / scale).to(kv_cache_ptr.type.element_ty)
    if LAYOUT == "SHUFFLE":
        dst_ptr = (
            kv_cache_ptr
            + block_id * kv_cache_value_stride
            + tile_block_id * BLOCK_TILE_SIZE * head_dim
            + tile_block_offset * HEAD_TILE_SIZE
        )
    else:
        dst_ptr = (
            kv_cache_ptr + block_id * kv_cache_value_stride + block_offset * head_dim
        )
    tl.store(dst_ptr + tile_store_offset, fp8_val)
    dst_scale_ptr = kv_cache_scale_ptr + block_id * kv_cache_scale_stride + block_offset
    tl.store(dst_scale_ptr, scale)


def indexer_k_quant_and_cache_triton(
    k: torch.Tensor,
    kv_cache: torch.Tensor,  # [num_blocks, block_size, head_dim + 4]
    slot_mapping: torch.Tensor,
    quant_block_size,
    scale_fmt,
    block_tile_size=16,
    head_tile_size=16,
):
    num_blocks = kv_cache.shape[0]
    head_dim = k.shape[-1]
    num_tokens = slot_mapping.shape[0]
    block_size = kv_cache.shape[1]
    # In real layout, we store the first portion as kv cache value
    # and second portion as kv cache scale
    kv_cache = kv_cache.view(num_blocks, -1)
    kv_cache_value = kv_cache[:, : block_size * head_dim]
    kv_cache_scale = kv_cache[:, block_size * head_dim :].view(torch.float32)
    head_tile_size = head_tile_size // kv_cache.element_size()
    grid = (num_tokens,)
    _indexer_k_quant_and_cache_kernel[grid](
        k,
        kv_cache_value,
        kv_cache_scale,
        slot_mapping,
        kv_cache_scale.stride(0),
        kv_cache_value.stride(0),
        block_size,
        num_tokens,
        head_dim,
        "NHD",
        block_tile_size,
        head_tile_size,
        IS_FNUZ=current_platform.fp8_dtype() == torch.float8_e4m3fnuz,
        USE_UE8M0=scale_fmt == "ue8m0",
    )


@triton.jit
def _cp_gather_indexer_quant_cache_kernel(
    kv_cache_ptr,  # [n_blks,blk_size//tile_blk,head_dim//16B,tile_blk,16B]
    # [n_blks, blk_size, head_dim]
    kv_cache_scale_ptr,  # [n_blks, blk_size]
    k_fp8_ptr,  # [num_tokens, head_dim]
    k_scale_ptr,  # [num_tokens]
    block_table_ptr,  # [batch_size, block_table_stride]
    cu_seqlen_ptr,  # [batch_size + 1]
    token_to_seq_ptr,  # [num_tokens]
    block_size,
    block_table_stride,
    kv_cache_stride,
    kv_cache_scale_stride,
    LAYOUT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_TILE_SIZE: tl.constexpr,
    HEAD_TILE_SIZE: tl.constexpr,
):
    tid = tl.program_id(0)
    offset = tl.arange(0, HEAD_DIM)
    batch_id = tl.load(token_to_seq_ptr + tid)
    batch_start = tl.load(cu_seqlen_ptr + batch_id)
    batch_end = tl.load(cu_seqlen_ptr + batch_id + 1)
    batch_offset = tid - batch_start
    if tid >= batch_end:
        return
    block_table_id = batch_offset // block_size
    block_offset = batch_offset % block_size
    block_table_offset = batch_id * block_table_stride + block_table_id
    block_id = tl.load(block_table_ptr + block_table_offset)
    tiled_block_id = block_offset // BLOCK_TILE_SIZE
    tiled_block_offset = block_offset % BLOCK_TILE_SIZE
    if LAYOUT == "SHUFFLE":
        src_cache_offset = (
            block_id * kv_cache_stride
            + tiled_block_id * HEAD_DIM * BLOCK_TILE_SIZE
            + tiled_block_offset * HEAD_TILE_SIZE
        )
    else:
        src_cache_offset = block_id * kv_cache_stride + block_offset * HEAD_DIM
    src_scale_offset = block_id * kv_cache_scale_stride + block_offset
    dst_offset = tid * HEAD_DIM
    src_scale_ptr = kv_cache_scale_ptr + src_scale_offset
    src_cache_ptr = kv_cache_ptr + src_cache_offset
    dst_k_ptr = k_fp8_ptr + dst_offset
    scale_val = tl.load(src_scale_ptr)
    tl.store(k_scale_ptr + tid, scale_val)
    if LAYOUT == "SHUFFLE":
        tiled_src_offset = (
            offset // HEAD_TILE_SIZE * HEAD_TILE_SIZE * BLOCK_TILE_SIZE
            + offset % HEAD_TILE_SIZE
        )
    else:
        tiled_src_offset = offset
    val = tl.load(src_cache_ptr + tiled_src_offset)
    tl.store(dst_k_ptr + offset, val)


def cp_gather_indexer_k_quant_cache_triton(
    k_cache: torch.Tensor,  # [num_blocks, block_size, head_dim + 4]
    k_fp8: torch.Tensor,
    k_fp8_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlen: torch.Tensor,
    token_to_seq: torch.Tensor,
    block_tile_size: int = 16,
    head_tile_size: int = 16,
):
    num_tokens = k_fp8.size(0)
    block_size = k_cache.size(1)
    block_table_stride = block_table.stride(0)
    head_dim = k_fp8.shape[-1]
    num_blocks = k_cache.shape[0]
    # we assume the kv cache already been split to 2 portion
    k_cache = k_cache.view(num_blocks, -1)
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_value = k_cache[:, : block_size * head_dim].view(fp8_dtype)
    k_cache_scale = k_cache[:, block_size * head_dim :].view(torch.float32)
    grid = (num_tokens,)
    k_fp8_scale = k_fp8_scale.view(torch.float32)
    _cp_gather_indexer_quant_cache_kernel[grid](
        k_cache_value,
        k_cache_scale,
        k_fp8,
        k_fp8_scale,
        block_table,
        cu_seqlen,
        token_to_seq,
        block_size,
        block_table_stride,
        k_cache_value.stride(0),
        k_cache_scale.stride(0),
        "NHD",
        head_dim,
        block_tile_size,
        head_tile_size,
    )


# Taken from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py#L156.
# Left here as a reference, very slow for large contexts, not currently used:
# all pathways use triton or aiter
def fp8_paged_mqa_logits_torch(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
):
    from vllm.utils.math_utils import cdiv

    fp8_dtype = current_platform.fp8_dtype()
    batch_size, next_n, _, dim = q.size()
    block_size = kv_cache.shape[1]
    N = kv_cache.shape[0]
    kv_cache = kv_cache.reshape([N, (dim+4)*block_size])
    kv_cache, scale = kv_cache[:, :dim*block_size], kv_cache[:, dim*block_size:]
    kv_cache = kv_cache.reshape([N, block_size, 1, dim])
    scale = scale.reshape([N, block_size, 1, 4])
    scale = scale.contiguous().view(torch.float)
    q = q.float()
    kv_cache = kv_cache.view(fp8_dtype).float() * scale
    num_block, block_size, _, dim = kv_cache.size()
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    context_lens = context_lens.tolist()
    for i in range(batch_size):
        context_len = context_lens[i]
        q_offsets = torch.arange(context_len - next_n, context_len, device="cuda")
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        for block_rk in range(cdiv(context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache[block_idx]
            k_offsets = torch.arange(
                block_rk * block_size, (block_rk + 1) * block_size, device="cuda"
            )
            mask = (k_offsets[None, :] < context_len) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype
                ),
                0.0,
            )
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            logits[
                i * next_n : (i + 1) * next_n,
                block_rk * block_size : (block_rk + 1) * block_size,
            ] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))
    return logits


@functools.lru_cache
def paged_mqa_logits_module():
    paged_mqa_logits_module_path = None
    if find_spec("aiter.ops.triton.pa_mqa_logits") is not None:
        paged_mqa_logits_module_path = "aiter.ops.triton.pa_mqa_logits"
    elif find_spec("aiter.ops.triton.attention.pa_mqa_logits") is not None:
        paged_mqa_logits_module_path = "aiter.ops.triton.attention.pa_mqa_logits"

    if paged_mqa_logits_module_path is not None:
        try:
            module = importlib.import_module(paged_mqa_logits_module_path)
            return module
        except ImportError:
            return None
    return None

@triton.jit
def _fp8_paged_mqa_logits_kernel(
    q_ptr,            # [B, next_n, H, D] fp8
    kv_fp8_ptr,       # [num_blocks, (D_actual+4)*BLOCK_SIZE] fp8 — full packed row, fp8 data first
    kv_scale_ptr,     # [num_blocks, BLOCK_SIZE] float32 — non-contiguous view into same buffer
    weights_ptr,      # [B * next_n, H] float32
    context_lens_ptr, # [B] int32
    block_tables_ptr, # [B, max_blocks_per_seq] int32
    logits_ptr,       # [B * next_n, max_model_len] float32
    next_n,
    max_model_len,
    max_blocks_per_seq,
    q_stride_b, q_stride_n, q_stride_h, q_stride_d,
    logits_stride_m,
    kv_fp8_row_stride,    # fp8 elements per physical block row: (D_actual + 4) * BLOCK_SIZE
    kv_scale_row_stride,  # float32 elements per physical block row: (D_actual + 4) * BLOCK_SIZE // 4
    BLOCK_SIZE: tl.constexpr,    # KV cache block size (positions per physical block)
    D: tl.constexpr,             # head dim, padded to next power of 2 by caller
    D_actual: tl.constexpr,      # true head dim before padding; used for strides and masking
    H: tl.constexpr,             # number of query heads
    BLOCK_N: tl.constexpr,       # MFMA KV tile — must be ≥ BLOCK_SIZE and a power of 2
):
    tile_rk    = tl.program_id(0)  # which BLOCK_N-sized KV tile
    i          = tl.program_id(1)  # batch item
    t          = tl.program_id(2)  # speculative token index
    query_idx  = i * next_n + t

    context_len = tl.load(context_lens_ptr + i)
    logi_start  = tile_rk * BLOCK_N

    if logi_start >= context_len:
        return

    q_pos = context_len - next_n + t

    h_offs    = tl.arange(0, H)
    d_offs    = tl.arange(0, D)
    # Mask padding lanes when D_actual is not a power-of-2 (zero-overhead when D==D_actual).
    d_mask    = d_offs < D_actual
    logi_offs = logi_start + tl.arange(0, BLOCK_N)   # [BLOCK_N] logical KV positions

    # Map each logical position to a (physical_block, within-block offset) pair.
    # Works for any BLOCK_SIZE: for BLOCK_SIZE=1, log_blk_rk == logi_offs.
    log_blk_rk = logi_offs // BLOCK_SIZE              # [BLOCK_N]
    within_blk  = logi_offs  % BLOCK_SIZE              # [BLOCK_N]
    blk_mask    = log_blk_rk < max_blocks_per_seq
    phys_blk    = tl.load(
        block_tables_ptr + i * max_blocks_per_seq + log_blk_rk,
        mask=blk_mask, other=0,
    )  # [BLOCK_N] physical block indices

    kv_mask = logi_offs < context_len
    # kv_fp8_ptr has row stride kv_fp8_row_stride = (D_actual+4)*BLOCK_SIZE; fp8 data is
    # at the start of each row, packed as [BLOCK_SIZE, D_actual] (position-major).
    k_blk = tl.load(
        kv_fp8_ptr + phys_blk[:, None] * kv_fp8_row_stride
                   + within_blk[:, None] * D_actual
                   + d_offs[None, :],
        mask=kv_mask[:, None] & d_mask[None, :], other=0.0,
    )  # [BLOCK_N, D] fp8

    # kv_scale_ptr is a float32 view of the same buffer offset to the scale region.
    # Its row stride is kv_scale_row_stride = (D_actual+4)*BLOCK_SIZE//4.
    scale = tl.load(
        kv_scale_ptr + phys_blk * kv_scale_row_stride + within_blk,
        mask=kv_mask, other=1.0,
    )  # [BLOCK_N] float32

    # Load all H query heads at once → [H, D] fp8; stays in registers.
    q_blk = tl.load(
        q_ptr + i * q_stride_b + t * q_stride_n
              + h_offs[:, None] * q_stride_h + d_offs[None, :] * q_stride_d,
        mask=d_mask[None, :],
        other=0.0,
        cache_modifier=".cg",
    )  # [H, D] fp8

    # MFMA: [H, D] × [D, BLOCK_N] → [H, BLOCK_N]  (fp8 × fp8, fp32 accumulate)
    scores = tl.dot(q_blk, k_blk.T, input_precision="ieee", out_dtype=tl.float32)
    scores = scores * scale[None, :]   # apply per-token dequant scale

    w = tl.load(weights_ptr + query_idx * H + h_offs)  # [H]
    scores = tl.maximum(scores, 0.0) * w[:, None]      # [H, BLOCK_N]
    accum  = tl.sum(scores, axis=0)                    # [BLOCK_N]

    valid = kv_mask & (logi_offs <= q_pos)
    accum = tl.where(valid, accum, float("-inf"))

    tl.store(
        logits_ptr + query_idx * logits_stride_m + logi_offs,
        accum,
        mask=logi_offs < max_model_len,
    )


def fp8_paged_mqa_logits_triton(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Triton implementation of fp8_paged_mqa_logits_torch."""
    fp8_dtype = current_platform.fp8_dtype()
    batch_size, next_n, H, D = q.shape
    N = kv_cache.shape[0]
    block_size = kv_cache.shape[1]

    # Unpack kv_cache [N, block_size, 1, D+4] uint8 without copying.
    # Memory layout (written by indexer_k_quant_and_cache): within each physical
    # block the D*block_size fp8 bytes come first (all positions, all dims packed),
    # followed by 4*block_size bytes for per-position float32 scales.
    kv_2d = kv_cache.reshape(N, (D + 4) * block_size)  # contiguous; reshape never copies here

    # Zero-copy fp8 view of the full packed row (stride[0] = (D+4)*block_size fp8 elements).
    # The kernel uses kv_fp8_row_stride to skip over the trailing scale bytes in each row.
    kv_fp8 = kv_2d.view(fp8_dtype)   # [N, (D+4)*block_size] fp8, same storage

    # Zero-copy float32 view starting at the scale region of each row.
    # kv_2d viewed as float32: shape [N, (D+4)*block_size//4], stride[0]=(D+4)*block_size//4.
    # Slicing columns [D*block_size//4:] gives a non-contiguous view that Triton handles
    # via the explicit kv_scale_row_stride parameter below.
    kv_scale = kv_2d.view(torch.float32)[:, D * block_size // 4:]  # [N, block_size] f32

    M = batch_size * next_n
    logits = torch.full((M, max_model_len), float("-inf"), device=q.device, dtype=torch.float32)

    max_blocks_per_seq = block_tables.shape[1]
    BLOCK_D = triton.next_power_of_2(D)
    # BLOCK_N = MFMA KV tile; must be ≥ block_size so each tile contains whole cache blocks.
    BLOCK_N = max(128, triton.next_power_of_2(block_size))
    # grid.x: BLOCK_N tiles (large dim, no overflow).  grid.y/z: batch × spec-tokens.
    grid = (triton.cdiv(max_model_len, BLOCK_N), batch_size, next_n)

    _fp8_paged_mqa_logits_kernel[grid](
        q, kv_fp8, kv_scale, weights, context_lens, block_tables, logits,
        next_n,
        max_model_len,
        max_blocks_per_seq,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        logits.stride(0),
        kv_fp8.stride(0),    # (D+4)*block_size — fp8 elements per block row
        kv_scale.stride(0),  # (D+4)*block_size//4 — float32 elements per block row
        BLOCK_SIZE=block_size,
        D=BLOCK_D,
        D_actual=D,
        H=H,
        BLOCK_N=BLOCK_N,
    )
    return logits

def rocm_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Compute FP8 MQA logits using paged KV-cache.

    Args:
        q_fp8: Query tensor of shape [B, next_n, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv_cache_fp8: Paged KV-cache in packed FP8+scale layout with shape
            [num_blocks, block_size, 1, D+4], dtype `torch.uint8`. The last
            4 bytes per (block,pos) store the `float` dequant scale.
        weights: Tensor of shape [B * next_n, H], dtype `torch.float32`.
        context_lens: Tensor of shape [B], dtype int32; effective context length
            for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        schedule_metadata: Returned by `get_paged_mqa_logits_metadata`;
            used to distribute work across SMs.
        max_model_len: Maximum sequence length used to size the logits output.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        `torch.float32`.
    """

    force_triton = (os.environ.get('PAGED_MQA_TRITON','1')=='1')

    block_size = kv_cache_fp8.shape[1]
    logger.info_once(f"rocm_fp8_paged_mqa_logits: KV cache "
                     f"block size {block_size}, force_triton {force_triton}")
    # As of 2026/04/01 (commit 381129b0), deepgemm_fp8_paged_mqa_logits_stage1
    # (the aiter triton path) triggers an obscure gluon compiler error when
    # block_size=16.  stage1 correctly handles the kv_cache packed format
    # (splitting [num_blocks, block_size, 1, D+4] into fp8 data + float32 scale),
    # so format is not the concern — the gluon error is block-size-specific.
    # Our triton kernel is confirmed correct at all block sizes and faster than
    # stage1 in all cases, so always use it unless a specific block size is known
    # to work with stage1 and batch > 1 makes triton less efficient.
    if block_size != 1 or force_triton:
        return fp8_paged_mqa_logits_triton(
            q_fp8, kv_cache_fp8, weights, context_lens, block_tables, max_model_len
        )

    from vllm._aiter_ops import rocm_aiter_ops

    aiter_paged_mqa_logits_module = None
    if rocm_aiter_ops.is_enabled():
        aiter_paged_mqa_logits_module = paged_mqa_logits_module()

    if aiter_paged_mqa_logits_module is not None:
        deepgemm_fp8_paged_mqa_logits_stage1 = (
            aiter_paged_mqa_logits_module.deepgemm_fp8_paged_mqa_logits_stage1
        )
        batch_size, next_n, heads, _ = q_fp8.shape
        out_qk = torch.full(
            (heads, batch_size * next_n, max_model_len),
            float("-inf"),
            device="cuda",
            dtype=torch.float32,
        )
        ChunkQ = 64
        while (heads % ChunkQ):
           ChunkQ = ChunkQ // 2
        deepgemm_fp8_paged_mqa_logits_stage1(
            q_fp8,
            kv_cache_fp8,
            weights,
            out_qk,
            context_lens,
            block_tables,
            max_model_len,
            ChunkQ
        )
        return out_qk.sum(dim=0)
    else:
        return fp8_paged_mqa_logits_triton(
            q_fp8, kv_cache_fp8, weights, context_lens, block_tables, max_model_len
        )


# Take from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py#L84
def fp8_mqa_logits_torch(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    k_fp8, scale = kv
    seq_len_kv = k_fp8.shape[0]
    k = (k_fp8.to(torch.float32) * scale).to(torch.bfloat16)
    q = q.to(torch.bfloat16)

    mask_lo = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(0, seq_len_kv, device="cuda")[None, :] < cu_seqlen_ke[:, None]
    )
    mask = mask_lo & mask_hi

    score = torch.einsum("mhd,nd->hmn", q, k).float()
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))

    return logits


@functools.lru_cache
def mqa_logits_module():
    mqa_logits_module_path = None
    if find_spec("aiter.ops.triton.fp8_mqa_logits") is not None:
        mqa_logits_module_path = "aiter.ops.triton.fp8_mqa_logits"
    elif find_spec("aiter.ops.triton.attention.fp8_mqa_logits") is not None:
        mqa_logits_module_path = "aiter.ops.triton.attention.fp8_mqa_logits"

    if mqa_logits_module_path is not None:
        try:
            module = importlib.import_module(mqa_logits_module_path)
            return module
        except ImportError:
            return None
    return None




@triton.jit
def _fp8_mqa_logits_non_aiter_kernel(
    q_ptr,        # [M, H, D] fp8 — contiguous
    k_ptr,        # [N, D] fp8 — contiguous
    scale_ptr,    # [N] float32
    weights_ptr,  # [M, H] float32 — contiguous
    cu_ks_ptr,    # [M] int32 — valid KV start per query (inclusive)
    cu_ke_ptr,    # [M] int32 — valid KV end per query (exclusive)
    logits_ptr,   # [M, N] float32 — contiguous
    N,
    q_stride_m: tl.int64,
    k_stride_n: tl.int64,
    logits_stride_m: tl.int64,
    SplitKV,      # KV splits per query; grid.y — ceil(num_SMs / M) for full occupancy
    BLOCK_N: tl.constexpr,
    H: tl.constexpr,         # number of query heads — must be ≥ 16 and power of 2
    D: tl.constexpr,         # head dim — padded to next power of 2 by caller
    D_actual: tl.constexpr,  # true head dim before padding; used for strides and masking
):
    # Reverse query order: largest windows (later queries in causal context) run first,
    # reducing tail effect when valid windows vary in length.
    m  = tl.num_programs(0) - tl.program_id(0) - 1
    sv = tl.program_id(1)
    tl.assume(m >= 0)

    tl.assume(q_stride_m > 0)
    tl.assume(k_stride_n > 0)
    tl.assume(logits_stride_m > 0)

    h_offs = tl.arange(0, H)
    d_offs = tl.arange(0, D)
    # Mask padding lanes when D_actual is not a power-of-2 (zero-overhead when D==D_actual).
    d_mask = d_offs < D_actual

    ks = tl.load(cu_ks_ptr + m)
    ke = tl.load(cu_ke_ptr + m)
    ks = tl.maximum(ks, 0)
    ke = tl.minimum(ke, N)

    window = ke - ks
    if window <= 0:
        return

    # Split [ks, ke) across SplitKV programs on grid.y; each owns a contiguous slice.
    # No reduction needed — each logits[m, n] is written by exactly one program.
    chunk     = tl.cdiv(window, SplitKV)
    sv_ks     = ks + sv * chunk
    sv_ke     = tl.minimum(sv_ks + chunk, ke)
    sv_window = sv_ke - sv_ks
    if sv_window <= 0:
        return

    # Load q[m, :, :] → [H, D] fp8 once; lives in registers for the entire KV loop.
    # Use D_actual as the head stride (real layout), mask padding cols with d_mask.
    q_blk = tl.load(
        q_ptr + m * q_stride_m + h_offs[:, None] * D_actual + d_offs[None, :],
        mask=d_mask[None, :],
        other=0.0,
        cache_modifier=".cg",
    )  # [H, D] fp8

    w = tl.load(
        weights_ptr + m * H + h_offs,
        cache_modifier=".cg",
    ).to(tl.float32)  # [H]

    full_tiles  = sv_window // BLOCK_N * BLOCK_N
    # kv_offs drives both k and logits addressing; base+arange keeps pointer patterns
    # simple for Triton's axis-info analysis inside the num_stages=2 pipelined loop.
    kv_offs     = sv_ks + tl.arange(0, BLOCK_N)
    logits_base = logits_ptr + m * logits_stride_m + sv_ks + tl.arange(0, BLOCK_N)

    # Hot loop: full BLOCK_N tiles, no masking.
    # Software pipelining is controlled by num_stages at the kernel launch level.
    for _ in tl.range(0, full_tiles, BLOCK_N):
        # k layout is [N, D_actual]; use D_actual as row stride, mask padding cols.
        k_blk = tl.load(
            k_ptr + d_offs[:, None] + kv_offs[None, :] * k_stride_n,
            mask=d_mask[:, None],
            other=0.0,
        )  # [D, BLOCK_N] fp8

        scale  = tl.load(scale_ptr + kv_offs)
        scores = tl.dot(q_blk, k_blk, input_precision="ieee", out_dtype=tl.float32)
        scores = scores * scale[None, :]
        scores = tl.maximum(scores, 0.0) * w[:, None]
        accum  = tl.sum(scores, axis=0)
        tl.store(logits_base, accum)

        kv_offs     += BLOCK_N
        logits_base += BLOCK_N

    # Tail tile: remainder of this split's window.
    tail_mask = kv_offs < sv_ke
    k_blk = tl.load(
        k_ptr + d_offs[:, None] + kv_offs[None, :] * k_stride_n,
        mask=d_mask[:, None] & tail_mask[None, :],
        other=0.0,
    )  # [D, BLOCK_N] fp8

    scale  = tl.load(scale_ptr + kv_offs, mask=tail_mask, other=1.0)
    scores = tl.dot(q_blk, k_blk, input_precision="ieee", out_dtype=tl.float32)
    scores = scores * scale[None, :]
    scores = tl.maximum(scores, 0.0) * w[:, None]
    accum  = tl.sum(scores, axis=0)
    tl.store(logits_base, accum, mask=tail_mask)


def fp8_mqa_logits_triton(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Triton implementation of fp8_mqa_logits_torch."""
    k, scale = kv
    # scale arrives as [N, 1] float32 (4-byte view of uint8 k_scale buffer)
    scale = scale.reshape(-1)

    M, H, D = q.shape
    N = k.shape[0]

    logger.info_once(f"fp8_mqa_logits_triton: q.shape {q.shape}, k.shape {k.shape}")

    logits = torch.full((M, N), float("-inf"), device=q.device, dtype=torch.float32)

    BLOCK_N = 64
    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    # grid.x: at most num_sms programs; when M > num_sms each program strides over
    # multiple queries (1 wave).  When M ≤ num_sms: grid.x = M, 1 program per query.
    # grid.y: extra KV splits to fill unused CUs when M is small.
    SplitKV = max(1, num_sms // M)

    # matrix_instr_nonkdim selects the MFMA tile size on CDNA3 (gfx942/MI325).
    # 32 for large M (better utilization), 16 for small M (finer granularity).
    matrix_instr_nonkdim = 32 if M > 1024 else 16

    _fp8_mqa_logits_non_aiter_kernel[(M, SplitKV)](
        q, k, scale, weights, cu_seqlen_ks, cu_seqlen_ke, logits,
        N,
        q.stride(0),
        k.stride(0),
        logits.stride(0),
        SplitKV,
        BLOCK_N=BLOCK_N,
        H=H,
        D=triton.next_power_of_2(D),
        D_actual=D,
        num_warps=4,
        num_stages=2,
        waves_per_eu=2,
        matrix_instr_nonkdim=matrix_instr_nonkdim,
    )
    return logits

def rocm_fp8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """

    if os.environ.get('MQA_TRITON','0')=='1':
        return fp8_mqa_logits_triton(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)

    # TODO(ganyi): Temporarily workaround, will remove the module check and reference
    # path after aiter merge this kernel into main
    from vllm._aiter_ops import rocm_aiter_ops

    aiter_mqa_logits_module = None
    if rocm_aiter_ops.is_enabled():
        aiter_mqa_logits_module = mqa_logits_module()

    if aiter_mqa_logits_module is not None:
        fp8_mqa_logits = aiter_mqa_logits_module.fp8_mqa_logits
        k_fp8, scale = kv
        logger.info_once(f"fp8_mqa_logits: q.shape {q.shape}, kv.shape {k_fp8.shape}")
        return fp8_mqa_logits(q, k_fp8, scale, weights, cu_seqlen_ks, cu_seqlen_ke)
    else:
        logger.info_once(f"fp8_mqa_logits_torch: q.shape {q.shape}, kv.shape {kv.shape}")
        return fp8_mqa_logits_torch(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke)


def rocm_aiter_sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
) -> torch.Tensor:
    # profile run
    # NOTE(Chen): create the max possible flattened_kv. So that
    # profile_run can get correct memory usage.
    _flattened_kv = torch.empty(
        [total_seq_lens, head_dim + 4], device=k.device, dtype=torch.uint8
    )
    fp8_dtype = current_platform.fp8_dtype()
    _k_fp8 = _flattened_kv[..., :head_dim].view(fp8_dtype).contiguous()
    _k_scale = _flattened_kv[..., head_dim:].view(torch.float32).contiguous()
    return topk_indices_buffer


def rocm_aiter_sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        return rocm_aiter_sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_fp8,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
        )
    layer_attn_metadata = attn_metadata[k_cache_prefix]
    assert isinstance(layer_attn_metadata, DeepseekV32IndexerMetadata)
    assert topk_indices_buffer is not None
    assert scale_fmt is not None
    slot_mapping = layer_attn_metadata.slot_mapping
    has_decode = layer_attn_metadata.num_decodes > 0
    has_prefill = layer_attn_metadata.num_prefills > 0
    num_decode_tokens = layer_attn_metadata.num_decode_tokens

    ops.indexer_k_quant_and_cache(
        k,
        kv_cache,
        slot_mapping,
        quant_block_size,
        scale_fmt,
    )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = layer_attn_metadata.prefill
        assert prefill_metadata is not None
        for chunk in prefill_metadata.chunks:
            k_fp8 = torch.empty(
                [chunk.total_seq_lens, head_dim],
                device=k.device,
                dtype=fp8_dtype,
            )
            k_scale = torch.empty(
                [chunk.total_seq_lens, 4],
                device=k.device,
                dtype=torch.uint8,
            )

            ops.cp_gather_indexer_k_quant_cache(
                kv_cache,
                k_fp8,
                k_scale,
                chunk.block_table,
                chunk.cu_seq_lens,
            )

            logits = rocm_fp8_mqa_logits(
                q_fp8[chunk.token_start : chunk.token_end],
                (k_fp8, k_scale.view(torch.float32)),
                weights[chunk.token_start : chunk.token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
            )
            num_rows = logits.shape[0]
            assert topk_tokens == 2048, "top_k_per_row assumes size 2048"
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
            torch.ops._C.top_k_per_row_prefill(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )

    if has_decode:
        decode_metadata = layer_attn_metadata.decode
        assert decode_metadata is not None
        # kv_cache size requirement [num_block, block_size, n_head, head_dim],
        # we only have [num_block, block_size, head_dim],
        kv_cache = kv_cache.unsqueeze(-2)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens)
            padded_q_fp8_decode_tokens = pack_seq_triton(
                q_fp8[:num_decode_tokens], decode_lens
            )
        else:
            padded_q_fp8_decode_tokens = q_fp8[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_fp8.shape[1:]
            )
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_fp8_decode_tokens.shape[0]
        next_n = padded_q_fp8_decode_tokens.shape[1]
        assert batch_size == decode_metadata.seq_lens.shape[0]
        num_padded_tokens = batch_size * next_n

        logits = rocm_fp8_paged_mqa_logits(
            padded_q_fp8_decode_tokens,
            kv_cache,
            weights[:num_padded_tokens],
            decode_metadata.seq_lens,
            decode_metadata.block_table,
            decode_metadata.schedule_metadata,
            max_model_len=max_model_len,
        )

        num_rows = logits.shape[0]
        assert topk_tokens == 2048, "top_k_per_row assumes size 2048"
        topk_indices = topk_indices_buffer[:num_decode_tokens, :topk_tokens]
        torch.ops._C.top_k_per_row_decode(
            logits,
            next_n,
            decode_metadata.seq_lens,
            topk_indices,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            topk_tokens,
        )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[:num_decode_tokens, : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer
