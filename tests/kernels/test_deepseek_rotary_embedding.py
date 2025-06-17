# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for Triton-fused DeepseekScalingRotaryEmbedding
"""
import pytest
import torch

from vllm.model_executor.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding)
from vllm.platforms import current_platform


def test_deepseek_rotary_embedding():
    device = torch.device("cuda:0")
    current_platform.seed_everything(0)
    torch.set_default_device("cuda:0")
    batch_size = 10
    base = 10000
    num_heads = 8
    max_position = 4096
    is_neox_style = False
    rotary_dim = 32
    head_size = 64
    scaling_factor = 40.0

    rot = DeepseekScalingRotaryEmbedding(head_size,
                                         rotary_dim,
                                         max_position,
                                         base,
                                         is_neox_style,
                                         scaling_factor,
                                         torch.float32,
                                         reference=False).to(device)

    rot_ref = DeepseekScalingRotaryEmbedding(head_size,
                                             rotary_dim,
                                             max_position,
                                             base,
                                             is_neox_style,
                                             scaling_factor,
                                             torch.float32,
                                             reference=True).to(device)

    positions = torch.randint(0, max_position, (batch_size, ), device=device)
    # query is [batch, num_heads, head_size]
    # key is [batch, 1, head_size]
    # cos_sin is [batch, head_size]
    query = torch.randn(batch_size,
                        num_heads,
                        head_size,
                        dtype=torch.float32,
                        device=device)
    key = torch.randn(batch_size,
                      1,
                      head_size,
                      dtype=torch.float32,
                      device=device)
    ref_query, ref_key = rot_ref.forward(positions, query, key)
    out_query, out_key = rot.forward(positions, query, key)
    torch.testing.assert_close(out_key.cpu(),
                               ref_key.cpu(),
                               atol=1e-4,
                               rtol=1e-4)
    torch.testing.assert_close(out_query.cpu(),
                               ref_query.cpu(),
                               atol=1e-4,
                               rtol=1e-4)
