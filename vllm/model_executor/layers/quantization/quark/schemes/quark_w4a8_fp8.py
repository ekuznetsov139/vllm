from typing import Callable, List, Optional

import torch
from torch.nn import Parameter

from vllm.model_executor.layers.quantization.quark.schemes import QuarkScheme
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    apply_fp4_fp8_linear, cutlass_fp8_supported, normalize_e4m3fn_to_e4m3fnuz,
    requantize_with_max_scale)
from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter,
                                           GroupQuantScaleParameter)
from vllm.platforms import current_platform

__all__ = ["QuarkW4A8Fp8"]

class QuarkW4A8Fp8(QuarkScheme):

    def __init__(self, qscheme: str, is_static_input_scheme: Optional[bool], axis: int, group_size: int):
        self.qscheme = qscheme
        self.is_static_input_scheme = is_static_input_scheme
        self.cutlass_fp8_supported = cutlass_fp8_supported()
        self.group_scheme = (axis,group_size)

    @classmethod
    def get_min_capability(cls) -> int:
        # lovelace and up (copy&paste from W8A8Fp8)
        return 89

    def process_weights_after_loading(self, layer) -> None:
        # If per tensor, when we have a fused module (e.g. QKV) with per
        # tensor scales (thus N scales being passed to the kernel),
        # requantize so we can always run per tensor
        assert current_platform.is_rocm()
        assert self.qscheme == "per_group"
        weight = layer.weight
        input_scale=layer.input_scale
        if input_scale is not None:
            input_scale = input_scale * 2.0
            layer.input_scale = Parameter(input_scale,
                                          requires_grad=False)

        # We would like to transpose weights (it has to be done either
        # here and only once, or in apply_fp4_fp8_linear() and repeatedly.)
        # But, since they are "really" packed fp4 and not fp8, simply
        # writing .t() would not transpose them correctly.
        # For simplicity, we kick the can down to apply_fp4_fp8_linear().
        #
        #layer.weight = Parameter(weight.t(), requires_grad=False)
        #assert len(weight.shape) == 2
        #if self.group_scheme[0] >= 0:
        #   self.group_scheme = (1-self.group_scheme[0], self.group_scheme[1])
        #else:
        #   self.group_scheme = (1-(self.group_scheme[0]+2), self.group_scheme[1])

        # INPUT SCALE
        if self.is_static_input_scheme:
            layer.input_scale = Parameter(layer.input_scale.max(),
                                          requires_grad=False)
        else:
            layer.input_scale = None

    def create_weights(self, layer: torch.nn.Module,
                       output_partition_sizes: List[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       input_size, output_size,
                       **kwargs):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes

        weight_shape = [output_size_per_partition, input_size_per_partition]
        weight_scale_shape = weight_shape[:]
        weight_scale_shape[self.group_scheme[0]] //= self.group_scheme[1]
        weight_shape[1] //= 2
        # WEIGHT
        weight = ModelWeightParameter(
            data=torch.empty(weight_shape, dtype=torch.uint8),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader)
        layer.register_parameter("weight", weight)

        assert self.qscheme == "per_group"
        weight_scale = GroupQuantScaleParameter(output_dim = 0, input_dim = 1,
            data=torch.empty(weight_scale_shape, dtype=torch.int8),
                                                weight_loader=weight_loader)
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE
        if self.is_static_input_scheme:
            input_scale = PerTensorScaleParameter(data=torch.empty(
                len(output_partition_sizes), dtype=torch.float32),
                                                  weight_loader=weight_loader)
            input_scale[:] = torch.finfo(torch.float32).min
            layer.register_parameter("input_scale", input_scale)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        return apply_fp4_fp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias,
            cutlass_fp8_supported=self.cutlass_fp8_supported,
            use_per_token_if_dynamic=True, 
            group_scheme = self.group_scheme)
