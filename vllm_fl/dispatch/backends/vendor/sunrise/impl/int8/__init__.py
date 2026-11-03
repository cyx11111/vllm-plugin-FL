# Copyright (c) 2026 BAAI. All rights reserved.

"""Sunrise INT8 kernel implementations for the vLLM-native W8A8 path."""

from . import deepgemm_moe, dense_gemm
from .scaled_int8_quant import dynamic_scaled_int8_quant, scaled_int8_quant
from .scaled_mm import scaled_mm
from .silu_and_mul_quant import silu_and_mul_quant_int8

__all__ = [
    "deepgemm_moe",
    "dense_gemm",
    "dynamic_scaled_int8_quant",
    "scaled_int8_quant",
    "scaled_mm",
    "silu_and_mul_quant_int8",
]
