# Copyright (c) 2026 BAAI. All rights reserved.
"""Sunrise MLA attention backend for decode MQA and prefill MHA."""

from __future__ import annotations

from typing import Optional, Union

import torch

from flag_gems import concat_and_cache_mla, flash_attn_varlen_func, flash_mla
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
)
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionLayer, AttentionType

logger = init_logger(__name__)


def torch_gather_and_maybe_dequant_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    token_to_seq: torch.Tensor,
    num_tokens: int,
    kv_cache_dtype: str,
    scale: torch.Tensor,
    seq_starts: torch.Tensor | None = None,
) -> None:
    """BF16/FP16 PyTorch fallback for ``_C_cache_ops.gather_and_maybe_dequant_cache``.

    Matches vLLM ``cache_kernels.cu::gather_and_maybe_dequant_cache`` for the
    non-FP8 path used by MLA chunked-prefill / prefix-cache context gather.
    """
    if kv_cache_dtype.startswith("fp8"):
        raise NotImplementedError(
            "Sunrise MLA: FP8 gather_and_maybe_dequant_cache not supported"
        )

    if num_tokens <= 0:
        return

    # [num_blocks, block_size, D] or [num_blocks, block_size, 1, D]
    if src_cache.ndim == 4:
        src = src_cache.squeeze(2)
    else:
        src = src_cache
    assert src.ndim == 3, f"unexpected MLA cache shape {tuple(src_cache.shape)}"
    block_size = src.size(1)
    hidden = src.size(-1)
    flat = src.reshape(-1, hidden)

    token_ids = torch.arange(num_tokens, device=src.device, dtype=torch.long)
    batch_ids = token_to_seq[:num_tokens].to(dtype=torch.long)
    batch_start = cu_seq_lens[batch_ids].to(dtype=torch.long)
    batch_offset = token_ids - batch_start
    if seq_starts is not None:
        batch_offset = batch_offset + seq_starts[batch_ids].to(dtype=torch.long)

    block_table_ids = torch.div(batch_offset, block_size, rounding_mode="floor")
    slot_ids = torch.remainder(batch_offset, block_size)
    # block_table: [batch, max_blocks]; stride may exceed size(1)
    block_ids = block_table[batch_ids, block_table_ids].to(dtype=torch.long)
    flat_idx = block_ids * block_size + slot_ids
    dst[:num_tokens].copy_(flat[flat_idx])


def _ensure_mla_flash_attn_importable() -> None:
    """MLACommonImpl.__init__ requires a module-level FA symbol.

    On Sunrise/PTPU ``vllm_flash_attn`` is unavailable, so inject FlagGems'
    varlen FA before ``super().__init__`` so the common FA prefill branch
    can finish setup. Prefill still goes through our
    ``_flash_attn_varlen_diff_headdims`` override.
    """
    import vllm.model_executor.layers.attention.mla_attention as mla_mod

    if mla_mod.flash_attn_varlen_func is None:
        mla_mod.flash_attn_varlen_func = flash_attn_varlen_func
        logger.info_once(
            "Sunrise MLA: injected FlagGems flash_attn_varlen_func into "
            "mla_attention (vllm_flash_attn unavailable on PTPU)."
        )


class SunriseMLABackend(MLACommonBackend):
    """Sunrise-owned MLA backend entry for vendor dispatch."""

    @staticmethod
    def get_name() -> str:
        return "SUNRISE_MLA"

    @staticmethod
    def get_impl_cls() -> type["SunriseMLAImpl"]:
        return SunriseMLAImpl


class SunriseMLAImpl(MLACommonImpl[MLACommonMetadata]):
    # Sunrise flash_mla does not emit LSE; DCP>1 would need a kernel upgrade.
    can_return_lse_for_decode: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float],
        attn_type: str,
        kv_sharing_target_layer_name: Optional[str],
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
        _ensure_mla_flash_attn_importable()
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "SunriseMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for SunriseMLAImpl"
            )

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "Sunrise MLA with FP8 KV cache not yet supported"
            )

        # Prefer FlagGems FA for prefill (diff head dims + PTPU). Parent may
        # have set CUDA FA kwargs; our override ignores self.flash_attn_varlen_func.
        self._pad_v = True

    def _flash_attn_varlen_diff_headdims(
        self, q, k, v, return_softmax_lse=False, softmax_scale=None, **kwargs
    ):
        maybe_padded_v = v
        if self._pad_v:
            maybe_padded_v = torch.nn.functional.pad(
                v, [0, q.shape[-1] - v.shape[-1]], value=0
            )

        kwargs["return_softmax_lse"] = return_softmax_lse
        attn_out = flash_attn_varlen_func(
            q=q,
            k=k,
            v=maybe_padded_v,
            softmax_scale=softmax_scale,
            **kwargs,
        )

        lse = None
        if isinstance(attn_out, tuple):
            attn_out, lse = attn_out[0], attn_out[1]

        if return_softmax_lse:
            return attn_out, lse
        return attn_out

    def do_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
    ) -> None:
        if kv_cache.numel() == 0:
            return
        # PTPU builds do not ship ``torch.ops._C_cache_ops.concat_and_cache_mla``.
        concat_and_cache_mla(
            kv_c_normed,
            k_pe.squeeze(1),
            kv_cache,
            slot_mapping.flatten(),
            kv_cache_dtype,
            k_scale,
        )

    def forward_mqa(
        self,
        q: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Decode-path MQA over ``kv_c ‖ k_pe`` cache via Sunrise ``flash_mla``.

        Expected layouts (GLM-4.7-Flash):
        * ``q``: ``[B, H, kv_lora + rope]`` (= ``[B, H, 576]``) or
          ``(q_nope, q_pe)`` with dims 512 / 64
        * ``kv_c_and_k_pe_cache``: ``[num_blocks, block_size, 576]``
        * out: ``[B, H, kv_lora_rank]`` (= ``[B, H, 512]``)
        """
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError("FP8 Sunrise MLA not yet supported")

        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        assert isinstance(q, torch.Tensor)
        # Common path: [B, H, D]. flash_mla wants [B, s_q, H, D].
        if q.ndim == 3:
            b, h, d = q.shape
            s_q = 1
            q4 = q.unsqueeze(1)
        elif q.ndim == 4:
            b, s_q, h, d = q.shape
            q4 = q
        else:
            raise ValueError(f"SunriseMLAImpl.forward_mqa: unexpected q.ndim={q.ndim}")

        head_dim_v = self.kv_lora_rank
        if d <= head_dim_v:
            raise ValueError(
                f"MLA decode expects D=kv_lora+rope > kv_lora; got D={d}, "
                f"kv_lora_rank={head_dim_v}"
            )

        # Cache is [num_blocks, block_size, D] (or with an extra head dim).
        kv_cache = kv_c_and_k_pe_cache
        if kv_cache.ndim == 4:
            # [num_blocks, block_size, 1, D] → treat as 3D for page size.
            page_size = kv_cache.size(1)
        elif kv_cache.ndim == 3:
            page_size = kv_cache.size(1)
        else:
            raise ValueError(
                f"Unexpected MLA kv cache ndim={kv_cache.ndim}, "
                f"shape={tuple(kv_cache.shape)}"
            )

        # flash_mla returns [B, s_q, H, dv]; common layer wants [B, H, dv]
        # (s_q==1 for pure decode). Pass vLLM softmax scale when supported.
        # Current Sunrise FlagGems ``flash_mla`` hardcodes 1/sqrt(d); optional
        # ``sm_scale`` is accepted when newer FlagGems exposes it.
        flash_mla_kwargs = {}
        try:
            import inspect

            if "sm_scale" in inspect.signature(flash_mla).parameters:
                flash_mla_kwargs["sm_scale"] = self.scale
        except (TypeError, ValueError):
            pass
        o = flash_mla(
            q4,
            attn_metadata.decode.block_table,
            kv_cache,
            None,
            page_size,
            b,
            s_q,
            attn_metadata.decode.seq_lens,
            h,
            None,
            d,
            head_dim_v,
            True,
            **flash_mla_kwargs,
        )
        if s_q == 1:
            o = o.squeeze(1)
        else:
            # Spec/multi-token decode: flatten query dim into batch like FA MLA.
            o = o.reshape(b * s_q, h, head_dim_v)

        return o, None
