# Copyright (c) 2026 BAAI. All rights reserved.

import ctypes
import importlib
import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)
_patches_applied = False


def apply_sunrise_patches():
    """Apply Sunrise/PTPU patches that must run before model construction."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    patch_flagcx_stream_adapter()
    patch_distributed_runtime()
    patch_op_cls()
    patch_accelerator_empty_cache()
    patch_memory_profiling_for_plain_allocator()
    patch_fused_topk_bias_router()


def patch_flagcx_stream_adapter():
    """Patch FlagCX wrapper to accept PTPU streams."""
    try:
        from vllm.platforms import current_platform

        if getattr(current_platform, "device_type", None) != "ptpu":
            return

        flagcx_path = os.getenv("FLAGCX_PATH")
        if flagcx_path and os.path.isdir(flagcx_path) and flagcx_path not in sys.path:
            sys.path.append(flagcx_path)

        flagcx_wrapper = importlib.import_module("plugin.interservice.flagcx_wrapper")
        FLAGCXLibrary = flagcx_wrapper.FLAGCXLibrary
        flagcxStream_t = flagcx_wrapper.flagcxStream_t

        if getattr(FLAGCXLibrary, "_sunrise_stream_patch_applied", False):
            return

        def _to_void_p(raw_stream_ptr):
            if isinstance(raw_stream_ptr, ctypes.c_void_p):
                return raw_stream_ptr
            if raw_stream_ptr is None:
                raise ValueError("Stream pointer is None.")
            return ctypes.c_void_p(int(raw_stream_ptr))

        def _extract_raw_stream_ptr(old_stream):
            if isinstance(old_stream, (int, ctypes.c_void_p)):
                return old_stream

            for attr in ("ptpu_stream", "cuda_stream"):
                stream_ptr = getattr(old_stream, attr, None)
                if stream_ptr is not None:
                    return stream_ptr

            stream_fn = getattr(old_stream, "stream", None)
            if callable(stream_fn):
                stream_ptr = stream_fn()
                if stream_ptr is not None:
                    return stream_ptr

            raise AttributeError(
                "Unsupported stream object: expected a raw pointer or one of "
                "`ptpu_stream`, `cuda_stream`, or callable `stream()`."
            )

        def _adaptor_stream_copy(self, old_stream):
            new_stream = flagcxStream_t()
            raw_stream_ptr = _extract_raw_stream_ptr(old_stream)
            self.FLAGCX_CHECK(
                self.handler.contents.devHandle.contents.streamCopy(
                    ctypes.byref(new_stream), _to_void_p(raw_stream_ptr)
                )
            )
            return new_stream

        FLAGCXLibrary.adaptor_stream_copy = _adaptor_stream_copy
        FLAGCXLibrary._sunrise_stream_patch_applied = True
        logger.info("Patched FlagCX stream adapter for Sunrise/PTPU")
    except Exception as e:
        logger.warning("Failed to patch FlagCX stream adapter for Sunrise: %s", e)


def patch_distributed_runtime():
    """Keep FlagCX path while mapping torch ProcessGroup backend to pccl."""
    try:
        from vllm.platforms import current_platform
        from vllm.distributed.device_communicators.base_device_communicator import (
            DeviceCommunicatorBase,
        )
        from vllm_fl.distributed.communicator import CommunicatorFL
        from vllm_fl.worker import worker as worker_mod

        platform_cls = (
            current_platform
            if isinstance(current_platform, type)
            else current_platform.__class__
        )

        if getattr(current_platform, "device_type", None) != "ptpu":
            return

        # Preserve the original Sunrise/FlagCX communicator selection logic.
        platform_cls.dist_backend = "flagcx"
        current_platform.dist_backend = "flagcx"

        if not getattr(CommunicatorFL, "_sunrise_all_gather_patched", False):
            def _all_gather(self, input_: torch.Tensor, dim: int = -1):
                world_size = self.world_size
                if world_size == 1:
                    return input_

                assert -input_.dim() <= dim < input_.dim(), (
                    f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
                )
                if dim < 0:
                    dim += input_.dim()

                pyflagcx_comm = getattr(self, "pyflagcx_comm", None)
                if pyflagcx_comm is None or pyflagcx_comm.disabled:
                    return DeviceCommunicatorBase.all_gather(self, input_, dim)

                output_tensor = self.all_gatherv(input_, dim=0, sizes=None)
                if dim == 0:
                    return output_tensor

                input_size = input_.size()
                output_tensor = output_tensor.reshape((world_size,) + input_size)
                output_tensor = output_tensor.movedim(0, dim)
                output_tensor = output_tensor.reshape(
                    input_size[:dim]
                    + (world_size * input_size[dim],)
                    + input_size[dim + 1 :]
                )
                return output_tensor

            CommunicatorFL.all_gather = _all_gather
            CommunicatorFL._sunrise_all_gather_patched = True

        init_dist = worker_mod.init_worker_distributed_environment
        if not getattr(init_dist, "_sunrise_backend_patched", False):
            def _init_worker_distributed_environment(
                vllm_config,
                rank,
                distributed_init_method=None,
                local_rank=-1,
                backend="nccl",
            ):
                backend_for_pg = backend
                if backend in ("flagcx", "nccl"):
                    backend_for_pg = "pccl"
                return init_dist(
                    vllm_config,
                    rank,
                    distributed_init_method=distributed_init_method,
                    local_rank=local_rank,
                    backend=backend_for_pg,
                )

            _init_worker_distributed_environment._sunrise_backend_patched = True
            worker_mod.init_worker_distributed_environment = (
                _init_worker_distributed_environment
            )

        logger.info(
            "Configured Sunrise/PTPU to use FlagCX communicator with pccl PGs"
        )
    except Exception as e:
        logger.warning("Failed to configure Sunrise distributed runtime: %s", e)


def patch_op_cls():
    """Register Sunrise replacements for upstream custom ops."""
    try:
        from vllm.model_executor.custom_op import PluggableLayer

        from .impl.vocab_parallel_embedding import SunriseVocabParallelEmbedding

        PluggableLayer.register_oot(
            _decorated_layer_cls=SunriseVocabParallelEmbedding,
            name="VocabParallelEmbedding",
        )
        logger.info("Patched VocabParallelEmbedding for Sunrise/PTPU")
    except Exception as e:
        logger.warning("Failed to patch VocabParallelEmbedding for Sunrise: %s", e)


def patch_accelerator_empty_cache():
    """Redirect torch.accelerator.empty_cache() to torch.ptpu.empty_cache().

    torch.accelerator.empty_cache() requires a DeviceAllocator interface that
    PTPU does not implement. torch.ptpu.empty_cache() works correctly instead.
    """
    try:
        import torch.accelerator as _accel

        if getattr(_accel, "_sunrise_empty_cache_patched", False):
            return
        _accel.empty_cache = torch.ptpu.empty_cache
        _accel._sunrise_empty_cache_patched = True
        logger.info("Patched torch.accelerator.empty_cache for Sunrise/PTPU")
    except Exception as e:
        logger.warning("Failed to patch torch.accelerator.empty_cache: %s", e)


def patch_fused_topk_bias_router():
    """Replace CUDA-only ``vllm_topk_{sigmoid,softmax}`` with a native
    implementation on Sunrise/PTPU.
    """

    try:
        from vllm.platforms import current_platform

        if getattr(current_platform, "device_type", None) != "ptpu":
            return

        import vllm.envs as envs
        from vllm.model_executor.layers.fused_moe.router import (
            fused_topk_bias_router as router_mod,
        )

        if getattr(router_mod, "_sunrise_topk_patched", False):
            return

        def _native_topk_with_bias(
            topk_weights: torch.Tensor,
            topk_indices: torch.Tensor,
            token_expert_indices: torch.Tensor,
            gating_output: torch.Tensor,
            renormalize: bool,
            e_score_correction_bias: torch.Tensor | None,
            scoring_func: str,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if scoring_func == "softmax":
                scores = gating_output.softmax(dim=-1)
            elif scoring_func == "sigmoid":
                scores = gating_output.sigmoid()
            else:
                raise ValueError(
                    f"Unsupported scoring function: {scoring_func}"
                )

            if e_score_correction_bias is not None:
                scores_for_choice = scores + e_score_correction_bias.to(
                    scores.dtype
                ).unsqueeze(0)
            else:
                scores_for_choice = scores

            topk = topk_weights.shape[-1]
            use_sorted = getattr(envs, "VLLM_BATCH_INVARIANT", False)
            chosen_indices = torch.topk(
                scores_for_choice, k=topk, dim=-1, sorted=use_sorted
            ).indices
            chosen_weights = scores.gather(dim=-1, index=chosen_indices)
            if renormalize:
                chosen_weights = chosen_weights / chosen_weights.sum(
                    dim=-1, keepdim=True
                )

            topk_weights.copy_(chosen_weights.to(topk_weights.dtype))
            topk_indices.copy_(chosen_indices.to(topk_indices.dtype))
            return topk_weights, topk_indices

        def _patched_vllm_topk_sigmoid(
            topk_weights, topk_indices, token_expert_indices,
            gating_output, renormalize=False, e_score_correction_bias=None,
        ):
            return _native_topk_with_bias(
                topk_weights, topk_indices, token_expert_indices,
                gating_output, renormalize, e_score_correction_bias,
                scoring_func="sigmoid",
            )

        def _patched_vllm_topk_softmax(
            topk_weights, topk_indices, token_expert_indices,
            gating_output, renormalize=False, e_score_correction_bias=None,
        ):
            return _native_topk_with_bias(
                topk_weights, topk_indices, token_expert_indices,
                gating_output, renormalize, e_score_correction_bias,
                scoring_func="softmax",
            )

        router_mod.vllm_topk_sigmoid = _patched_vllm_topk_sigmoid
        router_mod.vllm_topk_softmax = _patched_vllm_topk_softmax
        router_mod._sunrise_topk_patched = True

        logger.info(
            "Patched fused_topk_bias_router to use native topk on PTPU"
        )
    except Exception as e:
        logger.warning(
            "Failed to patch fused_topk_bias_router for Sunrise: %s", e
        )


def patch_memory_profiling_for_plain_allocator():
    """Fix KV-cache OOM caused by weights being double-counted on PTPU.

    ``torch_ptpu`` ships a "plain"-style device allocator that does not register
    with PyTorch's caching allocator, so ``torch.memory_reserved()`` is always
    ~0 on PTPU. The shared :func:`memory_profiling_fl` computes
    ``non_torch = cuda_used - torch_reserved``; with ``torch_reserved == 0``
    this leaks the model weights into ``non_torch``. The final
    ``non_kv_cache = weights + peak_act + non_torch`` then counts weights
    twice, yielding a phantom OOM in ``determine_available_memory`` even when
    the device has plenty of free memory.
    """
    try:
        from vllm.platforms import current_platform

        if getattr(current_platform, "device_type", None) != "ptpu":
            return
    except Exception as e:
        logger.warning("Skip PTPU memory profiling patch: %s", e)
        return

    try:
        from contextlib import contextmanager

        from vllm_fl.worker import worker as worker_mod

        original = worker_mod.memory_profiling_fl
        if getattr(original, "_sunrise_mem_profile_patched", False):
            return

        gib = float(2**30)
        _logged = {"once": False}

        @contextmanager
        def memory_profiling_fl_ptpu(baseline_snapshot, weights_memory):
            with original(
                baseline_snapshot, weights_memory=weights_memory
            ) as result:
                yield result

            if (
                result.after_profile.torch_memory == 0
                and result.weights_memory > 0
            ):
                actual_used = (
                    result.after_profile.cuda_memory
                    - result.before_create.cuda_memory
                )
                corrected_non_torch = max(0, actual_used - result.weights_memory)
                result.non_torch_increase = corrected_non_torch
                result.non_kv_cache_memory = (
                    corrected_non_torch
                    + result.torch_peak_increase
                    + result.weights_memory
                )

                if not _logged["once"]:
                    _logged["once"] = True
                    logger.info(
                        "PTPU plain-allocator memory accounting fix applied: "
                        "weights=%.2f GiB peak_act=%.2f GiB actual_used=%.2f GiB "
                        "-> non_torch=%.2f GiB non_kv_cache=%.2f GiB",
                        result.weights_memory / gib,
                        result.torch_peak_increase / gib,
                        actual_used / gib,
                        corrected_non_torch / gib,
                        result.non_kv_cache_memory / gib,
                    )

        memory_profiling_fl_ptpu._sunrise_mem_profile_patched = True
        worker_mod.memory_profiling_fl = memory_profiling_fl_ptpu
        logger.info(
            "Patched memory_profiling_fl for PTPU plain allocator "
            "(prevents weights double-counting in KV cache sizing)"
        )
    except Exception as e:
        logger.warning(
            "Failed to patch memory_profiling_fl for Sunrise/PTPU: %s", e
        )
