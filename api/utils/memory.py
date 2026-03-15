"""
api/utils/memory.py
===================
Blackwell-optimised CUDA memory helpers for the LTX 2.3 FastAPI server.
"""

from __future__ import annotations

import gc
import logging

logger = logging.getLogger(__name__)


def apply_blackwell_torch_flags() -> None:
    """
    Apply Blackwell / Ampere+ torch optimisation flags.

    Must be called AFTER torch is imported but BEFORE any model is loaded.
    The PYTORCH_CUDA_ALLOC_CONF env var should already be set by config before
    torch is imported; this function handles the remaining runtime flags.
    """
    import torch

    if torch.cuda.is_available():
        # TF32 — gives ~3× speedup on Blackwell matmul with negligible precision loss
        torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # auto-select fastest convolution kernels

        # Flash-attention v2 is auto-selected when available on sm_90+
        # Explicitly enable memory-efficient attention fallback
        try:
            torch.backends.cuda.enable_flash_sdp(True)  # type: ignore[attr-defined]
            torch.backends.cuda.enable_mem_efficient_sdp(True)  # type: ignore[attr-defined]
        except AttributeError:
            pass  # older torch versions don't expose these flags

        logger.info(
            "Blackwell CUDA flags applied: TF32=True, cudnn.benchmark=True, Flash-SDP=True"
        )
    else:
        logger.warning("CUDA not available — Blackwell optimisations skipped.")


def aggressive_cleanup() -> None:
    """
    Run a thorough GPU memory cleanup cycle.

    Call this after every generation job to reclaim VRAM that PyTorch keeps
    in its caching allocator. On 96 GB Blackwell cards this ensures the
    allocator doesn't balloon across concurrent requests.
    """
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def log_vram_usage(label: str = "") -> dict[str, float]:
    """
    Log and return current VRAM statistics.

    Returns a dict with keys:
        allocated_gb  — memory currently held by tensors
        reserved_gb   — memory held by PyTorch caching allocator
        free_gb       — driver-reported free VRAM
        total_gb      — total device VRAM
    """
    import torch

    if not torch.cuda.is_available():
        return {}

    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free = total - reserved

    stats = {
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "free_gb": round(free, 2),
        "total_gb": round(total, 2),
    }

    prefix = f"[{label}] " if label else ""
    logger.info(
        "%sVRAM — allocated: %.2f GB | reserved: %.2f GB | free: %.2f GB / %.2f GB",
        prefix,
        allocated,
        reserved,
        free,
        total,
    )
    return stats


def peak_vram_gb() -> float:
    """Return the peak VRAM allocated since last reset (in GB)."""
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def reset_peak_vram() -> None:
    """Reset the peak VRAM counter (call before each generation job)."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
