"""
api/routes/health.py
====================
Liveness, readiness, and metrics endpoints.

  GET /health   — always 200 (liveness probe)
  GET /ready    — 200 only after pipeline is loaded (readiness probe)
  GET /metrics  — JSON snapshot of VRAM, job counters, concurrency state
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Response, status

from api.models import HealthResponse, MetricsResponse, ReadyResponse
from api.pipeline_manager import pipeline_manager
from api.utils.memory import log_vram_usage

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health & Metrics"])


# ---------------------------------------------------------------------------
# GET /health  — Kubernetes liveness probe
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe — always returns 200 if the process is alive.",
)
async def health() -> HealthResponse:
    import torch

    return HealthResponse(
        status="ok",
        pipeline_ready=pipeline_manager.ready,
        gpu_available=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# GET /ready  — Kubernetes readiness probe
# ---------------------------------------------------------------------------

@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness probe — returns 503 until the pipeline has finished loading.",
)
async def ready(response: Response) -> ReadyResponse:
    if pipeline_manager.ready:
        return ReadyResponse(status="ready")

    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadyResponse(status="not_ready", detail="Pipeline is still loading.")


# ---------------------------------------------------------------------------
# GET /metrics  — operational JSON metrics
# ---------------------------------------------------------------------------

@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="JSON snapshot of GPU memory, job counters, and concurrency state.",
)
async def metrics() -> MetricsResponse:
    import torch

    vram: dict = {}
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        free = total - reserved
        vram = {
            "vram_allocated_gb": round(allocated, 2),
            "vram_reserved_gb": round(reserved, 2),
            "vram_free_gb": round(free, 2),
            "vram_total_gb": round(total, 2),
        }
    else:
        vram = {
            "vram_allocated_gb": 0.0,
            "vram_reserved_gb": 0.0,
            "vram_free_gb": 0.0,
            "vram_total_gb": 0.0,
        }

    counts = pipeline_manager.job_count
    semaphore_value = pipeline_manager.semaphore_free

    from api.config import settings

    return MetricsResponse(
        pipeline_ready=pipeline_manager.ready,
        gpu_available=torch.cuda.is_available(),
        **vram,
        jobs_queued=counts["queued"],
        jobs_processing=counts["processing"],
        jobs_completed=counts["completed"],
        jobs_failed=counts["failed"],
        avg_generation_time_s=pipeline_manager.avg_generation_time,
        concurrent_slots_total=settings.max_concurrent_jobs,
        concurrent_slots_free=semaphore_value,
    )
