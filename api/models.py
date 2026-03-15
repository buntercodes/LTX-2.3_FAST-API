"""
api/models.py
=============
Pydantic v2 request / response schemas for the LTX 2.3 FastAPI server.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Generation request (form-based; images handled separately as UploadFile)
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    """
    Parameters for a single text-to-video (or image-conditioned) generation.

    Resolution constraints (enforced by DistilledPipeline):
      - height and width must be divisible by 64 (two-stage pipeline).
      - num_frames must satisfy: (num_frames - 1) % 8 == 0.
    """

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="Text prompt describing the desired video content.",
    )
    seed: int = Field(default=42, description="Random seed for reproducibility.")
    height: int = Field(
        default=1024,
        ge=256,
        le=2160,
        description="Output video height in pixels. Must be divisible by 64.",
    )
    width: int = Field(
        default=1536,
        ge=256,
        le=3840,
        description="Output video width in pixels. Must be divisible by 64.",
    )
    num_frames: int = Field(
        default=121,
        ge=9,
        le=257,
        description="Number of frames. Must satisfy (num_frames - 1) % 8 == 0.",
    )
    frame_rate: float = Field(
        default=24.0,
        ge=8.0,
        le=60.0,
        description="Output frame rate (fps).",
    )
    enhance_prompt: bool = Field(
        default=False,
        description="Use Gemma to enhance the prompt before encoding.",
    )
    tiling: bool = Field(
        default=True,
        description="Enable VAE tiling to reduce peak VRAM during decode.",
    )

    # Image conditioning strengths
    image_start_strength: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Conditioning strength for the start-frame image.",
    )
    image_end_strength: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Conditioning strength for the end-frame image.",
    )

    @field_validator("height", "width")
    @classmethod
    def _divisible_by_64(cls, v: int) -> int:
        if v % 64 != 0:
            raise ValueError(f"Value {v} must be divisible by 64 for the two-stage distilled pipeline.")
        return v

    @field_validator("num_frames")
    @classmethod
    def _valid_frame_count(cls, v: int) -> int:
        if (v - 1) % 8 != 0:
            raise ValueError(
                f"num_frames={v} is invalid. It must satisfy (num_frames - 1) % 8 == 0 "
                f"(e.g. 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121 …)."
            )
        return v

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Job tracking
# ---------------------------------------------------------------------------

class JobInfo(BaseModel):
    """Internal state stored for each generation job."""
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    request: dict[str, Any] = Field(default_factory=dict)
    video_url: str | None = None
    duration_s: float | None = None
    gpu_mem_peak_gb: float | None = None
    error: str | None = None
    created_at: float = 0.0
    completed_at: float | None = None

    model_config = {"extra": "ignore"}


# ---------------------------------------------------------------------------
# API responses
# ---------------------------------------------------------------------------

class GenerateResponse(BaseModel):
    """Returned immediately by POST /generate."""
    job_id: str
    status: JobStatus
    message: str = "Generation job queued."


class JobStatusResponse(BaseModel):
    """Returned by GET /generate/{job_id}."""
    job_id: str
    status: JobStatus
    video_url: str | None = None
    duration_s: float | None = None
    gpu_mem_peak_gb: float | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    gpu_available: bool


class ReadyResponse(BaseModel):
    status: str
    detail: str | None = None


class MetricsResponse(BaseModel):
    """Prometheus-friendly JSON metrics snapshot."""
    pipeline_ready: bool
    gpu_available: bool
    vram_allocated_gb: float
    vram_reserved_gb: float
    vram_free_gb: float
    vram_total_gb: float
    jobs_queued: int
    jobs_processing: int
    jobs_completed: int
    jobs_failed: int
    avg_generation_time_s: float | None = None
    concurrent_slots_total: int
    concurrent_slots_free: int
