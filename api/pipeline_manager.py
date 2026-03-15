"""
api/pipeline_manager.py
=======================
Singleton lifecycle manager for the LTX 2.3 DistilledPipeline.

Responsibilities:
  - Load the pipeline once at server startup (FastAPI lifespan).
  - Apply all Blackwell-specific CUDA optimisations.
  - Optionally torch.compile the transformer for maximum throughput.
  - Accept generation requests via an async job queue.
  - Track job state (queued → processing → done/error).
  - Cap concurrency with an asyncio.Semaphore (default 4 on 96 GB VRAM).
  - Always run cleanup_memory() after each job.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from api.config import settings
from api.models import JobInfo, JobStatus
from api.utils.file_io import get_output_path
from api.utils.memory import (
    aggressive_cleanup,
    apply_blackwell_torch_flags,
    log_vram_usage,
    peak_vram_gb,
    reset_peak_vram,
)

if TYPE_CHECKING:
    from ltx_pipelines.distilled import DistilledPipeline

logger = logging.getLogger(__name__)


class PipelineManager:
    """
    Global singleton that owns the DistilledPipeline instance and manages
    async job dispatch with per-job concurrency control.
    """

    def __init__(self) -> None:
        self._pipeline: DistilledPipeline | None = None
        self._ready: bool = False

        # Concurrency control
        self._semaphore: asyncio.Semaphore | None = None

        # Thread pool for blocking pipeline execution (keeps event loop free)
        self._executor = ThreadPoolExecutor(
            max_workers=settings.max_concurrent_jobs,
            thread_name_prefix="ltx-gen",
        )

        # Job registry
        self._jobs: dict[str, JobInfo] = {}
        self._completed_count: int = 0
        self._failed_count: int = 0
        self._generation_times: deque[float] = deque(maxlen=20)  # rolling EMA window

    # ------------------------------------------------------------------
    # Startup / shutdown
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Initialise the pipeline.  Must be called from within an async
        context (FastAPI lifespan) AFTER the event loop is running, so
        that the asyncio.Semaphore is created on the correct loop.
        """
        logger.info("Applying Blackwell/CUDA optimisation flags …")
        apply_blackwell_torch_flags()

        # Import torch only after env vars have been set by config
        import torch
        from ltx_core.model.video_vae import TilingConfig
        from ltx_core.quantization import QuantizationPolicy
        from ltx_pipelines.distilled import DistilledPipeline

        # Build quantization policy
        quant_policy: QuantizationPolicy | None = None
        if settings.quantization == "fp8-cast":
            quant_policy = QuantizationPolicy.fp8_cast()
        elif settings.quantization == "fp8-scaled-mm":
            amax = settings.fp8_amax_path  # may be None → auto-calibrate
            quant_policy = QuantizationPolicy.fp8_scaled_mm(amax)

        # Build LoRA list
        loras: list = []
        if settings.lora_path:
            from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
            loras.append(
                LoraPathStrengthAndSDOps(
                    settings.lora_path,
                    settings.lora_strength,
                    LTXV_LORA_COMFY_RENAMING_MAP,
                )
            )

        logger.info("Loading DistilledPipeline (checkpoint=%s) …", settings.distilled_checkpoint_path)
        self._pipeline = DistilledPipeline(
            distilled_checkpoint_path=settings.distilled_checkpoint_path,
            gemma_root=settings.gemma_root,
            spatial_upsampler_path=settings.spatial_upsampler_path,
            loras=tuple(loras),
            quantization=quant_policy,
        )

        # Optional torch.compile on the transformer for Blackwell kernel fusion
        if settings.torch_compile:
            logger.info(
                "torch.compile(mode='max-autotune') enabled. "
                "First generation will incur a ~60–120 s warm-up …"
            )
            try:
                self._pipeline.model_ledger.transformer_builder  # ensure attribute exists
                # Wrap the internal transformer builder call; compile happens on first call
                _orig_transformer = self._pipeline.model_ledger.transformer

                def _compiled_transformer() -> object:  # noqa: ANN202
                    model = _orig_transformer()
                    return torch.compile(model, mode="max-autotune", fullgraph=False)  # type: ignore[arg-type]

                self._pipeline.model_ledger.transformer = _compiled_transformer  # type: ignore[method-assign]
                logger.info("Transformer will be compiled on first use.")
            except Exception as exc:  # noqa: BLE001
                logger.warning("torch.compile setup failed (non-fatal): %s", exc)

        # Semaphore must be created inside a running event loop
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_jobs)
        self._ready = True
        logger.info(
            "PipelineManager ready. Concurrency=%d, Quantization=%s, torch.compile=%s",
            settings.max_concurrent_jobs,
            settings.quantization,
            settings.torch_compile,
        )
        log_vram_usage("after pipeline load")

    def shutdown(self) -> None:
        """Clean up resources on server shutdown."""
        logger.info("PipelineManager shutting down …")
        self._pipeline = None
        self._ready = False
        self._executor.shutdown(wait=False)
        aggressive_cleanup()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def job_count(self) -> dict[str, int]:
        queued = sum(1 for j in self._jobs.values() if j.status == JobStatus.QUEUED)
        processing = sum(1 for j in self._jobs.values() if j.status == JobStatus.PROCESSING)
        return {
            "queued": queued,
            "processing": processing,
            "completed": self._completed_count,
            "failed": self._failed_count,
        }

    @property
    def semaphore_free(self) -> int:
        if self._semaphore is None:
            return 0
        return self._semaphore._value  # type: ignore[attr-defined]

    @property
    def avg_generation_time(self) -> float | None:
        if not self._generation_times:
            return None
        return round(sum(self._generation_times) / len(self._generation_times), 2)

    # ------------------------------------------------------------------
    # Job management
    # ------------------------------------------------------------------

    def create_job(self, request_dict: dict) -> str:
        """Register a new job and return its job_id."""
        job_id = uuid.uuid4().hex
        self._jobs[job_id] = JobInfo(
            job_id=job_id,
            status=JobStatus.QUEUED,
            request=request_dict,
            created_at=time.time(),
        )
        return job_id

    def get_job(self, job_id: str) -> JobInfo | None:
        return self._jobs.get(job_id)

    async def submit(
        self,
        job_id: str,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        enhance_prompt: bool,
        tiling: bool,
        images: list,  # list[ImageConditioningInput]
    ) -> None:
        """
        Enqueue a generation job.  Returns immediately; the job runs in the
        background thread pool behind the concurrency semaphore.
        """
        asyncio.get_event_loop().create_task(
            self._run_job(
                job_id=job_id,
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                enhance_prompt=enhance_prompt,
                tiling=tiling,
                images=images,
            )
        )

    async def _run_job(
        self,
        job_id: str,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        enhance_prompt: bool,
        tiling: bool,
        images: list,
    ) -> None:
        """Internal coroutine: acquire semaphore, dispatch to thread pool, clean up."""
        assert self._semaphore is not None  # noqa: S101
        job = self._jobs[job_id]

        async with self._semaphore:
            job.status = JobStatus.PROCESSING
            t_start = time.monotonic()

            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self._executor,
                    self._generate_sync,
                    job_id,
                    prompt,
                    seed,
                    height,
                    width,
                    num_frames,
                    frame_rate,
                    enhance_prompt,
                    tiling,
                    images,
                )
                elapsed = time.monotonic() - t_start
                job.duration_s = round(elapsed, 2)
                job.status = JobStatus.DONE
                job.video_url = f"/generate/{job_id}/download"
                job.completed_at = time.time()
                self._completed_count += 1
                self._generation_times.append(elapsed)
                logger.info("Job %s completed in %.1f s", job_id, elapsed)

            except Exception as exc:  # noqa: BLE001
                elapsed = time.monotonic() - t_start
                job.status = JobStatus.ERROR
                job.error = str(exc)
                job.completed_at = time.time()
                self._failed_count += 1
                logger.exception("Job %s failed after %.1f s: %s", job_id, elapsed, exc)

    def _generate_sync(
        self,
        job_id: str,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        enhance_prompt: bool,
        tiling: bool,
        images: list,
    ) -> None:
        """
        Blocking generation call — runs in ThreadPoolExecutor.
        All VRAM management happens here.
        """
        import torch
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_pipelines.utils.media_io import encode_video

        assert self._pipeline is not None  # noqa: S101

        reset_peak_vram()
        log_vram_usage(f"job {job_id} start")

        tiling_cfg = TilingConfig.default() if tiling else None
        output_path = str(get_output_path(settings.output_dir, job_id))

        try:
            with torch.inference_mode():
                video_iter, audio = self._pipeline(
                    prompt=prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=frame_rate,
                    images=images,
                    tiling_config=tiling_cfg,
                    enhance_prompt=enhance_prompt,
                )

                video_chunks_number = get_video_chunks_number(num_frames, tiling_cfg)
                encode_video(
                    video=video_iter,
                    fps=frame_rate,
                    audio=audio,
                    output_path=output_path,
                    video_chunks_number=video_chunks_number,
                )

            peak = round(peak_vram_gb(), 2)
            self._jobs[job_id].gpu_mem_peak_gb = peak
            log_vram_usage(f"job {job_id} end")
            logger.info("Job %s: peak VRAM = %.2f GB, output = %s", job_id, peak, output_path)

        finally:
            aggressive_cleanup()


# ---------------------------------------------------------------------------
# Global singleton instance
# ---------------------------------------------------------------------------
pipeline_manager = PipelineManager()
