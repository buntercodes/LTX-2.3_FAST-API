"""
api/routes/generate.py
======================
Generation endpoints for the LTX 2.3 distilled pipeline.

  POST /generate          — submit a text-to-video job (returns job_id immediately)
  GET  /generate/{job_id} — poll job status
  GET  /generate/{job_id}/download — stream the finished MP4
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from api.config import settings
from api.models import GenerateRequest, GenerateResponse, JobStatus, JobStatusResponse
from api.pipeline_manager import pipeline_manager
from api.utils.file_io import cleanup_dir, get_job_temp_dir, get_output_path, save_upload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/generate", tags=["Generation"])


# ---------------------------------------------------------------------------
# POST /generate
# ---------------------------------------------------------------------------

@router.post(
    "/generate",
    response_model=GenerateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a video generation job",
    description=(
        "Accepts a text prompt and optional start/end frame conditioning images. "
        "Returns a `job_id` immediately — poll `GET /generate/{job_id}` for status."
    ),
)
async def submit_generation(
    # ---- text fields (Form) ----
    prompt: Annotated[str, Form(description="Text prompt describing the desired video.")],
    seed: Annotated[int, Form()] = settings.default_seed,
    height: Annotated[int, Form()] = settings.default_height,
    width: Annotated[int, Form()] = settings.default_width,
    num_frames: Annotated[int, Form()] = settings.default_num_frames,
    frame_rate: Annotated[float, Form()] = settings.default_frame_rate,
    enhance_prompt: Annotated[bool, Form()] = False,
    tiling: Annotated[bool, Form()] = True,
    image_start_strength: Annotated[float, Form()] = 1.0,
    image_end_strength: Annotated[float, Form()] = 0.8,
    # ---- optional image uploads ----
    image_start: Annotated[UploadFile | None, File(description="Optional start-frame conditioning image.")] = None,
    image_end: Annotated[UploadFile | None, File(description="Optional end-frame conditioning image.")] = None,
) -> GenerateResponse:
    if not pipeline_manager.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline is still loading. Please retry in a moment.",
        )

    # Validate via the Pydantic model (reuse all validators)
    try:
        req = GenerateRequest(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            enhance_prompt=enhance_prompt,
            tiling=tiling,
            image_start_strength=image_start_strength,
            image_end_strength=image_end_strength,
        )
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    # Register job
    job_id = pipeline_manager.create_job(req.model_dump())
    temp_dir = get_job_temp_dir(settings.temp_dir, job_id)

    # Persist uploaded images
    images: list = []
    try:
        from ltx_pipelines.utils.args import ImageConditioningInput  # deferred to avoid slow import at start

        if image_start is not None:
            start_path = await save_upload(image_start, temp_dir)
            images.append(
                ImageConditioningInput(
                    path=str(start_path),
                    frame_idx=0,
                    strength=req.image_start_strength,
                )
            )
        if image_end is not None:
            end_path = await save_upload(image_end, temp_dir)
            images.append(
                ImageConditioningInput(
                    path=str(end_path),
                    frame_idx=req.num_frames - 1,
                    strength=req.image_end_strength,
                )
            )
    except Exception as exc:
        cleanup_dir(temp_dir)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process uploaded images: {exc}",
        ) from exc

    # Fire-and-forget background task
    await pipeline_manager.submit(
        job_id=job_id,
        prompt=req.prompt,
        seed=req.seed,
        height=req.height,
        width=req.width,
        num_frames=req.num_frames,
        frame_rate=req.frame_rate,
        enhance_prompt=req.enhance_prompt,
        tiling=req.tiling,
        images=images,
    )

    logger.info("Job %s queued — prompt='%s…'", job_id, prompt[:60])
    return GenerateResponse(job_id=job_id, status=JobStatus.QUEUED)


# ---------------------------------------------------------------------------
# GET /generate/{job_id}  — status polling
# ---------------------------------------------------------------------------

@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    summary="Poll job status",
)
async def get_job_status(job_id: str) -> JobStatusResponse:
    job = pipeline_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        video_url=job.video_url,
        duration_s=job.duration_s,
        gpu_mem_peak_gb=job.gpu_mem_peak_gb,
        error=job.error,
    )


# ---------------------------------------------------------------------------
# GET /generate/{job_id}/download  — serve the MP4
# ---------------------------------------------------------------------------

@router.get(
    "/{job_id}/download",
    summary="Download the generated video",
    response_class=FileResponse,
)
async def download_video(job_id: str) -> FileResponse:
    job = pipeline_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job '{job_id}' not found.")
    if job.status != JobStatus.DONE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job '{job_id}' is not complete yet (status={job.status}).",
        )

    output = get_output_path(settings.output_dir, job_id)
    if not output.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video file for job '{job_id}' not found on disk.",
        )

    return FileResponse(
        path=str(output),
        media_type="video/mp4",
        filename=f"ltx_{job_id}.mp4",
    )
