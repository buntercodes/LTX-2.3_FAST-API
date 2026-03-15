"""
api/utils/file_io.py
====================
Temporary file and upload management helpers for the LTX 2.3 FastAPI server.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

import aiofiles
from fastapi import UploadFile

logger = logging.getLogger(__name__)


async def save_upload(upload: UploadFile, dest_dir: Path) -> Path:
    """
    Persist an uploaded file to *dest_dir* using its original extension.
    Returns the full path to the saved file.
    """
    suffix = Path(upload.filename or "upload").suffix or ".jpg"
    dest = dest_dir / f"{uuid.uuid4().hex}{suffix}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    async with aiofiles.open(dest, "wb") as f:
        while chunk := await upload.read(1024 * 1024):  # 1 MB chunks
            await f.write(chunk)

    logger.debug("Saved upload '%s' → %s", upload.filename, dest)
    return dest


def cleanup_dir(path: Path) -> None:
    """
    Safely remove a directory and all its contents.
    Silently ignores missing paths.
    """
    try:
        if path.exists():
            shutil.rmtree(path)
            logger.debug("Cleaned up temp dir: %s", path)
    except OSError as exc:
        logger.warning("Could not clean up %s: %s", path, exc)


def get_output_path(output_dir: str, job_id: str) -> Path:
    """Return the canonical output MP4 path for a given job."""
    return Path(output_dir) / f"{job_id}.mp4"


def get_job_temp_dir(temp_dir: str, job_id: str) -> Path:
    """Return a per-job temporary directory for uploaded conditioning images."""
    return Path(temp_dir) / job_id
