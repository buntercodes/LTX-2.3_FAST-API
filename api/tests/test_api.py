"""
api/tests/test_api.py
=====================
Integration tests for the LTX 2.3 FastAPI server.
Uses pytest-asyncio + httpx AsyncClient with a mocked pipeline.
"""

from __future__ import annotations

import asyncio
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_pipeline():
    """Return a MagicMock that mimics DistilledPipeline (no GPU needed)."""
    import torch

    mock = MagicMock()
    # pipeline.__call__ returns (video_iterator, audio)
    dummy_frame = torch.zeros(1, 64, 64, 3, dtype=torch.uint8)
    mock.return_value = (iter([dummy_frame]), None)
    return mock


@pytest.fixture(scope="module")
def test_client(mock_pipeline):
    """
    Create a TestClient with the pipeline_manager pre-seeded as ready,
    bypassing actual GPU model loading.
    """
    from api.main import app
    from api.pipeline_manager import pipeline_manager

    pipeline_manager._pipeline = mock_pipeline
    pipeline_manager._ready = True
    pipeline_manager._semaphore = asyncio.Semaphore(4)

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


# ---------------------------------------------------------------------------
# Health / Ready / Metrics
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_always_200(self, test_client):
        r = test_client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "pipeline_ready" in data
        assert "gpu_available" in data

    def test_ready_when_pipeline_loaded(self, test_client):
        r = test_client.get("/ready")
        # pipeline_manager._ready = True in fixture
        assert r.status_code == 200
        assert r.json()["status"] == "ready"

    def test_metrics_schema(self, test_client):
        r = test_client.get("/metrics")
        assert r.status_code == 200
        data = r.json()
        for key in (
            "pipeline_ready",
            "gpu_available",
            "vram_allocated_gb",
            "vram_reserved_gb",
            "vram_free_gb",
            "vram_total_gb",
            "jobs_queued",
            "jobs_processing",
            "jobs_completed",
            "jobs_failed",
            "concurrent_slots_total",
            "concurrent_slots_free",
        ):
            assert key in data, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Input validation (no GPU needed)
# ---------------------------------------------------------------------------

class TestGenerateValidation:
    def test_missing_prompt_returns_422(self, test_client):
        r = test_client.post("/generate", data={"height": "1024", "width": "1536"})
        assert r.status_code == 422

    def test_resolution_not_divisible_by_64_returns_422(self, test_client):
        r = test_client.post(
            "/generate",
            data={"prompt": "A cat", "height": "1000", "width": "1536"},
        )
        assert r.status_code == 422

    def test_invalid_frame_count_returns_422(self, test_client):
        # 120 frames → (120-1) % 8 = 7  (invalid)
        r = test_client.post(
            "/generate",
            data={"prompt": "A cat", "height": "1024", "width": "1536", "num_frames": "120"},
        )
        assert r.status_code == 422

    def test_valid_frame_counts_accepted(self, test_client):
        """Spot-check a few valid frame counts: n = 8k+1."""
        for n in (9, 25, 49, 121):
            r = test_client.post(
                "/generate",
                data={
                    "prompt": "A cat walking",
                    "height": "1024",
                    "width": "1536",
                    "num_frames": str(n),
                    "seed": "0",
                },
            )
            # Should be 202 Accepted (queued), not a 4xx validation error
            assert r.status_code in (202, 503), f"Unexpected {r.status_code} for num_frames={n}"


# ---------------------------------------------------------------------------
# Job polling
# ---------------------------------------------------------------------------

class TestJobPolling:
    def test_unknown_job_id_returns_404(self, test_client):
        r = test_client.get("/generate/nonexistent_job_id_xyz")
        assert r.status_code == 404

    def test_download_unknown_job_returns_404(self, test_client):
        r = test_client.get("/generate/nonexistent_job_id_xyz/download")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Image upload validation
# ---------------------------------------------------------------------------

class TestImageUpload:
    def test_start_image_accepted(self, test_client):
        fake_image = io.BytesIO(b"\xff\xd8\xff" + b"\x00" * 100)  # minimal JPEG header
        r = test_client.post(
            "/generate",
            data={"prompt": "A beach scene", "height": "1024", "width": "1536"},
            files={"image_start": ("start.jpg", fake_image, "image/jpeg")},
        )
        assert r.status_code in (202, 503)
