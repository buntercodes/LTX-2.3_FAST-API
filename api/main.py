"""
api/main.py
===========
FastAPI application factory for the LTX 2.3 Distilled Pipeline server.

Startup sequence:
  1. Apply PYTORCH_CUDA_ALLOC_CONF env var (must happen before any torch import).
  2. Load DistilledPipeline via PipelineManager.
  3. Mount routers and start serving requests.

Shutdown:
  - PipelineManager.shutdown() releases GPU resources cleanly.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Apply CUDA env flags BEFORE importing torch anywhere else
from api.config import settings
settings.apply_cuda_env_flags()  # noqa: E402  (must run before torch)

from api.pipeline_manager import pipeline_manager  # noqa: E402
from api.routes.generate import router as generate_router  # noqa: E402
from api.routes.health import router as health_router  # noqa: E402

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001
    logger.info("═══ LTX 2.3 Distilled FastAPI Server — Starting Up ═══")
    logger.info(
        "Config: checkpoint=%s | quantization=%s | concurrency=%d | torch_compile=%s",
        settings.distilled_checkpoint_path,
        settings.quantization,
        settings.max_concurrent_jobs,
        settings.torch_compile,
    )

    try:
        pipeline_manager.load()
        logger.info("Pipeline loaded ✓ — server ready to accept requests.")
    except Exception:
        logger.exception("FATAL: Pipeline failed to load. Server will respond with 503 on all generation requests.")

    yield  # ← Server is running here

    logger.info("═══ LTX 2.3 Distilled FastAPI Server — Shutting Down ═══")
    pipeline_manager.shutdown()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="LTX 2.3 Distilled Video Generation API",
        description=(
            "Professional-grade FastAPI server wrapping the official LTX 2.3 "
            "Two-Stage Distilled Pipeline. Generates high-quality video in 8 "
            "denoising steps. Optimised for NVIDIA Blackwell (RTX PRO 6000 96 GB)."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ---- CORS ----
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- Request timing middleware ----
    @app.middleware("http")
    async def add_request_timing(request: Request, call_next):  # noqa: ANN001, ANN202
        t0 = time.monotonic()
        response = await call_next(request)
        elapsed_ms = (time.monotonic() - t0) * 1000
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        logger.debug("%s %s → %d (%.1f ms)", request.method, request.url.path, response.status_code, elapsed_ms)
        return response

    # ---- Exception handlers ----
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:  # noqa: ARG001
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": str(exc)},
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:  # noqa: ARG001
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error. Check server logs for details."},
        )

    # ---- Routers ----
    app.include_router(health_router)
    app.include_router(generate_router)

    return app


# Module-level app instance (used by uvicorn)
app = create_app()
