@echo off
REM =============================================================================
REM start_server.bat  -  Start the LTX 2.3 Distilled FastAPI server (Windows)
REM =============================================================================
REM Usage: double-click or run from a Command Prompt / PowerShell
REM       Requires .env file with model paths configured.
REM =============================================================================

REM Must be set before Python/torch imports to prevent VRAM fragmentation
SET PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ECHO ============================================================
ECHO  LTX 2.3 Distilled Video Generation API
ECHO  Optimised for RTX PRO 6000 Blackwell 96 GB
ECHO ============================================================
ECHO  PYTORCH_CUDA_ALLOC_CONF=%PYTORCH_CUDA_ALLOC_CONF%
ECHO.

REM Single worker — GPU is shared via PipelineManager semaphore
uv run uvicorn api.main:app ^
    --host 0.0.0.0 ^
    --port 8000 ^
    --workers 1 ^
    --log-level info ^
    --access-log
