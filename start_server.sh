#!/usr/bin/env bash
# =============================================================================
# start_server.sh — Start the LTX 2.3 Distilled FastAPI server
# =============================================================================
# Usage:
#   chmod +x start_server.sh
#   ./start_server.sh
#
# Requires:
#   - .env file with all required model paths configured
#   - Python environment with ltx-core, ltx-pipelines, and API deps installed
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# CUDA memory allocator — must be set before Python starts
# Expandable segments prevent fragmentation across concurrent jobs
# ---------------------------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ---------------------------------------------------------------------------
# Blackwell / NCCL tuning (optional, set if running multi-GPU in future)
# ---------------------------------------------------------------------------
# export NCCL_P2P_DISABLE=0
# export NCCL_IB_DISABLE=0

echo "════════════════════════════════════════════════════════════"
echo " LTX 2.3 Distilled Video Generation API"
echo " Optimised for RTX PRO 6000 Blackwell 96 GB"
echo "════════════════════════════════════════════════════════════"
echo " PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo ""

# ---------------------------------------------------------------------------
# Run with uvicorn — single worker (GPU is shared across async jobs via
# the PipelineManager semaphore; multiple uvicorn workers would each load
# separate pipeline instances and exhaust VRAM)
# ---------------------------------------------------------------------------
exec uv run uvicorn api.main:app \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8000}" \
    --workers 1 \
    --loop uvloop \
    --http h11 \
    --log-level "${LOG_LEVEL:-info}" \
    --access-log \
    --no-use-colors
