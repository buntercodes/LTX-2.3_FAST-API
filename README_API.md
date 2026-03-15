# LTX 2.3 Distilled — Professional FastAPI Server

A production-grade FastAPI server wrapping the official **LTX 2.3 Two-Stage Distilled Pipeline** (`DistilledPipeline`). Generates high-quality video in just **8 denoising steps**, heavily optimised for **NVIDIA Blackwell architecture** (RTX PRO 6000 Blackwell Server Edition — 96 GB VRAM).

---

## Architecture Overview

```
api/
├── main.py               # FastAPI app factory + lifespan
├── config.py             # Pydantic-settings (env-driven)
├── models.py             # Request / response schemas
├── pipeline_manager.py   # Singleton pipeline + async job queue
├── routes/
│   ├── generate.py       # POST/GET /generate  (video gen + polling + download)
│   └── health.py         # /health, /ready, /metrics
└── utils/
    ├── memory.py          # Blackwell CUDA helpers (TF32, Flash-SDP, VRAM log)
    └── file_io.py         # Upload persistence + temp cleanup
```

---

## Quick Start

### 1. Install API dependencies

```bash
# From the repo root (uv workspace already has ltx-core / ltx-pipelines)
pip install -r requirements-api.txt
```

### 2. Configure model paths

```bash
cp .env.example .env
# Edit .env and fill in:
#   DISTILLED_CHECKPOINT_PATH
#   GEMMA_ROOT
#   SPATIAL_UPSAMPLER_PATH
```

### 3. Start the server

**Linux / WSL:**
```bash
./start_server.sh
```

**Windows (cmd / PowerShell):**
```bat
start_server.bat
```

**Manual:**
```bash
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

> ⚠️ Always use `--workers 1`. The pipeline is shared across concurrent async jobs via an internal `asyncio.Semaphore`. Multiple workers would load separate pipeline copies and exhaust VRAM.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe — always 200 |
| `GET` | `/ready` | Readiness probe — 503 until pipeline loads |
| `GET` | `/metrics` | JSON: VRAM, job counters, concurrency state |
| `POST` | `/generate` | Submit a generation job (returns `job_id` immediately) |
| `GET` | `/generate/{job_id}` | Poll job status |
| `GET` | `/generate/{job_id}/download` | Download the finished MP4 |

Full interactive docs: **`http://localhost:8000/docs`**

---

## Generating a Video

```bash
# Text-to-video
curl -X POST http://localhost:8000/generate \
  -F "prompt=A golden retriever running on a beach at sunset" \
  -F "height=1024" \
  -F "width=1536" \
  -F "num_frames=121" \
  -F "frame_rate=24" \
  -F "seed=42"

# → {"job_id": "abc123...", "status": "queued", "message": "Generation job queued."}
```

```bash
# Poll status
curl http://localhost:8000/generate/abc123...

# → {"job_id": "abc123...", "status": "done", "video_url": "/generate/abc123.../download",
#    "duration_s": 45.2, "gpu_mem_peak_gb": 21.8}
```

```bash
# Download
curl -OJ http://localhost:8000/generate/abc123.../download
```

### With image conditioning

```bash
curl -X POST http://localhost:8000/generate \
  -F "prompt=A cat walking through a garden" \
  -F "height=1024" \
  -F "width=1536" \
  -F "num_frames=49" \
  -F "image_start=@/path/to/start_frame.jpg" \
  -F "image_start_strength=1.0" \
  -F "image_end=@/path/to/end_frame.jpg" \
  -F "image_end_strength=0.8"
```

---

## Resolution & Frame Constraints

| Constraint | Rule | Examples |
|---|---|---|
| Height & Width | Must be divisible by **64** | 512, 576, 640 … 1024, 1536, 2048 |
| Frame count | Must satisfy `(n-1) % 8 == 0` | 9, 17, 25, 33, 41, 49 … 121, 169, 257 |

---

## Blackwell Optimisations

| Feature | Setting | Notes |
|---|---|---|
| **Full BF16 (no quantization)** | `QUANTIZATION=none` | Default. All models natively load in BF16. FP8 only saves ~1-2 GB on the transformer; not worth it on 96 GB VRAM |
| **Expandable segments** | `CUDA_EXPANDABLE_SEGMENTS=true` | Prevents OOM from fragmentation |
| **TF32 matmul** | `ENABLE_TF32=true` | ~3× matmul speedup, negligible quality loss |
| **4× concurrency** | `MAX_CONCURRENT_JOBS=4` | ~22 GB/job × 4 = ~88 GB (safe on 96 GB) |
| **torch.compile** | `TORCH_COMPILE=false` (default) | Enable for production; ~60–120 s first-run warm-up |

---

## Running Tests

```bash
cd c:\LTX-2.3_FAST-API
python -m pytest api/tests/test_api.py -v
```

Tests use a mocked pipeline — **no GPU required**.

---

## Key Design Decisions

- **Single uvicorn worker + asyncio.Semaphore** — correct concurrency model when a single GPU is shared. Multiple processes would each load the pipeline and exhaust VRAM.
- **`torch.inference_mode()`** wraps every generation — disables autograd graph construction for maximum speed.
- **Job queue pattern** — `POST /generate` returns immediately with a `job_id`; client polls `GET /generate/{job_id}`. Prevents HTTP timeouts on long generations.
- **Per-job VRAM peak tracking** — every response includes `gpu_mem_peak_gb` so you can monitor headroom.
- **Aggressive cleanup** (`gc.collect + empty_cache + ipc_collect + synchronize`) runs in a `finally` block after every job.
