# LTX 2.3 Server Deployment Guide (packet.ai)

This guide takes you through deploying the LTX 2.3 FastAPI server step-by-step on a fresh GPU instance (e.g., Ubuntu on packet.ai) equipped with an RTX PRO 6000 Ada (96 GB VRAM).

## Prerequisites

Assuming you have just SSH'd into your fresh Ubuntu GPU instance from packet.ai. Check that your GPU is recognized:
```bash
nvidia-smi
```
You should see your RTX PRO 6000 Ada 96GB listed.

---

## Step 1: System Prep & CUDA Drivers

Most AI instances come with NVIDIA drivers installed, but let's ensure the baseline dependencies needed for PyTorch, Python, and Git.

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git wget curl build-essential libgl1-mesa-glx libglib2.0-0 ffmpeg python3-pip python3-venv tzdata
```

---

## Step 2: Install `uv` (Fast Python Package Manager)

The official LTX repository uses `uv` for lightning-fast workspace and dependency management.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

---

## Step 3: Clone the Repository & API Code

Since you have the updated codebase locally, the easiest way to get it onto the instance is to push it to a private GitHub repo and clone it. Assuming you have cloned it to `~/LTX-2.3_FAST-API` on the packet.ai instance:

```bash
git clone <your-repo-url> ~/LTX-2.3_FAST-API
cd ~/LTX-2.3_FAST-API
```

---

## Step 4: Setup the Environment and Install Dependencies

1. **Create the virtual environment** and sync the official workspace:
   ```bash
   uv venv
   source .venv/bin/activate
   uv sync
   ```

2. **Install the API-specific dependencies** (FastAPI, Uvicorn, etc.) directly into the same environment:
   ```bash
   uv pip install -r requirements-api.txt
   ```

---

## Step 5: Download the Model Weights

LTX 2.3 requires several model weights:
1. **Distilled Checkpoint:** The core video model
2. **Gemma:** The text encoder
3. **Spatial Upsampler:** Used for Stage 2

You must download these from HuggingFace.

### Install `huggingface_hub` and login

Since the weights are gated, you must accept the terms on HF and login.
```bash
uv pip install -U "huggingface_hub[cli]"
uv run hf auth login
# Paste your HuggingFace access token when prompted
```

### Download the weights
Create a folder to hold everything cleanly.

```bash
mkdir -p ~/models
cd ~/models

# 1. Download LTX Distilled Checkpoint
uv run hf download Lightricks/LTX-2.3 \
  ltx-2.3-22b-distilled.safetensors \
  --local-dir .

# 2. Download Gemma 3 (12B IT QAT Q4_0 unquantized) - Requires accepting terms on HF page first!
uv run hf download google/gemma-3-12b-it-qat-q4_0-unquantized \
  --local-dir ./gemma-3-12b-it-qat-q4_0-unquantized

# 3. Download Spatial Upsampler
uv run hf download Lightricks/LTX-2.3 \
  ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
  --local-dir .

cd ~/LTX-2.3_FAST-API
```

---

## Step 6: Configure `.env`

Now we point the FastAPI server to the downloaded weights.

1. **Copy the example file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env`:**
   Open the file with `nano .env` and update the paths to match your downloads:

   ```bash
   DISTILLED_CHECKPOINT_PATH=~/models/ltx-2.3-22b-distilled.safetensors
   GEMMA_ROOT=~/models/gemma-3-12b-it-qat-q4_0-unquantized
   SPATIAL_UPSAMPLER_PATH=~/models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors

   # Ensure BF16 default is set (since you have 96GB VRAM)
   QUANTIZATION=none

   MAX_CONCURRENT_JOBS=4
   CUDA_EXPANDABLE_SEGMENTS=true
   ENABLE_TF32=true
   ```

---

## Step 7: Exposing the Port & Keeping it Running (Optional)

If this is a permanent server, you should run it inside `tmux` or use `systemd` so it doesn't die when your SSH session drops.

### Using tmux:
```bash
tmux new -s ltx-server
```

Now start the server:
```bash
# Ensure the virtualenv is active!
source .venv/bin/activate
./start_server.sh
```
*(To leave the tmux session running in the background, press `Ctrl+B`, then `D`)*

---

## Step 8: Test the API

From your local machine or another terminal on the Server, send a test curl Request. Make sure port `8000` is exposed in your packet.ai firewall / security group settings.

```bash
curl -X POST http://<YOUR_INSTANCE_IP>:8000/generate \
  -F "prompt=A majestic lion roaring in the African savanna at sunrise" \
  -F "height=1024" \
  -F "width=1536" \
  -F "num_frames=25" \
  -F "seed=1337"
```

Look for the `job_id` in the response, then poll:
```bash
curl http://<YOUR_INSTANCE_IP>:8000/generate/<YOUR_JOB_ID>
```

When status is `"done"`, download the MP4:
```bash
curl -OJ http://<YOUR_INSTANCE_IP>:8000/generate/<YOUR_JOB_ID>/download
```

## Troubleshooting
- **Cannot download Gemma:** Ensure you have clicked "Agree and access repository" on the `google/gemma-3-12b-it-qat-q4_0-unquantized` HuggingFace page using the account associated with your HF token.
- **OOM Errors:** If you hit Out of Memory errors, lower `MAX_CONCURRENT_JOBS=2` in `.env`. The RTX PRO 6000 96GB should handle 4, but if other processes (like an X11 server) are eating VRAM, you may need to drop this slightly.
- **Port unreachable:** Verify your packet.ai networking setup allows incoming TCP traffic on port 8000.
