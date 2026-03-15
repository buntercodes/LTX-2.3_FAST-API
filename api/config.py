"""
api/config.py
=============
Environment-driven configuration for the LTX 2.3 Distilled FastAPI server.
All values can be set via environment variables or a .env file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Model paths (required)
    # -------------------------------------------------------------------------
    distilled_checkpoint_path: str = Field(
        ...,
        description="Absolute path to the LTX 2.3 distilled model checkpoint (.safetensors).",
    )
    gemma_root: str = Field(
        ...,
        description="Absolute path to the root directory of the Gemma text-encoder weights.",
    )
    spatial_upsampler_path: str = Field(
        ...,
        description="Absolute path to the spatial upsampler checkpoint (.safetensors).",
    )

    # -------------------------------------------------------------------------
    # Optional LoRA
    # -------------------------------------------------------------------------
    lora_path: str | None = Field(
        default=None,
        description="Optional path to a LoRA safetensors file.",
    )
    lora_strength: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="Strength applied to the LoRA adapter.",
    )

    # -------------------------------------------------------------------------
    # Quantization — fp8-scaled-mm is the best choice for Blackwell
    # -------------------------------------------------------------------------
    quantization: Literal["none", "fp8-cast", "fp8-scaled-mm"] = Field(
        default="none",
        description=(
            "Quantization policy for the transformer. 'none' = full BF16 (recommended on 96 GB VRAM). "
            "'fp8-scaled-mm' or 'fp8-cast' are available but only save ~1–2 GB on this GPU — "
            "all other models (Gemma, video/audio VAE, upsampler, vocoder) load in BF16 regardless."
        ),
    )
    fp8_amax_path: str | None = Field(
        default=None,
        description="Optional path to a pre-computed amax calibration file for fp8-scaled-mm.",
    )

    # -------------------------------------------------------------------------
    # Generation defaults
    # -------------------------------------------------------------------------
    default_height: int = Field(default=1024, description="Default output video height in pixels (divisible by 64).")
    default_width: int = Field(default=1536, description="Default output video width in pixels (divisible by 64).")
    default_num_frames: int = Field(default=121, description="Default frame count (must satisfy: (n-1) % 8 == 0).")
    default_frame_rate: float = Field(default=24.0, description="Default frame rate (fps).")
    default_seed: int = Field(default=42, description="Default random seed.")

    # -------------------------------------------------------------------------
    # Concurrency — RTX PRO 6000 Blackwell 96 GB supports 4 concurrent jobs
    # -------------------------------------------------------------------------
    max_concurrent_jobs: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Maximum number of simultaneous GPU generation jobs.",
    )

    # -------------------------------------------------------------------------
    # Blackwell / CUDA optimisations
    # -------------------------------------------------------------------------
    torch_compile: bool = Field(
        default=False,
        description=(
            "Enable torch.compile(mode='max-autotune') on the transformer. "
            "First request incurs a ~60–120 s warm-up; subsequent requests run "
            "fully compiled CUDA kernels. Disable during development."
        ),
    )
    cuda_expandable_segments: bool = Field(
        default=True,
        description="Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before importing torch.",
    )
    enable_tf32: bool = Field(
        default=True,
        description="Allow TF32 for matmul and cuDNN (significant speedup on Blackwell).",
    )

    # -------------------------------------------------------------------------
    # Storage
    # -------------------------------------------------------------------------
    output_dir: str = Field(
        default="./outputs",
        description="Directory where generated MP4 files are saved.",
    )
    temp_dir: str = Field(
        default="./tmp_uploads",
        description="Temporary directory for uploaded conditioning images.",
    )

    # -------------------------------------------------------------------------
    # Server
    # -------------------------------------------------------------------------
    host: str = Field(default="0.0.0.0", description="Bind host.")
    port: int = Field(default=8000, description="Bind port.")
    log_level: str = Field(default="info", description="Uvicorn/Python log level.")
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins. Use ['*'] for open access or restrict in production.",
    )

    # -------------------------------------------------------------------------
    # Validators
    # -------------------------------------------------------------------------
    @field_validator("distilled_checkpoint_path", "gemma_root", "spatial_upsampler_path", mode="before")
    @classmethod
    def _expand_path(cls, v: str) -> str:
        return str(Path(v).expanduser().resolve())

    @field_validator("output_dir", "temp_dir", mode="before")
    @classmethod
    def _expand_and_create(cls, v: str) -> str:
        p = Path(v).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return str(p)

    def apply_cuda_env_flags(self) -> None:
        """
        Must be called BEFORE any torch import to take effect.
        Sets PYTORCH_CUDA_ALLOC_CONF if enabled.
        """
        if self.cuda_expandable_segments:
            existing = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
            if "expandable_segments" not in existing:
                segment_flag = "expandable_segments:True"
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                    f"{existing},{segment_flag}" if existing else segment_flag
                )


# ---------------------------------------------------------------------------
# Global singleton — import and use `settings` everywhere in the server.
# ---------------------------------------------------------------------------
settings = Settings()  # type: ignore[call-arg]
