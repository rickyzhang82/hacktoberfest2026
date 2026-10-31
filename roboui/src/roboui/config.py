"""Environment-driven configuration. No secrets are hardcoded."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using default %d", name, raw, default)
        return default


@dataclass(frozen=True)
class Settings:
    backend: str = "mock"
    model_path: str = "models/gemma-4-e2b-it-Q4_K_M.gguf"
    mmproj_path: str = "models/mmproj-gemma-4-e2b-it-F16.gguf"
    max_ctx: int = 8192
    min_ctx: int = 1024
    n_gpu_layers: int = -1
    cloud_api_key: str = ""
    cloud_model: str = ""
    cloud_base_url: str = ""
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, dotenv: bool = True) -> "Settings":
        if dotenv:
            load_dotenv()
        return cls(
            backend=os.getenv("ROBOUI_BACKEND", "mock").strip().lower() or "mock",
            model_path=os.getenv("ROBOUI_MODEL_PATH", cls.model_path),
            mmproj_path=os.getenv("ROBOUI_MMPROJ_PATH", cls.mmproj_path),
            max_ctx=_int("ROBOUI_MAX_CTX", cls.max_ctx),
            min_ctx=_int("ROBOUI_MIN_CTX", cls.min_ctx),
            n_gpu_layers=_int("ROBOUI_N_GPU_LAYERS", cls.n_gpu_layers),
            cloud_api_key=os.getenv("ROBOUI_CLOUD_API_KEY", ""),
            cloud_model=os.getenv("ROBOUI_CLOUD_MODEL", ""),
            cloud_base_url=os.getenv("ROBOUI_CLOUD_BASE_URL", ""),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
