"""Hardware probing, GPU-layer offloading and context-window throttling."""
from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass

import psutil

logger = logging.getLogger(__name__)

# Rough KV-cache + activation cost per context token for a small (2-4B) quantized
# model. Deliberately conservative so we throttle early rather than OOM.
BYTES_PER_CTX_TOKEN = 160 * 1024
# Fraction of free memory we allow the model + KV cache to consume.
MEMORY_BUDGET_FRACTION = 0.6
CTX_STEP = 256


@dataclass(frozen=True)
class HardwareInfo:
    free_ram_bytes: int
    free_vram_bytes: int | None  # None when no discrete NVIDIA GPU is detected


def probe_hardware() -> HardwareInfo:
    return HardwareInfo(
        free_ram_bytes=psutil.virtual_memory().available,
        free_vram_bytes=_free_nvidia_vram(),
    )


def _free_nvidia_vram() -> int | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
        return max(int(line.strip()) for line in out.splitlines() if line.strip()) * 1024 * 1024
    except (subprocess.SubprocessError, ValueError, OSError) as exc:
        logger.warning("nvidia-smi probe failed (%s); assuming no discrete GPU", exc)
        return None


def throttle_context(
    requested_max: int,
    min_ctx: int,
    hw: HardwareInfo,
    model_bytes: int = 0,
) -> int:
    """Pick the largest context window (<= requested_max) that fits in memory.

    Memory pool = free VRAM if a GPU is present, otherwise free RAM. The model's
    own footprint is subtracted first; the remainder (times a safety fraction)
    is divided by the per-token cost. Result is clamped to [min_ctx, requested_max]
    and rounded down to a multiple of CTX_STEP.
    """
    pool = hw.free_vram_bytes if hw.free_vram_bytes else hw.free_ram_bytes
    budget = int(pool * MEMORY_BUDGET_FRACTION) - model_bytes
    fit = max(budget, 0) // BYTES_PER_CTX_TOKEN
    fit = (fit // CTX_STEP) * CTX_STEP
    ctx = max(min_ctx, min(requested_max, fit))
    if ctx < requested_max:
        logger.warning(
            "Context throttled %d -> %d tokens (pool=%.1f GiB, model=%.1f GiB)",
            requested_max, ctx, pool / 2**30, model_bytes / 2**30,
        )
    return ctx


def choose_gpu_layers(configured: int, hw: HardwareInfo, model_bytes: int) -> int:
    """-1 means auto: offload everything if it fits in VRAM, else a proportional share."""
    if configured >= 0:
        return configured
    if not hw.free_vram_bytes or model_bytes <= 0:
        # No discrete GPU detected: on Apple Silicon llama.cpp uses Metal with
        # unified memory, so full offload is fine; elsewhere it is ignored.
        return -1 if hw.free_vram_bytes is None else 0
    usable = hw.free_vram_bytes * MEMORY_BUDGET_FRACTION
    if usable >= model_bytes:
        return -1
    return max(0, int(32 * usable / model_bytes))  # ~32 layers assumed for E2B/E4B
