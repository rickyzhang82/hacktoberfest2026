"""Inference backends: Mock (default), llama.cpp (local Gemma 4 GGUF), optional cloud."""
from __future__ import annotations

import base64
import json
import logging
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

from .config import Settings
from .errors import BackendUnavailableError, ModelFileMissingError, OutOfMemoryError
from .hardware import choose_gpu_layers, probe_hardware, throttle_context

logger = logging.getLogger(__name__)


class InferenceBackend(ABC):
    name = "base"
    n_ctx = 4096

    @abstractmethod
    def generate(self, system: str, user: str, image_png: bytes | None, max_tokens: int) -> str:
        """Run one chat turn. `image_png` is provided for the vision stage only."""

    def close(self) -> None:  # pragma: no cover - trivial
        pass


class MockBackend(InferenceBackend):
    """Deterministic stand-in so the whole app runs with zero model weights."""

    name = "mock"

    def __init__(self, n_ctx: int = 4096):
        self.n_ctx = n_ctx
        self.calls: list[dict] = []

    def generate(self, system, user, image_png, max_tokens):
        self.calls.append({"system": system, "user": user, "has_image": image_png is not None,
                           "max_tokens": max_tokens})
        if image_png is not None:
            return (
                "1. PAGE PURPOSE: Landing page with a signup call-to-action.\n"
                "2. LAYOUT: Top navigation bar; centered hero with headline; three feature cards in a row; footer.\n"
                "3. COMPONENTS: Nav links (Home, About, Contact); primary button 'Get Started'; "
                "email input; three cards.\n"
                "4. INTERACTIONS: Clicking 'Get Started' validates the email and shows a confirmation."
            )
        return _MOCK_HTML


_MOCK_HTML = """Here is your page:
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RoboUI Mock Page</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-50 text-slate-800">
  <nav class="flex items-center justify-between px-6 py-4 bg-white shadow">
    <span class="font-bold text-indigo-600">RoboUI</span>
    <div class="space-x-4 text-sm"><a href="#" class="hover:text-indigo-600">Home</a>
      <a href="#" class="hover:text-indigo-600">About</a><a href="#" class="hover:text-indigo-600">Contact</a></div>
  </nav>
  <header class="text-center py-16 px-4">
    <h1 class="text-4xl font-extrabold mb-4">Sketch it. Ship it.</h1>
    <p class="mb-6 text-slate-600">Generated on-device from your hand-drawn wireframe (mock mode).</p>
    <form id="f" class="flex justify-center gap-2" onsubmit="return false">
      <input id="email" type="email" placeholder="you@example.com" class="border rounded px-3 py-2 w-64">
      <button id="go" class="bg-indigo-600 text-white rounded px-4 py-2 hover:bg-indigo-700">Get Started</button>
    </form>
    <p id="msg" class="mt-4 text-green-600 font-medium"></p>
  </header>
  <section class="grid gap-6 md:grid-cols-3 px-6 pb-16 max-w-5xl mx-auto">
    <div class="bg-white rounded-xl shadow p-6"><h3 class="font-semibold mb-2">Private</h3><p class="text-sm">Nothing leaves your machine.</p></div>
    <div class="bg-white rounded-xl shadow p-6"><h3 class="font-semibold mb-2">Fast</h3><p class="text-sm">Quantized Gemma 4 on consumer hardware.</p></div>
    <div class="bg-white rounded-xl shadow p-6"><h3 class="font-semibold mb-2">Simple</h3><p class="text-sm">HTML, Tailwind and vanilla JS.</p></div>
  </section>
  <footer class="text-center text-xs text-slate-500 pb-6">&copy; RoboUI</footer>
  <script>
    document.getElementById('go').addEventListener('click', function () {
      var v = document.getElementById('email').value;
      document.getElementById('msg').textContent = /^\\S+@\\S+\\.\\S+$/.test(v) ? 'Thanks! You are on the list.' : 'Please enter a valid email.';
    });
  </script>
</body>
</html>
```
"""


class LlamaCppBackend(InferenceBackend):
    """Quantized (e.g. Q4_K_M) Gemma 4 GGUF via llama-cpp-python, with throttled context."""

    name = "llama_cpp"

    def __init__(self, settings: Settings):
        self.settings = settings
        model, mmproj = settings.model_path, settings.mmproj_path
        for label, p in (("model", model), ("mmproj", mmproj)):
            if not os.path.isfile(p):
                raise ModelFileMissingError(
                    f"Gemma 4 {label} file not found at '{p}'. See the README 'Local model setup' "
                    "section, or set ROBOUI_BACKEND=mock."
                )
        try:
            from llama_cpp import Llama
            from llama_cpp import llama_chat_format as fmt
        except ImportError as exc:
            raise BackendUnavailableError(
                "llama-cpp-python is not installed. Run: pip install -r requirements-llm.txt"
            ) from exc

        hw = probe_hardware()
        model_bytes = os.path.getsize(model) + os.path.getsize(mmproj)
        self.n_ctx = throttle_context(settings.max_ctx, settings.min_ctx, hw, model_bytes)
        layers = choose_gpu_layers(settings.n_gpu_layers, hw, model_bytes)
        handler_cls = getattr(fmt, "Gemma3ChatHandler", None)
        if handler_cls is None:
            raise BackendUnavailableError(
                "Installed llama-cpp-python has no Gemma multimodal chat handler; upgrade it."
            )
        logger.info("Loading %s (n_ctx=%d, n_gpu_layers=%d)", model, self.n_ctx, layers)
        try:
            self._llm = Llama(
                model_path=model,
                chat_handler=handler_cls(clip_model_path=mmproj),
                n_ctx=self.n_ctx,
                n_gpu_layers=layers,
                use_mmap=True,   # page weights lazily instead of copying into RAM
                n_batch=256,
                verbose=False,
            )
        except (ValueError, RuntimeError, MemoryError) as exc:
            raise OutOfMemoryError(
                f"Could not load the model with n_ctx={self.n_ctx}, n_gpu_layers={layers}: {exc}. "
                "Lower ROBOUI_MAX_CTX, set ROBOUI_N_GPU_LAYERS=0, or use a smaller quant/E2B."
            ) from exc

    def generate(self, system, user, image_png, max_tokens):
        content: list[dict] = [{"type": "text", "text": user}]
        if image_png is not None:
            uri = "data:image/png;base64," + base64.b64encode(image_png).decode()
            content.insert(0, {"type": "image_url", "image_url": {"url": uri}})
        try:
            out = self._llm.create_chat_completion(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": content}],
                max_tokens=max_tokens,
                temperature=0.2,
            )
        except (ValueError, RuntimeError, MemoryError) as exc:
            # llama.cpp reports KV/alloc failures as ValueError/RuntimeError.
            raise OutOfMemoryError(f"Inference failed (likely out of memory): {exc}") from exc
        return out["choices"][0]["message"]["content"] or ""

    def close(self) -> None:
        self._llm = None


class CloudBackend(InferenceBackend):
    """Opt-in OpenAI-compatible fallback. Requires an explicit key in .env."""

    name = "cloud"

    def __init__(self, settings: Settings):
        if not (settings.cloud_api_key and settings.cloud_model and settings.cloud_base_url):
            raise BackendUnavailableError(
                "Cloud backend needs ROBOUI_CLOUD_API_KEY, ROBOUI_CLOUD_MODEL and ROBOUI_CLOUD_BASE_URL in .env."
            )
        self.s = settings
        self.n_ctx = settings.max_ctx

    def generate(self, system, user, image_png, max_tokens):
        content: list[dict] = [{"type": "text", "text": user}]
        if image_png is not None:
            uri = "data:image/png;base64," + base64.b64encode(image_png).decode()
            content.append({"type": "image_url", "image_url": {"url": uri}})
        body = json.dumps({
            "model": self.s.cloud_model, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
        }).encode()
        req = urllib.request.Request(
            self.s.cloud_base_url.rstrip("/") + "/chat/completions", data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.s.cloud_api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)["choices"][0]["message"]["content"]
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
            raise BackendUnavailableError(f"Cloud request failed: {exc}") from exc


def create_backend(settings: Settings) -> InferenceBackend:
    if settings.backend == "mock":
        return MockBackend(n_ctx=settings.max_ctx)
    if settings.backend in ("llama_cpp", "llama", "local"):
        return LlamaCppBackend(settings)
    if settings.backend == "cloud":
        return CloudBackend(settings)
    raise BackendUnavailableError(f"Unknown ROBOUI_BACKEND '{settings.backend}'.")
