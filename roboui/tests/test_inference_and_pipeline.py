import io
import socket

import pytest
from PIL import Image

from roboui.config import Settings
from roboui.errors import (BackendUnavailableError, ModelFileMissingError, NetworkBlockedError)
from roboui.hardware import HardwareInfo, choose_gpu_layers, throttle_context
from roboui.inference import LlamaCppBackend, MockBackend, create_backend
from roboui.pipeline import SketchToSitePipeline, plan_context, truncate_to_tokens
from roboui.sandbox import apply_csp, network_guard, sandboxed_iframe


def png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), "white").save(buf, format="PNG")
    return buf.getvalue()


# ---- mock inference ---------------------------------------------------------
def test_mock_vision_then_code():
    m = MockBackend()
    assert "LAYOUT" in m.generate("s", "u", png_bytes(), 100)
    assert "```html" in m.generate("s", "u", None, 100)


def test_factory_defaults_to_mock_and_rejects_unknown():
    assert isinstance(create_backend(Settings(backend="mock")), MockBackend)
    with pytest.raises(BackendUnavailableError):
        create_backend(Settings(backend="nope"))


def test_missing_model_file_is_graceful(tmp_path):
    s = Settings(backend="llama_cpp", model_path=str(tmp_path / "x.gguf"),
                 mmproj_path=str(tmp_path / "y.gguf"))
    with pytest.raises(ModelFileMissingError, match="README"):
        LlamaCppBackend(s)


def test_cloud_requires_credentials():
    with pytest.raises(BackendUnavailableError):
        create_backend(Settings(backend="cloud"))


# ---- pipeline ---------------------------------------------------------------
def test_pipeline_end_to_end_with_mock():
    backend = MockBackend(n_ctx=4096)
    res = SketchToSitePipeline(backend).run(png_bytes(), "dark theme")
    assert res.html.startswith("<!DOCTYPE html>") and "```" not in res.html
    assert "cdn.tailwindcss.com" in res.html
    assert [c["has_image"] for c in backend.calls] == [True, False]
    # vision output is carried into the code step
    assert "PAGE PURPOSE" in backend.calls[1]["user"]


def test_pii_in_notes_never_reaches_model():
    backend = MockBackend()
    res = SketchToSitePipeline(backend).run(png_bytes(), "contact bob@corp.com key sk-abcdefghijklmnop1234")
    sent = " ".join(c["user"] for c in backend.calls)
    assert "bob@corp.com" not in sent and "sk-abcdefghijklmnop1234" not in sent
    assert res.redactions == {"API_KEY": 1, "EMAIL": 1}


def test_context_plan_fits_window():
    for n in (1024, 2048, 8192):
        p = plan_context(n)
        assert p.description_budget + p.code_max_tokens < n
        assert p.vision_max_tokens <= p.description_budget


def test_description_truncated_to_budget():
    out = truncate_to_tokens("word " * 5000, 100)
    assert len(out) < 500 and "truncated" in out
    assert truncate_to_tokens("short", 100) == "short"


def test_small_context_still_works():
    res = SketchToSitePipeline(MockBackend(n_ctx=1024)).run(png_bytes())
    assert res.plan.n_ctx == 1024 and res.html


# ---- hardware ---------------------------------------------------------------
GiB = 2**30


def test_context_throttled_on_low_memory():
    low = HardwareInfo(free_ram_bytes=3 * GiB, free_vram_bytes=None)
    high = HardwareInfo(free_ram_bytes=64 * GiB, free_vram_bytes=None)
    assert throttle_context(8192, 1024, high, 2 * GiB) == 8192
    t = throttle_context(8192, 1024, low, 2 * GiB)
    assert 1024 <= t < 8192 and t % 256 == 0


def test_context_never_below_min():
    tiny = HardwareInfo(free_ram_bytes=1 * GiB, free_vram_bytes=None)
    assert throttle_context(8192, 1024, tiny, 4 * GiB) == 1024


def test_gpu_layer_offload_choice():
    assert choose_gpu_layers(5, HardwareInfo(GiB, 8 * GiB), 3 * GiB) == 5
    assert choose_gpu_layers(-1, HardwareInfo(GiB, 16 * GiB), 3 * GiB) == -1
    partial = choose_gpu_layers(-1, HardwareInfo(GiB, 2 * GiB), 4 * GiB)
    assert 0 <= partial < 32


# ---- sandbox ----------------------------------------------------------------
def test_network_guard_blocks_and_restores():
    with network_guard():
        with pytest.raises(NetworkBlockedError):
            socket.create_connection(("example.com", 80))
        s = socket.socket()
        with pytest.raises(NetworkBlockedError):
            s.connect(("127.0.0.1", 9))
        s.close()
    # restored: a real connect attempt now fails normally, not via our guard
    s = socket.socket()
    s.settimeout(0.5)
    try:
        s.connect(("127.0.0.1", 9))
    except NetworkBlockedError:  # pragma: no cover
        pytest.fail("guard not restored")
    except OSError:
        pass
    finally:
        s.close()


def test_csp_injected_and_iframe_sandboxed():
    page = "<html><head><title>x</title></head><body></body></html>"
    assert "connect-src &#x27;none&#x27;" in sandboxed_iframe(page) or "connect-src 'none'" in apply_csp(page)
    assert apply_csp(page).index("Content-Security-Policy") < apply_csp(page).index("<title>")
    frame = sandboxed_iframe(page)
    assert 'sandbox="allow-scripts"' in frame and "allow-same-origin" not in frame
    assert "Content-Security-Policy" in apply_csp("<div>fragment</div>")
