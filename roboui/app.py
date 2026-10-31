"""RoboUI Streamlit app: upload a sketch, get a website."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
import streamlit.components.v1 as components

from roboui.config import Settings, configure_logging
from roboui.errors import RoboUIError
from roboui.inference import create_backend
from roboui.pipeline import SketchToSitePipeline
from roboui.sandbox import sandboxed_iframe

logger = logging.getLogger("roboui.app")

st.set_page_config(page_title="RoboUI", page_icon="🤖", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load_backend(backend_name: str):
    settings = Settings.from_env()
    object.__setattr__(settings, "backend", backend_name)  # frozen dataclass override
    return create_backend(settings)


def main() -> None:
    settings = Settings.from_env()
    configure_logging(settings.log_level)

    st.title("🤖 RoboUI")
    st.caption("Hand-drawn sketch → website. Runs on-device with Gemma 4 + LangChain.")

    with st.sidebar:
        st.header("Settings")
        options = ["mock", "llama_cpp", "cloud"]
        backend_name = st.selectbox(
            "Inference backend", options,
            index=options.index(settings.backend) if settings.backend in options else 0,
            help="mock needs no model weights. llama_cpp runs quantized Gemma 4 locally.",
        )
        st.caption("🔒 Network is blocked during local inference; previews run under a no-network CSP.")

    left, right = st.columns(2)
    with left:
        upload = st.file_uploader("Upload your sketch", type=["png", "jpg", "jpeg", "webp"])
        notes = st.text_area("Optional notes", placeholder="e.g. dark theme, indigo accent")
        go = st.button("Generate website", type="primary", disabled=upload is None)
        if upload:
            st.image(upload, caption="Your sketch", use_container_width=True)

    if go and upload:
        try:
            backend = load_backend(backend_name)
            with st.spinner("Analyzing sketch and generating code..."):
                result = SketchToSitePipeline(backend).run(upload.getvalue(), notes)
            st.session_state["result"] = result
        except RoboUIError as exc:
            logger.error("Generation failed: %s", exc)
            st.error(f"{type(exc).__name__}: {exc}")
        except OSError as exc:
            logger.exception("Could not read upload")
            st.error(f"Could not process the uploaded image: {exc}")

    result = st.session_state.get("result")
    if result:
        with right:
            tab_preview, tab_code, tab_desc = st.tabs(["Preview", "HTML", "Vision analysis"])
            with tab_preview:
                components.html(sandboxed_iframe(result.html), height=720, scrolling=True)
            with tab_code:
                st.code(result.html, language="html")
                st.download_button("Download index.html", result.html, "index.html", "text/html")
            with tab_desc:
                st.text(result.description)
                if result.redactions:
                    st.warning(f"PII/secrets masked before inference: {result.redactions}")
                p = result.plan
                st.caption(f"Backend: {result.backend} · context {p.n_ctx} tokens · "
                           f"description budget {p.description_budget} · max output {p.code_max_tokens}")


main()
