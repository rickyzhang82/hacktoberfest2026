"""Two-stage LangChain pipeline: sketch -> structured description -> HTML.

    redact(notes) -> [vision prompt | model(image)] -> redact(description)
        -> fit description into context budget -> [code prompt | model] -> extract_html
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

from langchain_core.runnables import RunnableLambda

from .extract import extract_html
from .inference import InferenceBackend
from .pii import redact
from .prompts import CODE_HUMAN, CODE_SYSTEM, code_prompt, vision_prompt
from .sandbox import network_guard

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4  # coarse estimate; avoids needing the tokenizer
IMAGE_TOKENS = 512   # reserve for the encoded sketch in the vision turn
MAX_OUTPUT_TOKENS = 4096


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def truncate_to_tokens(text: str, budget: int) -> str:
    """Trim to a token budget, keeping the head (layout/purpose come first)."""
    if estimate_tokens(text) <= budget:
        return text
    keep = max(0, budget * CHARS_PER_TOKEN - 40)
    return text[:keep].rstrip() + "\n[...description truncated to fit context...]"


@dataclass
class ContextPlan:
    n_ctx: int
    vision_words: int
    vision_max_tokens: int
    description_budget: int
    code_max_tokens: int


def plan_context(n_ctx: int) -> ContextPlan:
    """Split the window between the two stages so the carried description always fits.

    code turn:   prompt overhead + description + generated HTML  <= n_ctx
    vision turn: prompt overhead + image + generated description <= n_ctx
    """
    overhead = estimate_tokens(CODE_SYSTEM + CODE_HUMAN) + 128  # template + user notes slack
    code_max = min(MAX_OUTPUT_TOKENS, max(256, n_ctx // 2))
    description_budget = max(64, n_ctx - overhead - code_max)
    vision_max = min(description_budget, 768)
    vision_words = max(40, int(vision_max * 0.6))
    return ContextPlan(n_ctx, vision_words, vision_max, description_budget, code_max)


@dataclass
class PipelineResult:
    html: str
    description: str
    redactions: dict[str, int] = field(default_factory=dict)
    plan: ContextPlan | None = None
    backend: str = ""


def _to_png(image_bytes: bytes, max_side: int = 1024) -> bytes:
    """Normalize any upload to a bounded-size PNG (less memory, consistent input)."""
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()


def _ocr_text(png: bytes) -> str:
    """Optional OCR so PII written in the drawing can be detected pre-inference."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""
    try:
        return pytesseract.image_to_string(Image.open(io.BytesIO(png)))
    except Exception as exc:  # pytesseract raises TesseractNotFoundError etc.
        logger.warning("OCR unavailable (%s); skipping image-text PII scan", exc)
        return ""


class SketchToSitePipeline:
    def __init__(self, backend: InferenceBackend, use_network_guard: bool = True):
        self.backend = backend
        # The explicit opt-in cloud backend legitimately needs the network.
        self.use_network_guard = use_network_guard and backend.name != "cloud"
        self._chain = (
            RunnableLambda(self._prepare)
            | RunnableLambda(self._vision)
            | RunnableLambda(self._carry_context)
            | RunnableLambda(self._codegen)
        )

    # ---- chain steps (each takes and returns a state dict) -------------------
    def _prepare(self, s: dict) -> dict:
        png = _to_png(s["image"])
        notes = redact(s.get("notes", "") or "")
        ocr = redact(_ocr_text(png))
        counts = dict(notes.counts)
        for k, v in ocr.counts.items():
            counts[k] = counts.get(k, 0) + v
        if ocr.total:
            # Text inside the picture contained secrets/PII. We cannot edit pixels
            # reliably, so the model is told to ignore any literal text values.
            notes_text = (notes.text + "\nIgnore any personal data or secrets visible in the image; "
                          "use generic placeholder text instead.").strip()
        else:
            notes_text = notes.text
        plan = plan_context(self.backend.n_ctx)
        return {"png": png, "notes": notes_text or "none", "redactions": counts, "plan": plan}

    def _vision(self, s: dict) -> dict:
        plan: ContextPlan = s["plan"]
        prompt = vision_prompt.invoke({"max_words": plan.vision_words, "user_notes": s["notes"]})
        system, human = prompt.messages[0].content, prompt.messages[1].content
        raw = self.backend.generate(system, human, s["png"], plan.vision_max_tokens)
        return {**s, "description": raw}

    def _carry_context(self, s: dict) -> dict:
        red = redact(s["description"])  # model output is untrusted too
        counts = dict(s["redactions"])
        for k, v in red.counts.items():
            counts[k] = counts.get(k, 0) + v
        desc = truncate_to_tokens(red.text, s["plan"].description_budget)
        return {**s, "description": desc, "redactions": counts}

    def _codegen(self, s: dict) -> dict:
        plan: ContextPlan = s["plan"]
        prompt = code_prompt.invoke({"description": s["description"], "user_notes": s["notes"]})
        system, human = prompt.messages[0].content, prompt.messages[1].content
        raw = self.backend.generate(system, human, None, plan.code_max_tokens)
        return {**s, "html": extract_html(raw)}

    # ---- public API ----------------------------------------------------------
    def run(self, image_bytes: bytes, notes: str = "") -> PipelineResult:
        state = {"image": image_bytes, "notes": notes}
        if self.use_network_guard:
            with network_guard():
                out = self._chain.invoke(state)
        else:
            out = self._chain.invoke(state)
        logger.info("Pipeline done: backend=%s redactions=%s", self.backend.name, out["redactions"])
        return PipelineResult(
            html=out["html"], description=out["description"],
            redactions=out["redactions"], plan=out["plan"], backend=self.backend.name,
        )
