"""Safely pull HTML out of raw LLM output (strips markdown fences and chatter)."""
from __future__ import annotations

import re

from .errors import CodeExtractionError

_FENCE = re.compile(r"```[ \t]*(?:html|HTML|htm)?[ \t]*\r?\n(.*?)(?:\r?\n)?```", re.DOTALL)
_UNCLOSED_FENCE = re.compile(r"```[ \t]*(?:html|HTML|htm)?[ \t]*\r?\n(.*)\Z", re.DOTALL)
_DOC = re.compile(r"(<!DOCTYPE html.*?</html\s*>|<html[\s>].*?</html\s*>)", re.DOTALL | re.IGNORECASE)
_TAGLIKE = re.compile(r"<\s*(?:html|body|div|main|section|header|head)\b", re.IGNORECASE)


def extract_html(raw: str) -> str:
    """Return clean HTML from model output.

    Handles: fenced ```html blocks (closed or truncated), bare documents with
    leading/trailing prose, and fragments. Raises CodeExtractionError if nothing
    HTML-like is found.
    """
    if not raw or not raw.strip():
        raise CodeExtractionError("Model returned empty output.")

    candidates = _FENCE.findall(raw)
    if candidates:
        # Prefer the fenced block that looks most like a full document, else the longest.
        body = max(candidates, key=lambda c: (bool(_DOC.search(c)), len(c)))
    else:
        m = _UNCLOSED_FENCE.search(raw)  # output truncated before closing fence
        body = m.group(1) if m else raw

    doc = _DOC.search(body)
    if doc:
        return doc.group(1).strip()
    if _TAGLIKE.search(body):
        return body.strip()
    raise CodeExtractionError("No HTML found in model output.")
