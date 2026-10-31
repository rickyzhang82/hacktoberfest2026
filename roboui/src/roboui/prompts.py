"""LangChain prompt templates for the two-stage pipeline."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

VISION_SYSTEM = (
    "You are a meticulous UI analyst. You are shown a hand-drawn wireframe sketch. "
    "Describe its layout precisely so a front-end developer can rebuild it without seeing it."
)

VISION_HUMAN = (
    "Analyze the attached sketch. Output a structured description with these sections:\n"
    "1. PAGE PURPOSE (one line)\n"
    "2. LAYOUT (top-to-bottom regions, columns, relative sizes)\n"
    "3. COMPONENTS (buttons, inputs, nav, cards, images, lists) with any visible labels\n"
    "4. INTERACTIONS implied by the sketch\n"
    "Be concise (under {max_words} words). Do not write code.\n"
    "Extra notes from the user: {user_notes}"
)

CODE_SYSTEM = (
    "You are an expert front-end engineer. Generate a single self-contained web page "
    "using ONLY HTML, Tailwind CSS via the CDN <script src=\"https://cdn.tailwindcss.com\"></script>, "
    "and vanilla JavaScript. No frameworks, no build step, no network requests "
    "(no fetch/XMLHttpRequest/WebSocket), no external images (use CSS or inline SVG). "
    "Return exactly one ```html fenced block and nothing else."
)

CODE_HUMAN = (
    "Build the website described below. Make it responsive, accessible and visually polished. "
    "Wire up the implied interactions with vanilla JavaScript.\n\n"
    "--- UI DESCRIPTION ---\n{description}\n--- END DESCRIPTION ---\n"
    "Extra notes from the user: {user_notes}"
)

vision_prompt = ChatPromptTemplate.from_messages(
    [("system", VISION_SYSTEM), ("human", VISION_HUMAN)]
)
code_prompt = ChatPromptTemplate.from_messages(
    [("system", CODE_SYSTEM), ("human", CODE_HUMAN)]
)
