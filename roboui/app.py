"""Turn a hand-drawn web page layout (JPEG) into a static HTML file.

Uses LangChain with a multimodal Gemma model served by Ollama.
"""
import argparse
import base64
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

OLLAMA_URL = "http://localhost:11434"
MODEL = "gemma4:e4b"

SYSTEM_PROMPT = """You are a front-end developer. The user gives you a hand-drawn \
sketch of a web page layout. Convert it into a single static HTML file.

Rules:
- Reproduce the layout using <div> elements (use flexbox or grid with inline <style>).
- Draw boxes labelled as buttons as <button> elements, keeping their text.
- Draw boxes marked with an X, a picture or "img" as <img> elements, using
  https://placehold.co/WIDTHxHEIGHT as src and a descriptive alt.
- Keep any text written in the sketch.
- Use only HTML and CSS: no JavaScript, no external libraries.
- Output ONLY the complete HTML document, starting with <!DOCTYPE html>. No explanations.
- Only consider the box only includes text: image, button, text. Ignore any other box"""


def extract_html(text: str) -> str:
    """Strip markdown code fences the model may wrap around the HTML."""
    match = re.search(r"```(?:html)?\s*(.*?)```", text, re.DOTALL)
    return (match.group(1) if match else text).strip()


def sketch_to_html(image_path: Path, model: str, base_url: str) -> str:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode()
    llm = ChatOllama(model=model, base_url=base_url, temperature=0.2)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=[
                {"type": "text", "text": "Convert this sketch into HTML."},
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{image_b64}",
                },
            ]
        ),
    ]
    return extract_html(llm.invoke(messages).content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="JPEG of the drawn layout")
    parser.add_argument("-o", "--output", type=Path, default=Path("output.html"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--base-url", default=OLLAMA_URL)
    args = parser.parse_args()

    if args.image.suffix.lower() not in {".jpg", ".jpeg"}:
        parser.error("input must be a JPEG (.jpg / .jpeg)")

    html = sketch_to_html(args.image, args.model, args.base_url)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
