# roboui

**Draw a web page. Get a web page.**

roboui turns a hand-drawn web page layout (a JPEG) into a working static HTML file with `div`, `button` and `img` elements. It is a LangChain app that runs Gemma 4 e4b locally through Ollama.

## Why roboui?

Building user interfaces has long been locked behind code. A designer or artist with a clear vision had to learn HTML and CSS, or hand the idea to a developer and hope it survived the translation.

roboui aims to **liberate UI technology for non-technical design artists**:

- **Sketch instead of code.** Draw boxes, buttons and image placeholders on paper or a tablet. That drawing is the only input.
- **Instant, tangible results.** Within seconds the sketch becomes a real page you can open in any browser, share and iterate on.
- **No toolchain to learn.** The output is plain HTML and CSS, with no frameworks, build steps or JavaScript.
- **Private and free.** The model runs on your own machine through Ollama. Your designs never leave your computer and there are no API fees.
- **A bridge to developers.** The generated HTML is readable and editable, so it works as a starting point that a developer can pick up and extend.

Creative people should be able to bring their interface ideas to life directly, so roboui makes the pencil the only tool they need.

## Setup

```bash
ollama pull gemma4:e4b      # Ollama must be running at http://localhost:11434
pip install -r requirements.txt
```

## Usage

```bash
python app.py layout.jpg -o output.html
```

Options: `--model` (default `gemma4:e4b`) and `--base-url` (default `http://localhost:11434`).

## Tips for drawing

- Write the label inside each button box.
- Mark image areas with a big X.
- Use dark ink on a white background.
