# roboui

LangChain app that turns a hand-drawn web page layout (JPEG) into a static HTML file with `div`, `button` and `img` elements. It uses Gemma 4 e4b served by Ollama.

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
