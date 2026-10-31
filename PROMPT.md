# TASK: Build "RoboUI" - On-Device Image-to-Code Web Generator

## 1. Objective & Core Workflow

Build a local, on-device Python application that takes a hand-drawn UI sketch, processes it using Gemma 4 (E2B or E4B) vision capabilities, and leverages LangChain to generate a functional frontend website.

- **Frontend/UI:** Use Streamlit or Gradio for the main user interface to enable rapid hackathon prototyping.
- **Orchestration:** Use Python 3 and LangChain to manage and chain the vision analysis prompt with the code-generation prompt.
- **Output:** The generated website must strictly use HTML, CSS (Tailwind via CDN), and vanilla JavaScript.

## 2. Architecture & Design Instructions

- **Image Ingestion:** A UI file uploader that accepts webcam captures or uploaded images of paper sketches.
- **Live Preview:** A sandboxed `iframe` or dynamic HTML rendering component within the app that instantly displays the generated website alongside the raw code.
- **Vision Parsing:** A LangChain prompt directing the Gemma 4 Vision model to identify layout structure, navigation bars, hero sections, buttons, and text areas from the sketch.

## 3. Safety & Privacy Architecture

- **Process-Level Network Isolation:** Wrap the core inference engine in a strict sandbox that denies outbound network traffic for the generated code, ensuring air-gapped processing.
- **On-the-Fly PII Redaction:** Implement a lightweight regex filter to mask standard PII (like API keys or emails) inadvertently included in the uploaded drawing before it hits the semantic processor.

## 4. Hardware Adaptation & Performance Controls

- **VRAM Management:** Implement aggressive memory offloading or model quantization (e.g., 4-bit GGUF via `llama-cpp-python`) to ensure the local Gemma 4 model can run smoothly on consumer hardware without Out of Memory (OOM) crashes.
- **Context Window Throttling:** Dynamically adjust the maximum context window to prevent memory exhaustion on lower-end devices.

## 5. Developer & Power-User Utilities

- **Bring Your Own Model (BYOM):** Provide a configuration `.env` or UI settings menu allowing users to sideload their own Gemma 4 `.gguf` or `.safetensors` weights.
- **Debug HUD:** Include a toggleable overlay detailing real-time inference metrics (tokens per second, VRAM utilization).

## 6. API Key & Secrets Instructions

- **Environment Templates:** Never hardcode secrets. Utilize a `.env` file for any fallback API logic and generate a `.env.example` file.
- **Graceful Degradation (Mock Mode):** Implement a `MockInferenceService`. If the local Gemma 4 model is not loaded, gracefully mock the generation process by returning a hardcoded HTML template after a 2-second delay. Do not attempt to download multi-gigabyte weights during the automated build/test phase.

## 7. Execution, Testing & Verification

- **Unit Testing:** Write `pytest` test cases to verify the LangChain prompt templates, the mock inference service, and the code-extraction regex (ensuring the app safely strips markdown ` ```html ` tags from the LLM output).
- **Execution Loop:** Build and verify the application using `pytest` and `python -m venv`.
- **Mock Verification:** Launch the Streamlit/Gradio app using the mock service to verify the image upload and `iframe` rendering work perfectly before loading real local AI models.

## 8. Project Hygiene & GitHub Instructions

- **Git Strategy:** Configure a comprehensive `.gitignore` to block multi-gigabyte model weights. Break changes into atomic, structured commits (e.g., `feat: scaffold Streamlit UI`, `test: add pytest suite`) rather than a single massive commit.
- **Error Handling:** Avoid empty `catch` blocks. Implement graceful error boundaries and structured logging for missing local model files or VRAM exhaustion.
- **Comprehensive README.md:** Generate a production-grade README. Include a LangChain data flow diagram, local model setup instructions, and the trade-offs of using local Gemma 4 vs a cloud API.

## 9. Artifact Generation & Resume Assets

Once the mock verification loop is complete, generate an Antigravity Artifact featuring:

- A clean terminal output summary showing the successful build and `pytest` execution.
- A technical overview explaining how the local LangChain pipeline manages token limits while maintaining context between the vision and coding steps.
- 3 high-impact, ready-to-use resume bullet points detailing the implementation of on-device multimodal AI, LangChain orchestration, and rapid UI prototyping.
