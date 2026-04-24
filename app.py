import os
import re
import math
import tempfile
import torch
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)
from threading import Thread

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

LLAMA_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
OPENAI_TRANSCRIPTION_MODEL = "whisper-1"
OPENAI_MINUTES_MODEL = "gpt-4o"
CHUNK_DURATION_MS = 10 * 60 * 1000

SYSTEM_PROMPT = """
You produce professional minutes of meetings from transcripts.
Return clean markdown (no code fences) that includes:
- A summary section with attendees, location, and date (if detectable)
- Key discussion points
- Takeaways
- Action items with owners and deadlines where mentioned
Be concise, professional, and precise.
""".strip()

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Add it to your .env file.")
    return OpenAI(api_key=api_key)


def clean_llama_response(raw: str) -> str:
    for marker in ["<|assistant|>", "[/INST]", "assistant\n"]:
        if marker in raw:
            raw = raw.split(marker)[-1]
    raw = re.sub(r"<\|.*?\|>", "", raw)
    return raw.strip()


def clean_transcription(text: str) -> str:
    """Remove non-English/non-ASCII noise (e.g. Japanese hallucinations from Whisper)."""
    # Keep only printable ASCII + common punctuation
    cleaned = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Collapse multiple spaces
    cleaned = re.sub(r' +', ' ', cleaned)
    return cleaned.strip()


def split_audio(audio_path: str, chunk_ms: int = CHUNK_DURATION_MS):
    try:
        from pydub import AudioSegment
    except ImportError:
        return None

    audio = AudioSegment.from_file(audio_path)
    duration_ms = len(audio)

    if duration_ms <= chunk_ms:
        return [audio_path]

    chunks = []
    num_chunks = math.ceil(duration_ms / chunk_ms)

    for i in range(num_chunks):
        start = i * chunk_ms
        end = min((i + 1) * chunk_ms, duration_ms)
        chunk = audio[start:end]
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        chunk.export(tmp.name, format="mp3")
        chunks.append(tmp.name)

    return chunks


# ── Transcription ─────────────────────────────────────────────────────────────

def transcribe_whisper_local(audio_path: str) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-medium.en",
        dtype=dtype,
        device=device,
        return_timestamps=True,
        chunk_length_s=30,
    )
    result = pipe(audio_path)
    return clean_transcription(result["text"])


def transcribe_openai(audio_path: str, progress_callback=None) -> str:
    client = get_openai_client()
    chunks = split_audio(audio_path)

    if chunks is None or chunks == [audio_path]:
        if progress_callback:
            progress_callback("🎙️ Transcribing audio…")
        with open(audio_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model=OPENAI_TRANSCRIPTION_MODEL,
                file=f,
                response_format="text",
            )
        return clean_transcription(result)

    transcriptions = []
    total = len(chunks)
    for i, chunk_path in enumerate(chunks):
        if progress_callback:
            progress_callback(f"🎙️ Transcribing chunk {i+1} of {total}…")
        with open(chunk_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model=OPENAI_TRANSCRIPTION_MODEL,
                file=f,
                response_format="text",
            )
            transcriptions.append(result)
        if chunk_path != audio_path:
            try:
                os.unlink(chunk_path)
            except Exception:
                pass

    return clean_transcription(" ".join(transcriptions))


# ── Minutes Generation ────────────────────────────────────────────────────────

def generate_minutes_openai(transcription: str) -> str:
    client = get_openai_client()
    response = client.chat.completions.create(
        model=OPENAI_MINUTES_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcription:\n\n{transcription}"},
        ],
    )
    return response.choices[0].message.content


def generate_minutes_llama(transcription: str):
    if not torch.cuda.is_available():
        yield "⚠️ LLaMA requires a CUDA GPU. Please choose OpenAI instead."
        return

    hf_token = os.getenv("HF_TOKEN")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Transcription:\n\n{transcription}"},
    ]

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL, device_map="auto", quantization_config=quant_config, token=hf_token,
    )

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    thread = Thread(target=model.generate, kwargs=dict(inputs=inputs, max_new_tokens=2000, streamer=streamer))
    thread.start()

    output = ""
    for chunk in streamer:
        output += chunk
        yield output


# ── Gradio Pipeline ───────────────────────────────────────────────────────────

def run_pipeline(audio_file, transcription_engine, minutes_engine, progress=gr.Progress(track_tqdm=True)):
    if audio_file is None:
        yield "⚠️ Please upload an audio file.", "", ""
        return

    audio_path = audio_file if isinstance(audio_file, str) else audio_file.name

    yield "🎙️ Starting transcription…", "", ""
    try:
        if transcription_engine == "OpenAI Whisper API":
            status_holder = ["🎙️ Starting transcription…"]
            def on_progress(msg):
                status_holder[0] = msg
            transcription = transcribe_openai(audio_path, progress_callback=on_progress)
            yield status_holder[0], "", ""
        else:
            yield "🎙️ Transcribing locally with Whisper (this may take a while)…", "", ""
            transcription = transcribe_whisper_local(audio_path)
    except Exception as e:
        yield f"❌ Transcription error: {e}", "", ""
        return

    yield "✅ Transcription done! Generating minutes…", transcription, ""

    try:
        if minutes_engine == "OpenAI GPT-4o":
            yield "⚙️ Sending to GPT-4o…", transcription, ""
            minutes = generate_minutes_openai(transcription)
            yield "✅ Done!", transcription, minutes
        else:
            partial = ""
            for partial in generate_minutes_llama(transcription):
                yield "⚙️ Generating with LLaMA…", transcription, partial
            yield "✅ Done!", transcription, partial
    except Exception as e:
        yield f"❌ Minutes generation error: {e}", transcription, ""


# ── UI ────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --primary: #1a1a2e;
    --accent: #e8c547;
    --surface: #16213e;
    --surface2: #0f3460;
    --text: #eaeaea;
    --text-muted: #8892a4;
    --radius: 12px;
}

body, .gradio-container {
    background: var(--primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

.app-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid rgba(232,197,71,0.2);
    margin-bottom: 2rem;
}

.app-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: var(--accent);
    margin: 0 0 0.4rem;
    letter-spacing: -0.5px;
}

.app-header p {
    font-size: 1rem;
    color: var(--text-muted);
    font-weight: 300;
    margin: 0;
}

.gr-box, .gr-panel, .gr-form, .gr-block {
    background: var(--surface) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: var(--radius) !important;
}

label, .gr-radio-row label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
}

button.primary {
    background: var(--accent) !important;
    color: var(--primary) !important;
    font-family: 'DM Mono', monospace !important;
    font-weight: 500 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 2rem !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
}

button.primary:hover { opacity: 0.85 !important; }

.status-box {
    background: var(--surface2) !important;
    border-left: 3px solid var(--accent) !important;
    border-radius: 6px !important;
    padding: 0.75rem 1rem !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    color: var(--text) !important;
}

textarea {
    background: #0d1b2a !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    border-radius: 8px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

/* ── Minutes markdown readability fix ── */
.minutes-panel, .minutes-panel * {
    color: var(--text) !important;
}

.minutes-panel h1, .minutes-panel h2, .minutes-panel h3 {
    color: var(--accent) !important;
    font-family: 'DM Serif Display', serif !important;
    margin-top: 1.2rem !important;
}

.minutes-panel h2 { font-size: 1.3rem !important; }
.minutes-panel h3 { font-size: 1.1rem !important; }

.minutes-panel p, .minutes-panel li {
    color: #d4dce8 !important;
    line-height: 1.7 !important;
    font-size: 0.95rem !important;
}

.minutes-panel strong {
    color: var(--accent) !important;
}

.minutes-panel ul, .minutes-panel ol {
    padding-left: 1.4rem !important;
}

.minutes-panel li {
    margin-bottom: 0.3rem !important;
}

.minutes-panel hr {
    border-color: rgba(232,197,71,0.2) !important;
    margin: 1rem 0 !important;
}
"""


def build_ui():
    with gr.Blocks(css=CUSTOM_CSS, title="AI Meeting Minutes") as demo:

        gr.HTML("""
        <div class="app-header">
            <h1>🎙️ AI Meeting Minutes</h1>
            <p>Upload a meeting recording — get structured, professional minutes in seconds.</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Upload Meeting Audio",
                    type="filepath",
                    sources=["upload"],
                )
                transcription_engine = gr.Radio(
                    choices=["OpenAI Whisper API", "Local Whisper (GPU required)"],
                    value="OpenAI Whisper API",
                    label="Transcription Engine",
                )
                minutes_engine = gr.Radio(
                    choices=["OpenAI GPT-4o", "Local LLaMA 3.2 (GPU required)"],
                    value="OpenAI GPT-4o",
                    label="Minutes Generation Engine",
                )
                run_btn = gr.Button("Generate Minutes", variant="primary")

            with gr.Column(scale=2):
                status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    elem_classes=["status-box"],
                    lines=1,
                )
                with gr.Tabs():
                    with gr.Tab("📄 Meeting Minutes"):
                        minutes_output = gr.Markdown(
                            label="Minutes",
                            elem_classes=["minutes-panel"],
                        )
                    with gr.Tab("📝 Raw Transcription"):
                        transcription_output = gr.Textbox(
                            label="Transcription",
                            lines=20,
                            interactive=False,
                        )

        run_btn.click(
            fn=run_pipeline,
            inputs=[audio_input, transcription_engine, minutes_engine],
            outputs=[status, transcription_output, minutes_output],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(share=True, inbrowser=True)
