# 🎙️ AI Meeting Minutes Generator

Automatically transcribes meeting audio and generates structured, professional meeting minutes using AI.

Built with **OpenAI Whisper**, **GPT-4o**, and **LLaMA 3.2** — with a clean **Gradio** UI.

---

## Features

- 🎧 **Audio transcription** via OpenAI Whisper API or local Whisper (Hugging Face)
- 📝 **Minutes generation** via GPT-4o or local LLaMA 3.2 (4-bit quantized)
- ⚡ **Streaming output** — see minutes appear in real time with LLaMA
- 🖥️ **Clean Gradio UI** — upload audio, choose engines, get minutes instantly
- 🔒 **Local-first option** — run fully offline with Whisper + LLaMA on GPU

---

## Output Format

Each set of minutes includes:
- **Summary** — attendees, location, date
- **Key Discussion Points**
- **Takeaways**
- **Action Items** with owners and deadlines

---

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/ghaderimaryam/ai-meeting-minutes.git
cd ai-meeting-minutes
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
```
Edit `.env` and add your keys:
```
OPENAI_API_KEY=your_openai_api_key_here
HF_TOKEN=your_huggingface_token_here   # Only needed for LLaMA
```

### 4. Run the app
```bash
python app.py
```
Then open `http://localhost:7860` in your browser.

---

## 🛠️ Engine Options

| Engine | Requires | Speed | Cost |
|---|---|---|---|
| OpenAI Whisper API | API key | Fast | ~$0.006/min |
| Local Whisper | GPU (CUDA) | Medium | Free |
| OpenAI GPT-4o | API key | Fast | Pay-per-use |
| Local LLaMA 3.2 | GPU + HF token | Slower | Free |

>  **Recommended for most users:** OpenAI Whisper API + GPT-4o

---

##  Project Structure

```
ai-meeting-minutes/
├── app.py               # Main Gradio application
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── .gitignore
└── README.md
```

---

##  Tech Stack

- [OpenAI API](https://platform.openai.com/) — Whisper transcription + GPT-4o minutes
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) — Local Whisper + LLaMA
- [Gradio](https://gradio.app/) — Web UI
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes) — 4-bit quantization for LLaMA

---

##  Sample Audio

You can test with this Denver City Council meeting extract:  
🔗 https://drive.google.com/file/d/1N_kpSojRR5RYzupz6nqM8hMSoEF_R7pU/view?usp=sharing

