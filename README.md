# 🎙️ AI-Powered Dubbing

This project is an **AI-powered dubbing web app** built with Flask.  
It takes an input audio file, transcribes it using Whisper, translates the speech with Google's Gemini,  
synthesizes the translated text into speech with Coqui TTS, and finally applies **RVC (Retrieval-Based Voice Conversion)** to match a target speaker's voice.  

---

## 🚀 Features

- 🎧 **Speech-to-Text**: Uses **Faster Whisper** for high-quality transcription.
- 🌍 **Translation**: Integrates with **Google Gemini** for accurate multilingual translation.
- 🗣️ **Text-to-Speech**: Uses **Coqui XTTS** to generate speech in the target language.
- 🎤 **Voice Cloning**: Applies **RVC voice conversion** to mimic a specific speaker's voice.
- 🎛️ **Audio Alignment**: Maintains natural timing and rhythm for professional-quality dubbing.
- 🌐 **Flask Web App**: Simple UI for uploading files and downloading dubbed output.

---

## 🏗️ Tech Stack

| Component                     | Technology                                |
|-------------------------------|------------------------------------------|
| **Backend Framework**        | Flask (Python)                           |
| **Transcription**            | Faster Whisper (large-v3 model)          |
| **Translation**              | Google Gemini Generative Model           |
| **Text-to-Speech (TTS)**     | Coqui XTTS v2                            |
| **Voice Conversion (VC)**    | RVC (Retrieval-Based Voice Conversion)   |
| **Audio Processing**         | Pydub, SciPy                             |
| **Hardware Support**         | CUDA GPU acceleration (optional)         |

---

## 🧩 Workflow

1. **Upload Audio** + Target Speaker Models (`.pth` RVC models).
2. **Transcription**: Whisper model converts speech → text.
3. **Translation**: Gemini translates sentences into target language.
4. **Text-to-Speech**: Coqui XTTS generates target-language audio.
5. **Voice Conversion**: RVC applies the target speaker's timbre.
6. **Output**: Download the final dubbed audio.
