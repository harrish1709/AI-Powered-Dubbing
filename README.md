# AI-Powered Dubbing Pipeline

A multilingual AI dubbing system that integrates speech recognition, translation, and text-to-speech synthesis for cross-lingual voice conversion.

## Architecture

1. **Automatic Speech Recognition (ASR)** — [Whisper v3 Large](https://github.com/openai/whisper) for high-accuracy speech-to-text transcription
2. **Translation** — [Gemini 2.5 Flash](https://deepmind.google/technologies/gemini/) for real-time cross-lingual translation
3. **Text-to-Speech (TTS)** — [XTTSv2](https://github.com/coqui-ai/TTS) for natural-sounding speech synthesis in the target language

## Features

- End-to-end dubbing pipeline from source audio to translated speech
- Support for multiple language pairs
- Maintains speaker voice characteristics through TTS
- Modular architecture — each component can be swapped independently

## Tech Stack

- Python
- OpenAI Whisper v3 Large
- Gemini 2.5 Flash
- XTTSv2
- FFmpeg (audio processing)

## Getting Started

```bash
# Clone the repository
git clone https://github.com/harrish1709/AI-Powered-Dubbing.git
cd AI-Powered-Dubbing

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python main.py --input audio.mp3 --source-lang en --target-lang ta
```
