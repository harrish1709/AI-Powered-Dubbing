from flask import Flask, render_template, request, send_file 
import os
import uuid
from faster_whisper import WhisperModel
from rvc_python.infer import RVCInference
from types import MethodType
import scipy.io.wavfile as wavfile
import torch
from torch.serialization import add_safe_globals
from fairseq.data.dictionary import Dictionary
from pydub import AudioSegment
from TTS.api import TTS
import re
import google.generativeai as genai
import time
import json

genai.configure(api_key="AIzaSyDxt6LK_wn_WMCJ-a9Oc9qQgKlsbzinT9o")

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
add_safe_globals({Dictionary, XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs})

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "cloned_audio_files")
os.makedirs(UPLOAD_DIR, exist_ok=True)

whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2",gpu=True)

def patched_infer_file(self, input_path, output_path):
    if not self.current_model:
        raise ValueError("Model not loaded.")
    model_info = self.models[self.current_model]
    file_index = model_info.get("index", "")
    result = self.vc.vc_single(
        sid=0,
        input_audio_path=input_path,
        f0_up_key=self.f0up_key,
        f0_method=self.f0method,
        file_index=file_index,
        index_rate=self.index_rate,
        filter_radius=self.filter_radius,
        resample_sr=self.resample_sr,
        rms_mix_rate=self.rms_mix_rate,
        protect=self.protect,
        f0_file="",
        file_index2=""
    )
    wav = result[0] if isinstance(result, tuple) else result
    wavfile.write(output_path, self.vc.tgt_sr, wav)

def extract_gemini_text(response):
    try:
        if hasattr(response, "text") and response.text:
            return response.text.strip()
        
        if hasattr(response, "candidates") and response.candidates:
            parts = []
            for cand in response.candidates:
                if hasattr(cand, "content") and hasattr(cand.content, "parts"):
                    for part in cand.content.parts:
                        if hasattr(part, "text") and part.text:
                            parts.append(part.text)
                        elif hasattr(part, "data") and isinstance(part.data, str):
                            parts.append(part.data)
            if parts:
                return "\n".join(parts).strip()

        raise ValueError("No usable text parts found")

    except Exception as e:
        print(f"⚠️ Gemini parsing error: {e}")
        try:
            with open("gemini_debug.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(response.to_dict(), indent=2, ensure_ascii=False))
            print("📄 Raw Gemini response dumped to gemini_debug.json")
        except Exception as dump_err:
            print(f"⚠️ Failed to dump raw response: {dump_err}")
        return ""

def translate_with_gemini_batched(sentences,source_lang="English",target_lang="Arabic",retries=3,batch_size=5): 
    all_translations = []
    model = genai.GenerativeModel("gemini-2.5-flash")

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]

        prompt = (
        f"Translate the following {source_lang} sentences into {target_lang}.\n"
        "⚠️ Very Important: Do not add, remove, or summarize any content - — every cultural or historical detail must appear.\n"
        "Translate every word and phrase exactly as it appears and use fluent {target_lang}, but keep all place names and historical references intact.\n"
        "Keep the same number of sentences and the same order.\n"
        "Return only the translations, one per line, without commentary.\n\n"
    )

        prompt += "\n".join(batch)

        translated_lines = []
        for attempt in range(retries):
            try:
                response = model.generate_content(prompt)
                output_text = extract_gemini_text(response)

                if not output_text.strip():
                    raise ValueError("Empty response from Gemini")

                translated_lines = [re.sub(r'^\d+[\.\)]\s*', '', line).strip()
                                     for line in output_text.split("\n") if line.strip()]

                if len(translated_lines) != len(batch):
                    # Pad or truncate
                    while len(translated_lines) < len(batch):
                        translated_lines.append("")
                    translated_lines = translated_lines[:len(batch)]

                break

            except Exception as e:
                print(f"⚠️ Gemini error (attempt {attempt+1}): {e}")
                time.sleep(2)

        if not translated_lines:
            translated_lines = [""] * len(batch)

        all_translations.extend(translated_lines)

    return all_translations

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])                     
def process():
    audio = request.files.get("audio")                       
    model_files = request.files.getlist("models")           
    target_lang = request.form.get("target_lang")            

    if not audio or not model_files or not target_lang:      
        return "Missing required inputs", 400

    input_filename = f"input_{uuid.uuid4().hex}.wav"         
    input_path = os.path.join(UPLOAD_DIR, input_filename)     
    audio.save(input_path)                                    

    segments_en, info = whisper_model.transcribe(
        input_path,
        word_timestamps=True
    )
    segments = list(segments_en)

    if not segments:
        return "No valid segments found in transcription.", 400

    aligned_segments = []
    for seg in segments:
        words = list(seg.words) if seg.words else []
        if not words:
            continue

        gaps = [(words[i].start - words[i-1].end) for i in range(1, len(words))]
        avg_gap = sum(gaps) / len(gaps) if gaps else 0.3
        std_gap = (sum((g - avg_gap) ** 2 for g in gaps) / len(gaps)) ** 0.5 if gaps else 0.1
        gap_threshold = avg_gap + std_gap

        word_durations = [(w.end - w.start) for w in words]
        avg_word_duration = sum(word_durations) / len(word_durations) if word_durations else 0.4

        phrase_duration = words[-1].end - words[0].start
        est_len = int(phrase_duration / avg_word_duration) if avg_word_duration > 0 else 8
        max_phrase_len = max(5, min(15, est_len))

        phrase_groups = []
        current_phrase = [words[0]]
        for i in range(1, len(words)):
            gap = words[i].start - words[i-1].end
            if gap > gap_threshold or len(current_phrase) >= max_phrase_len:
                phrase_groups.append(current_phrase)
                current_phrase = []
            current_phrase.append(words[i])
        if current_phrase:
            phrase_groups.append(current_phrase)

        source_phrases = [
            " ".join([w.word for w in grp])
            for grp in phrase_groups
        ]

        translated_phrases = translate_with_gemini_batched(   
            source_phrases,
            source_lang="English",
            target_lang="Arabic"
        )

        for grp, translated in zip(phrase_groups, translated_phrases):  
            if not translated.strip():
                continue

            phrase_start = grp[0].start                     
            phrase_end = grp[-1].end                        

            wav_path = os.path.join(
                UPLOAD_DIR, f"tts_phrase_{uuid.uuid4().hex}.wav"
            )
            try:
                tts.tts_to_file(
                    text=translated,
                    file_path=wav_path,
                    speaker_wav=input_path,
                    language=target_lang
                )
                phrase_audio = AudioSegment.from_wav(wav_path) 

                original_duration_ms = int((phrase_end - phrase_start) * 1000)
                gap_ms = max(0, original_duration_ms - len(phrase_audio))  
                if gap_ms > 0:
                    phrase_audio += AudioSegment.silent(duration=gap_ms)    

                aligned_segments.append((phrase_start, phrase_audio))       

            except Exception as e:
                print(f"❌ Error during phrase TTS: {e}")      

    if not aligned_segments:                                  
        return "TTS synthesis failed", 500

    aligned_segments.sort(key=lambda x: x[0])                  
    final_audio = AudioSegment.silent(duration=0)              
    last_end_ms = 0                                            
    for start_time, phrase_audio in aligned_segments:          
        start_ms = int(start_time * 1000)                      
        if start_ms > last_end_ms:                             
            final_audio += AudioSegment.silent(duration=start_ms - last_end_ms)
        final_audio += phrase_audio                            
        last_end_ms = len(final_audio)                         

    tts_output_path = os.path.join(
        UPLOAD_DIR, f"tts_{uuid.uuid4().hex}.wav"
    )
    final_audio.export(tts_output_path, format="wav")          

    duration_ms = len(final_audio)
    chunk_duration = duration_ms // len(model_files)
    output_filenames = []

    for idx, model_file in enumerate(model_files):
        model_filename = f"model_{uuid.uuid4().hex}.pth"
        model_path = os.path.join(UPLOAD_DIR, model_filename)
        model_file.save(model_path)

        start = idx * chunk_duration
        end = (idx + 1) * chunk_duration if idx < len(model_files) - 1 else duration_ms
        chunk = final_audio[start:end]
        chunk_filename = f"chunk_{uuid.uuid4().hex}.wav"
        chunk_path = os.path.join(UPLOAD_DIR, chunk_filename)
        chunk.export(chunk_path, format="wav")

        output_filename = f"cloned_{uuid.uuid4().hex}.wav"
        output_path = os.path.join(UPLOAD_DIR, output_filename)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        rvc = RVCInference(model_path=model_path, device=device)
        rvc.infer_file = MethodType(patched_infer_file, rvc)
        rvc.set_params(f0up_key=0, index_rate=0.75)
        rvc.infer_file(input_path=chunk_path, output_path=output_path)

        output_filenames.append(output_filename)

    return render_template("result.html", filenames=output_filenames)

@app.route("/download/<filename>")
def download_file(filename):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        return "File not found.", 404
    return send_file(file_path, as_attachment=True, download_name=filename, mimetype="audio/wav")

if __name__ == "__main__":
    app.run(debug=True)