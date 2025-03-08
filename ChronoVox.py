import streamlit as st
import sounddevice as sd
import numpy as np
import wave
import os
import json
import time
from vosk import Model, KaldiRecognizer
from datetime import datetime

# -----------------------------
# 🔹 Vosk Model Auto-Download
# -----------------------------
MODEL_PATH = "vosk_model"
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"

if not os.path.exists(MODEL_PATH):
    import zipfile, requests, shutil
    st.info("Downloading Vosk Model (First-time Setup)...")
    response = requests.get(MODEL_URL, stream=True)
    with open("vosk_model.zip", "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
    with zipfile.ZipFile("vosk_model.zip", "r") as zip_ref:
        zip_ref.extractall(".")
    os.rename("vosk-model-small-en-us-0.15", MODEL_PATH)
    os.remove("vosk_model.zip")
    st.success("Vosk Model Installed Successfully!")

model = Model(MODEL_PATH)

# -----------------------------
# 🔹 Streamlit App UI
# -----------------------------
st.set_page_config(page_title="Speech Recorder", layout="centered")

st.title("🗣️ Meeting Speech Recorder")
st.image("meeting_highlight.png", width=250)  # Meeting Highlighted Logo

# User identifier
user_name = st.text_input("Enter Your Name (Unique ID):", max_chars=50)

# -----------------------------
# 🔹 Audio Recording Parameters
# -----------------------------
DURATION = 600  # Max 10 min (in seconds)
SAMPLE_RATE = 16000
CHANNELS = 1

# -----------------------------
# 🔹 Audio Recording Function
# -----------------------------
def record_audio(duration):
    st.info("Recording... Speak Now 🎤")
    audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16")
    sd.wait()
    return audio_data

# -----------------------------
# 🔹 Speech-to-Text with Vosk
# -----------------------------
def transcribe_audio(audio_data):
    wf = wave.open("temp_audio.wav", "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)  # 16-bit
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(audio_data.tobytes())
    wf.close()

    wf = wave.open("temp_audio.wav", "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)  # Enable word timestamps

    transcript = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if "result" in result:
                for word in result["result"]:
                    transcript.append(f"{word['word']} [{round(word['start'],2)}s]")

    os.remove("temp_audio.wav")  # Clean up
    return transcript

# -----------------------------
# 🔹 Start Recording Button
# -----------------------------
if st.button("🎙️ Start Recording"):
    if not user_name:
        st.warning("Please enter your name before recording!")
    else:
        st.session_state["recording"] = True
        audio_data = record_audio(DURATION)
        transcript = transcribe_audio(audio_data)

        # Save the transcript as a TXT file
        filename = f"{user_name}_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, "w") as f:
            f.write("\n".join(transcript))

        st.success("✅ Recording & Transcription Completed!")
        st.download_button("📥 Download Transcript", data="\n".join(transcript), file_name=filename, mime="text/plain")

        st.subheader("🔹 Transcribed Text with Timestamps:")
        st.write("\n".join(transcript))
