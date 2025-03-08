import os
import queue
import json
import wave
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import vosk

# ---- CONFIG ----
MODEL_PATH = "./Voice2Text"  # Ensure the correct path in your GitHub repo
OUTPUT_FILE = "transcription.txt"  # Store transcriptions

# ---- DOWNLOAD CHECK ----
if not os.path.exists(MODEL_PATH):
    st.error("Vosk model not found! Please ensure it's uploaded to the correct folder in your GitHub repository.")
    st.stop()

# Load the Vosk model
asr_model = vosk.Model(MODEL_PATH)

# ---- AUDIO PROCESSOR ----
class SpeechRecognitionProcessor(AudioProcessorBase):
    def __init__(self):
        self.q = queue.Queue()
        self.rec = vosk.KaldiRecognizer(asr_model, 16000)
        self.transcriptions = []

    def recv(self, frame: av.AudioFrame):
        audio_data = frame.to_ndarray()
        self.q.put(audio_data)

        # Convert audio frame to speech
        if self.rec.AcceptWaveform(audio_data.tobytes()):
            result = json.loads(self.rec.Result())
            text = result.get("text", "")

            if text:
                self.transcriptions.append(text)
                with open(OUTPUT_FILE, "a") as f:
                    f.write(text + "\n")

        return frame

# ---- STREAMLIT UI ----
st.title("🗣️ Voice2Text: Real-time Speech-to-Text Transcription")
st.write("This app captures speech from your microphone and transcribes it using Vosk.")

# Start WebRTC
webrtc_ctx = webrtc_streamer(
    key="speech-recognition",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=SpeechRecognitionProcessor,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

# Stop Button
if st.button("Stop Recording"):
    webrtc_ctx.stop()
    st.success("Recording stopped.")

# Display Transcription
st.subheader("Live Transcription:")
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r") as f:
        st.text_area("Transcription", f.read(), height=200)

st.write("📄 The transcription is saved in `transcription.txt` for later processing.")
