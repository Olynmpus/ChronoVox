import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import vosk
import json
import numpy as np
import queue
import wave
import os

# Load the Vosk ASR Model
MODEL_PATH = "model"  # Ensure you have a Vosk model downloaded
if not os.path.exists(MODEL_PATH):
    st.error("Vosk model not found. Please download and place it in the 'model' folder.")
else:
    asr_model = vosk.Model(MODEL_PATH)

# Define Audio Processing Class
class SpeechRecognitionProcessor(AudioProcessorBase):
    def __init__(self):
        self.q = queue.Queue()
        self.rec = vosk.KaldiRecognizer(asr_model, 16000)

    def recv(self, frame):
        audio_data = frame.to_ndarray()
        self.q.put(audio_data)
        return frame

    def process_audio(self):
        if not self.q.empty():
            audio_data = self.q.get()
            if self.rec.AcceptWaveform(audio_data.tobytes()):
                result = json.loads(self.rec.Result())
                return result.get("text", "")

        return ""

# Streamlit UI
st.title("🎤 ChronoVox: Real-Time Speech Transcription")

st.write("This app captures speech from your microphone and transcribes it in real-time using Vosk ASR.")

# WebRTC Audio Stream
webrtc_ctx = webrtc_streamer(
    key="speech-recognition",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=SpeechRecognitionProcessor,
    media_stream_constraints={"video": False, "audio": True},
)

if webrtc_ctx.audio_processor:
    transcription = webrtc_ctx.audio_processor.process_audio()
    if transcription:
        st.write("📝 **Transcription:** ", transcription)
