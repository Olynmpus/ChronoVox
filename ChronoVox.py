import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import vosk
import json
import numpy as np
import queue
import os

# Define Vosk Model Path (Using a Smaller Model)
MODEL_PATH = "model/vosk-model-small-en-us"

# Check if Model Exists
if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
    st.error("⚠️ Vosk model not found! Please upload the small Vosk model to the 'model/' folder.")
    st.stop()  # Stop execution if model is missing
else:
    st.success("✅ Small Vosk model loaded successfully!")

# Session state for stopping transcription
if "stop_transcription" not in st.session_state:
    st.session_state.stop_transcription = False

# Define Speech Recognition Processor
class SpeechRecognitionProcessor(AudioProcessorBase):
    def __init__(self):
        self.q = queue.Queue()

        # Log model path
        st.write(f"🔍 Checking model path: {MODEL_PATH}")

        # Try loading the model
        try:
            self.model = vosk.Model(MODEL_PATH)
            st.success("✅ Vosk model successfully loaded!")
        except Exception as e:
            st.error(f"❌ Failed to load Vosk model: {str(e)}")
            st.stop()
        
        self.rec = vosk.KaldiRecognizer(self.model, 16000)
        self.transcriptions = []

    def recv(self, frame):
        if st.session_state.stop_transcription:
            return frame  # Stop processing if user clicks "Stop"

        audio_data = frame.to_ndarray()
        self.q.put(audio_data)
        return frame

    def process_audio(self):
        if not self.q.empty():
            audio_data = self.q.get()
            if self.rec.AcceptWaveform(audio_data.tobytes()):
                result = json.loads(self.rec.Result())
                text = result.get("text", "")
                if text:
                    self.transcriptions.append(text)
                    return text

        return ""

# Streamlit UI
st.title("🎤 ChronoVox: Real-Time Speech Transcription (Small Model)")

st.write("This app captures speech from your microphone and transcribes it in real-time using the **smaller Vosk ASR model**.")

# WebRTC Audio Stream
webrtc_ctx = webrtc_streamer(
    key="speech-recognition",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=SpeechRecognitionProcessor,
    media_stream_constraints={"video": False, "audio": True},
)

# Stop Button
if st.button("🛑 Stop Transcription"):
    st.session_state.stop_transcription = True
    st.success("🔴 Transcription Stopped!")

# Display real-time transcription
if webrtc_ctx.audio_processor and not st.session_state.stop_transcription:
    transcription = webrtc_ctx.audio_processor.process_audio()
    if transcription:
        st.write("📝 **Live Transcription:** ", transcription)

# Show full transcription history
if webrtc_ctx.audio_processor:
    st.subheader("📜 Full Transcription History")
    st.write(" ".join(webrtc_ctx.audio_processor.transcriptions))
