import streamlit as st
import os
import requests
import zipfile
import vosk
import json
import numpy as np
import queue
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

# Define the Model Directory and URL
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "vosk-model-small-en-us")
MODEL_ZIP = os.path.join(MODEL_DIR, "vosk-model-small-en-us.zip")
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"

# Function to Download and Extract the Model
def download_vosk_model():
    st.warning("🚀 Downloading the Vosk model... This may take a few minutes.")
    
    os.makedirs(MODEL_DIR, exist_ok=True)  # Create directory if not exists

    # Download model using requests
    response = requests.get(MODEL_URL, stream=True)
    with open(MODEL_ZIP, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Extract the model
    with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)

    # Rename extracted folder
    extracted_folder = os.path.join(MODEL_DIR, "vosk-model-small-en-us-0.15")
    if os.path.exists(extracted_folder):
        os.rename(extracted_folder, MODEL_PATH)

    # Remove the zip file
    os.remove(MODEL_ZIP)

# Check if Model Exists, Download if Missing
if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
    download_vosk_model()

st.success("✅ Vosk model is ready!")

# Session state for stopping transcription
if "stop_transcription" not in st.session_state:
    st.session_state.stop_transcription = False

# Define Speech Recognition Processor
class SpeechRecognitionProcessor(AudioProcessorBase):
    def __init__(self):
        self.q = queue.Queue()

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
