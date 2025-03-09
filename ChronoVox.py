import os
import queue
import json
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import vosk

# ---- CONFIG ----
MODEL_PATH = "./Voice2Text"
DEFAULT_SAVE_FOLDER = "Transcriptions"

# ---- CHECK VOSK MODEL ----
if not os.path.exists(MODEL_PATH):
    st.error("Vosk model not found! Ensure it's uploaded to your GitHub repository.")
    st.stop()

asr_model = vosk.Model(MODEL_PATH)

# ---- STREAMLIT UI ----
st.title("🗣️ Meeting Speech Recorder")
st.write("This app records speech and transcribes it using Vosk. Your transcription is saved instantly.")

# User enters their name
user_name = st.text_input("Enter Your Name (Unique ID):", max_chars=50).strip()
save_folder = st.text_input("Enter folder name to save transcriptions (leave blank for default):", "").strip()

# Ensure a valid folder path
if not save_folder:
    save_folder = DEFAULT_SAVE_FOLDER

if user_name:
    os.makedirs(save_folder, exist_ok=True)
    OUTPUT_FILE = os.path.join(save_folder, f"{user_name.replace(' ', '_')}_transcription.txt")
else:
    st.warning("⚠️ Please enter your name before starting the recording!")

# ---- AUDIO PROCESSOR ----
class SpeechRecognitionProcessor(AudioProcessorBase):
    def __init__(self):
        self.q = queue.Queue()
        self.rec = vosk.KaldiRecognizer(asr_model, 16000)
        self.transcriptions = []

    def recv(self, frame: av.AudioFrame):
        audio_data = frame.to_ndarray().astype("int16")  # Ensure correct format
        self.q.put(audio_data)

        st.write("🔹 Receiving audio frame...")  # Debugging message

        if self.rec.AcceptWaveform(audio_data.tobytes()):
            result = json.loads(self.rec.Result())
            text = result.get("text", "")

            if text:
                self.transcriptions.append(text)
                with open(OUTPUT_FILE, "a") as f:
                    f.write(f"{user_name}: {text}\n")

        return frame

# ---- START STREAMING ----
if user_name:
    webrtc_ctx = webrtc_streamer(
        key="speech-recognition",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=SpeechRecognitionProcessor,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

    # Debugging: Show WebRTC Status
    if webrtc_ctx and webrtc_ctx.state.playing:
        st.success("🎙️ Recording in progress...")
    else:
        st.warning("❗ WebRTC is not running. Check microphone permissions.")

    # Stop Button
    if st.button("⏹️ Stop Recording"):
        if webrtc_ctx:
            webrtc_ctx.stop()
        st.success(f"✅ Recording stopped. Transcription saved in `{OUTPUT_FILE}`.")

    # Display Transcription
    st.subheader("📜 Live Transcription:")
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            transcription_text = f.read()
            st.text_area("Transcription", transcription_text, height=250)

    st.write(f"📄 Your transcription is continuously saved in `{OUTPUT_FILE}`.")
else:
    st.warning("Please enter your name before starting the recording.")
