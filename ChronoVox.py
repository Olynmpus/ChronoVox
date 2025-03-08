import streamlit as st
from streamlit_webrtc import webrtc_streamer
import vosk
import json
import queue
import soundfile as sf

model = vosk.Model("model")  # Ensure the Vosk model is available in your directory

q = queue.Queue()

def audio_callback(frame):
    q.put(frame.to_ndarray())

webrtc_ctx = webrtc_streamer(
    key="speech-recognition",
    audio_receiver_size=256,
    mode="sendrecv",
    async_processing=True,
)

if webrtc_ctx.audio_receiver:
    while True:
        audio_frame = webrtc_ctx.audio_receiver.get_frame()
        audio_data = audio_frame.to_ndarray()

        # Process with Vosk
        rec = vosk.KaldiRecognizer(model, 16000)
        rec.AcceptWaveform(audio_data.tobytes())
        result = json.loads(rec.Result())

        st.write(result["text"])  # Display transcribed text
