import os
import streamlit as st
import numpy as np
# from sync_docs import sync_and_rebuild
# from connect_memory_with_llm import get_qa_chain

from ChatBot.localInterface.sync_docs import sync_and_rebuild
from ChatBot.localInterface.connect_memory_with_llm import get_qa_chain

import threading
import time
import schedule
import datetime

import speech_recognition as sr
import pyttsx3

from vosk import Model, KaldiRecognizer
import sounddevice as sd
import json

from dotenv import load_dotenv
load_dotenv()

ADMIN_PASSWORD = os.getenv("ADMIN_PASS", "change_me")
MODEL_PATH = "ChatBot/models/vosk/vosk-model-small-en-us-0.15"
VOSK_MODEL_PATH = "ChatBot/models/vosk"

@st.cache_resource
def load_vosk_model():
    return Model(MODEL_PATH)

def main():

    # ---------------- Scheduler ----------------
    def job():
        try:
            print("Running sync_pdfs.py task...")
            print( datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            sync_and_rebuild()
            print("Sync completed!")
        except Exception as e:
            print(f"Error during sync: {e}")

    def start_scheduler():
        schedule.every(2).hours.do(job)
        while True:
            schedule.run_pending()
            time.sleep(60)

    thread = threading.Thread(target=start_scheduler, daemon=True)
    thread.start()

    # ---------------- Voice Functions ----------------
    def voice_input():
        if not os.path.exists(VOSK_MODEL_PATH):
            st.error("❌ Vosk model not found!")
            return ""

        model = Model(VOSK_MODEL_PATH)
        recognizer = KaldiRecognizer(model, 16000)

        st.info("🎤 Listening... Speak now")

        audio_data = []

        def callback(indata, frames, time, status):
            if status:
                print(status)
            audio_data.append(indata.copy())

        try:
            with sd.InputStream(
                samplerate=16000,
                channels=1,
                dtype="int16",
                callback=callback
            ):
                sd.sleep(5000)  # listen for 5 seconds
        except Exception as e:
            st.error(f"Microphone error: {e}")
            return ""

        audio = np.concatenate(audio_data)

        if recognizer.AcceptWaveform(audio.tobytes()):
            result = json.loads(recognizer.Result())
            text = result.get("text", "")
        else:
            result = json.loads(recognizer.FinalResult())
            text = result.get("text", "")

        if text.strip():
            return text
        else:
            st.warning("⚠️ No speech detected. Please try again.")
            return ""

    def speak(text):
        engine = pyttsx3.init()
        engine.setProperty('rate', 200)      # adjust speaking rate
        engine.setProperty('volume', 1.0)    # volume (0.0 to 1.0)
        engine.say(text)
        engine.runAndWait()

    # ---------------- UI ----------------
    st.set_page_config(page_title="VNR VJIET Assistant", page_icon="🤖")

    hide_streamlit_style = """
        <style>
            /* Hide main menu (the top-right dots) */
            #MainMenu {display: none;}

            /* Hide deploy button */
            .st-emotion-cache-zq5wmm {display: none;}

            /* remove padding on top */
            .block-container {padding-top: 1rem;}

            /* reduce padding on bottom */
            .st-emotion-cache-139wi93 {padding-bottom: 2rem;}
        </style>
    """

    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.title("Welcome to VNR VJIET")
    st.write("I am your AI assistant, here to help you with your queries about EEE department and Engineering.")

    # ---- Sidebar ----
    st.sidebar.header("Admin Controls")
    if password := st.sidebar.text_input("Enter admin password", type="password"):
        if password != ADMIN_PASSWORD:  # Replace with secure password management in production
            st.sidebar.error("Incorrect password.")
        else: 
            if st.sidebar.button("🔄 Update latest data"):
                with st.spinner("Rebuilding database..."):
                    try:
                        sync_and_rebuild()
                        st.sidebar.success("✅ Database updated.")
                    except Exception as e:
                        st.sidebar.error(f"❌ Sync failed: {e}")

    # Initialize session
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display old messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # ---------------- Input Section ----------------
    # col1, col2 = st.columns([3, 1])
    # with col1:
    prompt = st.chat_input("What is your query?")
    # with col2:
    if st.button("🎙️ Speak"):
        prompt = voice_input()

    # ---------------- Response Handling ----------------
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try:
            with st.spinner("Thinking..."):
                qa_chain = get_qa_chain()
                response = qa_chain(prompt)
                result = response["result"] if isinstance(response, dict) else response

                st.chat_message("assistant").markdown(result)
                st.session_state.messages.append({'role': 'assistant', 'content': result})
                st.session_state.chat_history.append((prompt, result))

                speak(result)

        except Exception as e:
            st.error(f"An error occurred while processing your query: {e}")
