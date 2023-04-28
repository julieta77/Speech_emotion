############################################## We import the bookstore we will need #########################################################

import streamlit as st
import pandas as pd
import numpy as np  
import librosa 
import soundfile 
import os, glob
from joblib import load 
import time


model = load('speech_emotion.joblib') # We load the model 


def extract_feature(audio, mfcc, chroma, mel): # Audio preprocessing
    audio1, sample_rate= librosa.load(audio)
    if chroma:
        stft=np.abs(librosa.stft(audio1))
        result=np.array([])
    if mfcc:
        mfccs=np.mean(librosa.feature.mfcc(y=audio1, sr=sample_rate, n_mfcc=40).T, axis=0)
        result=np.hstack((result, mfccs))
    if chroma:
        chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma))
    if mel: 
        mel=np.mean(librosa.feature.melspectrogram(y=audio1, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
    return result


def predict_emotion(preprocessed_audio):
    # Get the prediction from the model
    prediction = model.predict([preprocessed_audio])
    # Decoding the prediction to get the emotion
    emotion = prediction[0]
    return  emotion



st.set_page_config(
   page_title="Speech emotion",
   page_icon="🗣️",  
   layout="wide")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title('Welcome to my emotion recognition project') 


audio_file = st.file_uploader("Upload audio file", type=['wav','.mp3','.mp4'])



if audio_file is not None:
    texto = st.empty() 
    texto.header("Analyzing...") 
    prep = extract_feature(audio_file,mfcc=True, chroma=True, mel=True)
    emotion = predict_emotion(prep)
    texto.empty()
    texto.subheader(f"La emoción detectada es {emotion}")
    time.sleep(5)
    texto.empty() 




#import av
#from streamlit_webrtc import (
#    AudioProcessorBase,
#    ClientSettings,
#    WebRtcMode,
#    webrtc_streamer,
#)
#
#class MicrophoneProcessor(AudioProcessorBase):
#    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
#        return frame
#
#client_settings = ClientSettings(
#    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
#    media_stream_constraints={
#        "audio": True,
#        "video": False,
#    },
#)
#
#webrtc_ctx = webrtc_streamer(
#    key="microphone",
#    mode=WebRtcMode.SENDRECV,
#    client_settings=client_settings,
#    audio_processor_factory=MicrophoneProcessor,
#)
#
#if webrtc_ctx.audio_receiver:
#    st.write('escuchando..')
#    st.audio(webrtc_ctx.audio_receiver.get_audio_queue(), format="audio/ogg", start_time=0) 
#
#
##import sounddevice as sd
#import wavio
#
#duration = 5  # duración de la grabación en segundos
#fs = 44100  # frecuencia de muestreo
#channels = 1  # número de canales (mono)
#
#def record_audio():
#    myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
#    sd.wait()  # espera hasta que termine la grabación
#    return myrecording
#
#if st.button("Grabar audio"):
#    audio = record_audio()
#    st.audio(audio, format="audio/wav", start_time=0)



