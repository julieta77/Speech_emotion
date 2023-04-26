############################################## We import the bookstore we will need #########################################################

import streamlit as st
import pandas as pd
import numpy as np  
import librosa 
import soundfile 
import os, glob
from joblib import load 


model = load('speech_emotion.joblib') # We load the model 


def extract_feature(file_name, mfcc, chroma, mel):
    X, sample_rate= librosa.load(file_name)
    if chroma:
        stft=np.abs(librosa.stft(X))
        result=np.array([])
    if mfcc:
        mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result=np.hstack((result, mfccs))
    if chroma:
        chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma))
    if mel: 
        mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
    return result


def predict_emotion(preprocessed_audio):
    # Obtener la predicción del modelo
    prediction = model.predict([preprocessed_audio])

    # Decodificar la predicción para obtener la emoción
    emotion = prediction[0]
    return  emotion


st.title('Prueba') 


audio_file = st.file_uploader("Upload audio file", type=['wav'])


if audio_file is not None:
    st.title("Analyzing...")
    #file_details = {"Filename": audio_file.name, "FileSize": audio_file.size}
    #st.write(file_details)
    prep = extract_feature(audio_file,mfcc=True, chroma=True, mel=True)
    emotion = predict_emotion(prep)
    st.write(f"La emoción detectada es: {emotion}")
