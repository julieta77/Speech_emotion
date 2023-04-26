import streamlit as st
import pandas as pd
import numpy as np  
import librosa 
import soundfile 
import os, glob
from joblib import load 



####
def extract_feature(audio, mfcc, chroma, mel):
    #with soundfile.SoundFile(file_name) as sound_file:
    X, sample_rate= librosa.load(audio)
    #sample_rate=sound_file.samplerate
    #stft = None
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


st.title('Prueba') 



audio_file = st.file_uploader("Upload audio file", type=['wav'])



if audio_file is not None:
    st.title("Analyzing...")
    file_details = {"Filename": audio_file.name, "FileSize": audio_file.size}
    st.write(file_details)
    # st.subheader(f"File {file_details['Filename']}"
    a = st.audio(audio_file, format='audio/wav', start_time=0)
    path = os.path.join("audio", audio_file.name)
    #save_audio(audio_file) 
    #print(path) 
    #print(path)
    prueba = extract_feature(a,mfcc=True, chroma=True, mel=True) 
#
    #emotion_model = load('speech_emotion.joblib')
    #pred = emotion_model.predict([prueba]) 
    #print(pred)