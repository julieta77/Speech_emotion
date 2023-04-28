############################################## We import the bookstore we will need #########################################################

import streamlit as st
import pandas as pd
import numpy as np  
import librosa 
import soundfile 
import os, glob
from joblib import load 



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



st.title('Welcome to my emotion recognition project') 

st.markdown('In this project we created a model for the detection of the three most important emotions :orange["Happy"], :blue["Sad"] and :red["Angry"]. It currently has an accuracy of 82.8%.')

st.markdown('To use this website, simply record your voice on any voice recorder and upload the file to predict your excitement.')





audio_file = st.file_uploader("Upload audio file", type=['wav','.mp3','.mp4']) #For the user to enter the sound file



if audio_file is not None:
    texto = st.empty() 
    texto.header("Analyzing...") 
    prep = extract_feature(audio_file,mfcc=True, chroma=True, mel=True)
    emotion = predict_emotion(prep)
    if emotion == 'happy':
        emoji = "😁"
    if emotion == 'sad':
        emoji = "😭"
    if emotion == 'angry':
        emoji = "😡"
    texto.empty()
    st.subheader(f"La emoción detectada es {emotion} {emoji}")
 
