import tensorflow as tf
import pandas as pd
import numpy as np
import os
import librosa
from tensorflow import keras

from sttPrac import devidingWav


def compute_melspectrogram_(audio_path, n_mels=128, hop_length=512, n_fft=2048):
    y, sr = librosa.load(audio_path)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)

    return mel_spec

def preprocess_audio_(audio_path, target_length=256):
    mel_spec = compute_melspectrogram_(audio_path)
    current_length = mel_spec.shape[1]
    if current_length < target_length:
        mel_spec = np.pad(mel_spec, ((0, 0), (0, target_length - current_length)), mode='constant')
    elif current_length > target_length:
        start = (current_length - target_length) // 2
        mel_spec = mel_spec[:, start:start+target_length]

    return mel_spec

def load_data(csv_file_path):
    data = pd.read_csv(csv_file_path)
    X = []
    y = []

    for index, row in data.iterrows():
        audio_path = row['PATH']
        mel_spectrogram = preprocess_audio_(audio_path)
        X.append(mel_spectrogram)

    X = np.array(X)

    return X

def process_predict(model_path, test_csv_path, output_csv_path):
    model = keras.models.load_model(model_path)
    X = load_data(test_csv_path)
    pred = model.predict(X)
    pred_classes = np.argmax(pred, axis=1)
    csv = pd.read_csv(test_csv_path)
    csv['pred'] = pred_classes
    csv.to_csv(output_csv_path, index=False)
    return csv

def work():
    devidingWav.work("output.wav")
    model_path = 'model_9.h5'
    test_csv_path = 'timestamps.csv'
    output_csv_path = 'result.csv'
    res = ""
    result = process_predict(model_path, test_csv_path, output_csv_path)

    for index, row in result.iterrows():    # 전체를 출력하는데 pred가 1이면 괄호안에 문장출력
        text = row['TEXT']
        prediction = row['pred']


        if prediction == 1:
            # print(f'({text})')
            res = res + text + "\n"
        else:
            print(text)
    return res