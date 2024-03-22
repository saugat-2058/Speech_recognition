import numpy as np
from hmmlearn import hmm
from fastdtw import fastdtw
import librosa
import sounddevice as sd
import tensorflow as tf
from tensorflow.keras import layers, models

# Global variables for models and reference MFCC
hmm_model = None
rnn_model = None
reference_mfcc = None

# Function to extract MFCC features from audio
def extract_mfcc(audio):
    mfccs = librosa.feature.mfcc(y=audio, sr=44100, n_mfcc=13)
    return mfccs

# Function to calculate DTW distance
def calculate_dtw_distance(reference, test):
    distance, _ = fastdtw(reference, test)
    return distance

# Function to create an HMM model
def create_hmm_model(n_components):
    return hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100)

# Function to train an HMM model
def train_hmm_model(model, data):
    model.fit(data)

# Function to create an RNN model
def create_rnn_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(layers.LSTM(128))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Main function for voice recognition
def voice_recognition(reference_path):
    global hmm_model, rnn_model, reference_mfcc

    # Extract features using MFCC from reference audio
    reference_audio, _ = librosa.load(reference_path, sr=44100)
    reference_mfcc = extract_mfcc(reference_audio)

    # Train HMM model
    n_components = 3  # Number of states in HMM
    hmm_model = create_hmm_model(n_components)
    train_hmm_model(hmm_model, reference_mfcc.T)

    # Create and train RNN model
    num_classes = 10  # Replace with your number of classes
    rnn_model = create_rnn_model((reference_mfcc.shape[1], reference_mfcc.shape[0]), num_classes)

    # Main loop for real-time voice recognition
    # with sd.InputStream(callback=process_audio, channels=1, samplerate=44100):
    #     print("Listening for input... Press Ctrl+C to exit.")
    #     sd.sleep(1000000)

# Callback function for processing audio in real-time
def process_audio(indata, frames, time, status):
    global hmm_model, rnn_model, reference_mfcc

    # Extract features using MFCC from real-time audio
    audio = indata.flatten()
    test_mfcc = extract_mfcc(audio)

    # Calculate DTW distance
    dtw_distance = calculate_dtw_distance(reference_mfcc.T, test_mfcc.T)
    print(f"DTW Distance: {dtw_distance}")

    # Score using HMM
    hmm_score = hmm_model.score(test_mfcc.T)
    print(f"HMM Log Likelihood: {hmm_score}")

    # Predict using the trained RNN model
    # rnn_predictions = rnn_model.predict(np.expand_dims(test_mfcc.T, axis=0))
    # print("RNN Predictions:", rnn_predictions)

if __name__ == "__main__":
    reference_audio_path = "recorded_audio.wav"
    voice_recognition(reference_audio_path)
