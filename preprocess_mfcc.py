import os
import numpy as np
# import librosa
import librosa.display
# import scipy.signal
import scipy.io.wavfile
from scipy.io.wavfile import write as wav_write
import time
import matplotlib.pyplot as plt
from functools import lru_cache


def folder():
    return f"static/sounds/{int(time.time())}"

fold=folder()

@lru_cache(maxsize=10)
def get_window(n, type='hamming'):
    coefs = np.arange(n)
    window = 0.54 - 0.46 * np.cos(2 * np.pi * coefs / (n - 1))
    return window
plt.plot(get_window(512))

def freq_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)
plt.plot(freq_to_mel(np.arange(8000)))

def mel_to_freq(mels):
    return 700.0 * (np.power(10.0, mels / 2595.0) - 1.0)

@lru_cache(maxsize=10)
def get_filterbank(numfilters, filterLen, lowFreq, highFreq, samplingFreq):
    minwarpfreq = freq_to_mel(lowFreq)
    maxwarpfreq = freq_to_mel(highFreq)
    dwarp = (maxwarpfreq - minwarpfreq) / (numfilters + 1)
    f = mel_to_freq(np.arange(numfilters + 2) * dwarp + minwarpfreq) * (filterLen - 1) * 2.0 / samplingFreq
    i = np.arange(filterLen)[None, :]
    f = f[:, None]
    hislope = (i - f[:numfilters]) / (f[1:numfilters+1] - f[:numfilters])
    loslope = (f[2:numfilters+2] - i) / (f[2:numfilters+2] - f[1:numfilters+1])
    H = np.maximum(0, np.minimum(hislope, loslope))
    return H
H = get_filterbank(numfilters=20, filterLen=257, lowFreq=0, highFreq=8000, samplingFreq=16000)
fig = plt.figure(figsize=(20,10))
for h in H:
  plt.plot(h)

def apply_preemphasis(y, preemCoef=0.97):
    y[1:] = y[1:] - preemCoef*y[:-1]
    y[0] *= (1 - preemCoef)
    return y

def normalized(y, threshold=0):
    y -= y.mean()
    stddev = y.std()
    if stddev > threshold:
        y /= stddev
    return y
def mfsc(y, sfr, window_size=0.025, window_stride=0.010, window='hamming', normalize=False, log=True, n_mels=20, preemCoef=0, melfloor=1.0):
    win_length = int(sfr * window_size)
    hop_length = int(sfr * window_stride)
    n_fft = 2048
    lowfreq = 0
    highfreq = sfr/2
    
    # get window
    window = get_window(win_length)
    padded_window = np.pad(window, (0, n_fft - win_length), mode='constant')[:, None]
    
    # preemphasis
    y = apply_preemphasis(y.copy(), preemCoef)

    # scale wave signal
    y *= 32768
    
    # get frames
    num_frames = 1 + (len(y) - win_length) // hop_length
    pad_after = num_frames*hop_length + (n_fft - hop_length) - len(y)
    if pad_after > 0:
        y = np.pad(y, (0, pad_after), mode='constant')
    frames = np.lib.stride_tricks.as_strided(y, shape=(n_fft, num_frames), strides=(y.itemsize, hop_length * y.itemsize), writeable=False)
    windowed_frames = padded_window * frames
    D = np.abs(np.fft.rfft(windowed_frames, axis=0))

    # mel filterbank
    filterbank = get_filterbank(n_mels, n_fft/2 + 1, lowfreq, highfreq, sfr)
    mf = np.dot(filterbank, D)
    mf = np.maximum(melfloor, mf)
    if log:
        mf = np.log(mf)
    if normalize:
        mf = normalized(mf)

    return mf

def mfsc2mfcc(S, n_mfcc=12, dct_type=2, norm='ortho', lifter=22, cms=True, cmvn=True):
    # Discrete Cosine Transform
    M = scipy.fftpack.dct(S, axis=0, type=dct_type, norm=norm)[:n_mfcc]

    # Ceptral mean subtraction (CMS) 
    if cms or cmvn:
        M -= M.mean(axis=1, keepdims=True)

    # Ceptral mean and variance normalization (CMVN)
    if cmvn:
        M /= M.std(axis=1, keepdims=True)
    
    # Liftering
    elif lifter > 0:
        lifter_window = 1 + (lifter / 2) * np.sin(np.pi * np.arange(1, 1 + n_mfcc, dtype=M.dtype) / lifter)[:, np.newaxis]
        M *= lifter_window

    return M

def r_spectrogram():
    raw_file_path = "recorded_audio.wav"
    # sr=16000
    y, sr = librosa.load(raw_file_path)

    # Compute the Short-Time Fourier Transform (STFT)
    n_fft = 2048  # Number of FFT points
    hop_length = 512  # Hop length
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    # Convert to magnitude scale
    magnitude = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(magnitude, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Raw Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()

    # Save the spectrogram as a PNG image
    plt.savefig(os.path.join(fold, "caudio_files.png"))

    plt.title('Raw Sampling freq')
    # sfr, y = scipy.io.wavfile.read(wav)
    y = y/32768
    # print(wav, 'Sampling frequency: ', sfr)
    fig = plt.subplot(4,2,2)
    plt.title('Sampling Rate of raw audio file')
    plt.plot(y)
    # plt.savefig(os.path.join(fold, "csampling_frequency.png"))


    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig = plt.subplot(4,2,5)
    librosa.display.specshow(D, y_axis='linear', sr=sr)
    plt.title('Linear-frequency power spectrogram of raw audio file')
    # plt.savefig(os.path.join(fold, "clinear_freq.png"))


    S = mfsc(y, sr)
    fig = plt.subplot(4,2,6)
    librosa.display.specshow(S - S.min())
    plt.title('Mel-scaled power spectrogram of raw audio file')


       # MFCC(5)
    M = mfsc2mfcc(S)
    fig = plt.subplot(4,2,8)
    plt.plot(M[1,:])
    plt.savefig(os.path.join(fold, "clinear_freq_all.png"))

    # Compute the Mel spectrogram
    n_fft = 2048  # Number of FFT points
    hop_length = 512  # Hop length
    n_mels = 128  # Number of Mel bands
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    # Convert to decibel scale
    S_db = librosa.power_to_db(S, ref=np.max)

    # Plot the Mel spectrogram
    # plt.figure(figsize=(10, 6))
    # librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='viridis')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel Spectrogram')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Mel Frequency')
    # plt.tight_layout()

    # # Save the Mel spectrogram as a PNG image
    # plt.savefig(os.path.join(fold, "rmel_spectrogram.png"))

    # Optionally, display the Mel spectrogram
    # plt.show()

    # Compute MFCCs
    n_mfcc = 13  # Number of MFCC coefficients
    mfccs = librosa.feature.mfcc(S=S_db, n_mfcc=n_mfcc)

    # Display the MFCCs
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap='viridis')
    plt.colorbar()
    plt.title('MFCC')
    plt.xlabel('Time (s)')
    plt.ylabel('Raw MFCC Coefficient')
    plt.tight_layout()
    if not os.path.exists(fold):
            os.makedirs(fold)
    # Save the MFCCs as a PNG image
    plt.savefig(os.path.join(fold, "cmfcc.png"))

    # Optionally, display the MFCCs
    # plt.show()

    # Optionally, display the spectrogram
    # plt.show()



            # Compute the Short-Time Fourier Transform (STFT)
        # D = librosa.stft(y)

        # # Convert to magnitude scale
        # magnitude = np.abs(D)

        # # Plot the spectrogram
        # plt.figure(figsize=(10, 4))
        # librosa.display.specshow(librosa.amplitude_to_db(magnitude, ref=np.max), sr=sr, x_axis='time', y_axis='log')
        # plt.colorbar(format='%+2.0f dB')
        # plt.title('Spectrogram')
        # plt.tight_layout()

        # # Save the spectrogram as a PNG image
        # plt.savefig(os.path.join(fold, "raudio_files.png"))

        # Optionally, display the spectrogram
        # plt.show()

    

    

def p_spectrogram():
    p_file_path=os.path.join(fold, "paudio_file.wav")

    y, sr = librosa.load(os.path.join(fold, "paudio_file.wav"))

    # Compute the Short-Time Fourier Transform (STFT)
    n_fft = 2048  # Number of FFT points
    hop_length = 512  # Hop length
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    # Convert to magnitude scale
    magnitude = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(magnitude, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Preprocessed Audio Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()

    # Save the spectrogram as a PNG image
    plt.savefig(os.path.join(fold, "paudio_files.png"))

    plt.title('preprocessed Sampling freq')
    # sfr, y = scipy.io.wavfile.read(wav)
    y = y/32768
    # print(wav, 'Sampling frequency: ', sfr)
    fig = plt.subplot(4,2,2)
    plt.title('Sampling Rate of preprocessed audio file')
    plt.plot(y)
    # plt.savefig(os.path.join(fold, "csampling_frequency.png"))


    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig = plt.subplot(4,2,5)
    librosa.display.specshow(D, y_axis='linear', sr=sr)
    plt.title('Linear-frequency power spectrogram of preprocessed audio file')
    # plt.savefig(os.path.join(fold, "clinear_freq.png"))


    S = mfsc(y, sr)
    fig = plt.subplot(4,2,6)
    librosa.display.specshow(S - S.min())
    plt.title('Mel-scaled power spectrogram of preprocessed audio file')


       # MFCC(5)
    M = mfsc2mfcc(S)
    fig = plt.subplot(4,2,8)
    plt.plot(M[1,:])
    plt.savefig(os.path.join(fold, "plinear_freq_all.png"))


    n_fft = 2048  # Number of FFT points
    hop_length = 512  # Hop length
    n_mels = 128  # Number of Mel bands
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    # # Convert to decibel scale
    S_db = librosa.power_to_db(S, ref=np.max)

    # # Plot the Mel spectrogram
    # plt.figure(figsize=(10, 6))
    # librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='viridis')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel Spectrogram')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Mel Frequency')
    # plt.tight_layout()

    # # Save the Mel spectrogram as a PNG image
    # plt.savefig(os.path.join(fold, "pmel_spectrogram.png"))

    # Optionally, display the Mel spectrogram
    # plt.show()

    # Compute MFCCs
    n_mfcc = 13  # Number of MFCC coefficients
    mfccs = librosa.feature.mfcc(S=S_db, n_mfcc=n_mfcc)

    # Display the MFCCs
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap='viridis')
    plt.colorbar()
    plt.title('MFCC')
    plt.xlabel('Time (s)')
    plt.ylabel('Preprocesed MFCC Coefficient')
    plt.tight_layout()

    # Save the MFCCs as a PNG image
    plt.savefig(os.path.join(fold, "pmfcc.png"))

    # Optionally, display the MFCCs
    # plt.show()

    

class AudioPreprocessor:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr

    def load_audio(self, audio_file_path):
        y, sr = librosa.load(audio_file_path, sr=self.target_sr)
        if not os.path.exists(fold):
         os.makedirs(fold)
    
        # Define the output file path
        output_file_path = os.path.join(fold, "audio_file.wav")
        
        # Save audio data to a WAV file
        wav_write(output_file_path, sr, y)
        return y, sr

    def preemphasis(self, y, coefficient=0.97):
        y_preemphasized = scipy.signal.lfilter([1, -coefficient], 1, y)
        return y_preemphasized

    def frame_blocking(self, y, sr, frame_length=0.025, hop_length=0.010):
        frame_length = int(frame_length * sr)
        hop_length = int(hop_length * sr)
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        return frames


'''


The AudioPreprocessor class you provided seems to be a part of a larger system for processing audio data. Let's break down each method and its purpose:

1.__init__ method: This is the constructor method for the AudioPreprocessor class. It initializes an instance of the class with a default target sampling rate of 16000 Hz, but you can also provide a different target sampling rate if needed.

2.load_audio method: This method loads an audio file from the specified file path using the librosa.load function. Librosa is a Python package for music and audio analysis, and it provides various utilities for loading, processing, and analyzing audio data. The load_audio method loads the audio file and resamples it to the target sampling rate specified during initialization. Resampling ensures that all audio data processed by the system has a consistent sampling rate.

3.preemphasis method: Pre-emphasis is a technique used in audio processing to emphasize high-frequency components in the signal. It applies a high-pass filter to the audio data to amplify high-frequency components relative to lower-frequency components. In this method, the input audio signal y is filtered using a first-order FIR filter with coefficients [1, -coefficient]. The coefficient parameter controls the strength of the pre-emphasis, with a default value of 0.97. Pre-emphasis is commonly used in speech processing to improve the performance of subsequent processing steps such as feature extraction.

4.frame_blocking method: This method performs frame blocking or segmentation of the audio signal into overlapping frames. Frame blocking is a common preprocessing step in audio signal processing and analysis. It divides the audio signal into short, overlapping segments called frames, which are typically used for feature extraction. Each frame is windowed using a window function to reduce spectral leakage and then processed individually. The frame_length parameter specifies the length of each frame in seconds, and the hop_length parameter specifies the hop size or the amount of overlap between consecutive frames. In this method, the librosa.util.frame function is used to perform frame blocking on the pre-emphasized audio signal y.

Overall, the AudioPreprocessor class provides methods for loading audio data, applying pre-emphasis, and performing frame blocking, which are essential preprocessing steps in audio signal processing pipelines. These preprocessing steps help prepare the audio data for further analysis and feature extraction, such as computing spectrograms, Mel-frequency cepstral coefficients (MFCCs), or other audio features used in tasks like speech recognition, audio classification, and sound synthesis.


'''


class FeatureExtractor:
    def __init__(self):
        pass

    def compute_mfcc(self, y, sr, n_mfcc=13):
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return mfccs

    def compute_delta_mfcc(self, mfccs):
        delta_mfccs = librosa.feature.delta(mfccs)
        delta_delta_mfccs = librosa.feature.delta(mfccs, order=2)
        return delta_mfccs, delta_delta_mfccs
    

'''
The FeatureExtractor class you provided seems to be focused on extracting Mel-frequency cepstral coefficients (MFCCs) and their delta and delta-delta coefficients from audio signals. Let's break down each method:

1.__init__ method: This is the constructor method for the FeatureExtractor class. It initializes an instance of the class. Currently, it doesn't have any specific initialization logic, as indicated by the pass statement.

2.compute_mfcc method: This method computes the Mel-frequency cepstral coefficients (MFCCs) of an audio signal. MFCCs are widely used in audio signal processing for tasks such as speech recognition and audio classification. The librosa.feature.mfcc function is used to compute the MFCCs from the input audio signal y and its sampling rate sr. The n_mfcc parameter specifies the number of MFCC coefficients to compute, with a default value of 13. The method returns the computed MFCCs as a numpy array.

3.compute_delta_mfcc method: This method computes the delta and delta-delta coefficients of the MFCCs. Delta coefficients represent the rate of change of the MFCCs over time, while delta-delta coefficients represent the acceleration of the MFCCs. These additional features capture temporal dynamics in the audio signal and are commonly used in speech and audio analysis tasks. The librosa.feature.delta function is used to compute the delta and delta-delta coefficients from the input MFCCs. The order parameter specifies the order of differentiation, with a default value of 1 for delta coefficients and 2 for delta-delta coefficients. The method returns the computed delta and delta-delta coefficients as numpy arrays.

Overall, the FeatureExtractor class provides methods for computing MFCCs and their temporal derivatives (delta and delta-delta coefficients) from audio signals. These features are essential for characterizing the spectral and temporal characteristics of audio signals and are commonly used as input features for machine learning models in various audio processing applications.

'''



class PreprocessedMFCCFeatures:
    def __init__(self, target_sr=16000):
        self.audio_preprocessor = AudioPreprocessor(target_sr=target_sr)
        self.feature_extractor = FeatureExtractor()


    def preprocess_audio(self, audio_file_path):
        # Load audio
        y, sr = self.audio_preprocessor.load_audio(audio_file_path)
        
        # Preprocess audio
        y_preemphasized = self.audio_preprocessor.preemphasis(y)
        frames = self.audio_preprocessor.frame_blocking(y_preemphasized, sr)
        
        # Compute MFCC features
        mfccs = self.feature_extractor.compute_mfcc(y_preemphasized, sr)
        delta_mfccs, delta_delta_mfccs = self.feature_extractor.compute_delta_mfcc(mfccs)
        
        # Save MFCC features
        filename = os.path.basename(audio_file_path)
        folder_name = fold
        os.makedirs(folder_name, exist_ok=True)
        np.save(os.path.join(folder_name, f"mfcc.npy"), mfccs)
        
        # Save preprocessed audio in int16 format
        y_int16 = (y_preemphasized * np.iinfo(np.int16).max).astype(np.int16)
        scipy.io.wavfile.write(os.path.join(folder_name, f"paudio_file.wav"), sr, y_int16)


        
            
        return mfccs, delta_mfccs, delta_delta_mfccs



'''
The PreprocessedMFCCFeatures class appears to be responsible for preprocessing audio files to extract MFCC features along with their delta and delta-delta coefficients. Let's break down each part of the preprocess_audio method:

1.Initialization: In the constructor (__init__), the PreprocessedMFCCFeatures class initializes an instance with a target sampling rate (target_sr) defaulting to 16000 Hz. It also initializes instances of the AudioPreprocessor and FeatureExtractor classes.

2.preprocess_audio method: This method takes an audio file path as input and performs the following steps:

3.Load Audio: It uses the load_audio method of the AudioPreprocessor class to load the audio file and resample it to the target sampling rate.

4.Preprocessing: The loaded audio is preprocessed using the preemphasis method of the AudioPreprocessor class to apply pre-emphasis. Then, it's divided into frames using the frame_blocking method to prepare it for feature extraction.

5.Compute MFCC Features: The preprocessed audio is passed to the compute_mfcc method of the FeatureExtractor class to compute the MFCC features.

6.Compute Delta and Delta-Delta MFCCs: The computed MFCC features are further processed using the compute_delta_mfcc method of the FeatureExtractor class to compute their delta and delta-delta coefficients.

7.Save Features: The extracted MFCC features are saved as a NumPy array in a folder named sounds with a timestamp as the folder name. The preprocessed audio is also saved in int16 format as a WAV file in the same folder.

8.Return Features: The method returns the computed MFCC features, along with their delta and delta-delta coefficients.

Overall, the PreprocessedMFCCFeatures class encapsulates the preprocessing steps required to extract MFCC features from audio files and provides a convenient interface to perform these operations. This class can be useful for preparing audio data for tasks such as speech recognition, speaker identification, and audio classification.

'''

def preprocess_feature(file_path):
    # Example usage:
    audio_file_path = file_path
    preprocessed_mfcc_features = PreprocessedMFCCFeatures()
    mfccs, delta_mfccs, delta_delta_mfccs = preprocessed_mfcc_features.preprocess_audio(audio_file_path)
    
    # Ensure all feature arrays have the same number of frames
    min_frames = min(mfccs.shape[1], delta_mfccs.shape[1], delta_delta_mfccs.shape[1])
    mfccs = mfccs[:, :min_frames]
    delta_mfccs = delta_mfccs[:, :min_frames]
    delta_delta_mfccs = delta_delta_mfccs[:, :min_frames]
    
    # Concatenate the features
    concatenated_features = np.concatenate((mfccs, delta_mfccs, delta_delta_mfccs), axis=0)
    
    return concatenated_features
    

'''

The preprocess_feature function you provided seems to be a convenient wrapper for preprocessing a single audio file and concatenating its MFCC features with their delta and delta-delta coefficients. Here's how the function works:

1.Input: The function takes a file path (file_path) as input, which specifies the path to the audio file to be processed.

Preprocessing: It creates an instance of the PreprocessedMFCCFeatures class and calls its preprocess_audio method with the provided file path. This method preprocesses the audio file, extracting MFCC features along with their delta and delta-delta coefficients.

2.Concatenation: After preprocessing, the function concatenates the computed MFCC features, delta MFCCs, and delta-delta MFCCs along the 0-axis using np.concatenate. This results in a single array containing all the extracted features stacked vertically.

3.Return: The function returns the concatenated features as a NumPy array.

This function provides a simple interface for preprocessing audio files and obtaining concatenated feature vectors, which can be directly used as input for machine learning models or other audio processing tasks. It abstracts away the details of the preprocessing pipeline and allows for easy integration into larger systems.


'''