import os
import struct
import sounddevice as sd
import tkinter as tk
import threading
import json
import pickle
import webbrowser
import requests
import re
import numpy as np
import queue
import warnings
from PIL import Image, ImageTk
import pyaudio
from preprocess_mfcc import preprocess_feature, fold, r_spectrogram, p_spectrogram
from model_load import model_load, recognize, logs
from hmm import hmm
from hmm.base import ConvergenceMonitor

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
log_value = -1
logs(log_value)  # for loging to be reset to not showing
warnings.filterwarnings("ignore")


class SpeechRecognizer:
    def __init__(self):
        # Initialize necessary variables
        self.CHUNK = 1024  # Number of audio samples per frame
        self.FORMAT = pyaudio.paInt16  # Audio format
        self.sample_rate = 44100  # Sample rate
        self.channels = 1  # Mono recording
        self.is_recording = False
        self.recording_frames = []
        self.path = "models/DecodeTrained"
        self.model = model_load(self.path)
        self.analyze = None
        # self.attrv = dir(self.model)
        # print(self.attrv)
        # for attr_name in dir(self.model):
        #     attr_value = getattr(self.model, attr_name)
        #     print(attr_value)

    def audio_stream(self, q):
        # Stream audio from microphone
        p = pyaudio.PyAudio()
        stream = p.open(
            format=self.FORMAT,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.CHUNK,
        )
        try:
            while True:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                q.put(data)
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def update_spectrum(self, canvas, q, lines):
        # Update spectrogram visualization
        if self.is_recording:
            canvas.pack()  # Show canvas during recording
        else:
            canvas.pack_forget()  # Hide canvas when recording stops

        if not q.empty():
            data = q.get()
            audio_data = np.frombuffer(data, dtype=np.int16)
            fft = np.fft.fft(audio_data)
            freq = np.fft.fftfreq(len(fft), 1.0 / self.sample_rate)
            freq = freq[: len(freq) // 2]
            fft = np.abs(fft[: len(fft) // 2])
            max_fft = np.max(fft)
            for i, line in enumerate(lines):
                x = i * (200 / len(lines)) + 10
                y = np.log1p(fft[i] / max_fft) * 100
                canvas.coords(line, x, 200, x, 200 - y)
        canvas.after(10, lambda: self.update_spectrum(canvas, q, lines))

    def transcribe_audio(self, frames, models):
        # Transcribe audio into text
        recognizer = recognize(self.sample_rate, self.model)
        recognizer.AcceptWaveform(frames)
        result = recognizer.Result()
        result_dict = json.loads(result)
        if "text" in result_dict:
            if result_dict["text"] == "":
                resp = "No Voice Detected! Please Try Again"
            else:
                resp = f"You Said : {result_dict['text']}"
            return resp
        else:
            resp = "Recognition failed."
            return resp

    def train_hmmv(self, features, n_components=3, covariance_type="diag", n_iter=100):
        # Train Hidden Markov Model
        modelv = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            algorithm="viterbi",
        )
        modelv.fit(features)
        return modelv

    def train_hmm(self, features, n_components=3, covariance_type="diag", n_iter=100):
        # Train Hidden Markov Model
        model = hmm.GaussianHMM(
            n_components=n_components, covariance_type=covariance_type, n_iter=n_iter
        )
        model.fit(features)
        return model

    def start_recording(self):
        # Start recording audio
        def callback(indata, frames, time, status):
            self.recording_frames.append(indata.copy())

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            callback=callback,
        ):
            while self.is_recording:
                sd.sleep(1000)

    def stop_recording(self, transcription_label):
        # Stop recording audio and transcribe
        self.is_recording = False
        frames = b"".join(self.recording_frames)

        with open("recorded_audio.wav", "wb") as f:
            f.write(b"RIFF")
            f.write(struct.pack("<L", len(frames) + 36))
            f.write(b"WAVE")
            f.write(b"fmt ")
            f.write(struct.pack("<L", 16))
            f.write(struct.pack("<H", 1))
            f.write(struct.pack("<H", 1))
            f.write(struct.pack("<L", self.sample_rate))
            f.write(struct.pack("<L", self.sample_rate * self.channels * 2))
            f.write(struct.pack("<H", self.channels * 2))
            f.write(struct.pack("<H", 16))
            f.write(b"data")
            f.write(struct.pack("<L", len(frames)))
            f.write(frames)

        mfcc_features = preprocess_feature("recorded_audio.wav")
        model_file = "hmm_model.pkl"

        if os.path.exists(model_file):
            with open(model_file, "rb") as file:
                model = self.train_hmm(mfcc_features)
                modelv = self.train_hmmv(mfcc_features)
                with open(model_file, "wb") as file:
                    pickle.dump(model, file)
        else:
            model = self.train_hmm(mfcc_features)
            modelv = self.train_hmmv(mfcc_features)
            with open(model_file, "wb") as file:
                pickle.dump(model, file)

        tsc = self.transcribe_audio(frames, model)
        match = re.search(r": (.*)$", tsc)  # Using regex to match everything after ": "
        if match:
            ext = match.group(1)
        wrd = ext.split()
        num_ = len(wrd)

        sequence_length = 10 * num_

        model_info = {}
        model_info_v = {}
        model_info["report of model"] = (
            "Report of Raw HMM (Hidden Markov Model Execution) without vertebri decoding"
        )
        model_info_v["report of model"] = (
            "Report of  HMM (Hidden Markov Model Execution) with vertebri decoding"
        )
        model_info["raw_audio_path"] = f"{fold}/raudio_file.wav"
        model_info["preprocessed_audio_path"] = f"{fold}/paudio_file.wav"
        model_info["raw_audio_spectrum"] = f"{fold}/raudio_files.png"
        model_info["preprocessed_audio_spectrum"] = f"{fold}/paudio_files.png"
        model_info["raw_audio_mfcc"] = f"{fold}/rmfcc.png"
        model_info["preprocessed_audio_mfcc"] = f"{fold}/pmfcc.png"
        model_info["mfcc_features"] = mfcc_features
        model_info["Sequence_length"] = sequence_length
        model_info["With_vertebri"] = False
        model_info_v["raw_audio_path"] = f"{fold}/raudio_file.wav"
        model_info_v["preprocessed_audio_path"] = f"{fold}/paudio_file.wav"
        model_info_v["raw_audio_spectrum"] = f"{fold}/raudio_files.png"
        model_info_v["preprocessed_audio_spectrum"] = f"{fold}/paudio_files.png"
        model_info_v["raw_audio_mfcc"] = f"{fold}/rmfcc.png"
        model_info_v["preprocessed_audio_mfcc"] = f"{fold}/pmfcc.png"
        model_info_v["mfcc_features"] = mfcc_features
        model_info_v["Sequence_length"] = sequence_length
        model_info_v["With_vertebri"] = True
        observed_sequence, _ = model.sample(n_samples=sequence_length)
        # observed_sequencev, _ = modelv.sample(n_samples=sequence_length)
        # print("Sequence:", observed_sequence)
        # print("Sequenceov:", observed_sequencev)
        # print(observed_sequence)
        # print("vert")
        # state_sequence = model.predict(algorithm='viterbi')
        # observed_sequences= model._generate_sample_from_state(state_sequence, n_samples=sequence_length)
        # print(state_sequence)
        # print(observed_sequences)
        _, states = model.sample(n_samples=sequence_length)
        # _, statesv = modelv.sample(n_samples=sequence_length)
        # print("Sequence of states:", states)
        # print("Sequence of statesv:", statesv)
        model_info["sequnce_state"] = states
        model_info["observeed_sequences"] = observed_sequence

        # Define the folder path
        output_folder = fold

        # print(output_folder)

        # Create the folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Define the file path within the folder
        output_file = os.path.join(output_folder, "report.json")

        words = ext.split()

        # Calculate the number of sequences per word
        # sequences_per_word = len(observed_sequence) // len(words)

        # Create a dictionary to store sequences per word
        sequences_per_word_dict = {word: [] for word in words}

        # Divide the sequences accordingly
        for i, sequence in enumerate(observed_sequence):
            word_index = i % len(
                words
            )  # Calculate the index of the word for the current sequence
            word = words[word_index]  # Get the corresponding word
            sequences_per_word_dict[word].append(sequence)

        # Print or use the sequences per word
        # for word, sequences in sequences_per_word_dict.items():
        #     print(f"Sequences for word '{word}':")
        #     for sequence in sequences:
        #         # print(sequence)
        #         pass
        # print()
        average_sequences_per_word = {
            word: np.mean(sequences, axis=0).tolist()
            for word, sequences in sequences_per_word_dict.items()
        }
        # print(average_sequences_per_word)
        # Open the text file for writing

        # model_info = {}
        # model_info["raw_audio_path"]=f"{fold}/raudio_file.wav"
        # model_info["preprocessed_audio_path"]=f"{fold}/paudio_file.wav"
        # model_file["raw_audio_spectrum"] = f"{fold}/raudio_files.png"
        # model_file["preprocessed_audio_spectrum"] = f"{fold}/paudio_files.png"
        # model_info["mfcc_features"]=mfcc_features
        # model_info["With vertebri"] = False
        # model_info["sequnce_state"]=states
        # model_info["onserveed_sequences"]=observed_sequence

        # Populate the dictionary with attribute names and values
        for attr_name in dir(model):
            attr_value = getattr(model, attr_name)
            if not callable(attr_value) and not isinstance(
                attr_value, ConvergenceMonitor
            ):  # Exclude methods and ConvergenceMonitor
                model_info[attr_name] = attr_value

        model_info["sequence_per_word"] = average_sequences_per_word
        model_info["Transcription"] = ext

        # Write the dictionary to a JSON file
        with open(output_file, "w") as json_file:
            json.dump(
                model_info, json_file, indent=4, default=str
            )  # Use default=str to handle non-serializable objects

        # print(f"Information saved to {output_file}")
        output_file_v = os.path.join(output_folder, "report_v.json")
        # model_info["with vertebri"]=True
        observed_sequence, _ = modelv.sample(n_samples=sequence_length)
        _, states = modelv.sample(n_samples=sequence_length)

        sequences_per_word_dict = {word: [] for word in words}

        # Divide the sequences accordingly
        for i, sequence in enumerate(observed_sequence):
            word_index = i % len(
                words
            )  # Calculate the index of the word for the current sequence
            word = words[word_index]  # Get the corresponding word
            sequences_per_word_dict[word].append(sequence)

        # Print or use the sequences per word
        # for word, sequences in sequences_per_word_dict.items():
        #     print(f"Sequences for word '{word}':")
        #     for sequence in sequences:
        #         # print(sequence)
        #         pass
        # print()
        average_sequences_per_word = {
            word: np.mean(sequences, axis=0).tolist()
            for word, sequences in sequences_per_word_dict.items()
        }

        # print(observed_sequence)
        model_info_v["sequnce_state"] = states
        model_info_v["observed_sequences"] = observed_sequence
        #  Populate the dictionary with attribute names and values
        for attr_name in dir(modelv):
            attr_value = getattr(modelv, attr_name)
            if not callable(attr_value) and not isinstance(
                attr_value, ConvergenceMonitor
            ):  # Exclude methods and ConvergenceMonitor
                model_info_v[attr_name] = attr_value
        model_info_v["sequence_per_word"] = average_sequences_per_word
        model_info_v["Transcription"] = ext

        # Write the dictionary to a JSON file
        with open(output_file_v, "w") as json_file:
            json.dump(
                model_info_v, json_file, indent=4, default=str
            )  # Use default=str to handle non-serializable objects

        #########################################
        # log_likelihoods = []
        # startprob_matrices = []
        # transmat_matrices = []
        # means_list = []
        # covars_list = []
        # posteriors_list = []
        # log_probabilities = []
        # print("MFCC Features Shape:", mfcc_features.shape)
        # print("Means Array Shape:", model.means_.shape)
        # for iteration in range(100):
        # Fit the model for one iteration

        # Store relevant values at each iteration
        # log_likelihoods.append(model.monitor_.history[-1])
        # startprob_matrices.append(model.startprob_)
        # transmat_matrices.append(model.transmat_)
        # means_list.append(model.means_[:, :37])
        # covars_list.append(model.covars_)

        # # Calculate posteriors and log probabilities
        # posteriors = model.predict_proba(mfcc_features)
        # posteriors_list.append(posteriors)
        # log_probability = model.score(mfcc_features)
        # log_probabilities.append(log_probability)

        # # Print information for each iteration
        # print("Iteration:", iteration + 1)
        # print("Log Likelihood:", log_likelihoods[-1])
        # print("Start Probability Matrix:")
        # print(startprob_matrices[-1])
        # print("Transition Matrix:")
        # print(transmat_matrices[-1])
        # print("Means:")
        # print(means_list[-1])
        # print("Covariances:")
        # print(covars_list[-1])
        # print("Posteriors:")
        # print(posteriors)
        # print("Log Probability:", log_probability)
        # print("---------------------------------------")

        ###########################################
        # for step in range(sequence_length):
        #     print(f"Step {step + 1}:")
        #     print("State:", states[step])
        # print("Log Likelihood:", model.monitor_.history[-1])
        # print("Start Probability Matrix:", model.startprob_)
        # print("Transition Matrix:", model.transmat_)
        # print("Means:", model.means_)
        # print("Covariances:", model.covars_)

        # print("Attributes and methods of the trained HMM model:")
        # print("Attribute values of the trained HMM model:")
        # for attr_name in dir(modelv):
        #     attr_value = getattr(modelv, attr_name)
        #     if not callable(attr_value):  # Exclude methods
        #         print(attr_name, ":", attr_value)

        with open(model_file, "wb") as file:
            pickle.dump(model, file)

        models = model
        transcription = self.transcribe_audio(frames, models)
        transcription_label.config(text=transcription)

    def show_welcome_window(self, username):
        # Display GUI window
        welcome_window = tk.Tk()
        welcome_window.title("Automatic Speech Recognition")
        welcome_window.geometry("700x600")
        # welcome_window.state('zoomed')
        # welcome_window.resizable(0, 0)
        welcome_window.configure(bg="#f0f0f0")

        # Function to logout
        def logout():
            welcome_window.destroy()  # Close the window after logout
            url = f"http://127.0.0.1:5000/logout"
            response = requests.get(url)
            if response.ok:
                webbrowser.open(url)
                pass

        def analyzer():
            output_folder = fold
            url = f"http://127.0.0.1:5000/result?folder_path={output_folder}"  # Replace with your URL
            data = {"folder_path": output_folder}  # Replace with your POST data

            # Send the POST request
            response = requests.get(url, data=data)

            # Open the URL in the web browser if the request is successful
            if response.ok:
                welcome_window.destroy()
                webbrowser.open(url)
                pass

        # Add logout button
        logout_button = tk.Button(
            welcome_window,
            text="Logout",
            command=logout,
            font=("Helvetica", 12),
            bg="#FF5733",
            fg="white",
            relief="flat",
        )
        logout_button.pack(anchor="ne", padx=20, pady=10)
        # analyze = None
        # analyze.pack(anchor="ne", padx=20, pady=10)

        canvas = tk.Canvas(welcome_window, width=800, height=400, bg="white")
        canvas.pack(
            fill="both", expand=True, padx=(20, 20), pady=(200, 0)
        )  # Adjust padx and pady for centering
        lines = [
            canvas.create_line(x, 200, x, 200, fill="green") for x in range(10, 300, 4)
        ]
        q = queue.Queue()
        threading.Thread(target=self.audio_stream, args=(q,), daemon=True).start()
        self.update_spectrum(canvas, q, lines)

        start_img = Image.open("images/start.png")
        start_img = start_img.resize((100, 100), Image.ANTIALIAS)
        start_image = ImageTk.PhotoImage(start_img)
        stop_img = Image.open("images/stop.png")
        stop_img = stop_img.resize((100, 100), Image.ANTIALIAS)
        stop_image = ImageTk.PhotoImage(stop_img)

        tk.Label(
            welcome_window,
            text=f"Hello, {username}!",
            font=("Helvetica", 20, "bold"),
            bg="#f0f0f0",
            fg="#333",
        ).pack(pady=20)
        transcription_label = tk.Label(
            welcome_window, text="", bg="#f0f0f0", fg="#333", font=("Helvetica", 12)
        )
        transcription_label.pack(pady=20)

        def toggle_recording():
            if not self.is_recording:
                self.is_recording = True
                mic_button.config(image=stop_image)
                info_label.config(text="Recording...", fg="red")
                self.recording_frames.clear()
                threading.Thread(target=self.start_recording).start()
                self.analyze.pack_forget()
            else:
                self.analyze = tk.Button(
                    welcome_window,
                    text="View Analysis",
                    command=analyzer,
                    font=("Helvetica", 12),
                    bg="#FF5733",
                    fg="white",
                    relief="flat",
                )
                self.analyze.pack(anchor="ne", padx=20, pady=10)
                mic_button.config(image=start_image)
                info_label.config(text="Click to start recording", fg="black")
                self.stop_recording(transcription_label)
                p_spectrogram()
                r_spectrogram()

        mic_button = tk.Button(
            welcome_window,
            image=start_image,
            command=toggle_recording,
            relief="flat",
            bd=0,
            highlightthickness=0,
        )
        mic_button.pack(pady=10)

        info_label = tk.Label(
            welcome_window,
            text="Click to start recording",
            fg="black",
            font=("Helvetica", 12),
        )
        info_label.pack(pady=5)
        welcome_window.lift()
        welcome_window.attributes("-topmost", True)
        welcome_window.mainloop()
        return welcome_window


if __name__ == "__main__":
    recognizer_gui = SpeechRecognizer()
    recognizer_gui.show_welcome_window("User")

# def initialize(user):
#     recon = SpeechRecognizer()
#     recon.show_welcome_window(user)

# initialize("saugat")
