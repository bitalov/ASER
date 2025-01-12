# Numerical and Data Handling
# utils.py
# This module provides various utility functions for audio processing, emotion recognition, and dataset handling.
# Functions:
# - get_log_mel_spectrogram(audio, sample_rate, n_fft=2048, hop_length=256, n_mels=128, fmin=80, fmax=7600, window='hann'):
# - get_audio_duration(file_path):
#     Get the duration of an audio file in seconds.
# - parse_filename(filename):
#     Parse the filename to extract relevant information such as emotion, gender, speaker, sentence, and trial.
# - load_speech_dataset(base_path):
#     Load a speech dataset from a given base path, extracting relevant information and returning a DataFrame.
# - EYASE_dataset(path):
#     Load the EYASE dataset from a given path, extracting relevant information and returning a DataFrame.
# - create_folds(Data, SAMPLE_RATE):
#     Create stratified k-folds for cross-validation from the given dataset.
# - sigToSpectogram(data, final_shape, duration=10, sample_rate=22500):
#     Perform signal-to-spectrogram conversion on audio files and compute mel spectrograms.
# - load_spectograms(train_df, test_df, num_augmentations, duration, SAMPLE_RATE):
#     Load spectrograms for training and testing datasets, performing necessary augmentations.
# - addAWGN(signal, num_bits=16, augmented_num=2, snr_low=15, snr_high=30):
#     Add Additive White Gaussian Noise (AWGN) to a signal.
# - addAWGN_with_duration(signal, duration=10, num_bits=16, augmented_num=2, snr_low=15, snr_high=30):
#     Add Additive White Gaussian Noise (AWGN) to a signal with a specified duration.
# - scale(X_train, X_test):
#     Scale the training and testing datasets using StandardScaler.
# - save_datasets(X_train, X_test, Y_train, Y_test, filename='dataset'):
#     Save training, validation, and test sets to a file.
# - load_datasets(filename='dataset'):
#     Load training, validation, and test sets from a file.

import numpy as np
import os
import glob
import os
import pandas as pd
#from tqdm.notebook import tqdm
import tqdm  
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Audio Processing and Visualization
import librosa
import librosa.display

# Plotting
import matplotlib.pyplot as plt
from adopt import ADOPT
import seaborn as sns


# Process files in parallel using joblib
from joblib import Parallel, delayed
    

# IPython Utilities for Jupyter Notebook
from IPython.display import Audio, Image

def get_log_mel_spectrogram(
    audio,
    sample_rate,
    n_fft=2048,
    hop_length=256,
    n_mels=128,
    fmin=80,
    fmax=7600,
    window='hann'
):
    """
    Compute the Log-Mel spectrogram of an audio signal.

    Parameters:
    - audio (np.ndarray): Audio time series.
    - sample_rate (int): Sampling rate of the audio signal.
    - n_fft (int): Length of the FFT window.
    - hop_length (int): Number of samples between successive frames.
    - n_mels (int): Number of Mel bands to generate.
    - fmin (float): Lowest frequency (in Hz).
    - fmax (float): Highest frequency (in Hz).
    - window (str): Window function to apply (e.g., 'hann', 'hamming').

    Returns:
    - mel_spec_db (np.ndarray): Log-Mel spectrogram (in dB).
    """
    # Compute Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0  # Power spectrogram (default)
    )

    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db

def get_audio_duration(file_path):
    """Get the duration of an audio file in seconds"""
    try:
        duration = librosa.get_duration(path=file_path)
        return duration
    except Exception as e:
        raise RuntimeError(f"Error processing file {file_path}: {str(e)}")


def parse_filename(filename):
    """Parse the filename to extract relevant information"""
    # Example filename: D05E03P104S01T01
    parts = filename.split('/')[-1]  # Get just the filename
    emotion = parts[3:6]  # Get E00-E05
    gender = '1' if parts[7] == '1' else '0'  # Get gender (0 male, 1 female)
    speaker = parts[7:9]  # Get speaker number
    sentence = parts[10:13]  # Get sentence number
    trial = parts[13:]  # Get trial number
    
    return {
        'emotion': emotion,
        'gender': gender,
        'speaker': speaker,
        'sentence': sentence,
        'trial': trial
    }

def load_ksu_dataset(base_path):
    data = []
    # Emotion encoding dictionary based on folder names (E00 to E05)
    emotion_encoding = {
        "E00": 0,  # Neutral
        "E01": 1,  # Happy
        "E02": 2,  # Sad
        "E03": 3,  # Surprise
        "E04": 4,  # Angry
        "E05": 5   # Fear
    }
    
    # Define phases
    phases = ['Phase_1', 'Phase_2']
    
    for phase in phases:
        phase_path = os.path.join(base_path, phase)
        
        # Get all emotion folders (E00 to E05)
        emotion_folders = glob.glob(os.path.join(phase_path, 'E*'))
        
        # Iterate over each emotion folder
        for emotion_folder in tqdm.tqdm(emotion_folders, desc=f"Processing {phase}"):
            emotion_code = os.path.basename(emotion_folder)  # Get E00-E05
            
            # Get all audio files in the emotion folder
            audio_files = glob.glob(os.path.join(emotion_folder, '*'))
            
            for audio_file in audio_files:
                file_info = parse_filename(audio_file)
            
             # Get audio duration
                duration = get_audio_duration(audio_file)
                
                # Create data entry
                data.append({
                    "Path": audio_file,
                    "Emotion": emotion_encoding.get(emotion_code, -1),
                    "Emotion_Label": emotion_code,
                    "Gender": file_info['gender'],
                    "Speaker": file_info['speaker'],
                    "Sentence": file_info['sentence'],
                    "Trial": file_info['trial'],
                    "Phase": phase,
                    "Duration": duration
                })
    
    # Convert the list into a DataFrame
    df = pd.DataFrame(data)
    return df


def EYASE_dataset(path):
    emotion_encoding = {
    "neu": 0,
    "hap": 1,
    "sad": 2,
    "ang": 3,
    # Add more emotions if necessary
    }
    data = []
    # Get all subfolders inside the main path
    all_folders_directory = glob.glob(os.path.join(path, '*'))
    
    # Iterate over each subfolder
    for folder in tqdm.tqdm(all_folders_directory, desc="Processing Folders"):
        # Get all files inside the subfolder
        sub_folder = glob.glob(os.path.join(folder, '*'))
        
        # Iterate over each audio file
        for audio_file in sub_folder:
            # Extract the emotion from the filename
            filename = os.path.basename(audio_file)
            emotion = filename.split('_')[-1].split(' ')[0].split('(')[0]
            
            # Print the raw emotion (this is useful for debugging)
            print(f"Raw emotion extracted: {emotion}")
            
            # Encode the emotion using the emotion_encoding dictionary
            encoded_emotion = emotion_encoding.get(emotion, -1)  # Default to -1 if emotion not found
            
            # Append the file data to the list
            data.append({
                "Path": audio_file,
                "Emotion": encoded_emotion,            # The raw emotion string
               
            })
    
    # Convert the list into a DataFrame
    df = pd.DataFrame(data)
    return df


def create_folds(Data, num_folds=5):

     # Assuming ksu_emotions_df is your DataFrame

    # Initialize StratifiedKFold with 5 splits
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=455)

    # Create a list to store the train and test DataFrames for each fold
    folds = []

    # Iterate through the folds
    for train_index, test_index in skf.split(Data, Data['Emotion']):
        train_df = Data.iloc[train_index].reset_index(drop=True)
        test_df = Data.iloc[test_index].reset_index(drop=True)
        folds.append((train_df, test_df))
    
    return folds



def load_and_preprocess_ksu_emotions(num_folds=5, include_augmentation=True):
    """
    Load and preprocess the KSU emotions dataset, optionally including data augmentation. and saves the files
    Parameters:
    include_augmentation (bool): If True, apply data augmentation to the training data.
    Returns:
    None
    """
    base_path = 'ksu_emotions/data/SPEECH'
    SAMPLE_RATE = 16_000  # Sample rate of the audio files
    duration = 10
    num_augmentations = 2 if include_augmentation else 0
    
    folds = create_folds(Data=load_ksu_dataset(base_path), num_folds=num_folds)
      
    
    for f, (train_df, test_df) in enumerate(folds):
        fold_num = f + 1
        print(f"Fold {fold_num}, {train_df.shape, test_df.shape}:")
        mel_spectrograms, signals, mel_spectrograms_test = load_spectograms(train_df, test_df, num_augmentations, duration, SAMPLE_RATE)
        k = train_df.shape[0]
        print(f"signals'shape {signals.shape}")

        if include_augmentation:
        # applying augmentations
            for i,signal in enumerate(signals):
                augmented_signals = add_augmentation(signal, augmented_num=num_augmentations)
                #augmented_signals = addAWGN(signal)
                for j in range(augmented_signals.shape[0]):
                    mel_spectrogram = get_log_mel_spectrogram(augmented_signals[j,:], sample_rate=SAMPLE_RATE)       
                    mel_spectrograms[k] = mel_spectrogram
                    k += 1
                    train_df = pd.concat([train_df, train_df.iloc[i:i+1]], ignore_index=True)
                print("\r Processed {}/{} files".format(i,len(signals)),end='')

        del signals

            

        X_train = np.expand_dims(mel_spectrograms,1)        
        Y_train = np.array(train_df.Emotion)
        X_test = np.expand_dims(mel_spectrograms_test, 1)
        Y_test = np.array(test_df.Emotion)
        X_train, X_test = scale(X_train, X_test)        
        save_datasets(X_train=X_train, X_test=X_test, Y_train=Y_train, Y_test=Y_test, filename=f"dataset_fold{fold_num}") 


    
    



def sigToSpectogram(data, final_shape, duration = 10 ,sample_rate=22500):
    """
    Perform sigToSpectogram on audio files and compute mel spectrograms.
    
    Args:
        data: DataFrame containing audio file paths
        final_shape: Shape of the final mel spectrogram array
        duration: Duration of the audio in seconds
        sample_rate: Audio sample rate
    
    Returns:
        tuple: (mel_spectrograms, signals)
        - mel_spectrograms: numpy array of mel spectrograms
        - signals: list of audio signals
    """
    
 
    
    # Preallocate arrays
    mel_spectrograms = np.empty(final_shape, dtype=np.float32)
    signals = []
    
    def process_file(file_path):
        try:
            # Load audio file
            audio, _ = librosa.load(
                file_path, 
                duration=duration, 
                offset=0.5, 
                sr=sample_rate,
                res_type='kaiser_fast'  # Faster resampling
            )
            
            # Zero-pad signal
            signal = np.zeros(int(sample_rate * duration), dtype=np.float32)
            signal[:len(audio)] = audio
            
            # Compute mel spectrogram
            mel_spec = get_log_mel_spectrogram(signal, sample_rate=sample_rate)
            
            return mel_spec, signal
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None, None

    # Process files in parallel
    results = Parallel(n_jobs=-1)(
        delayed(process_file)(file_path) 
        for file_path in tqdm.tqdm(data.Path, desc="Processing audio files")
    )
    
    # Store results in mel_spectrograms array and signals list
    for i, (mel_spec, signal) in enumerate(results):
        if mel_spec is not None:
            mel_spectrograms[i] = mel_spec
            signals.append(signal)
    
    # Convert signals list to numpy array for consistency
    signals = np.array(signals, dtype=np.float32)
    
    return mel_spectrograms, signals


def load_spectograms(train_df, test_df, num_augmentations, duration, SAMPLE_RATE):
    """
    Load and preprocess audio data to generate Log-Mel spectrograms for training and testing datasets.

    Parameters:
    train_df (pd.DataFrame): DataFrame containing the training data paths.
    test_df (pd.DataFrame): DataFrame containing the testing data paths.
    num_augmentations (int): Number of augmentations to apply to the training data.
    duration (float): Duration of the audio signal to be processed (in seconds).
    SAMPLE_RATE (int): Sample rate for loading the audio files.

    Returns:
    tuple: A tuple containing:
        - mel_spectrograms (np.ndarray): Array of Log-Mel spectrograms for the training data.
        - signals (np.ndarray): Array of processed audio signals for the training data.
        - mel_spectrograms_test (np.ndarray): Array of Log-Mel spectrograms for the testing data.
    """
    # Zero-padding the signal for a consistent 3-second input
    audio, sample_rate = librosa.load(train_df.loc[0, 'Path'], duration=duration, offset=0.5, sr=SAMPLE_RATE)
    signal = np.zeros(int(SAMPLE_RATE * duration))
    signal[:len(audio)] = audio
    # Compute the optimized Log-Mel spectrogram
    sample = get_log_mel_spectrogram(signal, SAMPLE_RATE)   
    final_shape = ((num_augmentations + 1) * train_df.shape[0], sample.shape[0], sample.shape[1])
    mel_spectrograms, signals = sigToSpectogram(data=train_df,final_shape=final_shape,duration=duration,sample_rate=SAMPLE_RATE)
    test_shape = (test_df.shape[0], sample.shape[0], sample.shape[1])
    mel_spectrograms_test, _ = sigToSpectogram(data=test_df,final_shape=test_shape,duration=duration,sample_rate=SAMPLE_RATE)
    print(signals.shape)
    return mel_spectrograms, signals, mel_spectrograms_test


def add_augmentation(signal, num_bits=16, augmented_num=2, snr_low=15, snr_high=30):
    """
    Add white Gaussian noise to a signal to create augmented versions of the signal.

    Parameters:
    signal (array-like): The input signal to be augmented.
    num_bits (int, optional): The number of bits for normalization. Default is 16.
    augmented_num (int, optional): The number of augmented signals to generate. Default is 2.
    snr_low (int, optional): The lower bound of the Signal-to-Noise Ratio (SNR) range in dB. Default is 15.
    snr_high (int, optional): The upper bound of the Signal-to-Noise Ratio (SNR) range in dB. Default is 30.

    Returns:
    numpy.ndarray: An array containing the augmented signals with added noise.
    """
    signal_len = len(signal)
    # Generate White Gaussian noise
    noise = np.random.normal(size=(augmented_num, signal_len))
    # Normalize signal and noise
    norm_constant = 2.0**(num_bits-1)
    signal_norm = signal / norm_constant
    noise_norm = noise / norm_constant
    # Compute signal and noise power
    s_power = np.sum(signal_norm ** 2) / signal_len
    n_power = np.sum(noise_norm ** 2, axis=1) / signal_len
    # Random SNR: Uniform [15, 30] in dB
    target_snr = np.random.randint(snr_low, snr_high)
    # Compute K (covariance matrix) for each noise
    K = np.sqrt((s_power / n_power) * 10 ** (- target_snr / 10))
    K = np.ones((signal_len, augmented_num)) * K
    # Generate noisy signal
    return signal + K.T * noise

def add_augmentation_with_duration(signal, duration=10, num_bits=16, augmented_num=2, snr_low=15, snr_high=30):
    """
    Adds white Gaussian noise to the input signal to create augmented versions of the signal.
    Parameters:
    signal (numpy.ndarray): The input signal to be augmented.
    duration (int, optional): The duration of the signal to be used for augmentation. Default is 10.
    num_bits (int, optional): The number of bits for normalization. Default is 16.
    augmented_num (int, optional): The number of augmented signals to generate. Default is 2.
    snr_low (int, optional): The lower bound of the Signal-to-Noise Ratio (SNR) in dB. Default is 15.
    snr_high (int, optional): The upper bound of the Signal-to-Noise Ratio (SNR) in dB. Default is 30.
    Returns:
    numpy.ndarray: The augmented signal with added noise.
    """
    full_length = len(signal)    
    signal_len = min(int((duration / 10) * full_length), full_length)
    # Generate White Gaussian noise
    noise = np.random.normal(size=(augmented_num, signal_len))
    # Normalize signal and noise
    norm_constant = 2.0**(num_bits-1)
    signal_norm = signal / norm_constant
    noise_norm = noise / norm_constant
    # Compute signal and noise power
    s_power = np.sum(signal_norm ** 2) / signal_len
    n_power = np.sum(noise_norm ** 2, axis=1) / signal_len
    # Random SNR: Uniform [15, 30] in dB
    target_snr = np.random.randint(snr_low, snr_high)
    # Compute K (covariance matrix) for each noise
    K = np.sqrt((s_power / n_power) * 10 ** (- target_snr / 10))
    K = np.ones((signal_len, augmented_num)) * K
    
    # Determine the target length
    target_length = signal.shape[0]  # 160000

    # Pad array2 and array3 with zeros
    k_T = np.pad(K.T, ((0, 0), (0, target_length - K.T.shape[1])), mode='constant', constant_values=0)
    noise = np.pad(noise, ((0, 0), (0, target_length - noise.shape[1])), mode='constant', constant_values=0)
    print(signal.shape, k_T.shape, noise.shape)
    return signal +  k_T  * noise



def scale(X_train, X_test):
    scaler = StandardScaler()

    b,c,h,w = X_train.shape
    X_train = np.reshape(X_train, newshape=(b,-1))
    X_train = scaler.fit_transform(X_train)
    X_train = np.reshape(X_train, newshape=(b,c,h,w))

    b,c,h,w = X_test.shape
    X_test = np.reshape(X_test, newshape=(b,-1))
    X_test = scaler.transform(X_test)
    X_test = np.reshape(X_test, newshape=(b,c,h,w))

    return X_train, X_test


def save_datasets(X_train, X_test, Y_train, Y_test, filename='dataset'):
    """
    Save training, validation and test sets
    
    Parameters:
    X_train, X_test: Features for train, test sets
    y_train, y_test: Labels for train, test sets
    filename: Name of the file to save (without extension)
    """
    np.savez(filename,
             X_train=X_train,             
             X_test=X_test,
             Y_train=Y_train,             
             Y_test=Y_test)
    print(f"Datasets saved to {filename}.npz")

def load_datasets(filename='dataset'):
    """
    Load training, validation and test sets
    
    Parameters:
    filename: Name of the file to load (without extension)
    
    Returns:
    X_train, X_test, y_train, y_test
    """
    data = np.load(f"{filename}.npz")
    return (data['X_train'], data['X_test'],
            data['Y_train'], data['Y_test'])

