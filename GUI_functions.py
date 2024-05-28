# Importing packages
import tensorflow as tf
import scipy.io
from scipy.signal import butter, filtfilt, find_peaks
import datetime
import pywt
import keras
import numpy as np
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from os.path import join as osj
import pandas as pd


lstm_model = tf.keras.models.load_model('/home/nakul/Documents/Python/Arrhythmia_Classification_PPG/lstm_model.h5')
bilstm_model = tf.keras.models.load_model('/home/nakul/Documents/Python/Arrhythmia_Classification_PPG/bilstm_model.h5')

def predict_arrhythmia(fs, signal, lstm_model_test, class_labels = [ 'Bradycardia'  ,  'Asystole', 'Ventricular Flutter Fib',  'Ventricular Tachycardia' ,'Tachycardia']):
    input_features = []
    
    segmented_signal = segment_signal(signal, 10, fs)
    
    for segment in segmented_signal:
        segment_processed = bandpass_filter(segment, lowcut=0.05, highcut=10, fs=fs)
        segment_processed = moving_average(segment_processed, window_size=3)
        segment_processed = remove_baseline_wandering(segment_processed)

        if not any(segment_processed):
            continue

        features, peaks = extract_ppg_features(segment_processed, fs)
        for peak in features:
            peak_array = [peak[key] for key in peak.keys()]
            input_features.append(peak_array)

    input_data = np.array(input_features)
    input_data = input_data.reshape(input_data.shape[0], input_data.shape[1], 1)
    raw_predictions = lstm_model_test.predict(input_data)
    average_prediction = np.mean(raw_predictions, axis=0)
    
    prediction_dict = {}
    for label, percentage in zip(class_labels, average_prediction):
        prediction_dict[label] = f"{percentage*100:.2f}%"

    
    
    return prediction_dict

def bandpass_filter(signal, lowcut, highcut, fs):

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    order = 3  # Filter order
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def moving_average(signal, window_size):

    smoothed_signal = np.convolve(signal, np.ones(window_size) / window_size, mode='same')
    return smoothed_signal


def remove_baseline_wandering(signal):

    wavelet = 'sym8'  # Wavelet type
    level = 4  # Decomposition level
    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Zero out the detail coefficients
    coeffs = [coeffs[0]] + [np.zeros_like(coeff) for coeff in coeffs[1:]]

    denoised_signal = pywt.waverec(coeffs, wavelet)
    return denoised_signal


def normalize_signal(signal):

    normalized_signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    return normalized_signal


def segment_signal(signal, segment_length, fs):

    num_samples_per_segment = int(segment_length * fs)
    num_segments = len(signal) // num_samples_per_segment
    segmented_signal = [signal[i * num_samples_per_segment: (i + 1) * num_samples_per_segment]
                        for i in range(num_segments)]
    return segmented_signal


def calculate_pulse_width(peak_window, fs):
    pw = len(peak_window) / fs
    return pw


def calculate_fwhm(peak_window, fs):
    max_index = np.argmax(peak_window)
    half_max = peak_window[max_index] / 2
    left_indices = np.where(peak_window[:max_index] <= half_max)[0]
    right_indices = np.where(peak_window[max_index:] <= half_max)[0] + max_index
    if len(left_indices) == 0 or len(right_indices) == 0:
        return 0

    left_index = left_indices[-1]
    right_index = right_indices[0]
    fwhm = (right_index - left_index) / fs

    return fwhm

def detect_prominent_peaks(ppg_signal, fs, prominence=0.5):
    # Calculate the peak height threshold based on a percentage of the maximum amplitude
    peak_height_threshold = prominence * np.max(ppg_signal)

    # Find peaks above the threshold
    peaks, _ = find_peaks(ppg_signal, height=peak_height_threshold)

    return peaks

import numpy as np
from scipy.signal import find_peaks
from scipy.integrate import simps
from scipy.fft import fft
from scipy import stats

def extract_ppg_features(ppg_signal, fs):
    # Peak Detection with higher threshold
    peak_height_threshold = 0.6  # Might have to adjust accordingly
    peaks, _ = find_peaks(ppg_signal, height=peak_height_threshold)

    peak_features = []

    prev_peak_index = None

    for peak_index in peaks:

        window_size = int(fs * 2) 
        peak_window = ppg_signal[max(0, peak_index - window_size):min(len(ppg_signal), peak_index + window_size)]

        # Morphological Features
        sa = np.max(peak_window)
        Da = np.min(peak_window)
        SA = np.trapz(peak_window, dx=1/fs)
        DA = np.trapz(ppg_signal, dx=1/fs) - SA
        St = peak_index / fs
        Dt = (len(ppg_signal) - peak_index) / fs

        # Frequency Domain Features
        fft_result = fft(peak_window)
        magnitude_spectrum = np.abs(fft_result)
        frequency_spectrum = np.fft.fftfreq(len(peak_window), d=1/fs)
        dominant_frequency = frequency_spectrum[np.argmax(magnitude_spectrum)]
        spectral_entropy = -np.sum(magnitude_spectrum * np.log2(magnitude_spectrum))

        # Additional features
        if prev_peak_index is not None:
            ppi = (peak_index - prev_peak_index) / fs
        else:
            ppi = 0

        pi = ppi
        pw = calculate_pulse_width(peak_window, fs)
        fwhm = calculate_fwhm(peak_window, fs)

        signal_area = np.trapz(ppg_signal, dx=1/fs)
        rise_time = St - (max(0, peak_index - window_size)) / fs
        fall_time = (min(len(ppg_signal), peak_index + window_size) - peak_index) / fs

        amplitude_modulation_depth = sa - Da
        energy = np.sum(peak_window ** 2)
        zero_crossing_rate = calculate_zero_crossing_rate(peak_window)

        # Statistical Features
        mean = np.mean(peak_window)
        median = np.median(peak_window)
        std_deviation = np.std(peak_window)
        skewness = stats.skew(peak_window)
        kurtosis = stats.kurtosis(peak_window)
        min_value = np.min(peak_window)
        max_value = np.max(peak_window)
        variance = np.var(peak_window)

        # Additional features
        slope = calculate_slope(peak_window, fs)
        peak_count = len(peaks)

        # Handle exceptions for divide by zero errors
        try:
            amplitude_ratio = sa / Da
            area_ratio = SA / DA
            interval_ratio = pi / ppi if ppi != 0 else 0
        except ZeroDivisionError:
            amplitude_ratio = 0
            area_ratio = 0
            interval_ratio = 0

        peak_features.append({
            'sa': sa, 'Da': Da, 'SA': SA, 'DA': DA, 'St': St, 'Dt': Dt,
            'PI': pi, 'PPI': ppi, 'PW': pw, 'FWHM': fwhm,
            'Dominant_frequency': dominant_frequency,
            'Spectral_entropy': spectral_entropy,
            'Signal_area': signal_area,
            'Rise_time': rise_time,
            'Fall_time': fall_time,
            'Amplitude_modulation_depth': amplitude_modulation_depth,
            'Energy': energy,
            'Zero_crossing_rate': zero_crossing_rate,
            'Mean': mean,
            'Median': median,
            'Standard_deviation': std_deviation,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'Min_value': min_value,
            'Max_value': max_value,
            'Variance': variance,
            'Slope': slope,
            'Peak_count': peak_count,
            'Amplitude_ratio': amplitude_ratio,
            'Area_ratio': area_ratio,
            'Interval_ratio': interval_ratio
        })

        prev_peak_index = peak_index

    return peak_features, peaks

def calculate_slope(signal, fs):
    time = np.arange(0, len(signal)) / fs
    slope, _ = np.polyfit(time, signal, 1)
    return slope

def calculate_pulse_width(peak_window, fs):
    # Calculate pulse width as the width at half of the maximum amplitude
    half_max_amplitude = 0.5 * np.max(peak_window)
    above_half = peak_window > half_max_amplitude
    return fs * np.sum(above_half) / fs

def calculate_fwhm(peak_window, fs):
    # Calculate full width half maximum (FWHM)
    half_max_amplitude = 0.5 * np.max(peak_window)
    peaks, _ = find_peaks(peak_window, height=half_max_amplitude)
    if len(peaks) < 2:
        return 0
    return (peaks[-1] - peaks[0]) / fs

def calculate_zero_crossing_rate(signal):
    # Calculate zero crossing rate
    return np.sum(np.diff(np.sign(signal)) != 0) / len(signal)