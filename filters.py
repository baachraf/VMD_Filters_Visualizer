"""
Filter Computation Module
Contains traditional signal processing filters
"""

import numpy as np
from scipy import signal
import pywt

def apply_butterworth(sig: np.ndarray, order: int, freq_min: float, 
                      freq_max: float, fps: float) -> np.ndarray:
    """
    Apply Butterworth bandpass filter
    """
    nyq = fps / 2.0
    low = freq_min / nyq
    high = freq_max / nyq
    
    # Ensure frequencies are in valid range
    low = max(0.01, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))
    
    b, a = signal.butter(order, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, sig)
    return filtered


def apply_lowpass(sig: np.ndarray, order: int, cutoff: float, fps: float) -> np.ndarray:
    """
    Apply Butterworth lowpass filter
    """
    nyq = fps / 2.0
    normal_cutoff = cutoff / nyq
    normal_cutoff = max(0.01, min(normal_cutoff, 0.99))
    
    b, a = signal.butter(order, normal_cutoff, btype='low')
    filtered = signal.filtfilt(b, a, sig)
    return filtered


def apply_highpass(sig: np.ndarray, order: int, cutoff: float, fps: float) -> np.ndarray:
    """
    Apply Butterworth highpass filter
    """
    nyq = fps / 2.0
    normal_cutoff = cutoff / nyq
    normal_cutoff = max(0.01, min(normal_cutoff, 0.99))
    
    b, a = signal.butter(order, normal_cutoff, btype='high')
    filtered = signal.filtfilt(b, a, sig)
    return filtered


def apply_notch(sig: np.ndarray, cutoff: float, q: float, fps: float) -> np.ndarray:
    """
    Apply Notch filter
    """
    nyq = fps / 2.0
    freq = cutoff / nyq
    freq = max(0.01, min(freq, 0.99))
    
    b, a = signal.iirnotch(freq, q)
    filtered = signal.filtfilt(b, a, sig)
    return filtered


def apply_chebyshev(sig: np.ndarray, order: int, ripple: float,
                    freq_min: float, freq_max: float, fps: float) -> np.ndarray:
    """
    Apply Chebyshev Type I bandpass filter
    """
    nyq = fps / 2.0
    low = freq_min / nyq
    high = freq_max / nyq
    
    low = max(0.01, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))
    
    b, a = signal.cheby1(order, ripple, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, sig)
    return filtered

def apply_cheby2(sig: np.ndarray, order: int, stopband_atten: float,
                 freq_min: float, freq_max: float, fps: float) -> np.ndarray:
    """
    Apply Chebyshev Type II bandpass filter
    """
    nyq = fps / 2.0
    low = freq_min / nyq
    high = freq_max / nyq
    
    low = max(0.01, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))
    
    b, a = signal.cheby2(order, stopband_atten, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, sig)
    return filtered


def apply_elliptic(sig: np.ndarray, order: int, passband_ripple: float,
                   stopband_atten: float, freq_min: float, freq_max: float, 
                   fps: float) -> np.ndarray:
    """
    Apply Elliptic bandpass filter
    """
    nyq = fps / 2.0
    low = freq_min / nyq
    high = freq_max / nyq
    
    low = max(0.01, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))
    
    b, a = signal.ellip(order, passband_ripple, stopband_atten, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, sig)
    return filtered


def apply_moving_average(sig: np.ndarray, window_size: int) -> np.ndarray:
    """
    Apply moving average filter
    """
    if window_size % 2 == 0:
        window_size += 1  # Make odd
    
    kernel = np.ones(window_size) / window_size
    filtered = np.convolve(sig, kernel, mode='same')
    return filtered


def apply_savgol(sig: np.ndarray, window_size: int, poly_order: int) -> np.ndarray:
    """
    Apply Savitzky-Golay filter
    """
    if window_size % 2 == 0:
        window_size += 1
    
    if poly_order >= window_size:
        poly_order = window_size - 1
    
    filtered = signal.savgol_filter(sig, window_size, poly_order)
    return filtered

def apply_wavelet(sig: np.ndarray, wavelet: str, level: int, threshold_mode: str = 'soft') -> np.ndarray:
    """
    Apply Wavelet Denoising
    """
    # Decompose
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    
    # Calculate threshold (VisuShrink)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(sig)))
    
    # Thresholding
    new_coeffs = [coeffs[0]] # Keep approximation coefficients
    for i in range(1, len(coeffs)):
        new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode=threshold_mode))
        
    # Reconstruct
    filtered = pywt.waverec(new_coeffs, wavelet)
    
    # Ensure length match
    if len(filtered) > len(sig):
        filtered = filtered[:len(sig)]
        
    return filtered