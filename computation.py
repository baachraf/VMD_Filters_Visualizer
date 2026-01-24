"""
Signal Processing Engine
Contains all VMD and analysis algorithms with enhanced auto-discovery
"""

import numpy as np
from scipy import signal, sparse
from scipy.sparse.linalg import spsolve
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.optimize import differential_evolution
from typing import Tuple, List, Dict, Any
import filters


def detrend_signal(sig: np.ndarray, lambda_val: int = 100) -> np.ndarray:
    """
    Detrend signal using Tarvainen-Ranta method (smoothness priors)
    """
    T = len(sig)
    I = np.eye(T)
    D2 = sparse.diags([1, -2, 1], [0, 1, 2], shape=(T-2, T)).toarray()
    A = I + (lambda_val**2) * (D2.T @ D2)
    trend = np.linalg.solve(A, sig)
    return sig - trend


def normalize_signal(sig: np.ndarray, method: str = 'z-score') -> np.ndarray:
    """
    Normalize signal
    """
    if method == 'z-score':
        return (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
    elif method == 'min-max':
        return (sig - np.min(sig)) / (np.max(sig) - np.min(sig) + 1e-8)
    else:
        return sig


def enhance_harmonics(pulse_signal: np.ndarray, fps: float, 
                      freq_min: float = 0.7, freq_max: float = 4.0, 
                      gain: float = 2.0) -> np.ndarray:
    """
    Enhance harmonic components of the pulse signal
    """
    if len(pulse_signal) < 4:
        return pulse_signal
    try:
        n = len(pulse_signal)
        # Ensure n is a multiple of 1024 for better FFT performance if possible, or at least even
        # But for signal reconstruction, we must keep original length.
        # We can pad for FFT calculation but must truncate back.
        # For now, we use the signal length directly.
        
        fft_signal = np.fft.fft(pulse_signal)
        freqs = np.fft.fftfreq(n, 1 / fps)
        
        hr_mask = (np.abs(freqs) >= freq_min) & (np.abs(freqs) <= freq_max)
        if not np.any(hr_mask): return pulse_signal
        
        hr_spectrum = np.abs(fft_signal[hr_mask])
        if len(hr_spectrum) == 0: return pulse_signal
        
        dominant_idx = np.argmax(hr_spectrum)
        dominant_freq = np.abs(freqs[hr_mask][dominant_idx])
        
        enhancement_mask = np.zeros_like(freqs, dtype=bool)
        
        for harmonic in range(1, 4):
            harmonic_freq = dominant_freq * harmonic
            if harmonic_freq <= fps / 2:
                harmonic_mask = (np.abs(freqs) >= harmonic_freq * 0.9) & (np.abs(freqs) <= harmonic_freq * 1.1)
                enhancement_mask |= harmonic_mask
                
        enhanced_fft = fft_signal.copy()
        enhanced_fft[enhancement_mask] *= gain
        
        bandpass_mask = (np.abs(freqs) >= freq_min * 0.5) & (np.abs(freqs) <= freq_max * 2.0)
        enhanced_fft[~bandpass_mask] *= 0.1
        
        return np.real(np.fft.ifft(enhanced_fft))
    except Exception as e:
        print(f"      [PostProc STAGE-ERROR] Harmonic enhancement failed: {e}")
        return pulse_signal


def run_vmd(sig: np.ndarray, K: int, alpha: float, tau: float, 
            DC: int, init: int, tol: float, fps: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Variational Mode Decomposition
    
    Args:
        sig: Input signal
        K: Number of modes
        alpha: Bandwidth constraint
        tau: Noise tolerance
        DC: Remove DC component (0 or 1)
        init: Initialization (0, 1, or 2)
        tol: Convergence tolerance
        fps: Sampling frequency (for fallback)
    
    Returns:
        modes: Array of shape (K, len(sig)) containing K modes
        center_freqs: Array of K center frequencies
    """
    try:
        # Try using vmdpy library if available
        from vmdpy import VMD
        modes, _, center_freqs = VMD(sig, alpha, tau, K, DC, init, tol)
        return modes, center_freqs
    except ImportError:
        # Fallback: Simple implementation using bandpass filters
        print("vmdpy not found, using simple bandpass decomposition fallback")
        return simple_vmd_fallback(sig, K, fps)


def simple_vmd_fallback(sig: np.ndarray, K: int, fps: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple VMD fallback using bandpass filters
    Not a true VMD implementation but provides similar output for testing
    """
    modes = []
    center_freqs = []
    
    # Divide frequency range into K bands (adaptive based on signal)
    freq_min, freq_max = 0.5, 4.0
    
    # Check signal for dominant frequency range
    # Use Hanning window for Welch to reduce spectral leakage
    freqs, psd = signal.welch(sig, fs=fps, window='hann', nperseg=min(1024, len(sig)))
    physio_mask = (freqs >= freq_min) & (freqs <= freq_max)
    
    if np.any(physio_mask):
        physio_freqs = freqs[physio_mask]
        physio_psd = psd[physio_mask]
        
        # Find dominant frequency range
        peak_idx = np.argmax(physio_psd)
        peak_freq = physio_freqs[peak_idx]
        
        # Adjust frequency range around peak
        freq_min = max(0.5, peak_freq - 1.5)
        freq_max = min(4.0, peak_freq + 1.5)
    
    freq_bands = np.linspace(freq_min, freq_max, K + 1)
    
    for i in range(K):
        freq_low = freq_bands[i]
        freq_high = freq_bands[i + 1]
        center_freq = (freq_low + freq_high) / 2
        
        # Create bandpass filter for this mode
        nyq = fps / 2.0
        low = freq_low / nyq
        high = freq_high / nyq
        
        if high >= 1.0:
            high = 0.99
        if low >= high:
            low = high - 0.1
        
        try:
            b, a = signal.butter(4, [low, high], btype='band')
            mode = signal.filtfilt(b, a, sig)
        except:
            mode = np.zeros_like(sig)
        
        modes.append(mode)
        center_freqs.append(center_freq)
    
    return np.array(modes), np.array(center_freqs)


def assess_signal_quality(sig: np.ndarray, fps: float) -> Dict[str, float]:
    """
    Assess rPPG signal quality metrics
    Guides parameter selection strategy
    """
    # SNR estimation
    signal_power = np.var(sig)
    try:
        b, a = signal.butter(4, 5.0, fs=fps, btype='high')
        high_freq_noise = signal.filtfilt(b, a, sig)
        noise_power = np.var(high_freq_noise)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-12))
    except:
        snr = 0.0
    
    # Spectral concentration (how focused is the power?)
    try:
        # Use Hanning window and nperseg=1024 (or closest power of 2)
        nperseg = min(1024, len(sig))
        freqs, psd = signal.welch(sig, fs=fps, window='hann', nperseg=nperseg)
        mask = (freqs >= 0.5) & (freqs <= 4.0)
        if np.any(mask):
            psd_physio = psd[mask]
            concentration = np.max(psd_physio) / (np.sum(psd_physio) + 1e-12)
        else:
            concentration = 0.0
    except:
        concentration = 0.0
    
    # Stationarity (variance of sliding window variances)
    window_size = int(2 * fps)  # 2-second windows
    if window_size < len(sig):
        n_windows = len(sig) // window_size
        window_vars = [np.var(sig[i*window_size:(i+1)*window_size]) 
                       for i in range(n_windows) if (i+1)*window_size <= len(sig)]
        if len(window_vars) > 1:
            stationarity = 1.0 - (np.std(window_vars) / (np.mean(window_vars) + 1e-12))
            stationarity = max(0.0, min(1.0, stationarity))
        else:
            stationarity = 0.5
    else:
        stationarity = 0.5
    
    quality_score = (min(snr/20, 1.0) + concentration + stationarity) / 3
    
    return {
        'snr': snr,
        'spectral_concentration': concentration,
        'stationarity': stationarity,
        'quality_score': quality_score
    }


def estimate_optimal_K(sig: np.ndarray, fps: float) -> int:
    """
    Estimate optimal number of modes based on signal complexity
    Uses spectral entropy and frequency content
    """
    try:
        # Compute power spectral density
        # Use Hanning window and nperseg=1024
        nperseg = min(1024, len(sig))
        freqs, psd = signal.welch(sig, fs=fps, window='hann', nperseg=nperseg)
        
        # Focus on physiological range (0.5-4 Hz)
        mask = (freqs >= 0.5) & (freqs <= 4.0)
        
        if not np.any(mask):
            return 5  # Default
        
        psd_physio = psd[mask]
        
        # Calculate spectral entropy
        psd_norm = psd_physio / (np.sum(psd_physio) + 1e-12)
        psd_norm = psd_norm[psd_norm > 0]  # Remove zeros for log
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
        
        # Higher entropy suggests more complex signal → more modes needed
        # Empirical mapping for rPPG signals
        if spectral_entropy < 2.0:
            return 3  # Simple signal
        elif spectral_entropy < 3.0:
            return 5  # Moderate complexity
        else:
            return 7  # Complex signal
    except:
        return 5  # Safe default


def estimate_optimal_alpha(sig: np.ndarray, fps: float) -> float:
    """
    Estimate alpha based on signal noise level
    Higher noise → higher alpha (more regularization)
    """
    try:
        # Estimate noise using high-frequency components
        nyq = fps / 2.0
        cutoff = min(0.1, 5.0 / nyq)
        b, a = signal.butter(4, cutoff, btype='high')
        noise_estimate = signal.filtfilt(b, a, sig)
        noise_std = np.std(noise_estimate)
        signal_std = np.std(sig)
        
        snr_estimate = signal_std / (noise_std + 1e-12)
        
        # Map SNR to alpha (empirical relationship)
        if snr_estimate > 10:
            return 500  # Low noise
        elif snr_estimate > 5:
            return 2000  # Moderate noise
        else:
            return 5000  # High noise
    except:
        return 2000  # Safe default


def get_center_frequency(mode: np.ndarray, fps: float) -> float:
    """
    Calculate center frequency of a mode using FFT
    """
    if len(mode) == 0:
        return 0.0
    
    # Compute FFT
    fft_vals = np.fft.fft(mode)
    freqs = np.fft.fftfreq(len(mode), 1/fps)
    
    # Only positive frequencies
    pos_mask = freqs > 0
    freqs_pos = freqs[pos_mask]
    fft_mag = np.abs(fft_vals[pos_mask])
    
    if len(freqs_pos) == 0 or np.sum(fft_mag) == 0:
        return 0.0
    
    # Weighted frequency average
    center_freq = np.sum(freqs_pos * fft_mag) / np.sum(fft_mag)
    return center_freq


def calculate_mode_energies(modes: np.ndarray) -> np.ndarray:
    """
    Calculate energy percentage for each mode
    """
    energies = []
    total_energy = sum(np.sum(mode**2) for mode in modes)
    
    if total_energy == 0:
        return np.zeros(len(modes))
    
    for mode in modes:
        mode_energy = np.sum(mode**2)
        energy_percent = (mode_energy / total_energy) * 100
        energies.append(energy_percent)
    
    return np.array(energies)


def calculate_correlation(mode: np.ndarray, reference: np.ndarray) -> float:
    """
    Calculate correlation between mode and reference signal
    """
    if len(mode) != len(reference):
        return 0.0
    
    if np.std(mode) == 0 or np.std(reference) == 0:
        return 0.0
    
    corr_matrix = np.corrcoef(mode, reference)
    return abs(corr_matrix[0, 1])


def calculate_kurtosis(mode: np.ndarray) -> float:
    """
    Calculate kurtosis of a mode
    """
    if len(mode) == 0 or np.std(mode) == 0:
        return 0.0
    
    return scipy_kurtosis(mode)


def adaptive_mode_selection(modes: np.ndarray, center_freqs: np.ndarray,
                            original_signal: np.ndarray, fps: float) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Adaptively select modes without manual threshold tuning
    Uses statistical outlier detection
    """
    energies = calculate_mode_energies(modes)
    
    # Calculate reference (bandpass filtered signal)
    reference = filters.apply_butterworth(original_signal, 4, 0.7, 4.0, fps)
    
    correlations = np.array([calculate_correlation(mode, reference) 
                            for mode in modes])
    kurtoses = np.array([calculate_kurtosis(mode) for mode in modes])
    
    # Adaptive thresholds using median absolute deviation (MAD)
    freq_in_range = (center_freqs >= 0.7) & (center_freqs <= 4.0)
    
    # Energy threshold: median - 0.5*MAD (select above-average energy modes)
    if len(energies) > 0:
        energy_median = np.median(energies)
        energy_mad = np.median(np.abs(energies - energy_median))
        energy_threshold = max(5.0, energy_median - 0.5 * energy_mad)
    else:
        energy_threshold = 5.0
    
    # Correlation threshold: median (select above-median correlation)
    if np.any(freq_in_range) and len(correlations[freq_in_range]) > 0:
        corr_threshold = np.median(correlations[freq_in_range])
    else:
        corr_threshold = 0.5
    
    # Kurtosis threshold: exclude extreme outliers (>10)
    kurtosis_threshold = 10.0
    
    # Select modes
    selected = (freq_in_range & 
               (energies >= energy_threshold) & 
               (correlations >= corr_threshold) &
               (kurtoses <= kurtosis_threshold))
    
    # Build mode info
    mode_info = []
    for i in range(len(modes)):
        mode_info.append({
            'center_freq': center_freqs[i],
            'energy': energies[i],
            'correlation': correlations[i],
            'kurtosis': kurtoses[i],
            'selected': selected[i]
        })
    
    return selected, mode_info


def select_modes(modes: np.ndarray, center_freqs: np.ndarray, 
                 original_signal: np.ndarray, selection_params: Dict[str, Any],
                 fps: float) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Select relevant modes based on criteria
    
    Returns:
        selected_indices: Boolean array indicating which modes are selected
        mode_info: List of dictionaries with mode information
    """
    # Check if adaptive mode is requested
    if selection_params.get('selection_method') == 'adaptive':
        return adaptive_mode_selection(modes, center_freqs, original_signal, fps)
    
    # Original manual threshold mode
    freq_min = selection_params['freq_min']
    freq_max = selection_params['freq_max']
    corr_threshold = selection_params['correlation_threshold']
    energy_threshold = selection_params['energy_threshold']
    kurtosis_max = selection_params['kurtosis_max']
    method = selection_params['selection_method']
    corr_ref = selection_params['correlation_reference']
    
    # Calculate mode properties
    energies = calculate_mode_energies(modes)
    
    # Determine reference signal for correlation
    if corr_ref == 'bandpass_filtered':
        reference = filters.apply_butterworth(original_signal, 4, freq_min, freq_max, fps)
    elif corr_ref == 'original':
        reference = original_signal
    else:  # selected_modes - will be handled iteratively
        reference = original_signal
    
    mode_info = []
    selected = np.zeros(len(modes), dtype=bool)
    
    # First pass: Identify fundamental mode candidates
    fundamental_candidates = []
    
    for i, mode in enumerate(modes):
        # Recalculate center frequency for accuracy
        cf = get_center_frequency(mode, fps)
        energy = energies[i]
        corr = calculate_correlation(mode, reference)
        kurt = calculate_kurtosis(mode)
        
        # Apply selection criteria
        freq_ok = freq_min <= cf <= freq_max
        corr_ok = corr >= corr_threshold
        energy_ok = energy >= energy_threshold
        kurt_ok = kurt <= kurtosis_max
        
        if method == 'frequency_only':
            is_selected = freq_ok
        elif method == 'frequency+correlation':
            is_selected = freq_ok and corr_ok
        elif method == 'frequency+energy':
            is_selected = freq_ok and energy_ok
        elif method == 'all_criteria':
            is_selected = freq_ok and corr_ok and energy_ok and kurt_ok
        elif 'harmonics' in method: # Handle harmonic methods by basic frequency check first
            is_selected = freq_ok
        else:
            is_selected = freq_ok
        
        selected[i] = is_selected
        
        if is_selected:
            fundamental_candidates.append((i, cf, energy))
            
        mode_info.append({
            'center_freq': cf,
            'energy': energy,
            'correlation': corr,
            'kurtosis': kurt,
            'selected': is_selected
        })
    
    # Harmonic Recombination Logic
    # Only run if explicitly requested in the method
    if 'harmonics' in method and np.any(selected) and len(fundamental_candidates) > 0:
        # Find the strongest fundamental mode (highest energy)
        fundamental_candidates.sort(key=lambda x: x[2], reverse=True)
        primary_idx, primary_freq, _ = fundamental_candidates[0]
        
        # Scan for harmonics (2nd and 3rd)
        for i, mode in enumerate(modes):
            if not selected[i]:
                cf = mode_info[i]['center_freq']
                
                # Check if close to 2nd harmonic (within 10% tolerance)
                is_2nd_harmonic = abs(cf - 2 * primary_freq) < (0.1 * 2 * primary_freq)
                # Check if close to 3rd harmonic
                is_3rd_harmonic = abs(cf - 3 * primary_freq) < (0.1 * 3 * primary_freq)
                
                if is_2nd_harmonic or is_3rd_harmonic:
                    selected[i] = True
                    mode_info[i]['selected'] = True
                    mode_info[i]['note'] = 'Harmonic Recombined'

    # If using selected_modes reference and we have at least one selection, recalculate
    if corr_ref == 'selected_modes' and np.any(selected):
        reference = np.sum(modes[selected], axis=0)
        
        for i, mode in enumerate(modes):
            if not selected[i]:  # Recalculate for non-selected modes
                corr = calculate_correlation(mode, reference)
                mode_info[i]['correlation'] = corr
    
    return selected, mode_info


def estimate_heart_rate(sig: np.ndarray, fps: float) -> float:
    """
    Estimate heart rate from signal using frequency domain method
    """
    if len(sig) < 2:
        return 0.0
    
    # Compute FFT
    fft_vals = np.fft.fft(sig)
    freqs = np.fft.fftfreq(len(sig), 1/fps)
    
    # Only positive frequencies in HR range [0.7, 4.0] Hz
    mask = (freqs > 0.7) & (freqs < 4.0)
    freqs_hr = freqs[mask]
    fft_mag = np.abs(fft_vals[mask])
    
    if len(freqs_hr) == 0:
        return 0.0
    
    # Find peak frequency
    peak_idx = np.argmax(fft_mag)
    peak_freq = freqs_hr[peak_idx]
    
    # Convert to BPM
    hr_bpm = peak_freq * 60
    return hr_bpm


def calculate_snr(original: np.ndarray, filtered: np.ndarray, fps: float = 30.0) -> float:
    """
    Calculate Spectral Signal-to-Noise Ratio (SNR) in dB.
    Measures the ratio of power around the dominant peak to the background noise.
    """
    # Use the filtered signal for quality assessment
    sig = filtered
    
    # Detrend and normalize for consistent PSD calculation
    sig = sig - np.mean(sig)
    if np.std(sig) > 0:
        sig = sig / np.std(sig)
    
    n = len(sig)
    if n < 2: return 0.0
    
    # Compute PSD
    freqs, psd = signal.welch(sig, fs=fps, window='hann', nperseg=min(n, 1024))
    
    # Define physiological range (e.g., 40-240 BPM -> 0.66 - 4.0 Hz)
    min_freq, max_freq = 0.65, 4.0
    mask = (freqs >= min_freq) & (freqs <= max_freq)
    
    if not np.any(mask):
        return 0.0
        
    freqs_roi = freqs[mask]
    psd_roi = psd[mask]
    
    # Find dominant peak
    peak_idx = np.argmax(psd_roi)
    peak_freq = freqs_roi[peak_idx]
    
    # Define signal band around peak (e.g., +/- 0.15 Hz)
    half_width = 0.15
    signal_mask = (freqs_roi >= peak_freq - half_width) & (freqs_roi <= peak_freq + half_width)
    
    # Calculate powers
    signal_power = np.sum(psd_roi[signal_mask])
    total_power = np.sum(psd_roi)
    noise_power = total_power - signal_power
    
    if noise_power <= 0:
        return 100.0 # Perfect signal
    
    if signal_power <= 0:
        return -10.0
        
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db


def calculate_time_lag(original: np.ndarray, filtered: np.ndarray, fps: float) -> float:
    """
    Calculate time lag (in ms) between original and filtered signal using cross-correlation.
    Positive lag means filtered signal is delayed.
    """
    if len(original) != len(filtered):
        # Truncate to shorter length
        min_len = min(len(original), len(filtered))
        original = original[:min_len]
        filtered = filtered[:min_len]
        
    # Normalize signals for correlation
    orig_norm = (original - np.mean(original)) / (np.std(original) + 1e-8)
    filt_norm = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
    
    correlation = signal.correlate(orig_norm, filt_norm, mode='full')
    lags = signal.correlation_lags(len(orig_norm), len(filt_norm), mode='full')
    
    lag_idx = lags[np.argmax(correlation)]
    
    # Convert lag index to time in ms
    # Note: signal.correlate(a, b) computes sum(a[k] * b[k+m])
    # If b is delayed version of a, peak will be at negative lag index in scipy's definition?
    # Let's verify: if b[t] = a[t-d], then peak is at lag d.
    # Scipy correlate(in1, in2):
    # If in2 is shifted by k relative to in1, the peak is at index corresponding to k.
    # Actually, let's just use the sign convention:
    # Lag > 0 means filtered is "ahead" (shifted left), Lag < 0 means filtered is "delayed" (shifted right).
    # Wait, usually we want "Delay".
    # If filtered is delayed, it matches original at a positive shift of original?
    # Let's stick to: Lag = (index of max correlation) / fps * 1000
    # But we need to be careful with the order of inputs.
    # correlate(original, filtered): peak at positive lag means filtered is shifted LEFT (ahead).
    # peak at negative lag means filtered is shifted RIGHT (delayed).
    
    # So, Delay = -lag_idx / fps * 1000
    
    time_lag_ms = -lag_idx / fps * 1000.0
    return time_lag_ms


def calculate_combined_score(sig: np.ndarray, extracted: np.ndarray, 
                             modes: np.ndarray, selected: np.ndarray,
                             fps: float) -> float:
    """
    Multi-objective score combining SNR, HR validity, sparsity, and spectral purity
    """
    # SNR component (40% weight)
    snr = calculate_snr(sig, extracted, fps)
    snr_score = min(snr / 20.0, 1.0) * 0.4  # Normalize to [0, 0.4]
    
    # HR validity component (30% weight)
    hr = estimate_heart_rate(extracted, fps)
    hr_valid = 1.0 if 42 <= hr <= 180 else 0.0  # Physiological range
    hr_score = hr_valid * 0.3
    
    # Sparsity component (20% weight) - prefer fewer modes
    n_selected = np.sum(selected)
    n_total = len(modes)
    sparsity = 1.0 - (n_selected / n_total)
    sparsity_score = sparsity * 0.2
    
    # Spectral purity component (10% weight)
    try:
        fft_vals = np.fft.fft(extracted)
        freqs = np.fft.fftfreq(len(extracted), 1/fps)
        mask = (freqs > 0.7) & (freqs < 4.0)
        psd = np.abs(fft_vals[mask])**2
        
        if np.sum(psd) > 0:
            # Ratio of peak power to total power
            peak_power = np.max(psd)
            total_power = np.sum(psd)
            purity = peak_power / total_power
            purity_score = min(purity, 1.0) * 0.1
        else:
            purity_score = 0.0
    except:
        purity_score = 0.0
    
    return snr_score + hr_score + sparsity_score + purity_score


def auto_optimize_vmd(sig: np.ndarray, K_range: List[int], alpha_range: List[float],
                      metric: str, selection_params: Dict[str, Any], 
                      fps: float, tau: float, DC: int, init: int, tol: float,
                      fft_size: int = 1024, progress_callback=None) -> Tuple[int, float, float]:
    """
    Perform grid search to find optimal VMD parameters
    
    Returns:
        best_K: Optimal K value
        best_alpha: Optimal alpha value
        best_score: Best metric score achieved
    """
    best_K = K_range[0]
    best_alpha = alpha_range[0]
    best_score = -float('inf')
    
    # Create grid
    K_values = list(range(K_range[0], K_range[1] + 1))
    alpha_values = np.linspace(alpha_range[0], alpha_range[1], 5)
    
    total_iterations = len(K_values) * len(alpha_values)
    current_iteration = 0
    
    for K in K_values:
        for alpha in alpha_values:
            try:
                # Run VMD
                modes, center_freqs = run_vmd(sig, K, alpha, tau, DC, init, tol, fps)
                
                # Select modes
                selected, mode_info = select_modes(modes, center_freqs, sig, selection_params, fps)
                
                # Extract signal
                if np.any(selected):
                    extracted = np.sum(modes[selected], axis=0)
                else:
                    extracted = modes[0]  # Fallback
                
                # Calculate metric
                if metric == 'snr':
                    score = calculate_snr(sig, extracted, fps)
                elif metric == 'reconstruction_error':
                    reconstructed = np.sum(modes, axis=0)
                    score = -np.mean((sig - reconstructed) ** 2)  # Negative MSE
                elif metric == 'spectral_purity':
                    # Use spectral entropy as a proxy for purity (lower is better, so negate)
                    # Use Hanning window and nperseg=1024
                    nperseg = min(fft_size, len(extracted))
                    freqs, psd = signal.welch(extracted, fs=fps, window='hann', nperseg=nperseg)
                    mask = (freqs >= 0.5) & (freqs <= 4.0)
                    if np.any(mask):
                        psd_physio = psd[mask]
                        psd_norm = psd_physio / (np.sum(psd_physio) + 1e-12)
                        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
                        score = -entropy # Maximize negative entropy
                    else:
                        score = -10.0
                elif metric == 'combined':
                    score = calculate_combined_score(sig, extracted, modes, selected, fps) * 100
                else:  # default to combined
                    score = calculate_combined_score(sig, extracted, modes, selected, fps) * 100
                
                # Update best
                if score > best_score:
                    best_score = score
                    best_K = K
                    best_alpha = alpha
                
            except Exception as e:
                print(f"Error at K={K}, alpha={alpha}: {e}")
            
            current_iteration += 1
            if progress_callback:
                progress_callback(current_iteration, total_iterations)
    
    return best_K, best_alpha, best_score


def auto_optimize_vmd_bayesian(sig: np.ndarray, selection_params: Dict[str, Any],
                                fps: float, tau: float, DC: int, init: int, 
                                tol: float, n_iterations: int = 20,
                                progress_callback=None) -> Tuple[int, float, float]:
    """
    Bayesian optimization for VMD parameters using differential evolution
    More efficient than grid search
    """
    # Start with intelligent initial guesses
    K_init = estimate_optimal_K(sig, fps)
    alpha_init = estimate_optimal_alpha(sig, fps)
    
    print(f"Initial estimates: K={K_init}, alpha={alpha_init:.0f}")
    
    best_score = -float('inf')
    best_K = K_init
    best_alpha = alpha_init
    
    iteration_count = [0]
    
    def objective(params):
        K = int(params[0])
        alpha = params[1]
        
        iteration_count[0] += 1
        if progress_callback:
            progress_callback(iteration_count[0], n_iterations)
        
        try:
            modes, center_freqs = run_vmd(sig, K, alpha, tau, DC, init, tol, fps)
            selected, mode_info = select_modes(modes, center_freqs, sig, 
                                              selection_params, fps)
            
            if np.any(selected):
                extracted = np.sum(modes[selected], axis=0)
            else:
                extracted = modes[0]
            
            score = calculate_combined_score(sig, extracted, modes, selected, fps)
            return -score  # Minimize negative score
            
        except Exception as e:
            print(f"Optimization iteration failed: {e}")
            return 1e6  # Penalty for failed decomposition
    
    # Define bounds
    bounds = [(3, 10), (500, 8000)]  # (K_range, alpha_range)
    
    # Run differential evolution (efficient global optimizer)
    result = differential_evolution(
        objective, 
        bounds, 
        maxiter=n_iterations//4,
        popsize=4, 
        seed=42,
        atol=0.01,
        tol=0.01
    )
    
    best_K = int(result.x[0])
    best_alpha = result.x[1]
    best_score = -result.fun
    
    return best_K, best_alpha, best_score


def smart_vmd_analysis(sig: np.ndarray, fps: float, 
                       mode: str = 'auto') -> Dict[str, Any]:
    """
    Smart VMD analysis that automatically determines optimal parameters
    
    Args:
        sig: Input signal
        fps: Sampling frequency
        mode: 'auto' for fully automatic, 'assisted' for suggestions
    
    Returns:
        Dictionary with analysis results and recommendations
    """
    # Assess signal quality
    quality = assess_signal_quality(sig, fps)
    
    # Estimate optimal parameters
    K_optimal = estimate_optimal_K(sig, fps)
    alpha_optimal = estimate_optimal_alpha(sig, fps)
    
    # Determine selection strategy based on quality
    if quality['quality_score'] > 0.7:
        selection_method = 'frequency_only'  # High quality - simple selection
    elif quality['quality_score'] > 0.4:
        selection_method = 'frequency+correlation'  # Medium - add correlation
    else:
        selection_method = 'adaptive'  # Low quality - adaptive thresholds
    
    recommendations = {
        'signal_quality': quality,
        'recommended_K': K_optimal,
        'recommended_alpha': alpha_optimal,
        'recommended_selection': selection_method,
        'rationale': {
            'K': f"Based on spectral complexity (entropy-based analysis)",
            'alpha': f"Based on estimated SNR (~{quality['snr']:.1f} dB)",
            'selection': f"Based on signal quality score ({quality['quality_score']:.2f})"
        }
    }
    
    return recommendations
