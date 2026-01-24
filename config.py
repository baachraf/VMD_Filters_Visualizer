"""
Configuration Management Module
Handles loading, saving, and validation of application settings
"""

import json
import os
from typing import Dict, Any, Tuple

CONFIG_DIR = "configs"
DEFAULT_CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
USER_CONFIG_FILE = os.path.join(CONFIG_DIR, "user_config.json")

def get_default_config() -> Dict[str, Any]:
    """
    Returns default configuration dictionary with all parameters
    """
    return {
        "rppg_column_name": "rPPG_signal",
        "fps": 30,
        "save_path": "saved_data",
        "preprocessing": {
            "detrend": {
                "enabled": True,
                "lambda": 100
            },
            "normalization": {
                "enabled": True,
                "method": "z-score"
            },
            "harmonics": {
                "enabled": False,
                "harmonics_gain": 2.0,
                "freq_min": 0.7,
                "freq_max": 4.0
            },
            "fft_size": 1024
        },
        "vmd": {
            "K": 5,
            "alpha": 2000,
            "tau": 0,
            "DC": 0,
            "init": 1,
            "tol": 1e-7
        },
        "mode_selection": {
            "freq_min": 0.7,
            "freq_max": 4.0,
            "correlation_threshold": 0.5,
            "energy_threshold": 5.0,
            "kurtosis_max": 3.0,
            "selection_method": "frequency_only",
            "correlation_reference": "bandpass_filtered"
        },
        "auto_optimize": {
            "K_range": [3, 7],
            "alpha_range": [500, 5000],
            "optimization_metric": "snr"
        },
        "traditional_filters": {
            "butterworth": {"enabled": True, "order": 4, "freq_min": 0.7, "freq_max": 4.0},
            "chebyshev": {"enabled": True, "order": 4, "ripple": 0.5, "freq_min": 0.7, "freq_max": 4.0},
            "cheby2": {"enabled": True, "order": 4, "stopband_atten": 40, "freq_min": 0.7, "freq_max": 4.0}, 
            "elliptic": {"enabled": True, "order": 4, "passband_ripple": 0.5, "stopband_atten": 40, "freq_min": 0.7, "freq_max": 4.0}, 
            "moving_average": {"enabled": True, "window_size": 15},
            "savgol": {"enabled": True, "window_size": 15, "poly_order": 3},
            "wavelet": {"enabled": True, "wavelet": "db4", "level": 4, "threshold_mode": "soft"},
            "harmonics": {"enabled": True, "harmonics_gain": 2.0, "freq_min": 0.7, "freq_max": 4.0}
        },
        "ui_state": {
            "left_panel_collapsed": False,
            "show_individual_modes": True,
            "chart_display_mode": "all"
        }
    }


def load_config(filepath: str = None) -> Dict[str, Any]:
    """
    Load configuration.
    If filepath is provided, it loads that specific file over the hardcoded defaults.
    Otherwise, it follows the priority:
    1. user_config.json (overrides defaults)
    2. config.json (overrides hardcoded defaults)
    3. Hardcoded defaults
    """
    config = get_default_config()

    if filepath:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    file_config = json.load(f)
                config = merge_configs(config, file_config)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
    else:
        # Load default config file if exists
        if os.path.exists(DEFAULT_CONFIG_FILE):
            try:
                with open(DEFAULT_CONFIG_FILE, 'r') as f:
                    default_file_config = json.load(f)
                config = merge_configs(config, default_file_config)
            except Exception as e:
                print(f"Error loading {DEFAULT_CONFIG_FILE}: {e}")

        # Load user config file if exists and override
        if os.path.exists(USER_CONFIG_FILE):
            try:
                with open(USER_CONFIG_FILE, 'r') as f:
                    user_file_config = json.load(f)
                config = merge_configs(config, user_file_config)
            except Exception as e:
                print(f"Error loading {USER_CONFIG_FILE}: {e}")
    
    return config


def save_config(config: Dict[str, Any], filepath: str = None) -> bool:
    """
    Save configuration to a JSON file.
    If filepath is not provided, it saves to user_config.json.
    Returns True if successful, False otherwise.
    """
    if filepath is None:
        filepath = USER_CONFIG_FILE
    
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config to {filepath}: {e}")
        return False


def validate_config(config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate configuration parameters
    Returns (is_valid, error_message)
    """
    try:
        # Check VMD parameters
        if config['vmd']['K'] < 2:
            return False, "VMD parameter K must be >= 2"
        
        if config['vmd']['alpha'] <= 0:
            return False, "VMD parameter alpha must be > 0"
        
        # Check frequency ranges
        if config['mode_selection']['freq_min'] >= config['mode_selection']['freq_max']:
            return False, "freq_max must be greater than freq_min"
        
        if config['mode_selection']['freq_min'] < 0:
            return False, "freq_min must be positive"
        
        # Check traditional filter parameters
        for filter_name, params in config['traditional_filters'].items():
            if 'freq_min' in params and 'freq_max' in params:
                if params['freq_min'] >= params['freq_max']:
                    return False, f"{filter_name}: freq_max must be greater than freq_min"
        
        # Check Savitzky-Golay parameters
        savgol = config['traditional_filters']['savgol']
        if savgol['window_size'] % 2 == 0:
            return False, "Savitzky-Golay window_size must be odd"
        
        if savgol['poly_order'] >= savgol['window_size']:
            return False, "Savitzky-Golay poly_order must be < window_size"
        
        # Check auto-optimization ranges
        if config['auto_optimize']['K_range'][0] >= config['auto_optimize']['K_range'][1]:
            return False, "Auto-optimize K_range: max must be > min"
        
        if config['auto_optimize']['alpha_range'][0] >= config['auto_optimize']['alpha_range'][1]:
            return False, "Auto-optimize alpha_range: max must be > min"
        
        return True, ""
    
    except KeyError as e:
        return False, f"Missing required config key: {e}"
    except Exception as e:
        return False, f"Validation error: {e}"


def get_config_value(config: Dict[str, Any], key_path: str) -> Any:
    """
    Get configuration value using dot notation path
    Example: get_config_value(config, "vmd.K") returns config['vmd']['K']
    """
    keys = key_path.split('.')
    value = config
    for key in keys:
        value = value[key]
    return value


def set_config_value(config: Dict[str, Any], key_path: str, value: Any) -> Dict[str, Any]:
    """
    Set configuration value using dot notation path
    Returns updated config dictionary
    """
    keys = key_path.split('.')
    current = config
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value
    return config


def merge_configs(default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge user config with default config
    User values override defaults, but missing keys are filled from defaults
    """
    merged = default.copy()
    
    for key, value in user.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged
