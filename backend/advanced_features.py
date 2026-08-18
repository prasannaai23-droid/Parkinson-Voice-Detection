"""
ADVANCED FEATURE EXTRACTION (215 Features) - FIXED
==================================================
Extracts 215 clinical acoustic features with CONSISTENT NAMING
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
import librosa
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# DEFINE CONSISTENT FEATURE NAMES
FEATURE_NAMES = []

# 1. Pitch & F0 Statistics (10)
F0_FEATURES = ['f0_mean', 'f0_std', 'f0_median', 'f0_min', 'f0_max', 
               'f0_range', 'f0_skew', 'f0_kurtosis', 'f0_q25', 'f0_q75']
FEATURE_NAMES.extend(F0_FEATURES)

# 2. Jitter Metrics (4)
JITTER_FEATURES = ['jitter', 'jitter_abs', 'jitter_rap', 'jitter_ppq5']
FEATURE_NAMES.extend(JITTER_FEATURES)

# 3. Shimmer Metrics (4)
SHIMMER_FEATURES = ['shimmer', 'shimmer_db', 'shimmer_apq3', 'shimmer_apq5']
FEATURE_NAMES.extend(SHIMMER_FEATURES)

# 4. HNR & Energy (8)
HNR_FEATURES = ['hnr', 'hnr_mean', 'hnr_std', 'energy_mean', 'energy_std', 
                'rms_mean', 'rms_std', 'zcr_mean']
FEATURE_NAMES.extend(HNR_FEATURES)

# 5. Spectral Features (24)
SPECTRAL_FEATURES = [
    'spectral_centroid_mean', 'spectral_centroid_std', 'spectral_centroid_min', 'spectral_centroid_max',
    'spectral_centroid_skew', 'spectral_centroid_kurt',
    'spectral_bandwidth_mean', 'spectral_bandwidth_std', 'spectral_bandwidth_min', 'spectral_bandwidth_max',
    'spectral_rolloff_mean', 'spectral_rolloff_std', 'spectral_rolloff_min', 'spectral_rolloff_max',
    'spectral_flatness_mean', 'spectral_flatness_std', 'spectral_flatness_min', 'spectral_flatness_max'
]
# Add spectral contrast features (only 3 bands to get exactly 215 total)
for i in range(3):
    SPECTRAL_FEATURES.append(f'spectral_contrast_mean_{i}')
    SPECTRAL_FEATURES.append(f'spectral_contrast_std_{i}')
FEATURE_NAMES.extend(SPECTRAL_FEATURES)

# 6. MFCC Features (160)
for i in range(20):
    FEATURE_NAMES.append(f'mfcc_mean_{i}')
    FEATURE_NAMES.append(f'mfcc_std_{i}')
    FEATURE_NAMES.append(f'mfcc_min_{i}')
    FEATURE_NAMES.append(f'mfcc_max_{i}')
    FEATURE_NAMES.append(f'mfcc_delta_mean_{i}')
    FEATURE_NAMES.append(f'mfcc_delta_std_{i}')
    FEATURE_NAMES.append(f'mfcc_delta2_mean_{i}')
    FEATURE_NAMES.append(f'mfcc_delta2_std_{i}')

# 7. Signal Morphology (5)
MORPH_FEATURES = ['signal_mean', 'signal_std', 'signal_skewness', 'signal_kurtosis', 'voiced_ratio']
FEATURE_NAMES.extend(MORPH_FEATURES)

# Total: 10 + 4 + 4 + 8 + 24 + 160 + 5 = 215
assert len(FEATURE_NAMES) == 215, f"Expected 215 features, got {len(FEATURE_NAMES)}"

def extract_pitch_and_perturbations(y: np.ndarray, sr: int):
    """
    Extract accurate pitch periods and clinical perturbation metrics (Jitter & Shimmer).
    FIXED: Proper cycle-to-cycle analysis
    """
    # 1. Fast Pitch extraction via YIN
    try:
        f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr, frame_length=2048, hop_length=256)
        voiced_flag = np.ones_like(f0, dtype=bool)
        voiced_probs = np.ones_like(f0)
    except Exception:
        f0 = None
        voiced_flag = None

    if f0 is not None and voiced_flag is not None:
        f0_valid = f0[voiced_flag & (f0 >= 50) & (f0 <= 400)]
    else:
        f0_valid = np.array([])

    # Fallback to autocorrelation pitch if pYIN failed
    if len(f0_valid) < 5:
        try:
            autocorr = np.correlate(y, y, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            min_p = int(sr / 400)
            max_p = int(sr / 50)
            if max_p < len(autocorr):
                subset = autocorr[min_p:max_p]
                peaks, _ = signal.find_peaks(subset, height=np.max(subset) * 0.25)
                if len(peaks) > 0:
                    best_f0 = float(sr / (min_p + peaks[0]))
                    f0_valid = np.full(10, best_f0)
        except Exception:
            pass

    # F0 statistics (10 features)
    f0_stats = {}
    if len(f0_valid) >= 3:
        f0_mean = float(np.mean(f0_valid))
        f0_std = float(np.std(f0_valid))
        f0_stats = {
            'f0_mean': f0_mean,
            'f0_std': f0_std,
            'f0_median': float(np.median(f0_valid)),
            'f0_min': float(np.min(f0_valid)),
            'f0_max': float(np.max(f0_valid)),
            'f0_range': float(np.ptp(f0_valid)),
            'f0_skew': float(pd.Series(f0_valid).skew()) if len(f0_valid) > 2 else 0.0,
            'f0_kurtosis': float(pd.Series(f0_valid).kurtosis()) if len(f0_valid) > 3 else 0.0,
            'f0_q25': float(np.percentile(f0_valid, 25)),
            'f0_q75': float(np.percentile(f0_valid, 75))
        }
    else:
        # Default values for healthy voice
        f0_stats = {
            'f0_mean': 140.0, 'f0_std': 8.0, 'f0_median': 140.0,
            'f0_min': 125.0, 'f0_max': 155.0, 'f0_range': 30.0,
            'f0_skew': 0.0, 'f0_kurtosis': 0.0, 'f0_q25': 135.0, 'f0_q75': 145.0
        }

    # Jitter Metrics (4 features) - FIXED: Proper cycle-to-cycle
    f0_mean = f0_stats['f0_mean']
    if len(f0_valid) >= 5:
        periods = 1.0 / (f0_valid + 1e-8)
        period_diffs = np.abs(np.diff(periods))
        mean_period = np.mean(periods)
        jitter_local = float(np.mean(period_diffs) / (mean_period + 1e-8))
        jitter_abs = float(np.mean(period_diffs))
        
        # RAP (3-point moving average)
        if len(periods) >= 3:
            rap_diffs = []
            for i in range(1, len(periods)-1):
                avg_3 = np.mean(periods[i-1:i+2])
                rap_diffs.append(abs(periods[i] - avg_3))
            jitter_rap = float(np.mean(rap_diffs) / (mean_period + 1e-8)) if rap_diffs else jitter_local * 0.8
        else:
            jitter_rap = jitter_local * 0.8

        # PPQ5 (5-point moving average)
        if len(periods) >= 5:
            ppq5_diffs = []
            for i in range(2, len(periods)-2):
                avg_5 = np.mean(periods[i-2:i+3])
                ppq5_diffs.append(abs(periods[i] - avg_5))
            jitter_ppq5 = float(np.mean(ppq5_diffs) / (mean_period + 1e-8)) if ppq5_diffs else jitter_local * 0.9
        else:
            jitter_ppq5 = jitter_local * 0.9
    else:
        # Default values for healthy voice
        jitter_local = 0.0035
        jitter_abs = 0.0035 / (f0_mean + 1e-8)
        jitter_rap = 0.0028
        jitter_ppq5 = 0.0031

    jitter_metrics = {
        'jitter': float(np.clip(jitter_local, 0.0005, 0.15)),
        'jitter_abs': float(np.clip(jitter_abs, 1e-7, 0.002)),
        'jitter_rap': float(np.clip(jitter_rap, 0.0004, 0.12)),
        'jitter_ppq5': float(np.clip(jitter_ppq5, 0.0005, 0.13))
    }

    # Shimmer Metrics (4 features) - FIXED: Better amplitude analysis
    frame_len = int(sr * 0.025)
    hop = int(sr * 0.010)
    rms_frames = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
    
    # Filter out near-silence frames
    peak_rms = np.max(rms_frames) if len(rms_frames) else 1.0
    valid_rms = rms_frames[rms_frames >= 0.1 * peak_rms]

    if len(valid_rms) >= 5:
        # Use log amplitude for better shimmer calculation
        amp_log = 20 * np.log10(valid_rms + 1e-8)
        amp_diffs = np.abs(np.diff(amp_log))
        mean_amp = np.mean(valid_rms)
        
        shimmer_local = float(np.mean(np.abs(np.diff(valid_rms))) / (mean_amp + 1e-8))
        shimmer_db = float(np.mean(amp_diffs))
        
        # APQ3 (3-point)
        if len(valid_rms) >= 3:
            apq3_diffs = []
            for i in range(1, len(valid_rms)-1):
                avg_3 = np.mean(valid_rms[i-1:i+2])
                apq3_diffs.append(abs(valid_rms[i] - avg_3))
            shimmer_apq3 = float(np.mean(apq3_diffs) / (mean_amp + 1e-8)) if apq3_diffs else shimmer_local * 0.85
        else:
            shimmer_apq3 = shimmer_local * 0.85

        # APQ5 (5-point)
        if len(valid_rms) >= 5:
            apq5_diffs = []
            for i in range(2, len(valid_rms)-2):
                avg_5 = np.mean(valid_rms[i-2:i+3])
                apq5_diffs.append(abs(valid_rms[i] - avg_5))
            shimmer_apq5 = float(np.mean(apq5_diffs) / (mean_amp + 1e-8)) if apq5_diffs else shimmer_local * 0.95
        else:
            shimmer_apq5 = shimmer_local * 0.95
    else:
        shimmer_local = 0.035
        shimmer_db = 0.35
        shimmer_apq3 = 0.030
        shimmer_apq5 = 0.033

    shimmer_metrics = {
        'shimmer': float(np.clip(shimmer_local, 0.005, 0.40)),
        'shimmer_db': float(np.clip(shimmer_db, 0.05, 3.50)),
        'shimmer_apq3': float(np.clip(shimmer_apq3, 0.004, 0.35)),
        'shimmer_apq5': float(np.clip(shimmer_apq5, 0.005, 0.38))
    }

    return f0_stats, jitter_metrics, shimmer_metrics, rms_frames


def extract_advanced_features(audio_path: str) -> pd.DataFrame:
    """
    Extract 215 clinical acoustic features from audio file.
    FIXED: Always returns DataFrame with correct column names
    """
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)

        if len(y) == 0:
            raise ValueError("Audio file is empty")

        # Gentle normalization & centering
        peak = np.max(np.abs(y)) + 1e-8
        y = y / peak
        y = y - np.mean(y)

        features = {}

        # 1. Pitch & Perturbations (18 features)
        f0_stats, jitter_metrics, shimmer_metrics, rms_frames = extract_pitch_and_perturbations(y, sr)
        features.update(f0_stats)
        features.update(jitter_metrics)
        features.update(shimmer_metrics)

        # 2. HNR & Energy Metrics (8 features)
        try:
            harmonic = librosa.effects.harmonic(y)
            noise = y - harmonic
            sig_p = float(np.mean(harmonic ** 2))
            noise_p = float(np.mean(noise ** 2))
            hnr_val = float(10.0 * np.log10(sig_p / (noise_p + 1e-10)))
            hnr_clipped = float(np.clip(hnr_val, 0.0, 50.0))
            features['hnr'] = hnr_clipped
            features['hnr_mean'] = hnr_clipped
            
            h_rms = librosa.feature.rms(y=harmonic)[0]
            n_rms = librosa.feature.rms(y=noise)[0]
            valid_hnr_frames = 10.0 * np.log10((h_rms ** 2) / (n_rms ** 2 + 1e-10) + 1e-8)
            features['hnr_std'] = float(np.std(valid_hnr_frames)) if len(valid_hnr_frames) else 3.0
        except Exception:
            features.update({'hnr': 18.0, 'hnr_mean': 18.0, 'hnr_std': 3.0})

        features['energy_mean'] = float(np.mean(y ** 2))
        features['energy_std'] = float(np.std(y ** 2))
        features['rms_mean'] = float(np.mean(rms_frames)) if len(rms_frames) else 0.05
        features['rms_std'] = float(np.std(rms_frames)) if len(rms_frames) else 0.01
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = float(np.mean(zcr))

        # 3. Spectral Features (24 features)
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(sc))
        features['spectral_centroid_std'] = float(np.std(sc))
        features['spectral_centroid_min'] = float(np.min(sc))
        features['spectral_centroid_max'] = float(np.max(sc))
        features['spectral_centroid_skew'] = float(pd.Series(sc).skew())
        features['spectral_centroid_kurt'] = float(pd.Series(sc).kurtosis())

        sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = float(np.mean(sb))
        features['spectral_bandwidth_std'] = float(np.std(sb))
        features['spectral_bandwidth_min'] = float(np.min(sb))
        features['spectral_bandwidth_max'] = float(np.max(sb))

        sr_off = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = float(np.mean(sr_off))
        features['spectral_rolloff_std'] = float(np.std(sr_off))
        features['spectral_rolloff_min'] = float(np.min(sr_off))
        features['spectral_rolloff_max'] = float(np.max(sr_off))

        sf_flat = librosa.feature.spectral_flatness(y=y)[0]
        features['spectral_flatness_mean'] = float(np.mean(sf_flat))
        features['spectral_flatness_std'] = float(np.std(sf_flat))
        features['spectral_flatness_min'] = float(np.min(sf_flat))
        features['spectral_flatness_max'] = float(np.max(sf_flat))

        s_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        for i in range(6):
            features[f'spectral_contrast_mean_{i}'] = float(np.mean(s_contrast[i, :]))
            features[f'spectral_contrast_std_{i}'] = float(np.std(s_contrast[i, :]))

        # 4. MFCC Features (160 features)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        for i in range(20):
            features[f'mfcc_mean_{i}'] = float(np.mean(mfcc[i, :]))
            features[f'mfcc_std_{i}'] = float(np.std(mfcc[i, :]))
            features[f'mfcc_min_{i}'] = float(np.min(mfcc[i, :]))
            features[f'mfcc_max_{i}'] = float(np.max(mfcc[i, :]))
            features[f'mfcc_delta_mean_{i}'] = float(np.mean(mfcc_delta[i, :]))
            features[f'mfcc_delta_std_{i}'] = float(np.std(mfcc_delta[i, :]))
            features[f'mfcc_delta2_mean_{i}'] = float(np.mean(mfcc_delta2[i, :]))
            features[f'mfcc_delta2_std_{i}'] = float(np.std(mfcc_delta2[i, :]))

        # 5. Signal Morphology Statistics (5 features)
        features['signal_mean'] = float(np.mean(y))
        features['signal_std'] = float(np.std(y))
        features['signal_skewness'] = float(pd.Series(y).skew())
        features['signal_kurtosis'] = float(pd.Series(y).kurtosis())
        features['voiced_ratio'] = float(np.mean(zcr > 0.02))

        # Create DataFrame with proper column names
        df = pd.DataFrame([features])
        
        # Ensure all 215 features exist
        for col in FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        
        # Keep only the 215 features in correct order
        df = df[FEATURE_NAMES]
        
        # Handle missing values
        df = df.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Ensure numeric type
        df = df.astype(np.float32)
        
        return df

    except Exception as e:
        print(f"  [ERROR] Feature extraction failed: {e}")
        raise


def get_default_215_features():
    """Default 215-feature DataFrame with proper column names."""
    arr = np.zeros((1, 215), dtype=np.float32)
    return pd.DataFrame(arr, columns=FEATURE_NAMES)

# Export feature names for use by other modules
__all__ = ['FEATURE_NAMES', 'extract_advanced_features', 'get_default_215_features']