import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy import signal


def extract_voice_features(audio_path):
    """
    Extract comprehensive voice features for Parkinson's detection.
    Uses robust methods that work with various audio files.
    """
    try:
        # Load audio file
        print(f"Loading audio from: {audio_path}")
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        print(f"Audio loaded: {len(y)} samples, sr={sr}")
        
        if len(y) == 0:
            print("⚠ Audio file is empty!")
            return get_default_voice_features()
        
        # Normalize audio
        y = y / (np.max(np.abs(y)) + 1e-8)
        
        features = {}
        
        # 1. ENERGY & AMPLITUDE FEATURES
        rms_energy = np.sqrt(np.mean(y ** 2))
        features['energy'] = float(rms_energy)
        print(f"✓ Energy: {rms_energy:.6f}")
        
        # 2. FUNDAMENTAL FREQUENCY - use multiple methods
        try:
            # Method 1: Zero crossing rate (simple, robust)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['zero_crossing_rate'] = float(np.mean(zcr))
            
            # Method 2: Spectral centroid
            S = np.abs(librosa.stft(y))
            spectral_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
            features['spectral_centroid'] = float(np.mean(spectral_centroid))
            
            # Method 3: Auto-correlation for pitch
            autocorr = np.correlate(y, y, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            f0_candidates = []
            
            # Look for peaks in autocorrelation
            min_period = int(sr / 300)  # Max 300 Hz
            max_period = int(sr / 50)   # Min 50 Hz
            
            if min_period < max_period and max_period < len(autocorr):
                autocorr_subset = autocorr[min_period:max_period]
                if len(autocorr_subset) > 0:
                    peaks, _ = signal.find_peaks(autocorr_subset, height=np.max(autocorr_subset) * 0.3)
                    if len(peaks) > 0:
                        best_peak = peaks[0]
                        period = min_period + best_peak
                        f0 = sr / period if period > 0 else 150.0
                        f0_candidates.append(f0)
            
            features['f0_mean'] = float(f0_candidates[0]) if f0_candidates else 150.0
            features['f0_std'] = float(np.std(f0_candidates)) if len(f0_candidates) > 1 else 20.0
            print(f"✓ F0 Mean: {features['f0_mean']:.1f}, F0 Std: {features['f0_std']:.1f}")
            
        except Exception as e:
            print(f"⚠ F0 extraction error: {e}")
            features['f0_mean'] = 150.0
            features['f0_std'] = 20.0
        
        # 3. VARIABILITY FEATURES (Jitter & Shimmer)
        # Use more robust methods for shimmer and jitter
        
        # Simple jitter: measure pitch stability in frequency domain
        # Use spectral peak tracking for rougher jitter estimate
        try:
            S = np.abs(librosa.stft(y))
            S_db = librosa.power_to_db(S, ref=np.max)
            
            # Track spectral centroid variation as jitter-like metric
            spectral_centroids = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
            centroid_std = np.std(spectral_centroids)
            centroid_mean = np.mean(spectral_centroids) 
            
            # Normalize to 0-0.1 range (realistic jitter range)
            features['jitter'] = float(min(0.1, (centroid_std / (centroid_mean + 1e-8)) * 0.01))
        except Exception as e:
            features['jitter'] = 0.01  # Default to slightly elevated jitter
        
        # Shimmer: amplitude variation in frames - use RMS energy variation
        # This is more conservative than coefficient of variation
        window_size = int(sr * 0.01)  # 10ms windows (smaller windows for shimmer)
        if window_size > 0 and len(y) > window_size * 3:
            # Split into frames
            n_frames = (len(y) - window_size) // (window_size // 2) + 1
            frame_amplitudes = []
            
            for i in range(n_frames - 1):
                frame_start = i * (window_size // 2)
                frame_end = frame_start + window_size
                if frame_end <= len(y):
                    frame = y[frame_start:frame_end]
                    # Use RMS as amplitude
                    rms_amplitude = np.sqrt(np.mean(frame ** 2))
                    frame_amplitudes.append(rms_amplitude)
            
            if len(frame_amplitudes) > 1:
                # Shimmer is std of amplitudes normalized by mean
                # But cap it to realistic range (0-0.2 for healthy voices)
                amp_std = np.std(frame_amplitudes)
                amp_mean = np.mean(frame_amplitudes)
                raw_shimmer = amp_std / (amp_mean + 1e-8)
                # Compress to more realistic range
                features['shimmer'] = float(min(0.25, raw_shimmer * 0.2))
                print(f"✓ Jitter: {features['jitter']:.6f}, Shimmer: {features['shimmer']:.6f}")
            else:
                features['jitter'] = 0.01
                features['shimmer'] = 0.05
        else:
            features['jitter'] = 0.01
            features['shimmer'] = 0.05
        
        # 4. HNR - Harmonic to Noise Ratio
        try:
            harmonic = librosa.effects.harmonic(y)
            noise = y - harmonic
            signal_power = np.mean(harmonic ** 2)
            noise_power = np.mean(noise ** 2)
            hnr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            features['hnr'] = float(np.clip(hnr, 0, 50))
            print(f"✓ HNR: {features['hnr']:.2f}")
        except Exception as e:
            print(f"⚠ HNR error: {e}")
            features['hnr'] = 20.0
        
        # 5. MFCCs - Mel-Frequency Cepstral Coefficients
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            for i in range(20):
                features[f'mfcc_mean_{i}'] = float(np.mean(mfcc[i, :]))
                features[f'mfcc_std_{i}'] = float(np.std(mfcc[i, :]))
            print(f"✓ MFCCs computed (20 coefficients)")
        except Exception as e:
            print(f"⚠ MFCC error: {e}")
            # Use dummy MFCCs
            for i in range(20):
                features[f'mfcc_mean_{i}'] = 0.0
                features[f'mfcc_std_{i}'] = 0.1 * (i + 1)
        
        # Ensure correct order
        expected_order = [
            'f0_mean', 'f0_std', 'jitter', 'shimmer', 'hnr', 'spectral_centroid', 'zero_crossing_rate',
            'mfcc_mean_0', 'mfcc_std_0', 'mfcc_mean_1', 'mfcc_std_1', 'mfcc_mean_2', 'mfcc_std_2',
            'mfcc_mean_3', 'mfcc_std_3', 'mfcc_mean_4', 'mfcc_std_4', 'mfcc_mean_5', 'mfcc_std_5',
            'mfcc_mean_6', 'mfcc_std_6', 'mfcc_mean_7', 'mfcc_std_7', 'mfcc_mean_8', 'mfcc_std_8',
            'mfcc_mean_9', 'mfcc_std_9', 'mfcc_mean_10', 'mfcc_std_10', 'mfcc_mean_11', 'mfcc_std_11',
            'mfcc_mean_12', 'mfcc_std_12', 'mfcc_mean_13', 'mfcc_std_13', 'mfcc_mean_14', 'mfcc_std_14',
            'mfcc_mean_15', 'mfcc_std_15', 'mfcc_mean_16', 'mfcc_std_16', 'mfcc_mean_17', 'mfcc_std_17',
            'mfcc_mean_18', 'mfcc_std_18', 'mfcc_mean_19', 'mfcc_std_19'
        ]
        
        ordered_features = {key: features.get(key, 0.0) for key in expected_order}
        result = pd.DataFrame([ordered_features])
        print(f"✓ Features extracted successfully: shape {result.shape}")
        return result
    
    except Exception as e:
        print(f"✗ Error extracting voice features: {e}")
        import traceback
        traceback.print_exc()
        return get_default_voice_features()


def extract_gait_features(gait_path):
    """Extract features from gait .txt or .csv file"""
    try:
        # Try to read as CSV first
        if gait_path.endswith('.csv'):
            df = pd.read_csv(gait_path)
        else:
            # Try tab-separated first, then comma-separated
            try:
                df = pd.read_csv(gait_path, sep='\t', header=None, engine='python')
            except:
                df = pd.read_csv(gait_path, header=None, engine='python')
        
        if df.shape[1] > 1:
            data = df.iloc[:, 1:].values
        else:
            data = df.values
    
        features = {}
        for col in range(min(data.shape[1], 8)):   # limit to first 8 sensors
            col_data = data[:, col]
            features[f'sensor_{col}_mean'] = float(np.mean(col_data))
            features[f'sensor_{col}_std'] = float(np.std(col_data))
            features[f'sensor_{col}_max'] = float(np.max(col_data))
        
        return pd.DataFrame([features])
    except Exception as e:
        # Fallback: create dummy features if file can't be read
        print(f"Warning: Could not process gait file {gait_path}: {e}")
        features = {}
        for col in range(8):
            features[f'sensor_{col}_mean'] = 0.0
            features[f'sensor_{col}_std'] = 0.0
            features[f'sensor_{col}_max'] = 0.0
        return pd.DataFrame([features])


def get_default_voice_features():
    """Return default voice features when extraction fails"""
    expected_order = [
        'f0_mean', 'f0_std', 'jitter', 'shimmer', 'hnr', 'spectral_centroid', 'zero_crossing_rate',
        'mfcc_mean_0', 'mfcc_std_0', 'mfcc_mean_1', 'mfcc_std_1', 'mfcc_mean_2', 'mfcc_std_2',
        'mfcc_mean_3', 'mfcc_std_3', 'mfcc_mean_4', 'mfcc_std_4', 'mfcc_mean_5', 'mfcc_std_5',
        'mfcc_mean_6', 'mfcc_std_6', 'mfcc_mean_7', 'mfcc_std_7', 'mfcc_mean_8', 'mfcc_std_8',
        'mfcc_mean_9', 'mfcc_std_9', 'mfcc_mean_10', 'mfcc_std_10', 'mfcc_mean_11', 'mfcc_std_11',
        'mfcc_mean_12', 'mfcc_std_12', 'mfcc_mean_13', 'mfcc_std_13', 'mfcc_mean_14', 'mfcc_std_14',
        'mfcc_mean_15', 'mfcc_std_15', 'mfcc_mean_16', 'mfcc_std_16', 'mfcc_mean_17', 'mfcc_std_17',
        'mfcc_mean_18', 'mfcc_std_18', 'mfcc_mean_19', 'mfcc_std_19'
    ]
    
    # Create dictionary with default values
    default_features = {}
    default_features['f0_mean'] = 150.0
    default_features['f0_std'] = 20.0
    default_features['jitter'] = 0.001
    default_features['shimmer'] = 0.02
    default_features['hnr'] = 20.0
    default_features['spectral_centroid'] = 2000.0
    default_features['zero_crossing_rate'] = 0.08
    
    # Add default MFCC values
    for i in range(20):
        default_features[f'mfcc_mean_{i}'] = 0.0
        default_features[f'mfcc_std_{i}'] = 0.1 * (i + 1)
    
    return pd.DataFrame([default_features])