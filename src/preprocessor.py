"""
EEG Signal Preprocessing Pipeline
- Bandpass filtering
- Artifact removal
- Normalization
- Spectrogram generation
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, welch
import pywt
from typing import Tuple, Optional


class EEGPreprocessor:
    """
    Preprocessing pipeline for EEG signals
    Optimized for seizure detection
    """
    
    def __init__(
        self,
        sampling_rate: int = 256,
        lowcut: float = 0.5,
        highcut: float = 50.0,
        notch_freq: float = 60.0  # Remove power line noise
    ):
        self.sampling_rate = sampling_rate
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq
        
    def bandpass_filter(self, data: np.ndarray, order: int = 4) -> np.ndarray:
        """
        Apply Butterworth bandpass filter
        
        Args:
            data: (n_channels, n_samples) array
            order: Filter order
            
        Returns:
            Filtered signal
        """
        nyquist = 0.5 * self.sampling_rate
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        
        b, a = butter(order, [low, high], btype='band')
        
        # Apply filter to each channel
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered[i] = filtfilt(b, a, data[i])
        
        return filtered
    
    def notch_filter(self, data: np.ndarray, quality_factor: float = 30.0) -> np.ndarray:
        """
        Remove power line noise (50/60 Hz)
        
        Args:
            data: (n_channels, n_samples) array
            quality_factor: Q factor for notch filter
        """
        nyquist = 0.5 * self.sampling_rate
        freq = self.notch_freq / nyquist
        
        b, a = signal.iirnotch(freq, quality_factor)
        
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered[i] = filtfilt(b, a, data[i])
        
        return filtered
    
    def remove_artifacts(self, data: np.ndarray, threshold: float = 4.0) -> np.ndarray:
        """
        Remove high-amplitude artifacts using z-score thresholding
        
        Args:
            data: (n_channels, n_samples) array
            threshold: Z-score threshold for artifact detection
        """
        cleaned = data.copy()
        
        for i in range(data.shape[0]):
            channel = data[i]
            mean = np.mean(channel)
            std = np.std(channel)
            
            # Find artifacts
            z_scores = np.abs((channel - mean) / std)
            artifacts = z_scores > threshold
            
            # Interpolate artifacts
            if np.any(artifacts):
                artifact_indices = np.where(artifacts)[0]
                valid_indices = np.where(~artifacts)[0]
                
                if len(valid_indices) > 0:
                    cleaned[i, artifact_indices] = np.interp(
                        artifact_indices,
                        valid_indices,
                        channel[valid_indices]
                    )
        
        return cleaned
    
    def normalize(self, data: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """
        Normalize EEG signals
        
        Args:
            data: (n_channels, n_samples) array
            method: 'zscore', 'minmax', or 'robust'
        """
        normalized = np.zeros_like(data)
        
        for i in range(data.shape[0]):
            channel = data[i]
            
            if method == 'zscore':
                normalized[i] = (channel - np.mean(channel)) / (np.std(channel) + 1e-8)
            
            elif method == 'minmax':
                min_val = np.min(channel)
                max_val = np.max(channel)
                normalized[i] = (channel - min_val) / (max_val - min_val + 1e-8)
            
            elif method == 'robust':
                median = np.median(channel)
                mad = np.median(np.abs(channel - median))
                normalized[i] = (channel - median) / (mad + 1e-8)
        
        return normalized
    
    def extract_spectrogram(
        self,
        data: np.ndarray,
        nperseg: int = 128,
        noverlap: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Short-Time Fourier Transform (STFT) spectrogram
        
        Args:
            data: (n_channels, n_samples) array
            nperseg: Length of each segment
            noverlap: Number of points to overlap
            
        Returns:
            f: Frequency array
            t: Time array
            Sxx: Spectrogram (n_channels, n_freq, n_time)
        """
        if noverlap is None:
            noverlap = nperseg // 2
        
        spectrograms = []
        
        for i in range(data.shape[0]):
            f, t, Sxx = signal.spectrogram(
                data[i],
                fs=self.sampling_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                scaling='density'
            )
            spectrograms.append(Sxx)
        
        return f, t, np.array(spectrograms)
    
    def extract_wavelet_features(
        self,
        data: np.ndarray,
        wavelet: str = 'db4',
        level: int = 5
    ) -> np.ndarray:
        """
        Extract wavelet decomposition features
        Useful for capturing transient seizure patterns
        
        Args:
            data: (n_channels, n_samples) array
            wavelet: Wavelet type ('db4', 'sym5', etc.)
            level: Decomposition level
            
        Returns:
            Wavelet coefficients (n_channels, n_features)
        """
        features = []
        
        for i in range(data.shape[0]):
            coeffs = pywt.wavedec(data[i], wavelet, level=level)
            
            # Extract statistical features from each level
            channel_features = []
            for coeff in coeffs:
                channel_features.extend([
                    np.mean(coeff),
                    np.std(coeff),
                    np.max(np.abs(coeff)),
                    np.sum(coeff**2)  # Energy
                ])
            
            features.append(channel_features)
        
        return np.array(features)
    
    def extract_power_spectral_density(
        self,
        data: np.ndarray,
        nperseg: int = 256
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Power Spectral Density (PSD)
        Extract band power features (delta, theta, alpha, beta, gamma)
        
        Args:
            data: (n_channels, n_samples) array
            
        Returns:
            freqs: Frequency array
            psd: PSD array (n_channels, n_freqs)
        """
        # Frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        psd_features = []
        
        for i in range(data.shape[0]):
            freqs, psd = welch(
                data[i],
                fs=self.sampling_rate,
                nperseg=nperseg,
                scaling='density'
            )
            
            # Calculate band powers
            band_powers = []
            for band_name, (low, high) in bands.items():
                idx = np.logical_and(freqs >= low, freqs <= high)
                band_power = np.trapz(psd[idx], freqs[idx])
                band_powers.append(band_power)
            
            psd_features.append(band_powers)
        
        return freqs, np.array(psd_features)
    
    def preprocess_pipeline(
        self,
        data: np.ndarray,
        apply_notch: bool = True,
        apply_artifact_removal: bool = True
    ) -> np.ndarray:
        """
        Complete preprocessing pipeline
        
        Args:
            data: (n_channels, n_samples) raw EEG
            
        Returns:
            Preprocessed EEG
        """
        # 1. Bandpass filter
        processed = self.bandpass_filter(data)
        
        # 2. Notch filter (remove power line noise)
        if apply_notch:
            processed = self.notch_filter(processed)
        
        # 3. Artifact removal
        if apply_artifact_removal:
            processed = self.remove_artifacts(processed)
        
        # 4. Normalize
        processed = self.normalize(processed, method='zscore')
        
        return processed


class FeatureExtractor:
    """
    Extract comprehensive features for seizure detection
    """
    
    def __init__(self, preprocessor: EEGPreprocessor):
        self.preprocessor = preprocessor
    
    def extract_all_features(self, data: np.ndarray) -> dict:
        """
        Extract all feature types
        
        Returns dict with:
        - raw: Preprocessed signal
        - spectrogram: Time-frequency representation
        - wavelet: Wavelet coefficients
        - psd: Power spectral density features
        """
        # Preprocess
        processed = self.preprocessor.preprocess_pipeline(data)
        
        # Extract features
        f_spec, t_spec, spectrogram = self.preprocessor.extract_spectrogram(processed)
        wavelet_features = self.preprocessor.extract_wavelet_features(processed)
        f_psd, psd_features = self.preprocessor.extract_power_spectral_density(processed)
        
        return {
            'raw': processed,
            'spectrogram': spectrogram,
            'spectrogram_freqs': f_spec,
            'spectrogram_times': t_spec,
            'wavelet': wavelet_features,
            'psd': psd_features,
            'psd_freqs': f_psd
        }


# Example usage
if __name__ == '__main__':
    # Simulate EEG data (23 channels, 5 seconds @ 256 Hz)
    np.random.seed(42)
    data = np.random.randn(23, 5 * 256) * 50  # microvolts
    
    # Initialize preprocessor
    preprocessor = EEGPreprocessor(sampling_rate=256)
    
    # Full pipeline
    processed = preprocessor.preprocess_pipeline(data)
    print(f"Input shape: {data.shape}")
    print(f"Output shape: {processed.shape}")
    print(f"Input range: [{data.min():.2f}, {data.max():.2f}]")
    print(f"Output range: [{processed.min():.2f}, {processed.max():.2f}]")
    
    # Extract features
    extractor = FeatureExtractor(preprocessor)
    features = extractor.extract_all_features(data)
    
    print("\nExtracted features:")
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape}")