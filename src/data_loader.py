"""CHB-MIT Scalp EEG Database Loader"""

import os
import numpy as np
import pandas as pd
import pyedflib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re


@dataclass
class SeizureEvent:
    patient_id: str
    file_name: str
    start_time: float
    end_time: float
    duration: float


class CHBMITLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.sampling_rate = 256
        self.num_channels = 23
        
    def get_patient_ids(self) -> List[str]:
        patients = []
        if not self.data_dir.exists():
            return patients
        for path in self.data_dir.iterdir():
            if path.is_dir() and path.name.startswith('chb'):
                patients.append(path.name)
        return sorted(patients)
    
    def load_seizure_annotations(self, patient_id: str) -> List[SeizureEvent]:
        summary_file = self.data_dir / patient_id / f"{patient_id}-summary.txt"
        if not summary_file.exists():
            return []
        
        seizures = []
        current_file = None
        
        with open(summary_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('File Name:'):
                    current_file = line.split(':')[1].strip()
                elif line.startswith('Seizure Start Time:'):
                    start_time = float(re.findall(r'\d+', line)[0])
                elif line.startswith('Seizure End Time:') and current_file:
                    end_time = float(re.findall(r'\d+', line)[0])
                    seizures.append(SeizureEvent(
                        patient_id=patient_id,
                        file_name=current_file,
                        start_time=start_time,
                        end_time=end_time,
                        duration=end_time - start_time
                    ))
        return seizures
    
    def load_edf_file(self, file_path: Path) -> Tuple[np.ndarray, Dict]:
        with pyedflib.EdfReader(str(file_path)) as f:
            n_channels = f.signals_in_file
            signals = np.zeros((n_channels, f.getNSamples()[0]))
            for i in range(n_channels):
                signals[i, :] = f.readSignal(i)
            
            metadata = {
                'sampling_rate': f.getSampleFrequency(0),
                'duration': f.getFileDuration(),
                'n_channels': n_channels,
                'channel_labels': f.getSignalLabels(),
            }
        return signals, metadata
    
    def create_windowed_dataset(
        self,
        patient_id: str,
        window_size: float = 5.0,
        overlap: float = 4.0,
        pre_ictal_window: float = 5.0,
        max_windows: int = 3000
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Create sliding windows (memory-efficient)"""
        seizures = self.load_seizure_annotations(patient_id)
        patient_dir = self.data_dir / patient_id
        
        X_list, y_list, meta_list = [], [], []
        
        for edf_file in sorted(patient_dir.glob('*.edf')):
            if len(X_list) >= max_windows:
                break
                
            try:
                signals, _ = self.load_edf_file(edf_file)
            except:
                continue
            
            file_seizures = [s for s in seizures if s.file_name == edf_file.name]
            
            window_samples = int(window_size * self.sampling_rate)
            overlap_samples = int(overlap * self.sampling_rate)
            stride = window_samples - overlap_samples
            
            for i in range(0, signals.shape[1] - window_samples, stride):
                if len(X_list) >= max_windows:
                    break
                
                window = signals[:, i:i+window_samples].astype(np.float32)
                
                start_time = i / self.sampling_rate
                end_time = (i + window_samples) / self.sampling_rate
                
                label = self._get_window_label(start_time, end_time, file_seizures, pre_ictal_window)
                
                X_list.append(window)
                y_list.append(label)
                meta_list.append({
                    'patient_id': patient_id,
                    'file': edf_file.name,
                    'start_time': start_time,
                    'label': label
                })
        
        if not X_list:
            return np.array([], dtype=np.float32).reshape(0, 23, 1280), np.array([]), []
        
        return np.array(X_list, dtype=np.float32), np.array(y_list), meta_list
    
    def _get_window_label(self, start_time, end_time, seizures, pre_ictal_window):
        for s in seizures:
            if start_time < s.end_time and end_time > s.start_time:
                return 2
            if s.start_time - pre_ictal_window <= start_time < s.start_time:
                return 1
        return 0
    
    def get_dataset_statistics(self) -> pd.DataFrame:
        stats = []
        for patient_id in self.get_patient_ids():
            seizures = self.load_seizure_annotations(patient_id)
            patient_dir = self.data_dir / patient_id
            n_files = len(list(patient_dir.glob('*.edf')))
            
            total_duration = 0
            for edf_file in patient_dir.glob('*.edf'):
                try:
                    _, meta = self.load_edf_file(edf_file)
                    total_duration += meta['duration']
                except:
                    pass
            
            stats.append({
                'patient_id': patient_id,
                'n_files': n_files,
                'n_seizures': len(seizures),
                'total_duration_hours': total_duration / 3600,
            })
        
        return pd.DataFrame(stats)