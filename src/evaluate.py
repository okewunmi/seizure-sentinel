"""
Comprehensive Evaluation Framework
Clinical metrics for seizure detection systems
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
from typing import Dict, List, Tuple
import json
from pathlib import Path


class SeizureDetectionEvaluator:
    """
    Evaluate seizure detection model with clinical metrics:
    - Sensitivity (True Positive Rate)
    - Specificity (True Negative Rate) 
    - False Alarm Rate per hour
    - Seizure detection latency
    - Precision-Recall curves
    """
    
    def __init__(self, class_names: List[str] = None):
        if class_names is None:
            self.class_names = ['Interictal', 'Pre-ictal', 'Ictal']
        else:
            self.class_names = class_names
    
    def evaluate_model(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> Dict:
        """
        Run complete evaluation
        """
        model.eval()
        model.to(device)
        
        all_labels = []
        all_preds = []
        all_probs = []
        all_metadata = []
        attention_weights_list = []
        
        with torch.no_grad():
            for x, y, metadata in dataloader:
                x = x.to(device)
                
                logits, attention_weights = model(x)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_labels.extend(y.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_metadata.extend(metadata)
                attention_weights_list.extend(attention_weights.cpu().numpy())
        
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        
        # Compute all metrics
        metrics = {
            'basic_metrics': self._compute_basic_metrics(all_labels, all_preds),
            'clinical_metrics': self._compute_clinical_metrics(
                all_labels, all_preds, all_metadata
            ),
            'per_class_metrics': self._compute_per_class_metrics(
                all_labels, all_preds, all_probs
            ),
            'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
        }
        
        return metrics, all_labels, all_preds, all_probs, attention_weights_list
    
    def _compute_basic_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """Standard classification metrics"""
        
        # Overall accuracy
        accuracy = np.mean(y_true == y_pred)
        
        # Binary classification: seizure-related (1,2) vs normal (0)
        y_true_binary = (y_true > 0).astype(int)
        y_pred_binary = (y_pred > 0).astype(int)
        
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        return {
            'accuracy': float(accuracy),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'f1_score': float(f1_score),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
    
    def _compute_clinical_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metadata: List[Dict]
    ) -> Dict:
        """
        Clinical metrics specific to seizure detection:
        - False alarm rate (per hour)
        - Seizure detection rate
        - Average detection latency
        """
        # Group predictions by patient and file
        patient_files = {}
        for i, meta in enumerate(metadata):
            key = f"{meta['patient_id']}_{meta['file']}"
            if key not in patient_files:
                patient_files[key] = {
                    'labels': [],
                    'preds': [],
                    'times': []
                }
            
            patient_files[key]['labels'].append(y_true[i])
            patient_files[key]['preds'].append(y_pred[i])
            patient_files[key]['times'].append(meta.get('start_time', 0))
        
        # Calculate metrics per file
        total_hours = 0
        false_alarms = 0
        seizures_detected = 0
        total_seizures = 0
        detection_latencies = []
        
        for key, data in patient_files.items():
            labels = np.array(data['labels'])
            preds = np.array(data['preds'])
            times = np.array(data['times'])
            
            # Calculate duration
            if len(times) > 0:
                duration_hours = (times[-1] - times[0]) / 3600
                total_hours += duration_hours
            
            # Count false alarms (predicted seizure when none occurred)
            false_alarm_mask = (labels == 0) & (preds > 0)
            false_alarms += np.sum(false_alarm_mask)
            
            # Count seizures (ground truth)
            # A seizure is a continuous sequence of ictal labels
            seizure_segments = self._find_seizure_segments(labels)
            total_seizures += len(seizure_segments)
            
            # Check if each seizure was detected
            for seg_start, seg_end in seizure_segments:
                # Check if any prediction in this segment was positive
                if np.any(preds[seg_start:seg_end+1] > 0):
                    seizures_detected += 1
                    
                    # Calculate detection latency
                    first_detection = np.where(preds[seg_start:seg_end+1] > 0)[0][0]
                    latency = times[seg_start + first_detection] - times[seg_start]
                    detection_latencies.append(latency)
        
        false_alarm_rate = false_alarms / total_hours if total_hours > 0 else 0
        seizure_detection_rate = seizures_detected / total_seizures if total_seizures > 0 else 0
        avg_latency = np.mean(detection_latencies) if detection_latencies else 0
        
        return {
            'false_alarm_rate_per_hour': float(false_alarm_rate),
            'seizure_detection_rate': float(seizure_detection_rate),
            'average_detection_latency_seconds': float(avg_latency),
            'total_seizures': int(total_seizures),
            'seizures_detected': int(seizures_detected),
            'total_false_alarms': int(false_alarms),
            'total_recording_hours': float(total_hours)
        }
    
    def _find_seizure_segments(self, labels: np.ndarray) -> List[Tuple[int, int]]:
        """Find continuous seizure segments (ictal periods)"""
        segments = []
        in_seizure = False
        start_idx = 0
        
        for i, label in enumerate(labels):
            if label == 2 and not in_seizure:  # Start of seizure
                in_seizure = True
                start_idx = i
            elif label != 2 and in_seizure:  # End of seizure
                in_seizure = False
                segments.append((start_idx, i-1))
        
        # Handle case where seizure extends to end
        if in_seizure:
            segments.append((start_idx, len(labels)-1))
        
        return segments
    
    def _compute_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray
    ) -> Dict:
        """Compute metrics for each class"""
        
        per_class = {}
        
        for i, class_name in enumerate(self.class_names):
            # Binary classification for this class
            true_binary = (y_true == i).astype(int)
            pred_binary = (y_pred == i).astype(int)
            
            tp = np.sum((true_binary == 1) & (pred_binary == 1))
            tn = np.sum((true_binary == 0) & (pred_binary == 0))
            fp = np.sum((true_binary == 0) & (pred_binary == 1))
            fn = np.sum((true_binary == 1) & (pred_binary == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # ROC curve
            fpr, tpr, _ = roc_curve(true_binary, y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            
            per_class[class_name] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'roc_auc': float(roc_auc),
                'support': int(np.sum(true_binary))
            }
        
        return per_class
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str = None
    ):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        save_path: str = None
    ):
        """Plot ROC curves for each class"""
        plt.figure(figsize=(10, 8))
        
        for i, class_name in enumerate(self.class_names):
            true_binary = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(true_binary, y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves', fontsize=16, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curves(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        save_path: str = None
    ):
        """Plot Precision-Recall curves"""
        plt.figure(figsize=(10, 8))
        
        for i, class_name in enumerate(self.class_names):
            true_binary = (y_true == i).astype(int)
            precision, recall, _ = precision_recall_curve(true_binary, y_probs[:, i])
            
            plt.plot(recall, precision, label=class_name, linewidth=2)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves', fontsize=16, fontweight='bold')
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(
        self,
        metrics: Dict,
        save_path: str = None
    ) -> str:
        """Generate comprehensive evaluation report"""
        
        report = []
        report.append("=" * 70)
        report.append("SEIZURE DETECTION MODEL EVALUATION REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Basic metrics
        report.append("BASIC METRICS")
        report.append("-" * 70)
        basic = metrics['basic_metrics']
        report.append(f"Accuracy:     {basic['accuracy']:.4f}")
        report.append(f"Sensitivity:  {basic['sensitivity']:.4f} (True Positive Rate)")
        report.append(f"Specificity:  {basic['specificity']:.4f} (True Negative Rate)")
        report.append(f"Precision:    {basic['precision']:.4f}")
        report.append(f"F1-Score:     {basic['f1_score']:.4f}")
        report.append("")
        
        # Clinical metrics
        report.append("CLINICAL METRICS")
        report.append("-" * 70)
        clinical = metrics['clinical_metrics']
        report.append(f"False Alarm Rate:        {clinical['false_alarm_rate_per_hour']:.2f} per hour")
        report.append(f"Seizure Detection Rate:  {clinical['seizure_detection_rate']:.2%}")
        report.append(f"Avg Detection Latency:   {clinical['average_detection_latency_seconds']:.2f} seconds")
        report.append(f"Total Seizures:          {clinical['total_seizures']}")
        report.append(f"Seizures Detected:       {clinical['seizures_detected']}")
        report.append(f"Total Recording Time:    {clinical['total_recording_hours']:.1f} hours")
        report.append("")
        
        # Per-class metrics
        report.append("PER-CLASS METRICS")
        report.append("-" * 70)
        for class_name, class_metrics in metrics['per_class_metrics'].items():
            report.append(f"\n{class_name}:")
            report.append(f"  Precision: {class_metrics['precision']:.4f}")
            report.append(f"  Recall:    {class_metrics['recall']:.4f}")
            report.append(f"  F1-Score:  {class_metrics['f1_score']:.4f}")
            report.append(f"  ROC-AUC:   {class_metrics['roc_auc']:.4f}")
            report.append(f"  Support:   {class_metrics['support']}")
        
        report.append("")
        report.append("=" * 70)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text


# Example usage
if __name__ == '__main__':
    from model import SeizureDetectionLSTM
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create dummy data
    n_samples = 200
    X = torch.randn(n_samples, 23, 1280)
    y = torch.randint(0, 3, (n_samples,))
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32)
    
    # Create model
    model = SeizureDetectionLSTM()
    
    # Evaluate
    evaluator = SeizureDetectionEvaluator()
    
    # Mock metadata
    metadata = [
        {'patient_id': 'chb01', 'file': 'test.edf', 'start_time': i * 5}
        for i in range(n_samples)
    ]
    
    # Since we need to modify dataloader to include metadata
    # This is simplified for demonstration
    print("Evaluation framework ready!")
    print("Use with actual model and data from training pipeline.")