"""
Training Pipeline for Seizure Detection
Handles class imbalance, patient-wise cross-validation, and evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from model import SeizureDetectionLSTM, FocalLoss
from preprocessor import EEGPreprocessor


class EEGDataset(Dataset):
    """PyTorch Dataset for EEG windows"""
    
    def __init__(
        self,
        X: np.ndarray,  # (n_samples, n_channels, n_timepoints)
        y: np.ndarray,  # (n_samples,)
        metadata: List[Dict],
        transform: Optional[callable] = None
    ):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.metadata = metadata
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        x = self.X[idx]
        y = self.y[idx]
        meta = self.metadata[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y, meta


class SeizureDetectionTrainer:
    """
    Complete training pipeline with:
    - Class imbalance handling
    - Patient-wise cross-validation
    - Early stopping
    - Model checkpointing
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        use_focal_loss: bool = True
    ):
        self.model = model.to(device)
        self.device = device
        
        # Loss function
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def compute_class_weights(self, y: np.ndarray) -> torch.Tensor:
        """
        Compute class weights for imbalanced dataset
        Seizures are rare events
        """
        class_counts = np.bincount(y)
        total_samples = len(y)
        
        # Inverse frequency weighting
        weights = total_samples / (len(class_counts) * class_counts)
        
        return torch.FloatTensor(weights).to(self.device)
    
    def create_weighted_sampler(self, y: np.ndarray) -> WeightedRandomSampler:
        """
        Create weighted sampler for balanced batches
        """
        class_weights = self.compute_class_weights(y).cpu().numpy()
        sample_weights = class_weights[y]
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        return sampler
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc='Training')
        for batch_idx, (x, y, _) in enumerate(pbar):
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(x)
            loss = self.criterion(logits, y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(dataloader)
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Tuple[float, Dict]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for x, y, _ in tqdm(dataloader, desc='Validating'):
            x = x.to(self.device)
            y = y.to(self.device)
            
            logits, _ = self.model(x)
            loss = self.criterion(logits, y)
            
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        
        # Compute metrics
        metrics = self.compute_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs)
        )
        
        return avg_loss, metrics
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_probs: np.ndarray
    ) -> Dict:
        """Compute comprehensive metrics"""
        
        # Classification report
        report = classification_report(
            y_true,
            y_pred,
            target_names=['Interictal', 'Pre-ictal', 'Ictal'],
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Sensitivity and specificity for seizure detection
        # Treat pre-ictal and ictal as "seizure-related"
        y_true_binary = (y_true > 0).astype(int)
        y_pred_binary = (y_pred > 0).astype(int)
        
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # ROC-AUC (one-vs-rest)
        try:
            auc_scores = {}
            for i, class_name in enumerate(['Interictal', 'Pre-ictal', 'Ictal']):
                y_true_class = (y_true == i).astype(int)
                auc = roc_auc_score(y_true_class, y_probs[:, i])
                auc_scores[class_name] = auc
        except:
            auc_scores = {'Interictal': 0, 'Pre-ictal': 0, 'Ictal': 0}
        
        return {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'sensitivity': sensitivity,
            'specificity': specificity,
            'auc_scores': auc_scores,
            'accuracy': report['accuracy']
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 50,
        early_stopping_patience: int = 10,
        save_dir: str = 'models'
    ) -> Dict:
        """
        Complete training loop with early stopping
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        writer = SummaryWriter(log_dir=save_path / 'tensorboard')
        
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(n_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{n_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Logging
            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Sensitivity: {val_metrics['sensitivity']:.4f}")
            print(f"Specificity: {val_metrics['specificity']:.4f}")
            
            writer.add_scalars('Loss', {
                'train': train_loss,
                'val': val_loss
            }, epoch)
            
            writer.add_scalar('Accuracy', val_metrics['accuracy'], epoch)
            writer.add_scalar('Sensitivity', val_metrics['sensitivity'], epoch)
            writer.add_scalar('Specificity', val_metrics['specificity'], epoch)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics
                }
                
                torch.save(checkpoint, save_path / 'best_model.pt')
                print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Best model from epoch {best_epoch+1}")
                break
        
        writer.close()
        
        # Load best model
        checkpoint = torch.load(save_path / 'best_model.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        return {
            'best_epoch': best_epoch,
            'best_val_loss': self.best_val_loss,
            'best_metrics': checkpoint['val_metrics'],
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(self.train_losses, label='Train')
        axes[0].plot(self.val_losses, label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training History')
        axes[0].legend()
        axes[0].grid(True)
        
        # Confusion matrix (from best model)
        # This would be populated from validation
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def patient_wise_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    metadata: List[Dict],
    n_folds: int = 5
) -> List[Dict]:
    """
    Perform patient-wise cross-validation
    Ensures patients don't appear in both train and validation sets
    """
    # Get unique patient IDs
    patient_ids = np.array([m['patient_id'] for m in metadata])
    unique_patients = np.unique(patient_ids)
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    results = []
    
    for fold, (train_patient_idx, val_patient_idx) in enumerate(kf.split(unique_patients)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{n_folds}")
        print(f"{'='*60}")
        
        train_patients = unique_patients[train_patient_idx]
        val_patients = unique_patients[val_patient_idx]
        
        # Split data by patient
        train_mask = np.isin(patient_ids, train_patients)
        val_mask = np.isin(patient_ids, val_patients)
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        
        print(f"Train patients: {train_patients}")
        print(f"Val patients: {val_patients}")
        print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
        
        # Create datasets
        train_dataset = EEGDataset(X_train, y_train, [m for i, m in enumerate(metadata) if train_mask[i]])
        val_dataset = EEGDataset(X_val, y_val, [m for i, m in enumerate(metadata) if val_mask[i]])
        
        # Create model
        model = SeizureDetectionLSTM(
            n_channels=X.shape[1],
            n_samples=X.shape[2],
            n_classes=3
        )
        
        trainer = SeizureDetectionTrainer(model)
        
        # Create weighted sampler for training
        sampler = trainer.create_weighted_sampler(y_train)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            sampler=sampler,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4
        )
        
        # Train
        fold_results = trainer.train(
            train_loader,
            val_loader,
            n_epochs=50,
            save_dir=f'models/fold_{fold+1}'
        )
        
        results.append({
            'fold': fold + 1,
            'train_patients': train_patients.tolist(),
            'val_patients': val_patients.tolist(),
            **fold_results
        })
    
    return results


# Example usage
if __name__ == '__main__':
    # This would use real data from data_loader.py
    # For demonstration, using synthetic data
    
    n_samples = 1000
    n_channels = 23
    n_timepoints = 1280
    
    X = np.random.randn(n_samples, n_channels, n_timepoints).astype(np.float32)
    y = np.random.choice([0, 1, 2], size=n_samples, p=[0.9, 0.05, 0.05])
    
    metadata = [
        {'patient_id': f'chb{(i % 5) + 1:02d}', 'file': 'test.edf'}
        for i in range(n_samples)
    ]
    
    # Simple train/val split
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    train_dataset = EEGDataset(X_train, y_train, metadata[:split_idx])
    val_dataset = EEGDataset(X_val, y_val, metadata[split_idx:])
    
    # Create model
    model = SeizureDetectionLSTM(
        n_channels=n_channels,
        n_samples=n_timepoints,
        n_classes=3
    )
    
    # Train
    trainer = SeizureDetectionTrainer(model)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    results = trainer.train(train_loader, val_loader, n_epochs=5)
    
    print("\nTraining completed!")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Best metrics: {results['best_metrics']}")