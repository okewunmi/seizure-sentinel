"""
Seizure Sentinel - Complete Training Pipeline
End-to-end workflow for training and evaluating seizure detection model

Usage:
    python main.py --data_dir data/raw/chb-mit --mode train
    python main.py --model_path models/best_model.pt --mode evaluate
"""
import sys
sys.path.insert(0, 'src')

import argparse
import json

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

from data_loader import CHBMITLoader
from preprocessor import EEGPreprocessor, FeatureExtractor
from model import SeizureDetectionLSTM
from train import SeizureDetectionTrainer, EEGDataset, patient_wise_cross_validation
from evaluate import SeizureDetectionEvaluator
from torch.utils.data import DataLoader


def load_and_preprocess_data(
    data_dir: str,
    patients: list = None,
    window_size: float = 5.0,
    pre_ictal_window: float = 5.0,
    cache_dir: str = 'data/processed'
) -> tuple:
    """Load CHB-MIT dataset and preprocess"""
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True, parents=True)
    
    cache_file = cache_path / f'preprocessed_w{window_size}_p{pre_ictal_window}.npz'
    
    # Check cache
    if cache_file.exists():
        print(f"Loading cached data from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        return data['X'], data['y'], data['metadata'].tolist()
    
    print("Loading and preprocessing raw data...")
    
    # Initialize loader
    loader = CHBMITLoader(data_dir)
    preprocessor = EEGPreprocessor()
    
    if patients is None:
        patients = loader.get_patient_ids()
    
    print(f"Processing {len(patients)} patients: {patients}")
    
    # Load data for each patient
    all_X = []
    all_y = []
    all_metadata = []
    
    for patient_id in patients:
        print(f"\nProcessing {patient_id}...")
        
        try:
            # Load windowed data with LESS overlap to reduce memory
            X, y, metadata = loader.create_windowed_dataset(
                patient_id,
                window_size=window_size,
                overlap=4.0,  # 50% overlap instead of 80%
                pre_ictal_window=pre_ictal_window,
                max_windows=3000
            )
            
            if len(X) == 0:
                continue
            
            
            # Preprocess each window
            X_processed = np.zeros(X.shape, dtype=np.float32)  # Use float32
            for i in range(len(X)):
                X_processed[i] = preprocessor.preprocess_pipeline(X[i]).astype(np.float32)
            
            all_X.append(X_processed)
            all_y.append(y)
            all_metadata.extend(metadata)
            
            print(f"  Windows: {len(X)}, Class distribution: {np.bincount(y)}")
            
        except Exception as e:
            print(f"  Error processing {patient_id}: {e}")
            continue
    
    if len(all_X) == 0:
        raise ValueError("No data loaded! Check your dataset directory.")
    
    # Combine all patients
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    print(f"\nTotal dataset:")
    print(f"  Shape: {X.shape}")
    print(f"  Class distribution: {np.bincount(y)}")
    print(f"  Interictal: {np.sum(y==0)} ({100*np.mean(y==0):.1f}%)")
    print(f"  Pre-ictal:  {np.sum(y==1)} ({100*np.mean(y==1):.1f}%)")
    print(f"  Ictal:      {np.sum(y==2)} ({100*np.mean(y==2):.1f}%)")
    
    # Cache preprocessed data
    print(f"\nCaching to {cache_file}")
    np.savez_compressed(
        cache_file,
        X=X,
        y=y,
        metadata=np.array(all_metadata, dtype=object)
    )
    
    return X, y, all_metadata

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    metadata: list,
    config: dict
) -> tuple:
    """
    Train seizure detection model
    
    Args:
        X: EEG windows (n_samples, n_channels, n_timepoints)
        y: Labels (n_samples,)
        metadata: Sample metadata
        config: Training configuration
        
    Returns:
        model, trainer, results
    """
    print("\n" + "="*70)
    print("TRAINING SEIZURE DETECTION MODEL")
    print("="*70)
    
    # Split data (80/20 train/val)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    split_idx = int(0.8 * n_samples)
    
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    X_train = X[train_idx]
    y_train = y[train_idx]
    metadata_train = [metadata[i] for i in train_idx]
    
    X_val = X[val_idx]
    y_val = y[val_idx]
    metadata_val = [metadata[i] for i in val_idx]
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    
    # Create datasets
    train_dataset = EEGDataset(X_train, y_train, metadata_train)
    val_dataset = EEGDataset(X_val, y_val, metadata_val)
    
    # Create model
    model = SeizureDetectionLSTM(
        n_channels=X.shape[1],
        n_samples=X.shape[2],
        hidden_dim=config['hidden_dim'],
        lstm_layers=config['lstm_layers'],
        dropout=config['dropout'],
        n_classes=3
    )
    
    print(f"\nModel architecture:")
    print(f"  Hidden dim: {config['hidden_dim']}")
    print(f"  LSTM layers: {config['lstm_layers']}")
    print(f"  Dropout: {config['dropout']}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Create trainer
    trainer = SeizureDetectionTrainer(
        model,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        use_focal_loss=config['use_focal_loss']
    )
    
    # Create weighted sampler for balanced batches
    sampler = trainer.create_weighted_sampler(y_train)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=sampler,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    # Train
    results = trainer.train(
        train_loader,
        val_loader,
        n_epochs=config['n_epochs'],
        early_stopping_patience=config['early_stopping_patience'],
        save_dir=config['save_dir']
    )
    
    # Plot training history
    trainer.plot_training_history(
        save_path=f"{config['save_dir']}/training_history.png"
    )
    
    return model, trainer, results


def evaluate_model(
    model_path: str,
    X: np.ndarray,
    y: np.ndarray,
    metadata: list,
    save_dir: str = 'results'
) -> dict:
    """
    Evaluate trained model
    
    Args:
        model_path: Path to saved model checkpoint
        X: EEG windows
        y: Labels
        metadata: Sample metadata
        save_dir: Directory to save results
        
    Returns:
        metrics dictionary
    """
    print("\n" + "="*70)
    print("EVALUATING MODEL")
    print("="*70)
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    
    # Load model
    checkpoint = torch.load(model_path)
    
    model = SeizureDetectionLSTM(
        n_channels=X.shape[1],
        n_samples=X.shape[2]
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\nLoaded model from {model_path}")
    print(f"Trained for {checkpoint['epoch']+1} epochs")
    print(f"Best validation loss: {checkpoint['val_loss']:.4f}")
    
    # Create dataset
    dataset = EEGDataset(X, y, metadata)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    # Evaluate
    evaluator = SeizureDetectionEvaluator()
    
    metrics, y_true, y_pred, y_probs, attention_weights = evaluator.evaluate_model(
        model,
        dataloader
    )
    
    # Generate report
    report = evaluator.generate_report(
        metrics,
        save_path=save_path / 'evaluation_report.txt'
    )
    
    print("\n" + report)
    
    # Save metrics
    with open(save_path / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot visualizations
    evaluator.plot_confusion_matrix(
        y_true,
        y_pred,
        save_path=save_path / 'confusion_matrix.png'
    )
    
    evaluator.plot_roc_curves(
        y_true,
        y_probs,
        save_path=save_path / 'roc_curves.png'
    )
    
    evaluator.plot_precision_recall_curves(
        y_true,
        y_probs,
        save_path=save_path / 'precision_recall.png'
    )
    
    print(f"\nResults saved to {save_dir}/")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Seizure Sentinel Training Pipeline')
    
    parser.add_argument('--data_dir', type=str, default='data/raw/chb-mit',
                        help='Path to CHB-MIT dataset')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'cross_validate'],
                        default='train', help='Operation mode')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint (for evaluation)')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration file')
    parser.add_argument('--patients', type=str, nargs='+', default=None,
                        help='Patient IDs to process (default: all)')
    
    args = parser.parse_args()
    
    # Load configuration
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            'window_size': 5.0,
            'pre_ictal_window': 5.0,
            'hidden_dim': 128,
            'lstm_layers': 2,
            'dropout': 0.5,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'batch_size': 32,
            'n_epochs': 50,
            'early_stopping_patience': 10,
            'use_focal_loss': True,
            'num_workers': 4,
            'save_dir': f'models/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        }
        
        # Save config
        with open(args.config, 'w') as f:
            json.dump(config, f, indent=2)
    
    print("Configuration:")
    print(json.dumps(config, indent=2))
    
    # Load data
    X, y, metadata = load_and_preprocess_data(
        args.data_dir,
        patients=args.patients,
        window_size=config['window_size'],
        pre_ictal_window=config['pre_ictal_window']
    )
    
    if args.mode == 'train':
        # Train model
        model, trainer, results = train_model(X, y, metadata, config)
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        print(f"Best validation loss: {results['best_val_loss']:.4f}")
        print(f"Best metrics:")
        print(json.dumps(results['best_metrics'], indent=2))
        
    elif args.mode == 'evaluate':
        if args.model_path is None:
            raise ValueError("--model_path required for evaluation mode")
        
        # Evaluate model
        metrics = evaluate_model(
            args.model_path,
            X, y, metadata,
            save_dir=f'results/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        
    elif args.mode == 'cross_validate':
        # Patient-wise cross-validation
        results = patient_wise_cross_validation(
            X, y, metadata,
            n_folds=5
        )
        
        print("\n" + "="*70)
        print("CROSS-VALIDATION COMPLETED")
        print("="*70)
        
        # Aggregate results
        avg_sensitivity = np.mean([r['best_metrics']['sensitivity'] for r in results])
        avg_specificity = np.mean([r['best_metrics']['specificity'] for r in results])
        
        print(f"Average Sensitivity: {avg_sensitivity:.4f}")
        print(f"Average Specificity: {avg_specificity:.4f}")
        
        # Save results
        with open(f'results/cross_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()