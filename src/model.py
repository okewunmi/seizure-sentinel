"""
Seizure Detection Model Architecture
Bidirectional LSTM with Attention Mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AttentionLayer(nn.Module):
    """
    Attention mechanism to focus on relevant temporal patterns
    Critical for detecting seizure precursors
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_dim)
            
        Returns:
            context: (batch, hidden_dim) - weighted representation
            weights: (batch, seq_len) - attention weights
        """
        # Calculate attention scores
        attention_scores = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_scores.squeeze(-1), dim=1)  # (batch, seq_len)
        
        # Apply attention weights
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, seq_len)
            lstm_output  # (batch, seq_len, hidden_dim)
        ).squeeze(1)  # (batch, hidden_dim)
        
        return context, attention_weights


class SeizureDetectionLSTM(nn.Module):
    """
    Bidirectional LSTM for seizure detection
    
    Architecture:
    1. Conv1D for spatial feature extraction across channels
    2. Bidirectional LSTM for temporal modeling
    3. Attention mechanism
    4. Fully connected classifier
    """
    
    def __init__(
        self,
        n_channels: int = 23,
        n_samples: int = 1280,  # 5 seconds @ 256 Hz
        hidden_dim: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.5,
        n_classes: int = 3  # Interictal, Pre-ictal, Ictal
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.hidden_dim = hidden_dim
        
        # 1. Spatial feature extraction (across channels)
        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        # Calculate LSTM input size after convolutions
        conv_output_length = n_samples // 4  # Two pooling layers
        
        # 2. Temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # 3. Attention mechanism
        self.attention = AttentionLayer(hidden_dim * 2)  # *2 for bidirectional
        
        # 4. Classifier
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, n_classes)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, n_channels, n_samples) - raw EEG
            
        Returns:
            logits: (batch, n_classes) - class predictions
            attention_weights: (batch, seq_len) - attention visualization
        """
        # Spatial feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # Reshape for LSTM: (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim*2)
        
        # Attention
        context, attention_weights = self.attention(lstm_out)
        
        # Classifier
        x = F.relu(self.fc1(context))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits, attention_weights


class SpectrogramCNN(nn.Module):
    """
    Alternative architecture using spectrogram input
    2D CNN for time-frequency analysis
    """
    
    def __init__(
        self,
        n_channels: int = 23,
        n_freq_bins: int = 65,
        n_time_bins: int = 39,
        n_classes: int = 3
    ):
        super().__init__()
        
        # Each channel has its own spectrogram
        # Input: (batch, n_channels, n_freq, n_time)
        
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        
        # Calculate flattened size
        final_freq = n_freq_bins // 8
        final_time = n_time_bins // 8
        flatten_size = 128 * final_freq * final_time
        
        self.fc1 = nn.Linear(flatten_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_channels, n_freq, n_time) - spectrogram
            
        Returns:
            logits: (batch, n_classes)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits


class EnsembleModel(nn.Module):
    """
    Ensemble of LSTM and CNN models
    Combines raw signal and spectrogram analysis
    """
    
    def __init__(
        self,
        n_channels: int = 23,
        n_samples: int = 1280,
        n_classes: int = 3
    ):
        super().__init__()
        
        self.lstm_model = SeizureDetectionLSTM(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes
        )
        
        self.cnn_model = SpectrogramCNN(
            n_channels=n_channels,
            n_classes=n_classes
        )
        
        # Fusion layer
        self.fusion = nn.Linear(n_classes * 2, n_classes)
    
    def forward(
        self,
        raw_signal: torch.Tensor,
        spectrogram: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            raw_signal: (batch, n_channels, n_samples)
            spectrogram: (batch, n_channels, n_freq, n_time)
            
        Returns:
            logits: (batch, n_classes)
        """
        lstm_logits, _ = self.lstm_model(raw_signal)
        cnn_logits = self.cnn_model(spectrogram)
        
        # Concatenate and fuse
        combined = torch.cat([lstm_logits, cnn_logits], dim=1)
        logits = self.fusion(combined)
        
        return logits


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Seizures are rare events (<<1% of data)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch, n_classes) - raw logits
            targets: (batch,) - class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


# Example usage and model summary
if __name__ == '__main__':
    # Create model
    model = SeizureDetectionLSTM(
        n_channels=23,
        n_samples=1280,
        hidden_dim=128,
        lstm_layers=2,
        n_classes=3
    )
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 23, 1280)
    
    logits, attention_weights = model(x)
    
    print("Model Architecture:")
    print(model)
    print(f"\nInput shape: {x.shape}")
    print(f"Output logits: {logits.shape}")
    print(f"Attention weights: {attention_weights.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test predictions
    with torch.no_grad():
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
    
    print(f"\nPredicted classes: {preds}")
    print(f"Prediction probabilities:\n{probs}")