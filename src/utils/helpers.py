"""Helper functions for training and evaluation."""

import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional


def set_seed(seed: int = 42):
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
):
    """Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Dictionary of metrics
        save_path: Path to save checkpoint
        scheduler: Learning rate scheduler (optional)
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = 'cpu'
) -> Dict:
    """Load model checkpoint.
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint file
        optimizer: Optimizer (optional)
        scheduler: Learning rate scheduler (optional)
        device: Device to load model to
    
    Returns:
        Dictionary containing checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}, Metrics: {checkpoint.get('metrics', {})}")
    
    return checkpoint


class EarlyStopping:
    """Early stopping handler to stop training when metric stops improving."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0, mode: str = 'max'):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for maximizing metric, 'min' for minimizing
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """Check if training should be stopped.
        
        Args:
            score: Current metric value
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


def get_device() -> torch.device:
    """Get the best available device.
    
    Returns:
        torch.device: CUDA if available, otherwise CPU
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Using CPU")
    return device
