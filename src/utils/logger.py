"""Logging utilities for training."""

import logging
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime


def setup_logger(
    name: str = 'industrial_defect_detection',
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Setup a logger with console and file handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class MetricLogger:
    """Logger for tracking training metrics."""
    
    def __init__(self, log_dir: str):
        """Initialize metric logger.
        
        Args:
            log_dir: Directory to save metric logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'train_iou': [],
            'val_iou': [],
            'learning_rate': []
        }
        
        self.current_epoch = 0
    
    def log_epoch(self, metrics: dict):
        """Log metrics for an epoch.
        
        Args:
            metrics: Dictionary of metric names and values
        """
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        
        self.current_epoch += 1
    
    def save(self, filename: str = 'metrics.json'):
        """Save metrics to JSON file.
        
        Args:
            filename: Name of the output file
        """
        save_path = self.log_dir / filename
        
        with open(save_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        print(f"Metrics saved to {save_path}")
    
    def load(self, filename: str = 'metrics.json'):
        """Load metrics from JSON file.
        
        Args:
            filename: Name of the input file
        """
        load_path = self.log_dir / filename
        
        if load_path.exists():
            with open(load_path, 'r') as f:
                self.metrics = json.load(f)
            
            # Infer current epoch
            self.current_epoch = len(self.metrics.get('train_loss', []))
            print(f"Metrics loaded from {load_path}")
        else:
            print(f"No metrics file found at {load_path}")
    
    def get_best_epoch(self, metric: str = 'val_dice', mode: str = 'max') -> int:
        """Get the epoch with the best metric value.
        
        Args:
            metric: Metric name to optimize
            mode: 'max' or 'min'
        
        Returns:
            Best epoch number (0-indexed)
        """
        if metric not in self.metrics or not self.metrics[metric]:
            return -1
        
        values = self.metrics[metric]
        
        if mode == 'max':
            return values.index(max(values))
        else:
            return values.index(min(values))
    
    def print_summary(self):
        """Print a summary of the training metrics."""
        print("\n" + "="*50)
        print("Training Summary")
        print("="*50)
        
        for metric_name, values in self.metrics.items():
            if values:
                print(f"{metric_name:20s}: Best = {max(values):.4f}, Last = {values[-1]:.4f}")
        
        print("="*50 + "\n")
