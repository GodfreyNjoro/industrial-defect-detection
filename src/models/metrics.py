"""Evaluation metrics for segmentation tasks."""

import torch
import numpy as np
from typing import Tuple


def dice_coefficient(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """Calculate Dice coefficient.
    
    Args:
        predictions: Predicted masks (B, C, H, W) or (B, H, W)
        targets: Ground truth masks (B, C, H, W) or (B, H, W)
        threshold: Threshold for binarization
        smooth: Smoothing factor
    
    Returns:
        Dice coefficient value
    """
    with torch.no_grad():
        predictions = torch.sigmoid(predictions)
        predictions = (predictions > threshold).float()
        
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)
        
        return dice.item()


def iou_score(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> float:
    """Calculate Intersection over Union (IoU) score.
    
    Args:
        predictions: Predicted masks (B, C, H, W) or (B, H, W)
        targets: Ground truth masks (B, C, H, W) or (B, H, W)
        threshold: Threshold for binarization
        smooth: Smoothing factor
    
    Returns:
        IoU score value
    """
    with torch.no_grad():
        predictions = torch.sigmoid(predictions)
        predictions = (predictions > threshold).float()
        
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        
        return iou.item()


def pixel_accuracy(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Calculate pixel-wise accuracy.
    
    Args:
        predictions: Predicted masks (B, C, H, W) or (B, H, W)
        targets: Ground truth masks (B, C, H, W) or (B, H, W)
        threshold: Threshold for binarization
    
    Returns:
        Pixel accuracy value
    """
    with torch.no_grad():
        predictions = torch.sigmoid(predictions)
        predictions = (predictions > threshold).float()
        
        correct = (predictions == targets).sum()
        total = torch.numel(predictions)
        
        accuracy = correct / total
        
        return accuracy.item()


def precision_recall_f1(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Tuple[float, float, float]:
    """Calculate precision, recall, and F1 score.
    
    Args:
        predictions: Predicted masks (B, C, H, W) or (B, H, W)
        targets: Ground truth masks (B, C, H, W) or (B, H, W)
        threshold: Threshold for binarization
    
    Returns:
        Tuple of (precision, recall, f1_score)
    """
    with torch.no_grad():
        predictions = torch.sigmoid(predictions)
        predictions = (predictions > threshold).float()
        
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        tp = ((predictions == 1) & (targets == 1)).sum().float()
        fp = ((predictions == 1) & (targets == 0)).sum().float()
        fn = ((predictions == 0) & (targets == 1)).sum().float()
        
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        return precision.item(), recall.item(), f1.item()


def confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """Calculate confusion matrix.
    
    Args:
        predictions: Predicted masks (B, C, H, W) or (B, H, W)
        targets: Ground truth masks (B, C, H, W) or (B, H, W)
        threshold: Threshold for binarization
    
    Returns:
        2x2 confusion matrix [[TN, FP], [FN, TP]]
    """
    with torch.no_grad():
        predictions = torch.sigmoid(predictions)
        predictions = (predictions > threshold).float()
        
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        tp = ((predictions == 1) & (targets == 1)).sum().item()
        fp = ((predictions == 1) & (targets == 0)).sum().item()
        fn = ((predictions == 0) & (targets == 1)).sum().item()
        tn = ((predictions == 0) & (targets == 0)).sum().item()
        
        return np.array([[tn, fp], [fn, tp]])
