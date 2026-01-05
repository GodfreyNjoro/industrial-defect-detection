"""Loss functions for segmentation tasks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation.
    
    The Dice coefficient is a measure of overlap between two samples.
    Loss = 1 - Dice Coefficient
    
    Args:
        smooth: Smoothing factor to avoid division by zero
    """
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate Dice loss.
        
        Args:
            predictions: Predicted masks (B, C, H, W)
            targets: Ground truth masks (B, C, H, W)
        
        Returns:
            Dice loss value
        """
        predictions = torch.sigmoid(predictions)
        
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    Paper: https://arxiv.org/abs/1708.02002
    
    Args:
        alpha: Weighting factor for positive class
        gamma: Focusing parameter
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate Focal loss.
        
        Args:
            predictions: Predicted logits (B, C, H, W)
            targets: Ground truth masks (B, C, H, W)
        
        Returns:
            Focal loss value
        """
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        
        # Get probabilities
        probs = torch.sigmoid(predictions)
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # Apply focal term
        focal_term = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        loss = alpha_t * focal_term * bce_loss
        
        return loss.mean()


class CombinedLoss(nn.Module):
    """Combined loss function: BCE + Dice Loss.
    
    This combination helps with both pixel-wise accuracy and overall
    region overlap.
    
    Args:
        bce_weight: Weight for binary cross-entropy loss
        dice_weight: Weight for Dice loss
    """
    
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate combined loss.
        
        Args:
            predictions: Predicted logits (B, C, H, W)
            targets: Ground truth masks (B, C, H, W)
        
        Returns:
            Combined loss value
        """
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class IoULoss(nn.Module):
    """Intersection over Union (IoU) Loss.
    
    Args:
        smooth: Smoothing factor to avoid division by zero
    """
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate IoU loss.
        
        Args:
            predictions: Predicted masks (B, C, H, W)
            targets: Ground truth masks (B, C, H, W)
        
        Returns:
            IoU loss value
        """
        predictions = torch.sigmoid(predictions)
        
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Intersection and union
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum() - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - iou
