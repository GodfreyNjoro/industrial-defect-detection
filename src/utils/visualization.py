"""Visualization utilities for model predictions and results."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Optional
import cv2
from pathlib import Path


def denormalize_image(image: torch.Tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) -> np.ndarray:
    """Denormalize an image tensor for visualization.
    
    Args:
        image: Normalized image tensor (C, H, W)
        mean: Mean values used for normalization
        std: Std values used for normalization
    
    Returns:
        Denormalized image as numpy array (H, W, C)
    """
    image = image.clone()
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
    image = torch.clamp(image, 0, 1)
    image = image.permute(1, 2, 0).cpu().numpy()
    return (image * 255).astype(np.uint8)


def visualize_predictions(
    images: torch.Tensor,
    masks: torch.Tensor,
    predictions: torch.Tensor,
    num_samples: int = 4,
    save_path: Optional[str] = None,
    threshold: float = 0.5
) -> None:
    """Visualize model predictions alongside ground truth.
    
    Args:
        images: Input images (B, C, H, W)
        masks: Ground truth masks (B, 1, H, W)
        predictions: Predicted masks (B, 1, H, W)
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization
        threshold: Threshold for binarizing predictions
    """
    num_samples = min(num_samples, images.size(0))
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_samples):
        # Denormalize image
        img = denormalize_image(images[idx])
        
        # Get masks
        gt_mask = masks[idx, 0].cpu().numpy()
        pred_mask = torch.sigmoid(predictions[idx, 0]).cpu().numpy()
        pred_binary = (pred_mask > threshold).astype(np.uint8)
        
        # Plot image
        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title('Input Image')
        axes[idx, 0].axis('off')
        
        # Plot ground truth mask
        axes[idx, 1].imshow(gt_mask, cmap='gray')
        axes[idx, 1].set_title('Ground Truth Mask')
        axes[idx, 1].axis('off')
        
        # Plot prediction (continuous)
        axes[idx, 2].imshow(pred_mask, cmap='jet', vmin=0, vmax=1)
        axes[idx, 2].set_title('Prediction (Probability)')
        axes[idx, 2].axis('off')
        
        # Plot binary prediction
        axes[idx, 3].imshow(pred_binary, cmap='gray')
        axes[idx, 3].set_title(f'Prediction (Binary, t={threshold})')
        axes[idx, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_history(
    history: dict,
    save_path: Optional[str] = None
) -> None:
    """Plot training history.
    
    Args:
        history: Dictionary containing training metrics
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training and validation loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot Dice coefficient
    if 'train_dice' in history:
        axes[0, 1].plot(history['train_dice'], label='Train Dice', linewidth=2)
        axes[0, 1].plot(history['val_dice'], label='Val Dice', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Coefficient')
        axes[0, 1].set_title('Dice Coefficient')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot IoU
    if 'train_iou' in history:
        axes[1, 0].plot(history['train_iou'], label='Train IoU', linewidth=2)
        axes[1, 0].plot(history['val_iou'], label='Val IoU', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU Score')
        axes[1, 0].set_title('IoU Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot learning rate
    if 'learning_rate' in history:
        axes[1, 1].plot(history['learning_rate'], linewidth=2, color='orange')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_prediction_grid(
    images: List[np.ndarray],
    masks: List[np.ndarray],
    predictions: List[np.ndarray],
    save_path: str,
    grid_size: tuple = (2, 2)
) -> None:
    """Save a grid of predictions.
    
    Args:
        images: List of input images
        masks: List of ground truth masks
        predictions: List of predicted masks
        save_path: Path to save the grid
        grid_size: Grid dimensions (rows, cols)
    """
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols * 3, figsize=(cols * 9, rows * 3))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= len(images):
                break
            
            col_offset = j * 3
            
            # Image
            axes[i, col_offset].imshow(images[idx])
            axes[i, col_offset].axis('off')
            if i == 0:
                axes[i, col_offset].set_title('Input')
            
            # Ground truth
            axes[i, col_offset + 1].imshow(masks[idx], cmap='gray')
            axes[i, col_offset + 1].axis('off')
            if i == 0:
                axes[i, col_offset + 1].set_title('Ground Truth')
            
            # Prediction
            axes[i, col_offset + 2].imshow(predictions[idx], cmap='gray')
            axes[i, col_offset + 2].axis('off')
            if i == 0:
                axes[i, col_offset + 2].set_title('Prediction')
            
            idx += 1
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Prediction grid saved to {save_path}")


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: tuple = (255, 0, 0)
) -> np.ndarray:
    """Overlay a binary mask on an image.
    
    Args:
        image: Input image (H, W, 3)
        mask: Binary mask (H, W)
        alpha: Transparency factor
        color: Color for the mask overlay (R, G, B)
    
    Returns:
        Image with mask overlay
    """
    overlay = image.copy()
    mask_colored = np.zeros_like(image)
    mask_colored[mask > 0] = color
    
    overlay = cv2.addWeighted(overlay, 1 - alpha, mask_colored, alpha, 0)
    return overlay
