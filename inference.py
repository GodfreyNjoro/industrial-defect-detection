#!/usr/bin/env python3
"""Inference script for industrial defect detection model."""

import argparse
import yaml
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from src.models import UNet, UNetPlusPlus
from src.utils.helpers import load_checkpoint


def load_image(image_path: str, image_size: tuple = (256, 256)):
    """Load and preprocess an image.
    
    Args:
        image_path: Path to the image
        image_size: Target image size
    
    Returns:
        Preprocessed image tensor and original image
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor, original_image


def predict(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5
):
    """Run inference on an image.
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor
        device: Device to run inference on
        threshold: Threshold for binarization
    
    Returns:
        Predicted mask (probability map) and binary mask
    """
    model.eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        
        # Get probability map
        prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # Binarize
        binary_mask = (prob_map > threshold).astype(np.uint8)
    
    return prob_map, binary_mask


def visualize_result(
    original_image: np.ndarray,
    prob_map: np.ndarray,
    binary_mask: np.ndarray,
    save_path: str = None,
    threshold: float = 0.5
):
    """Visualize prediction results.
    
    Args:
        original_image: Original input image
        prob_map: Predicted probability map
        binary_mask: Binary prediction mask
        save_path: Path to save visualization
        threshold: Threshold used for binarization
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')
    
    # Probability map
    im1 = axes[1].imshow(prob_map, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Defect Probability Map', fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # Binary mask
    axes[2].imshow(binary_mask, cmap='gray')
    axes[2].set_title(f'Binary Mask (t={threshold})', fontsize=14)
    axes[2].axis('off')
    
    # Overlay
    overlay = original_image.copy()
    if binary_mask.shape != overlay.shape[:2]:
        binary_mask_resized = cv2.resize(binary_mask, (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        binary_mask_resized = binary_mask
    
    mask_colored = np.zeros_like(overlay)
    mask_colored[binary_mask_resized > 0] = [255, 0, 0]  # Red overlay
    overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
    
    axes[3].imshow(overlay)
    axes[3].set_title('Defect Overlay', fontsize=14)
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Run inference with trained model')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, help='Path to save output visualization')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binarization')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print(f"Loading model: {config['model']['architecture']}")
    if config['model']['architecture'] == 'unet':
        model = UNet(
            n_channels=3,
            n_classes=1,
            bilinear=config['model']['bilinear'],
            base_channels=config['model']['base_channels']
        )
    elif config['model']['architecture'] == 'unetplusplus':
        model = UNetPlusPlus(
            n_channels=3,
            n_classes=1,
            base_channels=config['model']['base_channels']
        )
    else:
        raise ValueError(f"Unknown architecture: {config['model']['architecture']}")
    
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    load_checkpoint(model, args.checkpoint, device=str(device))
    
    # Load and preprocess image
    print(f"Loading image: {args.image}")
    image_tensor, original_image = load_image(
        args.image,
        image_size=tuple(config['data']['image_size'])
    )
    
    # Run inference
    print("Running inference...")
    prob_map, binary_mask = predict(
        model=model,
        image_tensor=image_tensor,
        device=device,
        threshold=args.threshold
    )
    
    # Determine if defect is present
    defect_percentage = (binary_mask.sum() / binary_mask.size) * 100
    has_defect = defect_percentage > 0.1  # If more than 0.1% pixels are defective
    
    print(f"\nResults:")
    print(f"Defect detected: {'Yes' if has_defect else 'No'}")
    print(f"Defect coverage: {defect_percentage:.2f}%")
    print(f"Max probability: {prob_map.max():.4f}")
    
    # Visualize results
    output_path = args.output if args.output else f"prediction_{Path(args.image).stem}.png"
    visualize_result(
        original_image=original_image,
        prob_map=prob_map,
        binary_mask=binary_mask,
        save_path=output_path,
        threshold=args.threshold
    )
    
    print(f"\nInference completed!")


if __name__ == '__main__':
    main()
