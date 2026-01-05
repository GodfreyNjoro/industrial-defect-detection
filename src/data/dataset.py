"""Dataset classes for MVTec AD anomaly detection."""

import os
from pathlib import Path
from typing import Optional, Tuple, Callable

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MVTecDataset(Dataset):
    """MVTec AD Dataset for anomaly detection and segmentation.
    
    The MVTec AD dataset contains images of various objects and textures with
    annotated defects for anomaly detection tasks.
    
    Args:
        root_dir: Root directory containing the MVTec AD dataset
        category: Product category (e.g., 'bottle', 'cable', 'capsule')
        split: 'train', 'val', or 'test'
        transform: Optional transform to be applied on images
        target_transform: Optional transform to be applied on masks
        image_size: Target size for images (height, width)
    """
    
    def __init__(
        self,
        root_dir: str,
        category: str = 'bottle',
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (256, 256)
    ):
        self.root_dir = Path(root_dir)
        self.category = category
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size
        
        # Build file paths
        self.image_paths = []
        self.mask_paths = []
        self.labels = []  # 0 for normal, 1 for anomaly
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset file paths and labels."""
        category_path = self.root_dir / self.category
        
        if self.split == 'train':
            # Training only contains normal images
            train_path = category_path / 'train' / 'good'
            if train_path.exists():
                for img_path in sorted(train_path.glob('*.png')):
                    self.image_paths.append(str(img_path))
                    self.mask_paths.append(None)  # No masks for normal images
                    self.labels.append(0)
        else:
            # Test/val contains both normal and anomaly images
            test_path = category_path / 'test'
            ground_truth_path = category_path / 'ground_truth'
            
            if test_path.exists():
                # Load normal test images
                good_path = test_path / 'good'
                if good_path.exists():
                    for img_path in sorted(good_path.glob('*.png')):
                        self.image_paths.append(str(img_path))
                        self.mask_paths.append(None)
                        self.labels.append(0)
                
                # Load anomaly images
                for defect_dir in sorted(test_path.iterdir()):
                    if defect_dir.is_dir() and defect_dir.name != 'good':
                        defect_type = defect_dir.name
                        
                        for img_path in sorted(defect_dir.glob('*.png')):
                            self.image_paths.append(str(img_path))
                            self.labels.append(1)
                            
                            # Find corresponding mask
                            mask_path = ground_truth_path / defect_type / f"{img_path.stem}_mask.png"
                            if mask_path.exists():
                                self.mask_paths.append(str(mask_path))
                            else:
                                self.mask_paths.append(None)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Get a sample from the dataset.
        
        Returns:
            image: Preprocessed image tensor
            mask: Binary mask tensor (all zeros if no defect)
            label: 0 for normal, 1 for anomaly
        """
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Load or create mask
        if self.mask_paths[idx] is not None:
            mask = Image.open(self.mask_paths[idx]).convert('L')
        else:
            # Create empty mask for normal images
            mask = Image.new('L', image.size, 0)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            mask = self.target_transform(mask)
        else:
            # Default mask transform
            mask = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor()
            ])(mask)
        
        # Binarize mask
        mask = (mask > 0.5).float()
        
        label = self.labels[idx]
        
        return image, mask, label


def get_data_loaders(
    root_dir: str,
    category: str = 'bottle',
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (256, 256)
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders.
    
    Args:
        root_dir: Root directory containing the MVTec AD dataset
        category: Product category
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        image_size: Target image size
    
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
    """
    from .transforms import get_train_transforms, get_val_transforms
    
    # Create datasets
    train_dataset = MVTecDataset(
        root_dir=root_dir,
        category=category,
        split='train',
        transform=get_train_transforms(image_size),
        image_size=image_size
    )
    
    val_dataset = MVTecDataset(
        root_dir=root_dir,
        category=category,
        split='test',
        transform=get_val_transforms(image_size),
        image_size=image_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
