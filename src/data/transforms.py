"""Data augmentation and transformation utilities."""

from typing import Tuple

from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_train_transforms(image_size: Tuple[int, int] = (256, 256)):
    """Get training data transformations with augmentation.
    
    Args:
        image_size: Target image size (height, width)
    
    Returns:
        Composed transforms for training
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_transforms(image_size: Tuple[int, int] = (256, 256)):
    """Get validation/test data transformations without augmentation.
    
    Args:
        image_size: Target image size (height, width)
    
    Returns:
        Composed transforms for validation/testing
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_albumentations_train_transforms(image_size: Tuple[int, int] = (256, 256)):
    """Get Albumentations training transforms for advanced augmentation.
    
    This is an alternative to torchvision transforms, offering more 
    sophisticated augmentation options.
    
    Args:
        image_size: Target image size (height, width)
    
    Returns:
        Albumentations compose object
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.GaussianBlur(blur_limit=3),
            A.MotionBlur(blur_limit=3),
        ], p=0.3),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05),
            A.GridDistortion(num_steps=5, distort_limit=0.05),
        ], p=0.2),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_albumentations_val_transforms(image_size: Tuple[int, int] = (256, 256)):
    """Get Albumentations validation transforms.
    
    Args:
        image_size: Target image size (height, width)
    
    Returns:
        Albumentations compose object
    """
    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
