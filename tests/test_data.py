"""Unit tests for data loading and preprocessing."""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path
from PIL import Image
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.transforms import get_train_transforms, get_val_transforms


class TestTransforms:
    """Tests for data transforms."""
    
    def test_train_transforms(self):
        """Test training transforms."""
        transform = get_train_transforms(image_size=(256, 256))
        
        # Create a dummy image
        dummy_image = Image.new('RGB', (512, 512), color='red')
        
        # Apply transform
        transformed = transform(dummy_image)
        
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, 256, 256)
        assert transformed.dtype == torch.float32
    
    def test_val_transforms(self):
        """Test validation transforms."""
        transform = get_val_transforms(image_size=(256, 256))
        
        # Create a dummy image
        dummy_image = Image.new('RGB', (512, 512), color='blue')
        
        # Apply transform
        transformed = transform(dummy_image)
        
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == (3, 256, 256)
        assert transformed.dtype == torch.float32
    
    def test_transform_normalization(self):
        """Test that transforms normalize images correctly."""
        transform = get_val_transforms(image_size=(256, 256))
        
        # Create a white image
        dummy_image = Image.new('RGB', (256, 256), color='white')
        
        # Apply transform
        transformed = transform(dummy_image)
        
        # Check that values are normalized (not in 0-255 range)
        assert transformed.max() < 10
        assert transformed.min() > -10
    
    def test_different_image_sizes(self):
        """Test transforms with different target sizes."""
        sizes = [(128, 128), (256, 256), (512, 512)]
        
        for size in sizes:
            transform = get_val_transforms(image_size=size)
            dummy_image = Image.new('RGB', (300, 300), color='green')
            transformed = transform(dummy_image)
            
            assert transformed.shape == (3, *size)


if __name__ == '__main__':
    pytest.main([__file__])
