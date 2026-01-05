"""Unit tests for model architectures."""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import UNet, UNetPlusPlus
from src.models.loss import DiceLoss, FocalLoss, CombinedLoss
from src.models.metrics import dice_coefficient, iou_score, pixel_accuracy


class TestUNet:
    """Tests for U-Net model."""
    
    def test_unet_creation(self):
        """Test U-Net model creation."""
        model = UNet(n_channels=3, n_classes=1)
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_unet_forward(self):
        """Test U-Net forward pass."""
        model = UNet(n_channels=3, n_classes=1)
        x = torch.randn(2, 3, 256, 256)
        output = model(x)
        
        assert output.shape == (2, 1, 256, 256)
    
    def test_unet_different_sizes(self):
        """Test U-Net with different input sizes."""
        model = UNet(n_channels=3, n_classes=1)
        
        sizes = [(128, 128), (256, 256), (512, 512)]
        
        for size in sizes:
            x = torch.randn(1, 3, *size)
            output = model(x)
            assert output.shape == (1, 1, *size)
    
    def test_unet_parameter_count(self):
        """Test U-Net parameter count."""
        model = UNet(n_channels=3, n_classes=1, base_channels=64)
        param_count = sum(p.numel() for p in model.parameters())
        
        # Should have around 31M parameters
        assert 30_000_000 < param_count < 35_000_000


class TestUNetPlusPlus:
    """Tests for U-Net++ model."""
    
    def test_unetplusplus_creation(self):
        """Test U-Net++ model creation."""
        model = UNetPlusPlus(n_channels=3, n_classes=1)
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_unetplusplus_forward(self):
        """Test U-Net++ forward pass."""
        model = UNetPlusPlus(n_channels=3, n_classes=1)
        x = torch.randn(2, 3, 256, 256)
        output = model(x)
        
        assert output.shape == (2, 1, 256, 256)
    
    def test_unetplusplus_deep_supervision(self):
        """Test U-Net++ with deep supervision."""
        model = UNetPlusPlus(n_channels=3, n_classes=1, deep_supervision=True)
        x = torch.randn(2, 3, 256, 256)
        outputs = model(x)
        
        assert isinstance(outputs, list)
        assert len(outputs) == 4
        for output in outputs:
            assert output.shape == (2, 1, 256, 256)


class TestLossFunctions:
    """Tests for loss functions."""
    
    def test_dice_loss(self):
        """Test Dice loss."""
        loss_fn = DiceLoss()
        
        predictions = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        
        loss = loss_fn(predictions, targets)
        
        assert loss.item() >= 0
        assert loss.item() <= 1
    
    def test_focal_loss(self):
        """Test Focal loss."""
        loss_fn = FocalLoss()
        
        predictions = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        
        loss = loss_fn(predictions, targets)
        
        assert loss.item() >= 0
    
    def test_combined_loss(self):
        """Test Combined loss."""
        loss_fn = CombinedLoss()
        
        predictions = torch.randn(2, 1, 64, 64)
        targets = torch.randint(0, 2, (2, 1, 64, 64)).float()
        
        loss = loss_fn(predictions, targets)
        
        assert loss.item() >= 0


class TestMetrics:
    """Tests for evaluation metrics."""
    
    def test_dice_coefficient(self):
        """Test Dice coefficient calculation."""
        # Perfect prediction
        predictions = torch.ones(1, 1, 64, 64) * 10  # High logits
        targets = torch.ones(1, 1, 64, 64)
        
        dice = dice_coefficient(predictions, targets)
        assert dice > 0.99
        
        # No overlap
        predictions = torch.ones(1, 1, 64, 64) * -10  # Low logits
        targets = torch.ones(1, 1, 64, 64)
        
        dice = dice_coefficient(predictions, targets)
        assert dice < 0.01
    
    def test_iou_score(self):
        """Test IoU score calculation."""
        # Perfect prediction
        predictions = torch.ones(1, 1, 64, 64) * 10
        targets = torch.ones(1, 1, 64, 64)
        
        iou = iou_score(predictions, targets)
        assert iou > 0.99
    
    def test_pixel_accuracy(self):
        """Test pixel accuracy calculation."""
        # Perfect prediction
        predictions = torch.ones(1, 1, 64, 64) * 10
        targets = torch.ones(1, 1, 64, 64)
        
        acc = pixel_accuracy(predictions, targets)
        assert acc > 0.99


if __name__ == '__main__':
    pytest.main([__file__])
