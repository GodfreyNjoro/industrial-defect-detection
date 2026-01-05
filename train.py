#!/usr/bin/env python3
"""Training script for industrial defect detection model."""

import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from src.models import UNet, UNetPlusPlus, CombinedLoss, dice_coefficient, iou_score
from src.data import get_data_loaders
from src.utils import (
    setup_logger,
    MetricLogger,
    set_seed,
    save_checkpoint,
    EarlyStopping,
    get_device,
    visualize_predictions,
    plot_training_history
)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger
) -> tuple:
    """Train for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        logger: Logger instance
    
    Returns:
        Tuple of (average_loss, average_dice, average_iou)
    """
    model.train()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, (images, masks, labels) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        dice = dice_coefficient(outputs, masks)
        iou = iou_score(outputs, masks)
        
        # Update statistics
        total_loss += loss.item()
        total_dice += dice
        total_iou += iou
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice:.4f}',
            'iou': f'{iou:.4f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_dice = total_dice / len(train_loader)
    avg_iou = total_iou / len(train_loader)
    
    logger.info(f'Train - Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}')
    
    return avg_loss, avg_dice, avg_iou


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    logger,
    save_vis: bool = False,
    vis_dir: str = None
) -> tuple:
    """Validate the model.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        logger: Logger instance
        save_vis: Whether to save visualizations
        vis_dir: Directory to save visualizations
    
    Returns:
        Tuple of (average_loss, average_dice, average_iou)
    """
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    # For visualization
    vis_images = None
    vis_masks = None
    vis_preds = None
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
    
    with torch.no_grad():
        for batch_idx, (images, masks, labels) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            dice = dice_coefficient(outputs, masks)
            iou = iou_score(outputs, masks)
            
            # Update statistics
            total_loss += loss.item()
            total_dice += dice
            total_iou += iou
            
            # Save first batch for visualization
            if batch_idx == 0 and save_vis:
                vis_images = images[:4]
                vis_masks = masks[:4]
                vis_preds = outputs[:4]
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice:.4f}',
                'iou': f'{iou:.4f}'
            })
    
    avg_loss = total_loss / len(val_loader)
    avg_dice = total_dice / len(val_loader)
    avg_iou = total_iou / len(val_loader)
    
    logger.info(f'Val - Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}')
    
    # Save visualizations
    if save_vis and vis_dir is not None and vis_images is not None:
        vis_path = Path(vis_dir) / f'predictions_epoch_{epoch}.png'
        visualize_predictions(
            vis_images, vis_masks, vis_preds,
            num_samples=4,
            save_path=str(vis_path)
        )
    
    return avg_loss, avg_dice, avg_iou


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train industrial defect detection model')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--data_dir', type=str, help='Path to dataset directory')
    parser.add_argument('--category', type=str, help='Product category')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.data_dir:
        config['data']['root_dir'] = args.data_dir
    if args.category:
        config['data']['category'] = args.category
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    # Setup
    set_seed(config['training']['seed'])
    device = get_device()
    
    # Create directories
    results_dir = Path(config['paths']['results_dir'])
    checkpoint_dir = results_dir / 'checkpoints'
    log_dir = results_dir / 'logs'
    vis_dir = results_dir / 'visualizations'
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(
        name='training',
        log_file=str(log_dir / 'training.log')
    )
    
    logger.info("="*50)
    logger.info("Industrial Defect Detection Training")
    logger.info("="*50)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Device: {device}")
    
    # Create data loaders
    logger.info("Loading data...")
    train_loader, val_loader = get_data_loaders(
        root_dir=config['data']['root_dir'],
        category=config['data']['category'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        image_size=tuple(config['data']['image_size'])
    )
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    logger.info(f"Creating model: {config['model']['architecture']}")
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
    
    from src.utils.helpers import count_parameters
    logger.info(f"Model parameters: {count_parameters(model):,}")
    
    # Create loss function
    criterion = CombinedLoss(
        bce_weight=config['training']['bce_weight'],
        dice_weight=config['training']['dice_weight']
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Create metric logger
    metric_logger = MetricLogger(log_dir=str(log_dir))
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience'],
        mode='max'
    )
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        from src.utils.helpers import load_checkpoint
        checkpoint = load_checkpoint(
            model=model,
            checkpoint_path=args.resume,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    logger.info("Starting training...")
    best_dice = 0.0
    
    for epoch in range(start_epoch, config['training']['epochs']):
        logger.info(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        # Train
        train_loss, train_dice, train_iou = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch+1,
            logger=logger
        )
        
        # Validate
        val_loss, val_dice, val_iou = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch+1,
            logger=logger,
            save_vis=(epoch % 5 == 0),
            vis_dir=str(vis_dir)
        )
        
        # Update learning rate
        scheduler.step(val_dice)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        metric_logger.log_epoch({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_dice': train_dice,
            'val_dice': val_dice,
            'train_iou': train_iou,
            'val_iou': val_iou,
            'learning_rate': current_lr
        })
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={'val_dice': val_dice, 'val_iou': val_iou},
                save_path=str(checkpoint_dir / 'best_model.pth'),
                scheduler=scheduler
            )
            logger.info(f"Best model saved with Dice: {best_dice:.4f}")
        
        # Save latest checkpoint
        if (epoch + 1) % config['training']['save_frequency'] == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={'val_dice': val_dice, 'val_iou': val_iou},
                save_path=str(checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'),
                scheduler=scheduler
            )
        
        # Early stopping
        if early_stopping(val_dice):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Save final model
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        metrics={'val_dice': val_dice, 'val_iou': val_iou},
        save_path=str(checkpoint_dir / 'final_model.pth'),
        scheduler=scheduler
    )
    
    # Save metrics and plot training history
    metric_logger.save()
    metric_logger.print_summary()
    
    plot_training_history(
        history=metric_logger.metrics,
        save_path=str(vis_dir / 'training_history.png')
    )
    
    logger.info("\nTraining completed!")
    logger.info(f"Best Dice score: {best_dice:.4f}")
    logger.info(f"Results saved to: {results_dir}")


if __name__ == '__main__':
    main()
