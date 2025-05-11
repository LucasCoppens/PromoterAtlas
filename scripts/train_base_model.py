#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import os
import argparse
import random

from promoter_atlas.data.sequence_dataset import SequenceDataset
from promoter_atlas.models.dna_transformer import DNATransformer

def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def create_batch_masks(sequence: torch.Tensor, config: dict) -> torch.Tensor:
    """Create randomized masks for a batch of sequences using vectorized operations."""
    batch_size, channels, seq_length = sequence.size()
    device = sequence.device
    
    # Initialize mask tensor with ones as float
    mask = torch.ones_like(sequence, dtype=torch.float32, device=device)
    positions = torch.arange(seq_length, device=device).view(1, 1, -1)
    
    if config['n_window_masks'] > 0:
        window_starts = torch.randint(0, seq_length - config['window_size'] + 1, 
                                    (batch_size, config['n_window_masks']), device=device)
        
        for i in range(config['n_window_masks']):
            start = window_starts[:, i:i+1].unsqueeze(-1)
            # Create window mask as float
            window_mask = ((positions >= start) & 
                         (positions < start + config['window_size'])).float()
            # Use multiplication instead of bitwise operations
            mask = mask * (1 - window_mask)
    
    if config['n_point_masks'] > 0:
        point_positions = torch.randint(0, seq_length, 
                                      (batch_size * config['n_point_masks'],), device=device)
        batch_indices = torch.arange(batch_size, device=device).repeat_interleave(config['n_point_masks'])
        # Create point mask as float
        point_mask = torch.zeros((batch_size, 1, seq_length), 
                               dtype=torch.float32, device=device)
        point_mask[batch_indices, 0, point_positions] = 1.0
        # Use multiplication instead of bitwise operations
        mask = mask * (1 - point_mask)
    
    return mask

def train_epoch(model, loader, optimizer, criterion, config, scaler):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    total_batches = len(loader)
    
    for i, batch in enumerate(loader):
        if i%100 == 0:
            t = datetime.now().strftime('%Y%m%d_%H%M%S')
            print(f"{t}: batch{i}")
        sequence = batch['sequence'].to(config['device'])
        mask = create_batch_masks(sequence, config)
        masked_sequence = sequence * mask
        
        # Mixed precision training
        with torch.cuda.amp.autocast():
            output, attention_scores = model(masked_sequence)
            loss = criterion(output * (1 - mask), sequence * (1 - mask))
        
        # Scale loss and backprop
        scaler.scale(loss).backward()
        
        if ((i + 1) % config['gradient_accumulation'] == 0) or (i + 1 == total_batches):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item()
    
    return total_loss / total_batches

def validate(model, loader, criterion, config):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            sequence = batch['sequence'].to(config['device'])
            mask = create_batch_masks(sequence, config)
            masked_sequence = sequence * mask
            
            # Mixed precision inference
            with torch.cuda.amp.autocast():
                output, attention_scores = model(masked_sequence)
                loss = criterion(output * (1 - mask), sequence * (1 - mask))
            
            total_loss += loss.item()
    
    return total_loss / len(loader)

def plot_losses(train_losses, val_losses, save_path):
    """Plot training and validation losses."""
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_window_masks', type=int, default=0,
                       help='Number of window masks per sequence')
    parser.add_argument('--window_size', type=int, default=6,
                       help='Size of each window mask')
    parser.add_argument('--n_point_masks', type=int, default=20,
                       help='Number of point masks per sequence')
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='Batch size')
    parser.add_argument('--data_path', type=str, default='data/processed/sequence_dataset.h5',
                       help='Path to the HDF5 dataset file')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') + f"_wm{args.n_window_masks}_pm{args.n_point_masks}"
    
    # Configure environment to allow expandable memory segments
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    config = {
        'data_path': Path(args.data_path),
        'output_dir': Path("data/training_runs") / timestamp / "output",
        'checkpoint_dir':  Path("data/training_runs") / timestamp / "checkpoint",
        'log_dir':  Path("data/training_runs") / timestamp / "log",    
        'subset_fraction': 1,
        'batch_size': args.batch_size,
        'gradient_accumulation': 1,
        'num_workers': 16,
        'learning_rate': 1e-3,
        'n_epochs': 2000,
        'patience': 15,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'n_window_masks': args.n_window_masks,
        'window_size': args.window_size,
        'n_point_masks': args.n_point_masks,
        'architecture': DNATransformer.__name__
    }
    
    # Create directories and save config
    for dir_path in [config['output_dir'], config['checkpoint_dir'], config['log_dir']]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    with open(config['output_dir'] / 'config.json', 'w') as f:
        config_save = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
        json.dump(config_save, f, indent=4)
    
    set_seeds()
    
    # Load dataset and background frequencies
    dataset = SequenceDataset(config['data_path'])
    
    train_size = int(0.9 * len(dataset) * config['subset_fraction'])
    val_size = int(0.1 * len(dataset) * config['subset_fraction'])
    
    indices = np.random.choice(len(dataset), train_size + val_size, replace=False)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    train_loader = DataLoader(
        train_subset, 
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=True,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=False,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    print(f"Training on {len(train_subset)} sequences")
    print(f"Validating on {len(val_subset)} sequences")
    
    model = DNATransformer().to(config['device'])
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    patience_counter = 0
    
    start_time = datetime.now()
    
    try:
        for epoch in range(config['n_epochs']):
            print(f"\nEpoch {epoch+1}/{config['n_epochs']}")
                
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, config, scaler
            )
            val_loss = validate(
                model, val_loader, criterion, config
            )
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            
            with open(config['log_dir'] / f'metrics_epoch_{epoch}.json', 'w') as f:
                json.dump(metrics, f, indent=4)
            
            print(f"Epoch {epoch}: train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                checkpoint_path = config['checkpoint_dir'] / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': config
                }, checkpoint_path)
                print(f"Saved best model with validation loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= config['patience']:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            lr_scheduler.step(val_loss)
            
            plot_losses(
                train_losses, val_losses,
                config['output_dir'] / 'loss_plot.png'
            )

        training_time = datetime.now() - start_time
        
        final_metrics = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'stopped_epoch': epoch + 1,
            'final_lr': optimizer.param_groups[0]['lr'],
            'training_time': str(training_time),
            'total_parameters': sum(p.numel() for p in model.parameters())
        }
        
        with open(config['output_dir'] / 'final_metrics.json', 'w') as f:
            json.dump(final_metrics, f, indent=4)
            
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise e

if __name__ == "__main__":
    main()