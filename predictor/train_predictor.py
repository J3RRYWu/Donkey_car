#!/usr/bin/env python3
"""
Training script for VAE Predictor
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from datetime import datetime
import json

from torch import amp
from vae_predictor import (
    VAEPredictor, TrajectoryDataset, train_epoch, validate_epoch,
    save_model, predictor_loss
)

def _parse_open_loop_schedule(spec: str):
    """
    Parse schedule spec like: "0:5,10:10,20:20,40:50"
    Returns list of (start_epoch_inclusive:int, steps:int) sorted by start_epoch.
    """
    items = []
    if not spec:
        return items
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Bad open_loop_schedule chunk: {chunk} (expected 'epoch:steps')")
        e_s, k_s = chunk.split(":", 1)
        e = int(e_s.strip())
        k = int(k_s.strip())
        items.append((e, k))
    items.sort(key=lambda x: x[0])
    return items

def _open_loop_steps_for_epoch(epoch0: int, schedule_items, default_steps: int) -> int:
    """epoch0 is 0-based epoch index."""
    steps = int(default_steps)
    for start_e, k in schedule_items:
        if epoch0 >= start_e:
            steps = int(k)
        else:
            break
    return steps

def _int_schedule_for_epoch(epoch0: int, schedule_items, default_value: int) -> int:
    """Generic schedule helper: pick value from (start_epoch,value) items."""
    v = int(default_value)
    for start_e, k in schedule_items:
        if epoch0 >= start_e:
            v = int(k)
        else:
            break
    return v


def main():
    parser = argparse.ArgumentParser(description='Train VAE Predictor')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='../npz_transfer',
                       help='Directory containing NPZ files')
    parser.add_argument('--npz_files', nargs='+', default=['traj1.npz', 'traj2.npz'],
                       help='NPZ files for training')
    parser.add_argument('--sequence_length', type=int, default=16,
                       help='Sequence length')
    parser.add_argument('--sequence_length_curriculum', action='store_true',
                       help='Enable curriculum schedule for sequence_length (rebuild dataset/dataloader at stage boundaries)')
    parser.add_argument('--sequence_length_schedule', type=str, default='0:32,20:64',
                       help="Sequence-length schedule as 'epoch:seq_len,...' (0-based epoch). Example: 0:32,20:64")
    
    # Model arguments
    parser.add_argument('--latent_dim', type=int, default=256,
                       help='Latent dimension (will be overridden if VAE is loaded)')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Image size (will be auto-detected from VAE)')
    parser.add_argument('--predictor', type=str, default='lstm', choices=['lstm', 'gru'],
                       help='Predictor type: lstm or gru')
    parser.add_argument('--hidden_size', type=int, default=256,
                       help='Hidden size for LSTM/GRU')
    parser.add_argument('--residual_prediction', action='store_true',
                       help='Use residual prediction')
    parser.add_argument('--vae_model_path', type=str, default=None,
                       help='Path to VAE model checkpoint')
    parser.add_argument('--freeze_vae', action='store_true', default=True,
                       help='Freeze VAE encoder/decoder')
    
    # Action arguments
    parser.add_argument('--use_actions', action='store_true',
                       help='Use action inputs')
    parser.add_argument('--action_dropout_prob', type=float, default=0.0,
                       help='Probability of dropping out actions during training')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=40,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--beta', type=float, default=1.0,
                       help='KL loss weight (not used for predictor, kept for compatibility)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers')
    
    # Checkpoint arguments
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Other arguments
    parser.add_argument('--open_loop_steps', type=int, default=0,
                       help='Number of open-loop steps during training (0 = closed loop)')
    parser.add_argument('--open_loop_weight', type=float, default=0.0,
                       help='Weight for open-loop loss')
    parser.add_argument('--open_loop_curriculum', action='store_true',
                       help='Enable curriculum schedule for open_loop_steps based on epoch')
    parser.add_argument('--open_loop_schedule', type=str, default='0:5,10:10,20:20,40:50',
                       help="Open-loop curriculum schedule as 'epoch:steps,...' (0-based epoch). Example: 0:5,10:10,20:20,40:50")
    parser.add_argument('--commit_weight', type=float, default=0.0,
                       help='Commitment loss weight (not used for predictor)')
    parser.add_argument('--commit_eta', type=float, default=0.25,
                       help='Commitment loss eta (not used)')
    parser.add_argument('--img_recon_weight', type=float, default=0.0,
                       help='Image reconstruction weight (not used for predictor)')
    parser.add_argument('--img_recon_size', type=int, default=224,
                       help='Image reconstruction size (not used)')
    parser.add_argument('--freeze_encoder_epochs', type=int, default=0,
                       help='Number of epochs to freeze encoder (not used for predictor)')
    parser.add_argument('--kl_warmup_epochs', type=int, default=0,
                       help='KL warmup epochs (not used)')
    parser.add_argument('--kl_warmup_start', type=float, default=0.0,
                       help='KL warmup start value (not used)')
    parser.add_argument('--free_bits', type=float, default=0.0,
                       help='Free bits for KL (not used)')
    parser.add_argument('--input_noise_std', type=float, default=0.0,
                       help='Input noise std (not used)')
    parser.add_argument('--target_jitter', type=float, default=0.0,
                       help='Target jitter (not used)')
    parser.add_argument('--detach_target', action='store_true',
                       help='Detach target (not used)')
    parser.add_argument('--use_amp', action='store_true',
                       help='Use automatic mixed precision')
    parser.add_argument('--amp_disable_open_loop_threshold', type=int, default=30,
                       help='If open_loop_steps >= this, disable AMP for stability (default: 30)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load dataset
    npz_paths = [os.path.join(args.data_dir, f) for f in args.npz_files]
    print(f"Loading data from: {npz_paths}")
    
    # Auto-detect image size from VAE checkpoint if provided
    image_size = args.image_size
    if args.vae_model_path and os.path.exists(args.vae_model_path):
        checkpoint = torch.load(args.vae_model_path, map_location='cpu')
        if 'args' in checkpoint and 'image_size' in checkpoint['args']:
            image_size = checkpoint['args']['image_size']
            print(f"Auto-detected image size from VAE checkpoint: {image_size}")
        elif 'image_size' in checkpoint:
            image_size = checkpoint['image_size']
            print(f"Auto-detected image size from VAE checkpoint: {image_size}")
    
    def build_loaders(seq_len: int):
        dataset_local = TrajectoryDataset(
            npz_paths=npz_paths,
            sequence_length=int(seq_len),
            image_size=image_size,
            normalize=True
        )

        total_size = len(dataset_local)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset_local, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader_local = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )

        val_loader_local = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )

        print(f"Dataset(seq_len={seq_len}) size: {total_size}")
        print(f"  Train: {train_size}, Val: {val_size}")
        print(f"Number of batches: Train={len(train_loader_local)}, Val={len(val_loader_local)}")

        return dataset_local, train_loader_local, val_loader_local

    # Initial dataset/loaders
    dataset, train_loader, val_loader = build_loaders(args.sequence_length)
    
    # Determine action dimension
    action_dim = 0
    if args.use_actions:
        # Check if dataset has actions
        sample = dataset[0]
        if 'actions' in sample:
            action_dim = sample['actions'].shape[-1]
            print(f"Detected action dimension: {action_dim}")
        else:
            print("Warning: --use_actions specified but dataset has no actions, setting action_dim=0")
            action_dim = 0
    
    # Create model
    model = VAEPredictor(
        latent_dim=args.latent_dim,
        image_size=args.image_size,
        channels=3,
        action_dim=action_dim,
        predictor_type=args.predictor,
        hidden_size=args.hidden_size,
        residual_prediction=args.residual_prediction,
        vae_model_path=args.vae_model_path,
        freeze_vae=args.freeze_vae
    )
    model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Setup optimizer - ONLY trainable parameters (LSTM/predictor, NOT frozen VAE)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    num_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable parameters: {num_trainable:,}")
    print(f"Frozen parameters (VAE): {num_frozen:,}")
    
    # Setup optimizer with ONLY trainable parameters
    optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=1e-5)
    
    # AMP scaler
    scaler = amp.GradScaler(enabled=(device.type == 'cuda' and args.use_amp))
    
    # Initialize optional attributes for action dropout
    model._act_drop = getattr(model, "_act_drop", 0.0)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')
    train_history = {
        'loss': [], 'recon_loss': [], 'kl_loss': [],
        'mu_mean': [], 'mu_std': [], 'logvar_mean': [], 'logvar_std': [],
        'open_loop_loss': [],
        'val_loss': [], 'val_mse': []
    }
    
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        train_history = checkpoint.get('train_history', train_history)
    
    # LR scheduler on validation loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)

    schedule_items = []
    if args.open_loop_curriculum:
        schedule_items = _parse_open_loop_schedule(args.open_loop_schedule)
        print(f"Open-loop curriculum enabled: {schedule_items}")

    seq_schedule_items = []
    if args.sequence_length_curriculum:
        seq_schedule_items = _parse_open_loop_schedule(args.sequence_length_schedule)
        print(f"Sequence-length curriculum enabled: {seq_schedule_items}")

    current_seq_len = int(args.sequence_length)
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)

        # Curriculum: possibly update sequence_length (rebuild dataset/loaders)
        if args.sequence_length_curriculum and seq_schedule_items:
            desired_seq_len = _int_schedule_for_epoch(epoch, seq_schedule_items, current_seq_len)
            if int(desired_seq_len) != int(current_seq_len):
                current_seq_len = int(desired_seq_len)
                print(f"\n[Curriculum] Switching sequence_length to {current_seq_len} at epoch {epoch+1}")
                dataset, train_loader, val_loader = build_loaders(current_seq_len)
        
        # Dynamic freeze/unfreeze encoder (only for non-VAE models)
        if model.vae_encoder is None:
            # Only freeze encoder if using default encoder (not VAE)
            freeze_enc = epoch < args.freeze_encoder_epochs
            for p in model.encoder.parameters():
                p.requires_grad_(not freeze_enc)
            for p in model.fc_mu.parameters():
                p.requires_grad_(not freeze_enc)
            for p in model.fc_logvar.parameters():
                p.requires_grad_(not freeze_enc)
        else:
            # VAE encoder/decoder are already frozen, skip
            freeze_enc = True
        
        # Set action dropout
        model._act_drop = max(0.0, min(1.0, args.action_dropout_prob))

        # Curriculum open-loop steps
        current_open_loop_steps = int(args.open_loop_steps)
        if args.open_loop_curriculum:
            current_open_loop_steps = _open_loop_steps_for_epoch(epoch, schedule_items, args.open_loop_steps)
        if current_open_loop_steps > 0:
            print(f"Open-loop steps this epoch: {current_open_loop_steps} (weight={args.open_loop_weight})")

        # AMP stability guard for long open-loop
        scaler_this_epoch = None
        if args.use_amp and device.type == "cuda":
            if current_open_loop_steps >= int(args.amp_disable_open_loop_threshold):
                scaler_this_epoch = None
                print(f"[AMP] Disabled for this epoch because open_loop_steps={current_open_loop_steps} >= {args.amp_disable_open_loop_threshold}")
            else:
                scaler_this_epoch = scaler
        
        # Train
        if model.vae_encoder is not None:
            # VAE is frozen: use MSE loss + open-loop rollout loss
            train_metrics = train_epoch(
                model, train_loader, optimizer, device,
                beta=0.0,  # No KL loss
                scaler=scaler_this_epoch,
                input_noise_std=args.input_noise_std,
                free_bits_nats=0.0,  # No free bits
                use_actions=args.use_actions,
                target_jitter_scale=0.0,  # No jitter
                detach_target=True,  # Always detach when frozen
                open_loop_steps=current_open_loop_steps,  # Enable open-loop rollout (curriculum)
                open_loop_weight=args.open_loop_weight
            )
        else:
            # Default encoder: use full VAE loss with all options
            train_metrics = train_epoch(
                model, train_loader, optimizer, device,
                beta=args.beta, scaler=scaler_this_epoch,
                input_noise_std=args.input_noise_std,
                free_bits_nats=args.free_bits,
                use_actions=args.use_actions,
                target_jitter_scale=args.target_jitter,
                detach_target=args.detach_target,
                open_loop_steps=current_open_loop_steps,
                open_loop_weight=args.open_loop_weight
            )
        
        # Store metrics
        for key, value in train_metrics.items():
            if key not in train_history:
                train_history[key] = []
            train_history[key].append(value)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, device)
        train_history['val_loss'].append(val_metrics['loss'])
        train_history['val_mse'].append(val_metrics['mse'])
        
        # Print metrics
        print(f"Loss: {train_metrics['loss']:.6f}")
        print(f"Reconstruction Loss (MSE z_pred vs z_target): {train_metrics['recon_loss']:.6f}")
        if 'open_loop_loss' in train_metrics and train_metrics['open_loop_loss'] > 0:
            print(f"Open-loop Rollout Loss: {train_metrics['open_loop_loss']:.6f}")
        print(f"KL Loss: {train_metrics['kl_loss']:.6f}")
        print(f"Val Loss: {val_metrics['loss']:.6f}, MSE: {val_metrics['mse']:.6f}")
        print(f"Enc mu(mean/std): {train_metrics['mu_mean']:.4f} / {train_metrics['mu_std']:.4f}  |  logvar(mean/std): {train_metrics['logvar_mean']:.4f} / {train_metrics['logvar_std']:.4f}")
        
        # Step LR scheduler on validation loss
        scheduler.step(val_metrics['loss'])

        # Save checkpoint (use validation loss for best)
        is_best = val_metrics['loss'] < best_loss
        if is_best:
            best_loss = val_metrics['loss']
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pt')
        save_model(
            model, checkpoint_path, epoch=epoch, optimizer=optimizer,
            best_loss=best_loss, train_history=train_history, args=vars(args)
        )
        
        # Save best model
        if is_best:
            best_model_path = os.path.join(args.save_dir, 'best_model.pt')
            save_model(
                model, best_model_path, epoch=epoch, optimizer=optimizer,
                best_loss=best_loss, train_history=train_history, args=vars(args)
            )
            print(f"  -> New best model saved (loss: {best_loss:.6f})")
        
        # Save predictor-only checkpoint (without VAE weights)
        predictor_state = {}
        for key, value in model.state_dict().items():
            if not key.startswith('vae_encoder.') and not key.startswith('vae_decoder.'):
                predictor_state[key] = value
        
        predictor_path = os.path.join(args.save_dir, f'vae_predictor_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': predictor_state,
            'args': vars(args),
            'best_loss': best_loss,
        }, predictor_path)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best validation loss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {args.save_dir}")


if __name__ == "__main__":
    main()

