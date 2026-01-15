#!/usr/bin/env python3
"""
Training script for VAE Predictor with Classification Auxiliary Loss

This version adds a classification loss to directly optimize the downstream task:
    Total Loss = MSE(z_pred, z_target) + λ * CrossEntropy(classifier(z_pred), true_label)

This ensures the predicted latents maintain semantic information for classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from datetime import datetime

from predictor.core.vae_predictor import VAEPredictor, TrajectoryDataset, save_model
from lane_classifier.latent_classifier import get_latent_classifier
from lane_classifier.dataset_visual import detect_red_line_position, get_visual_label


def get_visual_labels_from_frames(frames, device):
    """
    Extract visual labels from frames
    
    Args:
        frames: (B, 3, H, W) tensor in [0, 1]
        device: torch device
    
    Returns:
        labels: (B,) tensor of labels (0=Left, 1=Right)
    """
    B = frames.size(0)
    labels = []
    
    frames_np = frames.cpu().numpy()
    for i in range(B):
        # Convert to uint8 for OpenCV
        img = frames_np[i].transpose(1, 2, 0)  # (H, W, 3)
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Detect red line and get label
        red_x = detect_red_line_position(img_uint8)
        label = get_visual_label(red_x)
        labels.append(label)
    
    return torch.tensor(labels, dtype=torch.long, device=device)


def train_epoch_with_cls(model, classifier, dataloader, optimizer, device, 
                          lambda_cls=0.1, use_actions=False, teacher_forcing_prob=1.0):
    """
    Train one epoch with classification auxiliary loss
    
    Args:
        model: VAEPredictor
        classifier: Latent classifier
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device
        lambda_cls: Weight for classification loss
        use_actions: Whether to use actions
        teacher_forcing_prob: Teacher forcing probability
    
    Returns:
        Dictionary with loss statistics
    """
    model.train()
    classifier.eval()  # Keep classifier frozen
    
    total_loss = 0.0
    total_mse = 0.0
    total_cls = 0.0
    num_batches = 0
    
    for batch in dataloader:
        input_frames = batch['input_frames'].to(device)  # (B, T_in, 3, H, W)
        target_frames = batch['target_frames'].to(device)  # (B, T_tgt, 3, H, W)
        B, T_in = input_frames.shape[:2]
        T_tgt = target_frames.shape[1]
        
        # Get target offset
        target_offset = int(batch.get('target_offset', torch.tensor(1))[0].item()) if isinstance(batch.get('target_offset', 1), torch.Tensor) else int(batch.get('target_offset', 1))
        
        # Prepare actions if needed
        actions_seq = None
        if use_actions and 'actions' in batch:
            actions_full = batch['actions'].to(device)
            actions_seq = actions_full[:, 0:T_in, :]
        
        optimizer.zero_grad()
        
        # Encode input and target sequences
        with torch.no_grad():
            # Input frames
            in_flat = input_frames.reshape(B * T_in, *input_frames.shape[2:])
            mu_in, _ = model.encode(in_flat)
            z_input = mu_in
            z_input = z_input.reshape(B, T_in, *z_input.shape[1:])
            
            # Target frames
            tf_flat = target_frames.reshape(B * T_tgt, *target_frames.shape[2:])
            mu_t, _ = model.encode(tf_flat)
            z_target = mu_t.reshape(B, T_tgt, *mu_t.shape[1:])
        
        # Predict using teacher forcing
        teacher_forcing = (target_offset == 1 and T_tgt == T_in)
        
        if teacher_forcing:
            if teacher_forcing_prob >= 1.0:
                z_pred_seq = model.predict_teacher_forcing(z_input, actions_seq)
            else:
                z_pred_seq = model.predict_scheduled_sampling(
                    z_input, actions_seq, teacher_forcing_prob=teacher_forcing_prob
                )
            z_target_seq = z_target[:, 1:, ...]
        else:
            # Rollout mode
            z_pred_seq = model.rollout_from_context(
                z_context=z_input,
                steps=T_tgt,
                a_full=actions_seq,
                context_action_len=max(0, T_in - 1),
                start_action_index=max(0, target_offset - 1),
            )
            z_target_seq = z_target
        
        # 1. MSE Loss in latent space
        z_pred_flat = z_pred_seq.reshape(-1, z_pred_seq.shape[2] * z_pred_seq.shape[3] * z_pred_seq.shape[4])
        z_target_flat = z_target_seq.reshape(-1, z_target_seq.shape[2] * z_target_seq.shape[3] * z_target_seq.shape[4])
        mse_loss = F.mse_loss(z_pred_flat, z_target_flat)
        
        # 2. Classification Loss
        # Extract labels from target frames
        # Use the last predicted frame for classification
        z_pred_last = z_pred_seq[:, -1, ...]  # (B, C, H, W)
        target_frame_last = target_frames[:, -1, ...]  # (B, 3, H, W)
        
        # Get visual labels from target frames
        true_labels = get_visual_labels_from_frames(target_frame_last, device)  # (B,)
        
        # Classify predicted latent
        with torch.no_grad():
            # Don't backprop through classifier
            cls_logits = classifier(z_pred_last)
        
        # But we do want to backprop the classification loss to the predictor
        # So we need to compute it with predicted latents
        cls_logits_for_loss = classifier(z_pred_last)
        cls_loss = F.cross_entropy(cls_logits_for_loss, true_labels)
        
        # Total loss
        loss = mse_loss + lambda_cls * cls_loss
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Stats
        total_loss += loss.item()
        total_mse += mse_loss.item()
        total_cls += cls_loss.item()
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'mse': total_mse / num_batches,
        'cls': total_cls / num_batches,
    }


def validate_epoch_with_cls(model, classifier, dataloader, device, lambda_cls=0.1, use_actions=False):
    """Validate with classification loss"""
    model.eval()
    classifier.eval()
    
    total_loss = 0.0
    total_mse = 0.0
    total_cls = 0.0
    total_correct = 0
    total_samples = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_frames = batch['input_frames'].to(device)
            target_frames = batch['target_frames'].to(device)
            B, T_in = input_frames.shape[:2]
            T_tgt = target_frames.shape[1]
            
            target_offset = int(batch.get('target_offset', torch.tensor(1))[0].item()) if isinstance(batch.get('target_offset', 1), torch.Tensor) else int(batch.get('target_offset', 1))
            
            actions_seq = None
            if use_actions and 'actions' in batch:
                actions_full = batch['actions'].to(device)
                actions_seq = actions_full[:, 0:T_in, :]
            
            # Encode
            in_flat = input_frames.reshape(B * T_in, *input_frames.shape[2:])
            mu_in, _ = model.encode(in_flat)
            z_input = mu_in.reshape(B, T_in, *mu_in.shape[1:])
            
            tf_flat = target_frames.reshape(B * T_tgt, *target_frames.shape[2:])
            mu_t, _ = model.encode(tf_flat)
            z_target = mu_t.reshape(B, T_tgt, *mu_t.shape[1:])
            
            # Predict
            teacher_forcing = (target_offset == 1 and T_tgt == T_in)
            if teacher_forcing:
                z_pred_seq = model.predict_teacher_forcing(z_input, actions_seq)
                z_target_seq = z_target[:, 1:, ...]
            else:
                z_pred_seq = model.rollout_from_context(
                    z_context=z_input, steps=T_tgt, a_full=actions_seq,
                    context_action_len=max(0, T_in - 1),
                    start_action_index=max(0, target_offset - 1),
                )
                z_target_seq = z_target
            
            # MSE loss
            z_pred_flat = z_pred_seq.reshape(-1, z_pred_seq.shape[2] * z_pred_seq.shape[3] * z_pred_seq.shape[4])
            z_target_flat = z_target_seq.reshape(-1, z_target_seq.shape[2] * z_target_seq.shape[3] * z_target_seq.shape[4])
            mse_loss = F.mse_loss(z_pred_flat, z_target_flat)
            
            # Classification loss
            z_pred_last = z_pred_seq[:, -1, ...]
            target_frame_last = target_frames[:, -1, ...]
            true_labels = get_visual_labels_from_frames(target_frame_last, device)
            
            cls_logits = classifier(z_pred_last)
            cls_loss = F.cross_entropy(cls_logits, true_labels)
            
            # Accuracy
            pred_labels = cls_logits.argmax(1)
            correct = (pred_labels == true_labels).sum().item()
            
            loss = mse_loss + lambda_cls * cls_loss
            
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_cls += cls_loss.item()
            total_correct += correct
            total_samples += B
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'mse': total_mse / num_batches,
        'cls': total_cls / num_batches,
        'accuracy': 100.0 * total_correct / total_samples,
    }


def main():
    parser = argparse.ArgumentParser(description='Train VAE Predictor with Classification Loss')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='npz_data')
    parser.add_argument('--npz_files', nargs='+', default=['traj1_64x64.npz', 'traj2_64x64.npz'])
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--input_length', type=int, default=15)
    parser.add_argument('--target_length', type=int, default=15)
    parser.add_argument('--target_offset', type=int, default=1)
    
    # Model
    parser.add_argument('--vae_model_path', type=str, default='vae_recon/best_model.pt')
    parser.add_argument('--predictor_checkpoint', type=str, default=None,
                       help='Resume from predictor checkpoint')
    parser.add_argument('--classifier_path', type=str, default='lane_classifier/latent_classifier_checkpoints/best_model.pt')
    parser.add_argument('--predictor', type=str, default='lstm')
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--residual_prediction', action='store_true', default=True)
    parser.add_argument('--use_actions', action='store_true', default=True)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_cls', type=float, default=0.5,
                       help='Weight for classification loss (default: 0.5)')
    parser.add_argument('--teacher_forcing_prob', type=float, default=1.0)
    parser.add_argument('--val_split', type=float, default=0.15)
    
    # Output
    parser.add_argument('--save_dir', type=str, default='predictor/checkpoints_with_cls')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("="*70)
    print("Training VAE Predictor with Classification Auxiliary Loss")
    print("="*70)
    print(f"Device: {device}")
    print(f"Lambda_cls: {args.lambda_cls}")
    
    # Load VAE
    print(f"\n[1/4] Loading VAE...")
    vae_predictor = VAEPredictor(
        latent_dim=64,
        image_size=64,
        channels=3,
        action_dim=2,
        predictor_type=args.predictor,
        hidden_size=args.hidden_size,
        residual_prediction=args.residual_prediction,
        vae_model_path=args.vae_model_path,
        freeze_vae=True
    ).to(device)
    
    # Load or resume predictor
    start_epoch = 0
    best_val_loss = float('inf')
    if args.predictor_checkpoint:
        print(f"Resuming from {args.predictor_checkpoint}...")
        ckpt = torch.load(args.predictor_checkpoint, map_location=device, weights_only=False)
        vae_predictor.load_state_dict(ckpt['model_state_dict'], strict=False)
        start_epoch = ckpt.get('epoch', 0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
    
    # Load classifier
    print(f"[2/4] Loading latent classifier...")
    classifier_ckpt = torch.load(args.classifier_path, map_location=device, weights_only=False)
    classifier = get_latent_classifier(
        model_type=classifier_ckpt['args'].get('model_type', 'conv'),
        latent_dim=64,
        latent_spatial_size=4,
        dropout_rate=0.0
    ).to(device)
    classifier.load_state_dict(classifier_ckpt['model_state_dict'])
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False
    
    print(f"Classifier accuracy: {classifier_ckpt.get('best_val_acc', 'N/A')}")
    
    # Load data
    print(f"[3/4] Loading data...")
    npz_paths = [os.path.join(args.data_dir, f) for f in args.npz_files]
    
    full_dataset = TrajectoryDataset(
        npz_paths=npz_paths,
        sequence_length=args.sequence_length,
        image_size=64,
        normalize=True,
        input_length=args.input_length,
        target_length=args.target_length,
        target_offset=args.target_offset
    )
    
    # Split train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)} sequences")
    print(f"Val:   {len(val_dataset)} sequences")
    
    # Optimizer
    print(f"[4/4] Setting up training...")
    optimizer = optim.Adam(vae_predictor.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    print(f"\n{'='*70}")
    print("Starting Training")
    print("="*70)
    
    train_history = {'loss': [], 'mse': [], 'cls': [], 'val_loss': [], 'val_mse': [], 'val_cls': [], 'val_acc': []}
    
    for epoch in range(start_epoch + 1, args.epochs + 1):
        # Train
        train_stats = train_epoch_with_cls(
            vae_predictor, classifier, train_loader, optimizer, device,
            lambda_cls=args.lambda_cls, use_actions=args.use_actions,
            teacher_forcing_prob=args.teacher_forcing_prob
        )
        
        # Validate
        val_stats = validate_epoch_with_cls(
            vae_predictor, classifier, val_loader, device,
            lambda_cls=args.lambda_cls, use_actions=args.use_actions
        )
        
        # Update LR
        scheduler.step(val_stats['loss'])
        
        # Log
        train_history['loss'].append(train_stats['loss'])
        train_history['mse'].append(train_stats['mse'])
        train_history['cls'].append(train_stats['cls'])
        train_history['val_loss'].append(val_stats['loss'])
        train_history['val_mse'].append(val_stats['mse'])
        train_history['val_cls'].append(val_stats['cls'])
        train_history['val_acc'].append(val_stats['accuracy'])
        
        print(f"Epoch {epoch}/{args.epochs}:")
        print(f"  Train - Loss: {train_stats['loss']:.4f}, MSE: {train_stats['mse']:.4f}, Cls: {train_stats['cls']:.4f}")
        print(f"  Val   - Loss: {val_stats['loss']:.4f}, MSE: {val_stats['mse']:.4f}, Cls: {val_stats['cls']:.4f}, Acc: {val_stats['accuracy']:.2f}%")
        
        # Save best model
        if val_stats['loss'] < best_val_loss:
            best_val_loss = val_stats['loss']
            save_path = os.path.join(args.save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae_predictor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_acc': val_stats['accuracy'],
                'train_history': train_history,
                'args': vars(args)
            }, save_path)
            print(f"  [*] Best model saved! (loss: {best_val_loss:.4f}, acc: {val_stats['accuracy']:.2f}%)")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae_predictor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_history': train_history,
                'args': vars(args)
            }, save_path)
    
    # Save final model
    final_path = os.path.join(args.save_dir, 'final_model.pt')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': vae_predictor.state_dict(),
        'train_history': train_history,
        'args': vars(args)
    }, final_path)
    
    print(f"\n{'='*70}")
    print("Training Complete!")
    print("="*70)
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Models saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
