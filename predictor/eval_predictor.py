#!/usr/bin/env python3
"""
Evaluation script for VAE Predictor
Three checks:
1. Baseline vs LSTM one-step prediction
2. Multi-step open-loop rollout
3. Visualize: GT, VAE reconstruction, LSTM prediction
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("Warning: imageio not available, video generation will be disabled")

# Optional: Pillow for video text overlay
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from vae_predictor import VAEPredictor, TrajectoryDataset, load_model


def _overlay_sbs_labels(frame_u8: np.ndarray, left_label: str = "GT", right_label: str = "PR") -> np.ndarray:
    """Overlay 'GT'/'PR' labels on a side-by-side uint8 frame (H, W, 3).
    If Pillow is unavailable, returns the frame unchanged.
    """
    if not HAS_PIL:
        return frame_u8
    try:
        h, w, _ = frame_u8.shape
        img = Image.fromarray(frame_u8, mode="RGB")
        draw = ImageDraw.Draw(img)
        # Try a default font; fall back to PIL default
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        # Put labels near top-left of each half, with a dark background box.
        half = w // 2
        pad = 6
        y = 6

        def draw_label(x: int, text: str):
            bbox = draw.textbbox((x + pad, y), text, font=font)
            # Background rectangle slightly padded
            bx0, by0, bx1, by1 = bbox
            bg = (0, 0, 0)
            fg = (255, 255, 255)
            draw.rectangle((bx0 - 4, by0 - 2, bx1 + 4, by1 + 2), fill=bg)
            draw.text((x + pad, y), text, fill=fg, font=font)

        draw_label(0, left_label)
        draw_label(half, right_label)

        return np.array(img)
    except Exception:
        return frame_u8


def compute_baseline_vs_lstm(model: VAEPredictor, dataloader: DataLoader, device: torch.device, 
                              max_batches: int = None) -> Dict:
    """Check 1: Compare baseline (z_{t+1} ≈ z_t) vs LSTM one-step prediction"""
    print("\n" + "="*70)
    print("Check 1: Baseline vs LSTM One-step Prediction")
    print("="*70)
    
    total_batches = len(dataloader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
        print(f"Evaluating {total_batches} batches (limited from {len(dataloader)} total)")
    else:
        print(f"Evaluating {total_batches} batches")
    
    model.eval()
    total_baseline_mse = 0.0
    total_lstm_mse = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            input_frames = batch['input_frames'].to(device)  # (B, 15, 3, 64, 64)
            target_frames = batch['target_frames'].to(device)  # (B, 15, 3, 64, 64)
            B, T = input_frames.shape[:2]
            
            # Get actions if available
            actions_seq = None
            if 'actions' in batch and model.action_dim > 0:
                actions_seq = batch['actions'][:, :-1, :].to(device)  # (B, 15, action_dim)
            
            # Encode input sequence (use mu directly, no sampling)
            in_flat = input_frames.reshape(B*T, *input_frames.shape[2:])
            mu_in, _ = model.encode(in_flat)
            
            # Reshape
            if model.vae_encoder is not None:
                z_input = mu_in.reshape(B, T, *mu_in.shape[1:])  # (B, T, C, 4, 4)
            else:
                z_input = mu_in.reshape(B, T, -1)  # (B, T, D)
            
            # Encode target sequence
            target_flat = target_frames.reshape(B*T, *target_frames.shape[2:])
            mu_target, _ = model.encode(target_flat)
            
            if model.vae_encoder is not None:
                z_target = mu_target.reshape(B, T, *mu_target.shape[1:])  # (B, T, C, 4, 4)
            else:
                z_target = mu_target.reshape(B, T, -1)  # (B, T, D)
            
            # Baseline: z_pred_baseline = z_t (for each step t -> t+1)
            # z_t is at position t-1 in z_input (since input is frames 0-14, target is frames 1-15)
            baseline_mses = []
            lstm_mses = []
            
            for t in range(T):  # For each step in sequence
                # z_t (current latent)
                z_t = z_input[:, t, ...]  # (B, C, 4, 4) or (B, D)
                # z_{t+1} (target latent)
                z_tp1 = z_target[:, t, ...]  # (B, C, 4, 4) or (B, D)
                
                # Baseline: z_pred = z_t
                z_pred_baseline = z_t
                
                # LSTM prediction: predict z_{t+1} from z_t
                # Prepare input for single step prediction
                if model.vae_encoder is not None:
                    z_t_expanded = z_t.unsqueeze(1)  # (B, 1, C, 4, 4)
                else:
                    z_t_expanded = z_t.unsqueeze(1)  # (B, 1, D)
                
                # Get action for this step if available
                a_t = None
                if actions_seq is not None:
                    a_t = actions_seq[:, t:t+1, :]  # (B, 1, action_dim)
                
                # Predict
                z_pred_lstm = model.predict(z_t_expanded, a_t)  # (B, 1, C, 4, 4) or (B, 1, D)
                z_pred_lstm = z_pred_lstm.squeeze(1)  # (B, C, 4, 4) or (B, D)
                
                # Compute MSE (flatten for comparison)
                if model.vae_encoder is not None:
                    z_t_flat = z_t.reshape(B, -1)
                    z_tp1_flat = z_tp1.reshape(B, -1)
                    z_pred_baseline_flat = z_pred_baseline.reshape(B, -1)
                    z_pred_lstm_flat = z_pred_lstm.reshape(B, -1)
                else:
                    z_t_flat = z_t
                    z_tp1_flat = z_tp1
                    z_pred_baseline_flat = z_pred_baseline
                    z_pred_lstm_flat = z_pred_lstm
                
                baseline_mse = F.mse_loss(z_pred_baseline_flat, z_tp1_flat, reduction='mean').item()
                lstm_mse = F.mse_loss(z_pred_lstm_flat, z_tp1_flat, reduction='mean').item()
                
                baseline_mses.append(baseline_mse)
                lstm_mses.append(lstm_mse)
            
            total_baseline_mse += np.mean(baseline_mses)
            total_lstm_mse += np.mean(lstm_mses)
            num_samples += 1
            
            if batch_idx % 10 == 0 or (batch_idx + 1) == total_batches:
                progress = (batch_idx + 1) / total_batches * 100
                print(f"Batch {batch_idx+1}/{total_batches} ({progress:.1f}%): baseline_mse={np.mean(baseline_mses):.6f}, lstm_mse={np.mean(lstm_mses):.6f}")
    
    avg_baseline_mse = total_baseline_mse / num_samples
    avg_lstm_mse = total_lstm_mse / num_samples
    improvement = (avg_baseline_mse - avg_lstm_mse) / avg_baseline_mse * 100
    
    print(f"\nResults:")
    print(f"  Baseline MSE (z_t -> z_{{t+1}}): {avg_baseline_mse:.6f}")
    print(f"  LSTM MSE (z_pred_lstm -> z_{{t+1}}): {avg_lstm_mse:.6f}")
    print(f"  Improvement: {improvement:.2f}%")
    
    if avg_lstm_mse < avg_baseline_mse:
        print(f"  [OK] LSTM is better than baseline!")
    else:
        print(f"  [WARNING] LSTM is worse than baseline!")
    
    return {
        'baseline_mse': avg_baseline_mse,
        'lstm_mse': avg_lstm_mse,
        'improvement': improvement
    }


def compute_multi_step_rollout(model: VAEPredictor, dataloader: DataLoader, device: torch.device,
                               max_horizon: int = 15, max_batches: int = None) -> Dict:
    """Check 2: Multi-step open-loop rollout"""
    print("\n" + "="*70)
    print("Check 2: Multi-step Open-loop Rollout")
    print("="*70)
    
    total_batches = len(dataloader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
        print(f"Evaluating {total_batches} batches (limited from {len(dataloader)} total)")
    else:
        print(f"Evaluating {total_batches} batches")
    
    model.eval()
    horizon_mses = {h: [] for h in range(1, max_horizon + 1)}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            input_frames = batch['input_frames'].to(device)  # (B, 15, 3, 64, 64)
            target_frames = batch['target_frames'].to(device)  # (B, 15, 3, 64, 64)
            B, T = input_frames.shape[:2]
            
            # Get actions if available
            actions_seq = None
            if 'actions' in batch and model.action_dim > 0:
                actions_seq = batch['actions'][:, :-1, :].to(device)  # (B, 15, action_dim)
            
            # Encode first frame for rollout start
            first_frame = input_frames[:, 0, ...]  # (B, 3, 64, 64)
            mu_start, _ = model.encode(first_frame)
            
            # Encode target frames for comparison
            target_flat = target_frames.reshape(B*T, *target_frames.shape[2:])
            mu_target, _ = model.encode(target_flat)
            
            if model.vae_encoder is not None:
                z_start = mu_start  # (B, C, 4, 4)
                z_target = mu_target.reshape(B, T, *mu_target.shape[1:])  # (B, T, C, 4, 4)
            else:
                z_start = mu_start  # (B, D)
                z_target = mu_target.reshape(B, T, -1)  # (B, T, D)
            
            # Rollout: start from z_start, predict step by step
            z_current = z_start  # (B, C, 4, 4) or (B, D)
            
            for step in range(1, max_horizon + 1):
                # Prepare input for single step prediction
                if model.vae_encoder is not None:
                    z_current_expanded = z_current.unsqueeze(1)  # (B, 1, C, 4, 4)
                else:
                    z_current_expanded = z_current.unsqueeze(1)  # (B, 1, D)
                
                # Get action for this step if available
                a_step = None
                if actions_seq is not None:
                    # Use last available action if step exceeds sequence length
                    action_idx = min(step - 1, actions_seq.shape[1] - 1)
                    a_step = actions_seq[:, action_idx:action_idx+1, :]  # (B, 1, action_dim)
                
                # Predict next latent
                z_next = model.predict(z_current_expanded, a_step)  # (B, 1, C, 4, 4) or (B, 1, D)
                z_next = z_next.squeeze(1)  # (B, C, 4, 4) or (B, D)
                
                # Compare with target only if we have target frames (step <= T)
                if step <= T:
                    z_target_step = z_target[:, step-1, ...]  # (B, C, 4, 4) or (B, D)
                    
                    # Compute MSE
                    if model.vae_encoder is not None:
                        z_next_flat = z_next.reshape(B, -1)
                        z_target_step_flat = z_target_step.reshape(B, -1)
                    else:
                        z_next_flat = z_next
                        z_target_step_flat = z_target_step
                    
                    mse = F.mse_loss(z_next_flat, z_target_step_flat, reduction='mean').item()
                    horizon_mses[step].append(mse)
                
                # Use predicted latent for next step (open-loop)
                z_current = z_next
            
            if batch_idx % 10 == 0 or (batch_idx + 1) == total_batches:
                progress = (batch_idx + 1) / total_batches * 100
                print(f"Batch {batch_idx+1}/{total_batches} ({progress:.1f}%): Rollout completed")
    
    # Compute average MSE for each horizon (only where GT exists)
    avg_mses = {}
    for h in range(1, max_horizon + 1):
        if horizon_mses[h]:
            avg_mses[h] = float(np.mean(horizon_mses[h]))
        else:
            avg_mses[h] = None
    
    print(f"\nResults (MSE vs horizon):")
    for h in sorted(avg_mses.keys()):
        v = avg_mses[h]
        if v is None:
            print(f"  Horizon {h:2d}: N/A (no GT beyond sequence_length={T})")
        else:
            print(f"  Horizon {h:2d}: {v:.6f}")
    
    return avg_mses


def visualize_predictions(model: VAEPredictor, dataloader: DataLoader, device: torch.device,
                          num_samples: int = 3, rollout_steps: int = 15, save_dir: str = './eval_results'):
    """Check 3: Visualize GT, VAE reconstruction, LSTM prediction"""
    print("\n" + "="*70)
    print("Check 3: Visualize Predictions")
    print("="*70)
    
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    with torch.no_grad():
        batch = next(iter(dataloader))
        input_frames = batch['input_frames'].to(device)  # (B, 15, 3, 64, 64)
        target_frames = batch['target_frames'].to(device)  # (B, 15, 3, 64, 64)
        B = input_frames.shape[0]
        
        # Get actions if available
        actions_seq = None
        if 'actions' in batch and model.action_dim > 0:
            actions_seq = batch['actions'][:, :-1, :].to(device)  # (B, 15, action_dim)
        
        num_samples = min(num_samples, B)
        
        for sample_idx in range(num_samples):
            print(f"Processing sample {sample_idx + 1}/{num_samples}...")
            
            # Get single sample
            input_seq = input_frames[sample_idx:sample_idx+1]  # (1, 15, 3, 64, 64)
            target_seq = target_frames[sample_idx:sample_idx+1]  # (1, 15, 3, 64, 64)
            
            # Get actions for this sample
            a_seq = None
            if actions_seq is not None:
                a_seq = actions_seq[sample_idx:sample_idx+1]  # (1, 15, action_dim)
            
            # Encode first frame
            first_frame = input_seq[:, 0, ...]  # (1, 3, 64, 64)
            mu_start, _ = model.encode(first_frame)
            z_start = mu_start  # (1, C, 4, 4) or (1, D)
            
            # VAE reconstruction of first frame (for reference)
            if model.vae_encoder is not None:
                # Use full VAE forward to get proper reconstruction with skip connections
                vae = model.vae_encoder  # EnhancedVAE instance
                vae.eval()
                with torch.no_grad():
                    recon_first, _, _, _ = vae(first_frame)  # (1, 3, 64, 64)
                # Rescale from [-1,1] to [0,1] if needed
                if recon_first.min() < 0:
                    recon_first = (recon_first + 1.0) / 2.0
                recon_first = torch.clamp(recon_first, 0, 1)
            else:
                recon_first = model.decode_images(z_start)
                if recon_first.min() < 0:
                    recon_first = (recon_first + 1.0) / 2.0
                recon_first = torch.clamp(recon_first, 0, 1)
            
            # Rollout predictions
            predicted_images = []
            z_current = z_start
            
            for step in range(rollout_steps):
                # Predict next latent
                if model.vae_encoder is not None:
                    z_current_expanded = z_current.unsqueeze(1)  # (1, 1, C, 4, 4)
                else:
                    z_current_expanded = z_current.unsqueeze(1)  # (1, 1, D)
                
                # Get action if available
                a_step = None
                if a_seq is not None and a_seq.shape[1] > 0:
                    # Use last available action if step exceeds sequence length
                    action_idx = min(step, a_seq.shape[1] - 1)
                    a_step = a_seq[:, action_idx:action_idx+1, :]  # (1, 1, action_dim)
                
                # Predict
                z_next = model.predict(z_current_expanded, a_step)  # (1, 1, C, 4, 4) or (1, 1, D)
                z_next = z_next.squeeze(1)  # (1, C, 4, 4) or (1, D)
                
                # Decode to image (for visualization only)
                # Use GT frame's skip features to decode predicted latent (cheating for visualization)
                if model.vae_encoder is not None:
                    # Get skip features from corresponding GT frame
                    t_for_skip = min(step, target_seq.shape[1] - 1)
                    x_gt_t = target_seq[:, t_for_skip, ...]  # (1, 3, 64, 64)
                    with torch.no_grad():
                        _, _, skip_t = model.vae_encoder.encode(x_gt_t)  # Get encoder's skip features
                    # Decode predicted latent with GT skip features
                    pred_img = model.vae_decoder.decode(z_next, skip_t)  # Use z_next + skip_t
                else:
                    pred_img = model.decode_images(z_next)
                
                # Clamp and convert to numpy
                if pred_img.min() < 0:  # Tanh output
                    pred_img = (pred_img + 1.0) / 2.0
                pred_img = torch.clamp(pred_img, 0, 1)
                predicted_images.append(pred_img[0].cpu().numpy().transpose(1, 2, 0))  # (64, 64, 3)
                
                # Use predicted latent for next step
                z_current = z_next
            
            # Get GT images (target frames)
            gt_images = []
            for t in range(min(rollout_steps, target_seq.shape[1])):
                gt_img = target_seq[:, t, ...].cpu().numpy().transpose(0, 2, 3, 1)[0]  # (64, 64, 3)
                gt_images.append(gt_img)
            
            # Get VAE reconstructions of target frames (for comparison)
            vae_recon_images = []
            
            if model.vae_encoder is not None:
                # Use full VAE forward to get proper reconstruction with skip connections
                vae = model.vae_encoder  # EnhancedVAE instance
                vae.eval()
                with torch.no_grad():
                    # Process each target frame individually to get proper skip connections
                    for t in range(min(rollout_steps, target_seq.shape[1])):
                        x_t = target_seq[:, t, ...]  # (1, 3, 64, 64)
                        recon_t, _, _, _ = vae(x_t)  # Full forward with skip connections
                        
                        # Rescale from [-1,1] to [0,1] if needed
                        if recon_t.min() < 0:
                            recon_t = (recon_t + 1.0) / 2.0
                        recon_t = torch.clamp(recon_t, 0, 1)
                        
                        recon_img = recon_t[0].cpu().numpy().transpose(1, 2, 0)  # (64, 64, 3)
                        vae_recon_images.append(recon_img)
            else:
                # Fallback for non-VAE models
                target_flat = target_seq.reshape(-1, 3, 64, 64)
                mu_target, _ = model.encode(target_flat)
                z_target = mu_target
                recon_target = model.decode_images(z_target)  # (N, 3, 64, 64)
                
                if recon_target.min() < 0:  # Tanh output
                    recon_target = (recon_target + 1.0) / 2.0
                recon_target = torch.clamp(recon_target, 0, 1)
                
                for t in range(min(rollout_steps, target_seq.shape[1])):
                    recon_img = recon_target[t].cpu().numpy().transpose(1, 2, 0)  # (64, 64, 3)
                    vae_recon_images.append(recon_img)
            
            # Create visualization
            num_steps_to_show = min(rollout_steps, 10)  # Show first 10 steps
            fig, axes = plt.subplots(3, num_steps_to_show, figsize=(num_steps_to_show * 2, 6))
            
            if num_steps_to_show == 1:
                axes = axes.reshape(3, 1)
            
            for step in range(num_steps_to_show):
                # Row 1: GT images
                if step < len(gt_images):
                    axes[0, step].imshow(gt_images[step])
                axes[0, step].axis('off')
                if step == 0:
                    axes[0, step].set_title('GT', fontsize=10)
                
                # Row 2: VAE reconstruction
                if step < len(vae_recon_images):
                    axes[1, step].imshow(vae_recon_images[step])
                axes[1, step].axis('off')
                if step == 0:
                    axes[1, step].set_title('VAE Recon', fontsize=10)
                
                # Row 3: LSTM prediction
                if step < len(predicted_images):
                    axes[2, step].imshow(predicted_images[step])
                axes[2, step].axis('off')
                if step == 0:
                    axes[2, step].set_title('LSTM Pred', fontsize=10)
            
            plt.suptitle(f'Sample {sample_idx + 1}: GT vs VAE Recon vs LSTM Prediction (Open-loop)', fontsize=12)
            plt.tight_layout()
            
            save_path = os.path.join(save_dir, f'prediction_sample_{sample_idx + 1}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved visualization to {save_path}")


def visualize_rollout_images(model: VAEPredictor, dataset: TrajectoryDataset, device: torch.device, 
                             save_path: str, rollout_steps: int = 30, sample_idx: int = 0):
    """Visualize 30-step rollout: GT vs VAE Recon vs Predictor Rollout"""
    model.eval()
    
    if model.vae_encoder is None:
        print("Warning: VAE encoder not available, skipping rollout visualization")
        return
    
    vae = model.vae_encoder
    vae.eval()
    
    # Get sample from dataset
    sample = dataset[sample_idx]
    input_frames = sample['input_frames'].unsqueeze(0).to(device)  # (1, 15, 3, 64, 64)
    target_frames = sample['target_frames'].unsqueeze(0).to(device)  # (1, 15, 3, 64, 64)
    actions = sample.get('actions', None)
    if actions is not None:
        actions = actions.unsqueeze(0).to(device)  # (1, 15, action_dim)
    
    # Combine input and target frames for full sequence
    all_frames = torch.cat([input_frames, target_frames], dim=1)  # (1, 30, 3, 64, 64)
    T = all_frames.shape[1]
    
    with torch.no_grad():
        # Encode first frame
        first_frame = all_frames[:, 0, ...]  # (1, 3, 64, 64)
        mu0, logvar0, _ = vae.encode(first_frame)
        z0 = vae.reparameterize(mu0, logvar0)  # (1, C, 4, 4)
        
        # Build GT latents
        gt_latents = []
        for t in range(min(rollout_steps, T)):
            frame_t = all_frames[:, t, ...]  # (1, 3, 64, 64)
            mu_t, logvar_t, _ = vae.encode(frame_t)
            z_t = vae.reparameterize(mu_t, logvar_t)
            gt_latents.append(z_t)
        
        # Predictor rollout latents
        pred_latents = []
        z_prev = z0.clone()
        for t in range(rollout_steps):
            # Get action if available (use last available action if t exceeds sequence length)
            a = None
            if actions is not None and actions.shape[1] > 0:
                action_idx = min(t, actions.shape[1] - 1)
                a = actions[:, action_idx:action_idx+1, :]  # (1, 1, action_dim)
            
            # Predict next latent
            z_prev_expanded = z_prev.unsqueeze(1)  # (1, 1, C, 4, 4)
            z_pred = model.predict(z_prev_expanded, a)  # (1, 1, C, 4, 4)
            z_pred = z_pred.squeeze(1)  # (1, C, 4, 4)
            pred_latents.append(z_pred)
            z_prev = z_pred
        
        # Decode to images
        gt_imgs = []
        recon_imgs = []
        pred_imgs = []
        
        for t in range(rollout_steps):
            # Get frame for skip features (use last available frame if t exceeds sequence length)
            frame_idx = min(t, T - 1)
            frame_t = all_frames[:, frame_idx, ...]  # (1, 3, 64, 64)
            _, _, skip_t = vae.encode(frame_t)
            
            # GT image (only if we have GT latent for this step)
            if t < len(gt_latents):
                gt_img = vae.decode(gt_latents[t], skip_t)  # (1, 3, 64, 64)
                if gt_img.min() < 0:
                    gt_img = (gt_img + 1.0) / 2.0
                gt_img = torch.clamp(gt_img, 0, 1)
                gt_imgs.append(gt_img[0].cpu().permute(1, 2, 0).numpy())
                # VAE reconstruction (same as GT for visualization)
                recon_imgs.append(gt_img[0].cpu().permute(1, 2, 0).numpy())
            else:
                # Beyond GT sequence, use predicted latent decoded with last frame's skip features
                pred_img = vae.decode(pred_latents[t], skip_t)
                if pred_img.min() < 0:
                    pred_img = (pred_img + 1.0) / 2.0
                pred_img = torch.clamp(pred_img, 0, 1)
                # Use predicted as placeholder for GT/VAE rows
                gt_imgs.append(pred_img[0].cpu().permute(1, 2, 0).numpy())
                recon_imgs.append(pred_img[0].cpu().permute(1, 2, 0).numpy())
            
            # Predicted image (use GT skip features for better visualization)
            pred_img = vae.decode(pred_latents[t], skip_t)  # (1, 3, 64, 64)
            if pred_img.min() < 0:
                pred_img = (pred_img + 1.0) / 2.0
            pred_img = torch.clamp(pred_img, 0, 1)
            pred_imgs.append(pred_img[0].cpu().permute(1, 2, 0).numpy())
    
    # Plot
    fig, axes = plt.subplots(3, rollout_steps, figsize=(rollout_steps * 1.5, 9))
    if rollout_steps == 1:
        axes = axes.reshape(3, 1)
    
    fig.suptitle(f"30-step GT vs VAE Recon vs Predictor Rollout (Sample {sample_idx})", fontsize=16)
    
    for t in range(rollout_steps):
        if t < len(gt_imgs):
            axes[0, t].imshow(gt_imgs[t])
            axes[0, t].axis("off")
            if t == 0:
                axes[0, t].set_title("GT", fontsize=8)
            else:
                axes[0, t].set_title(f"{t}", fontsize=6)
        
        if t < len(recon_imgs):
            axes[1, t].imshow(recon_imgs[t])
            axes[1, t].axis("off")
            if t == 0:
                axes[1, t].set_title("VAE", fontsize=8)
            else:
                axes[1, t].set_title(f"{t}", fontsize=6)
        
        if t < len(pred_imgs):
            axes[2, t].imshow(pred_imgs[t])
            axes[2, t].axis("off")
            if t == 0:
                axes[2, t].set_title("Pred", fontsize=8)
            else:
                axes[2, t].set_title(f"{t}", fontsize=6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved 30-step rollout visualization to {save_path}")


def generate_prediction_video(model: VAEPredictor, dataset: TrajectoryDataset, device: torch.device,
                              save_path: str, rollout_steps: int = 100, sample_idx: int = 0, fps: int = 10,
                              debug_rollout: bool = False,
                              action_mode: str = "repeat_last",
                              video_layout: str = "pred"):
    """Generate video of long-term prediction rollout"""
    if not HAS_IMAGEIO:
        print("Warning: imageio not available, cannot generate video")
        return
    
    model.eval()
    
    if model.vae_encoder is None:
        print("Warning: VAE encoder not available, skipping video generation")
        return
    
    vae = model.vae_encoder
    vae.eval()
    
    # Get sample from dataset
    sample = dataset[sample_idx]
    input_frames = sample['input_frames'].unsqueeze(0).to(device)  # (1, 15, 3, 64, 64)
    target_frames = sample['target_frames'].unsqueeze(0).to(device)  # (1, 15, 3, 64, 64)
    actions = sample.get('actions', None)
    if actions is not None:
        actions = actions.unsqueeze(0).to(device)  # (1, 16, action_dim) (dataset returns length=sequence_length)

    # Optionally fetch a longer continuous action sequence directly from the underlying NPZ (memmap),
    # instead of repeating the last action when t exceeds dataset sequence length.
    actions_long = None
    gt_frames_u8 = None  # optional (rollout_steps, H, W, 3) uint8 for left side of video
    file_idx = None
    local_start = None
    f = None
    if action_mode == "from_npz":
        try:
            file_idx, local_start = dataset._locate(sample_idx)
            f = dataset.files[file_idx]
            s = int(local_start)
            # We want actions for transitions s->s+1, ..., s+rollout_steps-1 -> s+rollout_steps
            raw_actions = f["actions"]  # memmap (T, A)
            avail = int(raw_actions.shape[0] - s)
            take = min(rollout_steps, max(0, avail))
            if take <= 0:
                raise ValueError("No available actions for this sample index.")
            a_np = np.array(raw_actions[s:s+take], dtype=np.float32)  # (take, A)
            # Normalize actions exactly like TrajectoryDataset.__getitem__
            mean = dataset.act_mean.astype(np.float32)
            std = dataset.act_std.astype(np.float32)
            a_np = (a_np - mean) / std
            a_np = np.clip(a_np, -3.0, 3.0) / 3.0
            # Pad to rollout_steps by repeating last action if needed
            if take < rollout_steps:
                pad = np.repeat(a_np[-1:, :], rollout_steps - take, axis=0)
                a_np = np.concatenate([a_np, pad], axis=0)
            actions_long = torch.from_numpy(a_np).unsqueeze(0).to(device)  # (1, rollout_steps, A)
            print(f"[video] action_mode=from_npz: loaded continuous actions from {f.get('path','<npz>')} start={s}, steps={rollout_steps}")

            # Also load GT frames from the same NPZ for side-by-side comparison if requested.
            # pred_latents[t] corresponds to frame (t+1), so we take GT frames from s+1.
            if video_layout == "gt_pred":
                raw_frames = f["frames"]  # memmap (T, 3, H, W), uint8
                Tframes = int(raw_frames.shape[0])
                start_gt = s + 1
                end_gt = start_gt + rollout_steps
                if start_gt >= Tframes:
                    raise ValueError("GT frames not available for this start index.")
                gt_take = max(0, min(rollout_steps, Tframes - start_gt))
                gt = np.array(raw_frames[start_gt:start_gt + gt_take], dtype=np.uint8)  # (gt_take,3,H,W)
                # pad with last GT frame if short
                if gt_take < rollout_steps:
                    pad = np.repeat(gt[-1:, ...], rollout_steps - gt_take, axis=0)
                    gt = np.concatenate([gt, pad], axis=0)
                gt_frames_u8 = gt.transpose(0, 2, 3, 1)  # (rollout_steps,H,W,3)
        except Exception as e:
            print(f"[video] Warning: action_mode=from_npz failed ({e}); falling back to repeat_last.")
            actions_long = None
            action_mode = "repeat_last"
    
    # Combine input and target frames for full sequence
    all_frames = torch.cat([input_frames, target_frames], dim=1)  # (1, 30, 3, 64, 64)
    T = all_frames.shape[1]
    
    print(f"Generating {rollout_steps}-step prediction video...")
    
    with torch.no_grad():
        # Encode first frame
        first_frame = all_frames[:, 0, ...]  # (1, 3, 64, 64)
        mu0, logvar0, _ = vae.encode(first_frame)
        z0 = vae.reparameterize(mu0, logvar0)  # (1, C, 4, 4)
        
        # Build GT latents (only for available frames)
        gt_latents = []
        for t in range(min(rollout_steps, T)):
            frame_t = all_frames[:, t, ...]  # (1, 3, 64, 64)
            mu_t, logvar_t, _ = vae.encode(frame_t)
            z_t = vae.reparameterize(mu_t, logvar_t)
            gt_latents.append(z_t)
        
        # Predictor rollout latents
        pred_latents = []
        z_prev = z0.clone()
        for t in range(rollout_steps):
            # Get action if available (use last available action if t exceeds sequence length)
            a = None
            if actions_long is not None:
                a = actions_long[:, t:t+1, :]  # (1,1,A)
            elif actions is not None and actions.shape[1] > 0:
                action_idx = min(t, actions.shape[1] - 1)
                a = actions[:, action_idx:action_idx+1, :]  # (1, 1, action_dim)
            
            # Predict next latent
            z_prev_expanded = z_prev.unsqueeze(1)  # (1, 1, C, 4, 4)
            z_pred = model.predict(z_prev_expanded, a)  # (1, 1, C, 4, 4)
            z_pred = z_pred.squeeze(1)  # (1, C, 4, 4)
            # clone() avoids any chance of accidental aliasing
            pred_latents.append(z_pred.detach().clone())
            z_prev = z_pred
            
            if (t + 1) % 10 == 0:
                print(f"  Predicted step {t + 1}/{rollout_steps}")
        
        # Decode to images and create video frames
        print("Decoding images...")
        video_frames = []
        debug_stats = []
        prev_lat = None
        prev_img_f = None
        prev_img_u8 = None
        
        for t in range(rollout_steps):
            # Decode predicted latent without GT skip features (to see real prediction degradation)
            # Use None for skip_features to use the decoder's fallback (zero-padding)
            pred_img = vae.decode(pred_latents[t], None)  # (1, 3, 64, 64)
            if pred_img.min() < 0:
                pred_img = (pred_img + 1.0) / 2.0
            pred_img = torch.clamp(pred_img, 0, 1)
            pred_img_f = pred_img[0].detach().cpu()  # (3, H, W) float in [0,1]
            pred_img_np = pred_img_f.permute(1, 2, 0).numpy()  # (H, W, 3)
            pred_img_u8 = (pred_img_np * 255.0).round().astype(np.uint8)
            
            # Video layout
            if video_layout == "gt_pred":
                if gt_frames_u8 is None:
                    # Fallback: use available frames from dataset sample (input+target), padded.
                    # all_frames has shape (1, T_all, 3, H, W). Pred at t corresponds to frame index t+1.
                    frame_idx = min(t + 1, T - 1)
                    gt_img_u8 = (all_frames[0, frame_idx].detach().cpu().numpy().transpose(1, 2, 0)).astype(np.uint8)
                else:
                    gt_img_u8 = gt_frames_u8[t]
                # left=GT, right=Pred
                frame_u8 = np.concatenate([gt_img_u8, pred_img_u8], axis=1)
                frame_u8 = _overlay_sbs_labels(frame_u8, left_label="GT", right_label="PR")
                video_frames.append(frame_u8)
            else:
                # pred-only (default)
                video_frames.append(pred_img_u8)

            if debug_rollout:
                lat_max = None
                imgf_max = None
                imgu8_max = None
                if prev_lat is not None:
                    lat_max = float((pred_latents[t] - prev_lat).abs().max().item())
                if prev_img_f is not None:
                    imgf_max = float((pred_img_f - prev_img_f).abs().max().item())
                if prev_img_u8 is not None:
                    # uint8 difference after quantization (what you see in the mp4)
                    imgu8_max = int(np.abs(pred_img_u8.astype(np.int16) - prev_img_u8.astype(np.int16)).max())
                debug_stats.append({
                    "t": t,
                    "latent_max_abs_delta": lat_max,
                    "img_float_max_abs_delta": imgf_max,
                    "img_uint8_max_abs_delta": imgu8_max,
                })
                prev_lat = pred_latents[t]
                prev_img_f = pred_img_f
                prev_img_u8 = pred_img_u8
            
            if (t + 1) % 10 == 0:
                print(f"  Decoded step {t + 1}/{rollout_steps}")

        if debug_rollout:
            # Save debug deltas alongside the video
            try:
                import json
                debug_path = os.path.splitext(save_path)[0] + "_debug.json"
                with open(debug_path, "w", encoding="utf-8") as f:
                    json.dump(debug_stats, f, indent=2)
                print(f"Saved rollout debug stats to {debug_path}")
                # Quick hint if it 'freezes' due to uint8 quantization
                tail = [d for d in debug_stats[-10:] if d.get("img_uint8_max_abs_delta") is not None]
                if tail:
                    max_u8 = max(d["img_uint8_max_abs_delta"] for d in tail)
                    if max_u8 == 0:
                        print("Note: last frames are identical AFTER uint8 quantization (mp4). The float prediction may still change slightly.")
            except Exception as e:
                print(f"Warning: failed to write debug stats: {e}")
        
        # Save video
        print(f"Saving video to {save_path}...")
        imageio.mimsave(save_path, video_frames, fps=fps, codec='libx264', quality=8)
        print(f"Video saved to {save_path} ({len(video_frames)} frames, {fps} fps)")


def main():
    parser = argparse.ArgumentParser(description='Evaluate VAE Predictor')
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pt',
                       help='Path to predictor model checkpoint')
    parser.add_argument('--vae_model_path', type=str, default=None,
                       help='Optional: override VAE checkpoint path (needed for older predictor checkpoints)')
    parser.add_argument('--data_dir', type=str, default='../npz_transfer',
                       help='Directory containing NPZ files')
    parser.add_argument('--npz_files', nargs='+', default=['traj1.npz', 'traj2.npz'],
                       help='NPZ files for evaluation')
    parser.add_argument('--sequence_length', type=int, default=16,
                       help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--max_horizon', type=int, default=30,
                       help='Maximum horizon for multi-step rollout')
    parser.add_argument('--num_vis_samples', type=int, default=3,
                       help='Number of samples to visualize')
    parser.add_argument('--max_eval_batches', type=int, default=None,
                       help='Maximum number of batches to evaluate (None = all, useful for quick testing)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--save_dir', type=str, default='./eval_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--generate_video', action='store_true',
                       help='Generate prediction video (long-term rollout)')
    parser.add_argument('--video_steps', type=int, default=100,
                       help='Number of steps for video generation')
    parser.add_argument('--video_fps', type=int, default=10,
                       help='FPS for generated video')
    parser.add_argument('--video_sample_idx', type=int, default=0,
                       help='Sample index for video generation')
    parser.add_argument('--video_action_mode', type=str, default='repeat_last',
                       choices=['repeat_last', 'from_npz'],
                       help='Action mode for video rollout: repeat_last (default) or from_npz (use longer continuous actions from NPZ)')
    parser.add_argument('--video_layout', type=str, default='pred',
                       choices=['pred', 'gt_pred'],
                       help='Video layout: pred (prediction only) or gt_pred (left=GT, right=Prediction)')
    parser.add_argument('--debug_rollout', action='store_true',
                       help='If set: write per-step latent/image deltas to *_debug.json to diagnose \"frozen\" videos')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = load_model(args.model_path, device, vae_model_path_override=args.vae_model_path)
    model.eval()
    print("Model loaded!")
    
    # Create dataset
    npz_paths = [os.path.join(args.data_dir, f) for f in args.npz_files]
    print(f"\nLoading data from: {npz_paths}")
    dataset = TrajectoryDataset(npz_paths=npz_paths, sequence_length=args.sequence_length, normalize=True)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Run evaluations
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Check 1: Baseline vs LSTM
    baseline_results = compute_baseline_vs_lstm(model, dataloader, device, max_batches=args.max_eval_batches)
    
    # Check 2: Multi-step rollout
    rollout_results = compute_multi_step_rollout(model, dataloader, device, 
                                                  max_horizon=args.max_horizon,
                                                  max_batches=args.max_eval_batches)
    
    # Check 3: Visualization
    visualize_predictions(model, dataloader, device, 
                         num_samples=args.num_vis_samples, 
                         rollout_steps=args.max_horizon,
                         save_dir=args.save_dir)
    
    # Additional: 30-step rollout image visualization
    if args.max_horizon >= 30:
        print("\n" + "="*70)
        print("Creating 30-step Rollout Image Visualization")
        print("="*70)
        rollout_img_path = os.path.join(args.save_dir, 'rollout_30step.png')
        visualize_rollout_images(model, dataset, device, rollout_img_path, 
                                rollout_steps=30, sample_idx=0)
    
    # Plot rollout MSE vs horizon
    print("\n" + "="*70)
    print("Plotting Rollout MSE vs Horizon")
    print("="*70)
    
    # Only plot horizons where we actually have GT (value is not None)
    horizons = sorted([h for h, v in rollout_results.items() if v is not None])
    mses = [rollout_results[h] for h in horizons]
    
    plt.figure(figsize=(10, 6))
    plt.plot(horizons, mses, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Horizon (steps)', fontsize=12)
    plt.ylabel('MSE', fontsize=12)
    plt.title('Multi-step Open-loop Rollout: MSE vs Horizon', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(args.save_dir, 'rollout_mse_vs_horizon.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {plot_path}")
    
    # Save results
    results = {
        'baseline': baseline_results,
        'rollout': rollout_results
    }
    
    import json
    results_path = os.path.join(args.save_dir, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")
    
    # Generate prediction video if requested
    if args.generate_video:
        print("\n" + "="*70)
        print(f"Generating {args.video_steps}-step Prediction Video")
        print("="*70)
        video_path = os.path.join(args.save_dir, f'prediction_{args.video_steps}step.mp4')
        generate_prediction_video(model, dataset, device, video_path,
                                rollout_steps=args.video_steps,
                                sample_idx=args.video_sample_idx,
                                fps=args.video_fps,
                                debug_rollout=args.debug_rollout,
                                action_mode=args.video_action_mode,
                                video_layout=args.video_layout)
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)
    print(f"Results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()

