#!/usr/bin/env python3
"""
LSTM Open-loop Rollout Evaluation

Visualizes LSTM's multi-step prediction capability:
1. Given context frames (e.g., 10 frames)
2. LSTM predicts future latents for N steps
3. Decode predicted latents to images
4. Compare with ground truth images

This shows:
- Short-term prediction accuracy
- Long-term drift/error accumulation
- VAE decoding quality on predicted latents
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from predictor.core.vae_predictor import VAEPredictor
from vae_recon.vae_model_64x64 import SimpleVAE64x64


def load_models(vae_path, predictor_path, device):
    """Load VAE and LSTM predictor"""
    # Load VAE
    print(f"Loading VAE from {vae_path}...")
    vae = SimpleVAE64x64(latent_dim=64)
    vae_checkpoint = torch.load(vae_path, map_location=device, weights_only=False)
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    vae = vae.to(device)
    vae.eval()
    
    # Load LSTM Predictor
    print(f"Loading LSTM Predictor from {predictor_path}...")
    predictor_checkpoint = torch.load(predictor_path, map_location=device, weights_only=False)
    
    predictor = VAEPredictor(
        latent_dim=64,
        image_size=64,
        channels=3,
        action_dim=2,
        predictor_type='lstm',
        hidden_size=256,
        residual_prediction=True,
        vae_model_path=vae_path,
        freeze_vae=True
    ).to(device)
    
    predictor.load_state_dict(predictor_checkpoint['model_state_dict'], strict=False)
    predictor.eval()
    
    return vae, predictor


def open_loop_rollout(vae, predictor, context_frames, num_steps, device):
    """
    Perform open-loop rollout: predict multiple steps without ground truth
    
    Args:
        vae: VAE model
        predictor: LSTM predictor
        context_frames: (1, T_ctx, 3, H, W) tensor
        num_steps: Number of future steps to predict
        device: Device
    
    Returns:
        predicted_latents: (num_steps, C, H, W) tensor of predicted latents
        decoded_images: (num_steps, 3, H, W) tensor of decoded images
    """
    B = 1  # Single sequence
    T_ctx = context_frames.size(1)
    
    # Encode context frames to latent space
    with torch.no_grad():
        context_flat = context_frames.reshape(B * T_ctx, *context_frames.shape[2:])
        mu_ctx, logvar_ctx, _ = vae.encode(context_flat)
        z_context = vae.reparameterize(mu_ctx, logvar_ctx)
        z_context = z_context.reshape(B, T_ctx, *z_context.shape[1:])  # (1, T_ctx, C, H, W)
    
    # Open-loop rollout
    predicted_latents = []
    decoded_images = []
    
    # Start with context latents
    current_latents = z_context  # (1, T_ctx, C, H, W)
    
    with torch.no_grad():
        for step in range(num_steps):
            # Prepare LSTM input: flatten latents
            T = current_latents.size(1)
            latent_flat = current_latents.view(B, T, -1)  # (1, T, D_flat)
            
            # Add dummy actions (zeros)
            dummy_actions = torch.zeros(B, T, 2, device=device)
            lstm_input = torch.cat([latent_flat, dummy_actions], dim=-1)  # (1, T, D_flat+2)
            
            # LSTM forward
            out, _ = predictor.lstm(lstm_input)  # (1, T, hidden)
            out_last = predictor.lstm_out(out[:, -1])  # Last timestep -> (1, D_flat)
            
            # Reshape to spatial latent
            C, H, W = z_context.shape[2:]
            pred_latent = out_last.view(B, C, H, W)  # (1, C, H, W)
            
            # Decode to image
            decoded_img = vae.decode(pred_latent)
            decoded_img = torch.clamp(decoded_img, 0, 1)
            
            # Store results
            predicted_latents.append(pred_latent.squeeze(0).cpu())  # (C, H, W)
            decoded_images.append(decoded_img.squeeze(0).cpu())  # (3, H, W)
            
            # Update context: slide window and append predicted latent
            # Keep last (T_ctx-1) latents + new prediction
            current_latents = torch.cat([
                current_latents[:, 1:, ...],  # Remove oldest
                pred_latent.unsqueeze(1)      # Add newest
            ], dim=1)  # (1, T_ctx, C, H, W)
    
    return torch.stack(predicted_latents), torch.stack(decoded_images)


def load_test_sequences(npz_files, num_sequences=5):
    """Load test sequences from NPZ files"""
    sequences = []
    
    for npz_file in npz_files:
        data = np.load(npz_file)
        frames = data['frame']  # (N, 3, H, W)
        
        # Sample a few sequences
        for _ in range(num_sequences // len(npz_files)):
            start_idx = np.random.randint(0, len(frames) - 40)
            seq = frames[start_idx:start_idx + 40]  # 40 frames
            sequences.append(seq)
    
    return sequences


def visualize_rollout(context_imgs, true_future_imgs, pred_future_imgs, output_path, 
                     context_length=10, display_steps=20):
    """
    Visualize rollout results
    
    Args:
        context_imgs: (T_ctx, 3, H, W) context frames
        true_future_imgs: (N_steps, 3, H, W) ground truth future frames
        pred_future_imgs: (N_steps, 3, H, W) predicted future frames
        output_path: Save path
        context_length: Number of context frames
        display_steps: Number of future steps to display
    """
    fig, axes = plt.subplots(3, display_steps, figsize=(display_steps * 1.5, 5))
    
    # Row 1: Context frames (only show last few)
    show_ctx = min(context_length, display_steps)
    for i in range(display_steps):
        ax = axes[0, i]
        if i < show_ctx:
            ctx_idx = context_length - show_ctx + i
            img = context_imgs[ctx_idx].numpy().transpose(1, 2, 0)
            ax.imshow(img)
            ax.set_title(f'Ctx-{context_length-ctx_idx}', fontsize=8, color='blue')
        else:
            ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Row 2: Ground truth future
    for i in range(display_steps):
        ax = axes[1, i]
        if i < len(true_future_imgs):
            img = true_future_imgs[i].numpy().transpose(1, 2, 0)
            ax.imshow(img)
            ax.set_title(f'True+{i+1}', fontsize=8, color='green')
        else:
            ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Row 3: Predicted future
    for i in range(display_steps):
        ax = axes[2, i]
        if i < len(pred_future_imgs):
            img = pred_future_imgs[i].numpy().transpose(1, 2, 0)
            ax.imshow(img)
            ax.set_title(f'Pred+{i+1}', fontsize=8, color='red')
        else:
            ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add row labels
    axes[0, 0].set_ylabel('Context', fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel('Ground Truth', fontsize=10, fontweight='bold')
    axes[2, 0].set_ylabel('LSTM Predicted', fontsize=10, fontweight='bold')
    
    plt.suptitle('LSTM Open-Loop Rollout: Multi-Step Prediction', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[*] Visualization saved to {output_path}")


def calculate_rollout_metrics(vae, true_imgs, pred_imgs, device):
    """
    Calculate metrics for rollout prediction
    
    Returns:
        mse_per_step: MSE loss per prediction step
        psnr_per_step: PSNR per prediction step
    """
    mse_per_step = []
    psnr_per_step = []
    
    for i in range(len(pred_imgs)):
        true_img = true_imgs[i].numpy().transpose(1, 2, 0)
        pred_img = pred_imgs[i].numpy().transpose(1, 2, 0)
        
        # MSE
        mse = np.mean((true_img - pred_img) ** 2)
        mse_per_step.append(mse)
        
        # PSNR (Peak Signal-to-Noise Ratio)
        if mse > 0:
            psnr = 10 * np.log10(1.0 / mse)
        else:
            psnr = 100.0  # Perfect match
        psnr_per_step.append(psnr)
    
    return np.array(mse_per_step), np.array(psnr_per_step)


def main():
    parser = argparse.ArgumentParser(description='LSTM Open-Loop Rollout Evaluation')
    parser.add_argument('--vae_path', type=str, default='vae_recon/best_model.pt')
    parser.add_argument('--predictor_path', type=str, 
                       default='predictor/checkpoints_with_cls/best_model.pt')
    parser.add_argument('--npz_files', nargs='+', 
                       default=['npz_data/traj1_64x64.npz', 'npz_data/traj2_64x64.npz'])
    parser.add_argument('--context_length', type=int, default=10,
                       help='Number of context frames')
    parser.add_argument('--rollout_steps', type=int, default=20,
                       help='Number of steps to predict')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of test sequences')
    parser.add_argument('--output_dir', type=str, default='predictor/evaluation/lstm_rollout')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("LSTM Open-Loop Rollout Evaluation")
    print("="*70)
    print(f"Device: {device}")
    print(f"Context length: {args.context_length}")
    print(f"Rollout steps: {args.rollout_steps}")
    print(f"Number of samples: {args.num_samples}")
    
    # Load models
    vae, predictor = load_models(args.vae_path, args.predictor_path, device)
    
    # Load test sequences
    print(f"\nLoading test sequences from {len(args.npz_files)} files...")
    sequences = load_test_sequences(args.npz_files, args.num_samples)
    print(f"Loaded {len(sequences)} test sequences")
    
    # Evaluate each sequence
    all_mse = []
    all_ssim = []
    
    for seq_idx, seq_frames in enumerate(sequences):
        print(f"\n[{seq_idx+1}/{len(sequences)}] Processing sequence...")
        
        # Prepare data
        context_frames = seq_frames[:args.context_length]  # (T_ctx, 3, H, W)
        true_future_frames = seq_frames[args.context_length:args.context_length + args.rollout_steps]
        
        # Convert to tensor and normalize
        context_tensor = torch.from_numpy(context_frames).float() / 255.0
        context_tensor = context_tensor.unsqueeze(0).to(device)  # (1, T_ctx, 3, H, W)
        
        true_future_tensor = torch.from_numpy(true_future_frames).float() / 255.0
        
        # Perform rollout
        pred_latents, pred_images = open_loop_rollout(
            vae, predictor, context_tensor, args.rollout_steps, device
        )
        
        # Calculate metrics
        mse_per_step, psnr_per_step = calculate_rollout_metrics(
            vae, true_future_tensor, pred_images, device
        )
        all_mse.append(mse_per_step)
        all_ssim.append(psnr_per_step)  # Reuse variable name
        
        print(f"  MSE (step 1): {mse_per_step[0]:.6f}")
        print(f"  MSE (step {args.rollout_steps}): {mse_per_step[-1]:.6f}")
        print(f"  PSNR (step 1): {psnr_per_step[0]:.2f} dB")
        print(f"  PSNR (step {args.rollout_steps}): {psnr_per_step[-1]:.2f} dB")
        
        # Visualize this sequence
        output_path = os.path.join(args.output_dir, f'rollout_sample_{seq_idx+1}.png')
        visualize_rollout(
            context_tensor.squeeze(0).cpu(),  # (T_ctx, 3, H, W)
            true_future_tensor,
            pred_images,
            output_path,
            context_length=args.context_length,
            display_steps=min(args.rollout_steps, 20)
        )
    
    # Average metrics across all sequences
    avg_mse = np.mean(all_mse, axis=0)
    avg_psnr = np.mean(all_ssim, axis=0)  # all_ssim stores psnr
    
    # Plot combined metrics
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # MSE
    ax = axes[0]
    ax.plot(range(1, len(avg_mse) + 1), avg_mse, marker='o', color='red', linewidth=2, label='Average MSE')
    ax.set_xlabel('Prediction Step', fontsize=12)
    ax.set_ylabel('MSE (Image Space)', fontsize=12)
    ax.set_title('Prediction Error over Time', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # PSNR
    ax = axes[1]
    ax.plot(range(1, len(avg_psnr) + 1), avg_psnr, marker='o', color='blue', linewidth=2, label='Average PSNR')
    ax.set_xlabel('Prediction Step', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('Image Quality over Time', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'metrics_combined.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[*] Metrics plot saved to {os.path.join(args.output_dir, 'metrics_combined.png')}")
    
    # Save metrics to text file
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w', encoding='utf-8') as f:
        f.write("LSTM Open-Loop Rollout Metrics\n")
        f.write("="*50 + "\n\n")
        f.write(f"Context length: {args.context_length}\n")
        f.write(f"Rollout steps: {args.rollout_steps}\n")
        f.write(f"Number of samples: {len(sequences)}\n\n")
        
        f.write("Average MSE per step:\n")
        for i, mse in enumerate(avg_mse):
            f.write(f"  Step {i+1:2d}: {mse:.6f}\n")
        
        f.write("\nAverage PSNR per step:\n")
        for i, psnr_val in enumerate(avg_psnr):
            f.write(f"  Step {i+1:2d}: {psnr_val:.2f} dB\n")
        
        f.write(f"\nInitial prediction (step 1):\n")
        f.write(f"  MSE:  {avg_mse[0]:.6f}\n")
        f.write(f"  PSNR: {avg_psnr[0]:.2f} dB\n")
        
        f.write(f"\nLong-term prediction (step {args.rollout_steps}):\n")
        f.write(f"  MSE:  {avg_mse[-1]:.6f}\n")
        f.write(f"  PSNR: {avg_psnr[-1]:.2f} dB\n")
        
        f.write(f"\nError growth (MSE increase):\n")
        f.write(f"  {avg_mse[-1] / avg_mse[0]:.2f}x worse at step {args.rollout_steps}\n")
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)
    print(f"Results saved to: {args.output_dir}")
    print(f"\nSummary:")
    print(f"  Initial MSE (step 1):  {avg_mse[0]:.6f}")
    print(f"  Final MSE (step {args.rollout_steps}): {avg_mse[-1]:.6f}")
    print(f"  Error growth: {avg_mse[-1] / avg_mse[0]:.2f}x")
    print(f"  Initial PSNR: {avg_psnr[0]:.2f} dB")
    print(f"  Final PSNR:   {avg_psnr[-1]:.2f} dB")


if __name__ == '__main__':
    main()
