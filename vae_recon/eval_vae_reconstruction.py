#!/usr/bin/env python3
"""
VAE Reconstruction Quality Evaluation

Evaluates VAE's ability to reconstruct images:
1. Load test images
2. Encode to latent space
3. Decode back to images
4. Compare reconstruction quality with original

This shows the VAE's baseline reconstruction capability without LSTM.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from vae_recon.vae_model_64x64 import SimpleVAE64x64


def load_test_images(npz_files, num_samples=20):
    """Load random test images from NPZ files"""
    all_images = []
    
    for npz_file in npz_files:
        data = np.load(npz_file)
        frames = data['frame']  # (N, 3, H, W) uint8
        all_images.append(frames)
    
    all_images = np.concatenate(all_images, axis=0)
    
    # Sample random images
    indices = np.random.choice(len(all_images), num_samples, replace=False)
    sampled_images = all_images[indices]
    
    return sampled_images


def evaluate_vae_reconstruction(vae, images, device):
    """
    Evaluate VAE reconstruction quality
    
    Args:
        vae: VAE model
        images: (N, 3, H, W) numpy array, uint8
        device: Device
    
    Returns:
        original_imgs: (N, 3, H, W) tensor, normalized [0, 1]
        reconstructed_imgs: (N, 3, H, W) tensor, reconstructed [0, 1]
        mse_per_image: (N,) numpy array
        psnr_per_image: (N,) numpy array
    """
    vae.eval()
    
    # Normalize images
    images_tensor = torch.from_numpy(images).float() / 255.0
    images_tensor = images_tensor.to(device)
    
    with torch.no_grad():
        # Encode
        mu, logvar, _ = vae.encode(images_tensor)
        z = vae.reparameterize(mu, logvar)
        
        # Decode
        reconstructed = vae.decode(z)
        reconstructed = torch.clamp(reconstructed, 0, 1)
    
    # Calculate metrics per image
    mse_per_image = []
    psnr_per_image = []
    
    for i in range(len(images_tensor)):
        orig = images_tensor[i].cpu().numpy()
        recon = reconstructed[i].cpu().numpy()
        
        mse = np.mean((orig - recon) ** 2)
        mse_per_image.append(mse)
        
        if mse > 0:
            psnr = 10 * np.log10(1.0 / mse)
        else:
            psnr = 100.0
        psnr_per_image.append(psnr)
    
    return (images_tensor.cpu(), reconstructed.cpu(), 
            np.array(mse_per_image), np.array(psnr_per_image))


def visualize_reconstruction(original_imgs, reconstructed_imgs, mse_per_image, 
                             output_path, num_display=20):
    """
    Visualize original vs reconstructed images
    
    Args:
        original_imgs: (N, 3, H, W) tensor
        reconstructed_imgs: (N, 3, H, W) tensor
        mse_per_image: (N,) array
        output_path: Save path
        num_display: Number of samples to display
    """
    num_display = min(num_display, len(original_imgs))
    
    fig, axes = plt.subplots(2, num_display, figsize=(num_display * 1.5, 3.5))
    
    for i in range(num_display):
        # Original
        ax_orig = axes[0, i]
        img_orig = original_imgs[i].numpy().transpose(1, 2, 0)
        ax_orig.imshow(img_orig)
        ax_orig.set_title(f'Original {i+1}', fontsize=8, color='green')
        ax_orig.axis('off')
        
        # Reconstructed
        ax_recon = axes[1, i]
        img_recon = reconstructed_imgs[i].numpy().transpose(1, 2, 0)
        ax_recon.imshow(img_recon)
        mse = mse_per_image[i]
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100.0
        ax_recon.set_title(f'MSE: {mse:.4f}\nPSNR: {psnr:.1f}dB', 
                          fontsize=7, color='blue')
        ax_recon.axis('off')
    
    # Add row labels
    axes[0, 0].set_ylabel('Original', fontsize=10, fontweight='bold', rotation=0, 
                          ha='right', va='center')
    axes[1, 0].set_ylabel('VAE Recon', fontsize=10, fontweight='bold', rotation=0, 
                          ha='right', va='center')
    
    plt.suptitle('VAE Reconstruction Quality (Encode → Decode)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[*] Visualization saved to {output_path}")


def plot_metrics_distribution(mse_values, psnr_values, output_path):
    """Plot MSE and PSNR distributions"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # MSE histogram
    ax = axes[0]
    ax.hist(mse_values, bins=20, color='red', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(mse_values), color='darkred', linestyle='--', 
              linewidth=2, label=f'Mean: {np.mean(mse_values):.4f}')
    ax.set_xlabel('MSE (Mean Squared Error)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('MSE Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # PSNR histogram
    ax = axes[1]
    ax.hist(psnr_values, bins=20, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(psnr_values), color='darkblue', linestyle='--', 
              linewidth=2, label=f'Mean: {np.mean(psnr_values):.2f} dB')
    ax.set_xlabel('PSNR (Peak Signal-to-Noise Ratio) [dB]', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('PSNR Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[*] Metrics distribution saved to {output_path}")


def compare_with_lstm_rollout(vae_mse, vae_psnr, lstm_metrics_path, output_path):
    """Compare VAE reconstruction with LSTM+VAE prediction"""
    if not os.path.exists(lstm_metrics_path):
        print(f"[!] LSTM metrics not found at {lstm_metrics_path}, skipping comparison")
        return
    
    # Load LSTM metrics
    with open(lstm_metrics_path, 'r') as f:
        lines = f.readlines()
    
    # Parse first step metrics (most comparable to direct VAE)
    lstm_mse_step1 = None
    lstm_psnr_step1 = None
    for line in lines:
        if 'Step  1:' in line and 'MSE' not in line and 'PSNR' not in line:
            if lstm_mse_step1 is None:
                lstm_mse_step1 = float(line.split(':')[1].strip())
            elif lstm_psnr_step1 is None:
                parts = line.split(':')[1].strip().split()
                lstm_psnr_step1 = float(parts[0])
                break
    
    if lstm_mse_step1 is None:
        print("[!] Could not parse LSTM metrics")
        return
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # MSE comparison
    ax = axes[0]
    methods = ['VAE Direct\nReconstruction', 'LSTM+VAE\nPrediction (step 1)']
    mse_vals = [np.mean(vae_mse), lstm_mse_step1]
    colors = ['green', 'red']
    bars = ax.bar(methods, mse_vals, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('MSE (Mean Squared Error)', fontsize=12)
    ax.set_title('MSE Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, mse_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.4f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # PSNR comparison
    ax = axes[1]
    psnr_vals = [np.mean(vae_psnr), lstm_psnr_step1]
    bars = ax.bar(methods, psnr_vals, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('PSNR Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, psnr_vals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f} dB',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle('VAE vs LSTM+VAE: Reconstruction Quality', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[*] Comparison plot saved to {output_path}")
    
    # Print comparison
    print("\n" + "="*70)
    print("Comparison: VAE Direct vs LSTM+VAE")
    print("="*70)
    print(f"VAE Direct Reconstruction:")
    print(f"  MSE:  {np.mean(vae_mse):.6f}")
    print(f"  PSNR: {np.mean(vae_psnr):.2f} dB")
    print(f"\nLSTM+VAE Prediction (step 1):")
    print(f"  MSE:  {lstm_mse_step1:.6f}")
    print(f"  PSNR: {lstm_psnr_step1:.2f} dB")
    print(f"\nDifference (LSTM - VAE):")
    print(f"  MSE:  +{lstm_mse_step1 - np.mean(vae_mse):.6f} ({(lstm_mse_step1/np.mean(vae_mse) - 1)*100:+.1f}%)")
    print(f"  PSNR: {lstm_psnr_step1 - np.mean(vae_psnr):+.2f} dB")


def main():
    parser = argparse.ArgumentParser(description='VAE Reconstruction Quality Evaluation')
    parser.add_argument('--vae_path', type=str, default='vae_recon/best_model.pt')
    parser.add_argument('--npz_files', nargs='+', 
                       default=['npz_data/traj1_64x64.npz', 'npz_data/traj2_64x64.npz'])
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of test images')
    parser.add_argument('--output_dir', type=str, default='vae_recon/eval_reconstruction')
    parser.add_argument('--lstm_metrics', type=str, 
                       default='predictor/evaluation/lstm_rollout/metrics.txt',
                       help='Path to LSTM rollout metrics for comparison')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("VAE Reconstruction Quality Evaluation")
    print("="*70)
    print(f"Device: {device}")
    print(f"Number of test samples: {args.num_samples}")
    
    # Load VAE
    print(f"\nLoading VAE from {args.vae_path}...")
    vae = SimpleVAE64x64(latent_dim=64)
    vae_checkpoint = torch.load(args.vae_path, map_location=device, weights_only=False)
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    vae = vae.to(device)
    vae.eval()
    print("VAE loaded successfully")
    
    # Load test images
    print(f"\nLoading test images from {len(args.npz_files)} files...")
    test_images = load_test_images(args.npz_files, args.num_samples)
    print(f"Loaded {len(test_images)} test images")
    
    # Evaluate reconstruction
    print("\nEvaluating VAE reconstruction...")
    original_imgs, reconstructed_imgs, mse_per_image, psnr_per_image = \
        evaluate_vae_reconstruction(vae, test_images, device)
    
    # Print statistics
    print("\n" + "="*70)
    print("Results:")
    print("="*70)
    print(f"Average MSE:  {np.mean(mse_per_image):.6f} (std: {np.std(mse_per_image):.6f})")
    print(f"Average PSNR: {np.mean(psnr_per_image):.2f} dB (std: {np.std(psnr_per_image):.2f})")
    print(f"Min MSE:  {np.min(mse_per_image):.6f}")
    print(f"Max MSE:  {np.max(mse_per_image):.6f}")
    print(f"Min PSNR: {np.min(psnr_per_image):.2f} dB")
    print(f"Max PSNR: {np.max(psnr_per_image):.2f} dB")
    
    # Visualize reconstruction
    print("\nGenerating visualizations...")
    visualize_reconstruction(
        original_imgs, reconstructed_imgs, mse_per_image,
        os.path.join(args.output_dir, 'reconstruction_samples.png'),
        num_display=20
    )
    
    # Plot metrics distribution
    plot_metrics_distribution(
        mse_per_image, psnr_per_image,
        os.path.join(args.output_dir, 'metrics_distribution.png')
    )
    
    # Compare with LSTM rollout if available
    compare_with_lstm_rollout(
        mse_per_image, psnr_per_image,
        args.lstm_metrics,
        os.path.join(args.output_dir, 'vae_vs_lstm_comparison.png')
    )
    
    # Save metrics to text file
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w', encoding='utf-8') as f:
        f.write("VAE Reconstruction Quality Metrics\n")
        f.write("="*50 + "\n\n")
        f.write(f"Number of test images: {len(test_images)}\n\n")
        
        f.write("Summary Statistics:\n")
        f.write(f"  Average MSE:  {np.mean(mse_per_image):.6f}\n")
        f.write(f"  Std MSE:      {np.std(mse_per_image):.6f}\n")
        f.write(f"  Min MSE:      {np.min(mse_per_image):.6f}\n")
        f.write(f"  Max MSE:      {np.max(mse_per_image):.6f}\n")
        f.write(f"  Median MSE:   {np.median(mse_per_image):.6f}\n\n")
        
        f.write(f"  Average PSNR: {np.mean(psnr_per_image):.2f} dB\n")
        f.write(f"  Std PSNR:     {np.std(psnr_per_image):.2f} dB\n")
        f.write(f"  Min PSNR:     {np.min(psnr_per_image):.2f} dB\n")
        f.write(f"  Max PSNR:     {np.max(psnr_per_image):.2f} dB\n")
        f.write(f"  Median PSNR:  {np.median(psnr_per_image):.2f} dB\n\n")
        
        f.write("Quality Assessment:\n")
        avg_psnr = np.mean(psnr_per_image)
        if avg_psnr > 30:
            quality = "Excellent"
        elif avg_psnr > 25:
            quality = "Good"
        elif avg_psnr > 20:
            quality = "Fair"
        elif avg_psnr > 15:
            quality = "Poor"
        else:
            quality = "Very Poor"
        f.write(f"  Overall quality: {quality}\n")
        f.write(f"  (PSNR > 30dB: Excellent, 25-30: Good, 20-25: Fair, 15-20: Poor, <15: Very Poor)\n")
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
