"""
Test script for Conformal MPC controller
Demonstrates offline MPC optimization on recorded trajectories

Usage:
    python test_mpc.py --lstm_path checkpoints/best_model.pt \
                       --vae_path ../vae_recon/best_model.pt \
                       --cp_path eval_results/cp_quantiles.json \
                       --data_dir ../npz_data \
                       --npz_file traj1_64x64.npz
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

from predictor.core.vae_predictor import VAEPredictor, TrajectoryDataset
from predictor.mpc.conformal_mpc import ConformalMPC


def test_mpc_offline(args):
    """
    Test MPC on recorded trajectory (offline)
    """
    print("=" * 70)
    print("Testing Conformal MPC (Offline)")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ========================================================================
    # Step 1: Load Models
    # ========================================================================
    print("\n[1/5] Loading models...")
    
    # Load LSTM predictor (use load_model function)
    from predictor.core.vae_predictor import load_model
    
    model = load_model(
        path=args.lstm_path,
        device=device,
        vae_model_path_override=args.vae_path,
        freeze_vae_override=True
    )
    model.eval()
    
    # VAEPredictor includes both VAE (encoder/decoder) and LSTM
    # No separate .vae attribute needed
    
    print(f"✓ Model loaded: latent_dim={model.latent_dim}, hidden={model.hidden_size}")
    print(f"✓ VAE frozen: {model.freeze_vae}")
    print(f"✓ Action dim: {model.action_dim}")
    
    # ========================================================================
    # Step 2: Load Dataset
    # ========================================================================
    print("\n[2/5] Loading test data...")
    
    # Build npz_paths list
    import os
    npz_paths = [os.path.join(args.data_dir, args.npz_file)]
    
    dataset = TrajectoryDataset(
        npz_paths=npz_paths,
        sequence_length=16,
        image_size=64,
        normalize=True
    )
    
    print(f"✓ Dataset loaded: {len(dataset)} sequences")
    
    # ========================================================================
    # Step 3: Initialize MPC
    # ========================================================================
    print("\n[3/5] Initializing MPC controller...")
    
    # VAEPredictor contains both VAE and LSTM, pass the same model for both
    mpc = ConformalMPC(
        vae_model=model,  # VAEPredictor has encode/decode methods
        lstm_model=model,  # VAEPredictor has rollout_from_context method
        cp_quantiles_path=args.cp_path,
        horizon=args.horizon,
        device=device
    )
    
    print(f"✓ MPC initialized: horizon={args.horizon} steps")
    print(f"  CP alpha={mpc.cp_alpha}, norm={mpc.cp_norm}")
    print(f"  CP quantiles: q_1={mpc.q_t[0].item():.2f}, q_{args.horizon}={mpc.q_t[-1].item():.2f}")
    
    # ========================================================================
    # Step 4: Run MPC on Test Sequence
    # ========================================================================
    print("\n[4/5] Running MPC control loop...")
    
    # Get a test sequence
    test_idx = args.test_idx
    sample = dataset[test_idx]
    
    input_frames = sample['input_frames'].to(device)  # [15, 3, 64, 64]
    target_frames = sample['target_frames'].to(device)  # [1, 3, 64, 64]
    gt_actions = sample['actions'].to(device) if 'actions' in sample else None
    
    # Define goal: track a closer target
    # Option 1: Use last input frame as goal (track current state)
    # z_goal = mpc.compute_goal_latent(input_frames[-1])
    
    # Option 2: Average of last input and first target (smoother goal)
    goal_image = 0.7 * input_frames[-1] + 0.3 * target_frames[0]
    z_goal = mpc.compute_goal_latent(goal_image)
    
    print(f"Test sequence index: {test_idx}")
    print(f"Goal latent norm: {torch.norm(z_goal).item():.2f}")
    
    # Compute distance from current state to goal
    with torch.no_grad():
        z_current = mpc.compute_goal_latent(input_frames[-1])
        goal_distance = torch.norm(z_goal - z_current).item()
        print(f"Distance to goal: {goal_distance:.2f}")
    
    # Run single MPC step
    u_opt, info = mpc.control_step(
        images_history=input_frames,
        z_goal=z_goal
    )
    
    print(f"\n✓ MPC optimization complete!")
    print(f"  Optimal action: steering={u_opt[0].item():.3f}, throttle={u_opt[1].item():.3f}")
    if gt_actions is not None:
        print(f"  GT action: steering={gt_actions[0, 0].item():.3f}, throttle={gt_actions[0, 1].item():.3f}")
    print(f"  Final cost: {info['cost_final']:.4f}")
    print(f"  Tracking error @ horizon: {info['tracking_error'][-1]:.4f}")
    
    # ========================================================================
    # Step 5: Visualization
    # ========================================================================
    print("\n[5/5] Generating visualizations...")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot 1: Cost convergence
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Cost history
    ax = axes[0, 0]
    ax.plot(info['cost_history'], linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Total Cost', fontsize=12)
    ax.set_title('MPC Optimization Convergence', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Tracking error vs horizon
    ax = axes[0, 1]
    horizons = np.arange(1, args.horizon + 1)
    ax.plot(horizons, info['tracking_error'], 'o-', linewidth=2, label='MPC trajectory')
    ax.axhline(y=3.0, color='r', linestyle='--', alpha=0.7, label='MSE threshold')
    ax.set_xlabel('Horizon (steps)', fontsize=12)
    ax.set_ylabel('Tracking Error (L2 norm)', fontsize=12)
    ax.set_title('Predicted Trajectory Error', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Action sequence
    ax = axes[1, 0]
    ax.plot(horizons, info['u_seq'][:, 0], 'o-', linewidth=2, label='Steering')
    ax.plot(horizons, info['u_seq'][:, 1], 's-', linewidth=2, label='Throttle')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Horizon (steps)', fontsize=12)
    ax.set_ylabel('Action', fontsize=12)
    ax.set_title('Optimized Action Sequence', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1.2, 1.2])
    
    # CP quantiles
    ax = axes[1, 1]
    ax.plot(horizons, info['q_t'], 'o-', linewidth=2, color='orange', label='CP quantile (q_t)')
    ax.fill_between(horizons, 0, info['q_t'], alpha=0.3, color='orange')
    ax.set_xlabel('Horizon (steps)', fontsize=12)
    ax.set_ylabel('Safety Radius (latent space)', fontsize=12)
    ax.set_title('Conformal Prediction Uncertainty', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'mpc_test_results.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot: {plot_path}")
    
    # Save results to JSON (convert tensors to lists)
    mpc_params_serializable = {}
    for k, v in mpc.params.items():
        if isinstance(v, torch.Tensor):
            mpc_params_serializable[k] = v.cpu().tolist()
        else:
            mpc_params_serializable[k] = v
    
    results = {
        'test_idx': test_idx,
        'optimal_action': u_opt.cpu().tolist(),
        'gt_action': gt_actions[0].cpu().tolist() if gt_actions is not None else None,
        'final_cost': info['cost_final'],
        'tracking_error_mean': float(np.mean(info['tracking_error'])),
        'tracking_error_max': float(np.max(info['tracking_error'])),
        'mpc_params': mpc_params_serializable,
        'horizon': args.horizon,
        'cp_alpha': mpc.cp_alpha,
        'cp_norm': mpc.cp_norm
    }
    
    json_path = output_dir / 'mpc_test_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results: {json_path}")
    
    print("\n" + "=" * 70)
    print("✓ MPC test complete!")
    print("=" * 70)


def test_mpc_rollout(args):
    """
    Test MPC over multiple steps (closed-loop simulation)
    """
    print("=" * 70)
    print("Testing Conformal MPC (Closed-loop Rollout)")
    print("=" * 70)
    print("⚠ Not implemented yet - coming soon!")
    print("This will simulate MPC controlling over multiple steps with feedback.")


def main():
    parser = argparse.ArgumentParser(description='Test Conformal MPC')
    
    # Model paths
    parser.add_argument('--lstm_path', type=str, required=True,
                        help='Path to LSTM checkpoint')
    parser.add_argument('--vae_path', type=str, required=True,
                        help='Path to VAE checkpoint')
    parser.add_argument('--cp_path', type=str, required=True,
                        help='Path to CP quantiles JSON')
    
    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing NPZ files')
    parser.add_argument('--npz_file', type=str, required=True,
                        help='NPZ file name')
    parser.add_argument('--test_idx', type=int, default=100,
                        help='Test sequence index')
    
    # MPC parameters
    parser.add_argument('--horizon', type=int, default=17,
                        help='MPC prediction horizon')
    parser.add_argument('--mode', type=str, default='offline',
                        choices=['offline', 'rollout'],
                        help='Test mode')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./mpc_test_results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    if args.mode == 'offline':
        test_mpc_offline(args)
    elif args.mode == 'rollout':
        test_mpc_rollout(args)


if __name__ == '__main__':
    main()
