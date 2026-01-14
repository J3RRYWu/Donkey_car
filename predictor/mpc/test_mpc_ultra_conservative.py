"""
Ultra-conservative MPC test: 90% current + 10% future
Even more conservative goal for best stability
"""

import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse

def test_mpc_ultra_conservative(args):
    """MPC test with ULTRA-CONSERVATIVE goal (90% current + 10% future)"""
    print("=" * 70)
    print("Testing Conformal MPC - ULTRA CONSERVATIVE GOAL")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    print("\n[1/5] Loading models...")
    from predictor.core.vae_predictor import load_model
    
    model = load_model(
        path=args.lstm_path,
        device=device,
        vae_model_path_override=args.vae_path,
        freeze_vae_override=True
    )
    model.eval()
    
    print(f"✓ Model loaded: latent_dim={model.latent_dim}, hidden={model.hidden_size}")
    
    # Load data
    print("\n[2/5] Loading test data...")
    import os
    npz_paths = [os.path.join(args.data_dir, args.npz_file)]
    
    from predictor.core.vae_predictor import TrajectoryDataset
    dataset = TrajectoryDataset(
        npz_paths=npz_paths,
        sequence_length=16,
        image_size=64,
        normalize=True
    )
    
    print(f"✓ Dataset loaded: {len(dataset)} sequences")
    
    # Initialize MPC
    print("\n[3/5] Initializing MPC controller...")
    from predictor.mpc.conformal_mpc import ConformalMPC
    
    mpc = ConformalMPC(
        vae_model=model,
        lstm_model=model,
        cp_quantiles_path=args.cp_path,
        horizon=args.horizon,
        device=device
    )
    
    print(f"✓ MPC initialized: horizon={args.horizon} steps")
    print(f"  CP alpha={mpc.cp_alpha}, norm={mpc.cp_norm}")
    print(f"  CP quantiles: q_1={mpc.q_t[0].item():.2f}, q_{args.horizon}={mpc.q_t[-1].item():.2f}")
    
    # Get test sequence
    print("\n[4/5] Running MPC control loop...")
    test_idx = args.test_idx
    sample = dataset[test_idx]
    
    input_frames = sample['input_frames'].to(device)
    target_frames = sample['target_frames'].to(device)
    gt_actions = sample['actions'].to(device) if 'actions' in sample else None
    
    # ========== ULTRA-CONSERVATIVE GOAL ==========
    print("\n🎯 Using ULTRA-CONSERVATIVE goal strategy:")
    print("  Strategy: 90% current + 10% future (maximum stability)")
    
    goal_image = 0.9 * input_frames[-1] + 0.1 * target_frames[0]
    z_goal = mpc.compute_goal_latent(goal_image)
    
    # Compute distances
    with torch.no_grad():
        z_current = mpc.compute_goal_latent(input_frames[-1])
        z_target = mpc.compute_goal_latent(target_frames[0])
        
        dist_current_to_target = torch.norm(z_target - z_current).item()
        dist_current_to_goal = torch.norm(z_goal - z_current).item()
    
    print(f"\nTest sequence index: {test_idx}")
    print(f"Current state norm: {torch.norm(z_current).item():.2f}")
    print(f"Future target norm: {torch.norm(z_target).item():.2f}")
    print(f"MPC goal norm: {torch.norm(z_goal).item():.2f}")
    print(f"")
    print(f"📏 Distance analysis:")
    print(f"  Current → Original Target: {dist_current_to_target:.2f} (way too far!)")
    print(f"  Current → MPC Goal:        {dist_current_to_goal:.2f} (ultra close!)")
    print(f"  Reduction factor: {dist_current_to_target/dist_current_to_goal:.1f}x")
    
    # Run MPC
    u_opt, info = mpc.control_step(
        images_history=input_frames,
        z_goal=z_goal
    )
    
    print(f"\n✓ MPC optimization complete!")
    print(f"  Optimal action: steering={u_opt[0].item():.3f}, throttle={u_opt[1].item():.3f}")
    if gt_actions is not None:
        print(f"  GT action:      steering={gt_actions[0, 0].item():.3f}, throttle={gt_actions[0, 1].item():.3f}")
    print(f"  Final cost: {info['cost_final']:.4f}")
    print(f"  Tracking error @ horizon: {info['tracking_error'][-1]:.4f}")
    print(f"  Tracking error mean: {np.mean(info['tracking_error']):.4f}")
    print(f"  Tracking error max: {np.max(info['tracking_error']):.4f}")
    
    # Visualization
    print("\n[5/5] Generating visualizations...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Cost history
    ax = axes[0, 0]
    ax.plot(info['cost_history'], linewidth=2, color='blue', alpha=0.7)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Total Cost', fontsize=12)
    ax.set_title(f'MPC Optimization Convergence (Horizon={args.horizon})', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Tracking error
    ax = axes[0, 1]
    horizons = np.arange(1, args.horizon + 1)
    ax.plot(horizons, info['tracking_error'], 'o-', linewidth=2, label='MPC trajectory', color='blue')
    ax.axhline(y=3.0, color='r', linestyle='--', alpha=0.7, label='MSE threshold (3.0)', linewidth=2)
    ax.set_xlabel('Horizon (steps)', fontsize=12)
    ax.set_ylabel('Tracking Error (L2 norm)', fontsize=12)
    ax.set_title('Predicted Trajectory Error', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Action sequence
    ax = axes[1, 0]
    ax.plot(horizons, info['u_seq'][:, 0], 'o-', linewidth=2, label='Steering', color='blue')
    ax.plot(horizons, info['u_seq'][:, 1], 's-', linewidth=2, label='Throttle', color='orange')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axhline(y=1, color='r', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(y=-1, color='r', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_xlabel('Horizon (steps)', fontsize=12)
    ax.set_ylabel('Action', fontsize=12)
    ax.set_title('Optimized Action Sequence', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1.2, 1.2])
    
    # CP quantiles
    ax = axes[1, 1]
    ax.plot(horizons, info['q_t'], 'o-', linewidth=2, color='orange', label='CP quantile (q_t)')
    ax.fill_between(horizons, 0, info['q_t'], alpha=0.3, color='orange')
    ax.set_xlabel('Horizon (steps)', fontsize=12)
    ax.set_ylabel('Safety Radius (latent space)', fontsize=12)
    ax.set_title('Conformal Prediction Uncertainty', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / f'mpc_ultra_conservative_h{args.horizon}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot: {plot_path}")
    
    # Save JSON
    mpc_params_serializable = {}
    for k, v in mpc.params.items():
        if isinstance(v, torch.Tensor):
            mpc_params_serializable[k] = v.cpu().tolist()
        else:
            mpc_params_serializable[k] = v
    
    results = {
        'test_idx': test_idx,
        'horizon': args.horizon,
        'goal_strategy': '90% current + 10% future (ultra-conservative)',
        'optimal_action': u_opt.cpu().tolist(),
        'gt_action': gt_actions[0].cpu().tolist() if gt_actions is not None else None,
        'final_cost': info['cost_final'],
        'tracking_error_mean': float(np.mean(info['tracking_error'])),
        'tracking_error_max': float(np.max(info['tracking_error'])),
        'tracking_error_final': float(info['tracking_error'][-1]),
        'dist_current_to_target': float(dist_current_to_target),
        'dist_current_to_goal': float(dist_current_to_goal),
        'distance_reduction_factor': float(dist_current_to_target / dist_current_to_goal),
        'mpc_params': mpc_params_serializable,
        'cp_alpha': mpc.cp_alpha,
        'cp_norm': mpc.cp_norm
    }
    
    json_path = output_dir / f'mpc_ultra_conservative_h{args.horizon}.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results: {json_path}")
    
    print("\n" + "=" * 70)
    print("✓ MPC test complete!")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"  Horizon: {args.horizon} steps")
    print(f"  Goal distance: {dist_current_to_goal:.2f} (vs original {dist_current_to_target:.2f})")
    print(f"  Final cost: {info['cost_final']:.2f}")
    print(f"  Mean tracking error: {np.mean(info['tracking_error']):.2f}")
    print(f"  Actions: steering={u_opt[0].item():.3f}, throttle={u_opt[1].item():.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lstm_path', type=str, required=True)
    parser.add_argument('--vae_path', type=str, required=True)
    parser.add_argument('--cp_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--npz_file', type=str, required=True)
    parser.add_argument('--test_idx', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='./mpc_test_results')
    
    args = parser.parse_args()
    test_mpc_ultra_conservative(args)
