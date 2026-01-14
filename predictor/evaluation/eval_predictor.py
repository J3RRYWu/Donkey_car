#!/usr/bin/env python3
"""
Evaluation script for VAE Predictor - Main entry point

This script coordinates evaluation tasks across multiple modules:
- eval_metrics.py: Core evaluation (baseline vs LSTM, multi-step rollout)
- eval_visualization.py: Standard visualizations (predictions, videos)
- eval_conformal.py: Conformal Prediction evaluation and visualizations
- eval_utils.py: Utility functions (PSNR, SSIM, image processing)
"""

import torch
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
from typing import Dict, Optional
import matplotlib.pyplot as plt
import csv
import json

from predictor.core.vae_predictor import VAEPredictor, TrajectoryDataset, load_model
from predictor.evaluation.eval_metrics import compute_baseline_vs_lstm, compute_multi_step_rollout
from predictor.evaluation.eval_visualization import visualize_predictions, visualize_rollout_images, generate_prediction_video
from predictor.evaluation.eval_conformal import visualize_cp_trajectory_band, visualize_cp_boundary_decode
from predictor.conformal.conformal import CPQuantiles, nonconformity_scores, quantiles_per_horizon, coverage_per_horizon, set_size_summary
from predictor.evaluation.eval_utils import effective_horizon_from_curve

# Import rigorous safety evaluation (optional, only if --cp_safety is used)
try:
    from predictor.evaluation.eval_cp_safety import rigorous_cp_evaluation, export_safety_report
    HAS_CP_SAFETY = True
except ImportError:
    HAS_CP_SAFETY = False


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
    parser.add_argument('--input_length', type=int, default=15,
                       help='Number of context frames (input) in each sample')
    parser.add_argument('--target_length', type=int, default=15,
                       help='Number of target frames (GT) in each sample')
    parser.add_argument('--target_offset', type=int, default=1,
                       help='Start index of target frames within sampled window (default: 1)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--max_horizon', type=int, default=30,
                       help='Maximum horizon for multi-step rollout')
    parser.add_argument('--psnr_threshold', type=float, default=20.0,
                       help='PSNR threshold for effective horizon (image-space)')
    parser.add_argument('--ssim_threshold', type=float, default=0.5,
                       help='SSIM threshold for effective horizon (image-space)')
    parser.add_argument('--mse_threshold', type=float, default=0.01,
                       help='Image MSE threshold for effective horizon (image-space).')
    parser.add_argument('--mc_samples', type=int, default=1,
                       help='MC-dropout samples for uncertainty (1 disables; >1 enables).')
    parser.add_argument('--gt_from_npz', action='store_true',
                       help='If set: use future frames from the same NPZ (memmap) as GT for rollout metrics, '
                            'even when sequence_length is small.')

    # ---- Conformal Prediction (Split CP) ----
    parser.add_argument('--cp_alpha', type=float, default=0.05,
                       help='Conformal alpha (e.g., 0.05 for 95%% target coverage).')
    parser.add_argument('--cp_norm', type=str, default='l2', choices=['l2', 'linf'],
                       help='Norm used in nonconformity score: l2 or linf.')
    parser.add_argument('--cp_calib_size', type=int, default=500,
                       help='Number of sequences to use for calibration (split conformal).')
    parser.add_argument('--cp_seed', type=int, default=42,
                       help='Random seed for calibration/test split.')
    parser.add_argument('--cp_quantiles_path', type=str, default=None,
                       help='Path to save/load conformal quantiles JSON. Default: <save_dir>/cp_quantiles.json')
    parser.add_argument('--cp_calibrate', action='store_true',
                       help='If set: run conformal calibration and save quantiles.')
    parser.add_argument('--cp_eval', action='store_true',
                       help='If set: evaluate conformal coverage on the test split using saved quantiles (or freshly calibrated).')
    parser.add_argument('--cp_safety', action='store_true',
                       help='If set: run RIGOROUS safety evaluation with conservative CP, multiple validation, and detailed safety analysis.')
    parser.add_argument('--cp_traj_plot', action='store_true',
                       help='If set: plot a single test trajectory in 2D PCA latent space with CP band (q_t) around predictions.')
    parser.add_argument('--cp_traj_sample_idx', type=int, default=0,
                       help='Dataset index for CP trajectory band visualization.')
    parser.add_argument('--cp_traj_horizon', type=int, default=50,
                       help='Horizon (steps) for CP trajectory band visualization.')
    parser.add_argument('--cp_traj_pca_samples', type=int, default=500,
                       help='Number of random windows to fit PCA for CP trajectory plot.')
    parser.add_argument('--cp_traj_scale', type=float, default=0.01,
                       help='Scaling factor for CP band in 2D visualization (smaller = smaller circles). Default: 0.01')
    parser.add_argument('--cp_boundary_plot', action='store_true',
                       help='If set: pick a step t, sample a few points on CP L2-ball boundary around z_hat_t, decode with VAE, and save a PNG.')
    parser.add_argument('--cp_boundary_step', type=int, default=20,
                       help='Horizon step t for boundary sampling/decoding (1-indexed).')
    parser.add_argument('--cp_boundary_num', type=int, default=4,
                       help='Number of boundary samples to decode (excluding the center).')
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
                       help='If set: write per-step latent/image deltas to *_debug.json to diagnose "frozen" videos')

    # ---- Runtime controls (avoid re-running expensive eval) ----
    parser.add_argument('--skip_check1', action='store_true',
                       help='Skip Check 1 (baseline vs LSTM one-step).')
    parser.add_argument('--skip_check2', action='store_true',
                       help='Skip Check 2 (multi-step open-loop rollout).')
    parser.add_argument('--skip_visualize', action='store_true',
                       help='Skip visualization images (prediction_sample_*.png).')
    parser.add_argument('--skip_rollout_plot', action='store_true',
                       help='Skip plotting rollout MSE vs horizon figure.')
    parser.add_argument('--skip_exports', action='store_true',
                       help='Skip JSON/CSV exports (rollout_metrics.json, csv curves, etc.).')
    parser.add_argument('--only_cp', action='store_true',
                       help='Only run CP-related tasks (cp_calibrate/cp_eval/cp_traj_plot). Skips checks/plots/exports unless needed by CP.')
    
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
    dataset = TrajectoryDataset(
        npz_paths=npz_paths,
        sequence_length=args.sequence_length,
        normalize=True,
        input_length=int(args.input_length),
        target_length=int(args.target_length),
        target_offset=int(args.target_offset),
    )
    
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
    
    # Decide what to run
    run_check1 = (not args.skip_check1) and (not args.only_cp)
    run_check2 = (not args.skip_check2) and (not args.only_cp)
    run_vis = (not args.skip_visualize) and (not args.only_cp)
    run_rollout_plot = (not args.skip_rollout_plot) and (not args.only_cp)
    run_exports = (not args.skip_exports) and (not args.only_cp)

    baseline_results = None
    rollout_results = None

    if run_check1:
        baseline_results = compute_baseline_vs_lstm(model, dataloader, device, max_batches=args.max_eval_batches)
    
    if run_check2:
        rollout_results = compute_multi_step_rollout(
            model, dataloader, device,
                                                  max_horizon=args.max_horizon,
            max_batches=args.max_eval_batches,
            psnr_threshold=float(args.psnr_threshold),
            ssim_threshold=float(args.ssim_threshold),
            mc_samples=int(args.mc_samples),
            gt_from_npz=bool(args.gt_from_npz),
            mse_threshold=float(args.mse_threshold),
        )

    # ---- Conformal Prediction (Split CP): calibrate + evaluate ----
    cp_quant_path = args.cp_quantiles_path or os.path.join(args.save_dir, "cp_quantiles.json")
    cp_results = None
    
    # ==== RIGOROUS SAFETY EVALUATION ====
    if args.cp_safety:
        if not HAS_CP_SAFETY:
            print("\n⚠️  Warning: eval_cp_safety.py not found. Skipping safety evaluation.")
        else:
            print("\n" + "="*80)
            print("RUNNING RIGOROUS CP SAFETY EVALUATION")
            print("="*80)
            
            cp_standard, cp_conservative, safety_report = rigorous_cp_evaluation(
                model=model,
                dataset=dataset,
                device=device,
                max_horizon=int(args.max_horizon),
                alpha=float(args.cp_alpha),
                calib_size=int(args.cp_calib_size),
                test_size=int(args.cp_calib_size),  # Same size for balanced evaluation
                seed=int(args.cp_seed),
                gt_from_npz=bool(args.gt_from_npz),
                norm=str(args.cp_norm),
                conservative_factor=1.2,  # 20% safety margin
            )
            
            # Export safety report
            export_safety_report(
                safety_report=safety_report,
                cp_standard=cp_standard,
                cp_conservative=cp_conservative,
                save_dir=args.save_dir,
            )
            
            # Save both standard and conservative quantiles
            with open(os.path.join(args.save_dir, "cp_quantiles_standard.json"), "w") as f:
                json.dump(cp_standard.to_dict(), f, indent=2)
            with open(os.path.join(args.save_dir, "cp_quantiles_conservative.json"), "w") as f:
                json.dump(cp_conservative.to_dict(), f, indent=2)
            
            print(f"\n✅ Safety evaluation complete. Check {args.save_dir}/cp_safety_summary.txt")
            
            # Use conservative quantiles for visualization if safety is questionable
            if safety_report.safety_margin < 0 or not safety_report.bonferroni_corrected_passed:
                print("\n⚠️  Using CONSERVATIVE CP quantiles for subsequent visualizations")
                cp_quant_path = os.path.join(args.save_dir, "cp_quantiles_conservative.json")
            else:
                print("\n✅ Using STANDARD CP quantiles (safety requirements met)")
                cp_quant_path = os.path.join(args.save_dir, "cp_quantiles_standard.json")
    
    if args.cp_calibrate or args.cp_eval:
        # Deterministic split by indices
        total_n = len(dataset)
        calib_n = min(int(args.cp_calib_size), total_n)
        gen = torch.Generator().manual_seed(int(args.cp_seed))
        perm = torch.randperm(total_n, generator=gen).tolist()
        calib_idx = perm[:calib_n]
        test_idx = perm[calib_n:]

        calib_loader = DataLoader(torch.utils.data.Subset(dataset, calib_idx),
                                  batch_size=args.batch_size, shuffle=False, num_workers=0,
                                  pin_memory=True if device.type == 'cuda' else False)
        test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx),
                                 batch_size=args.batch_size, shuffle=False, num_workers=0,
                                 pin_memory=True if device.type == 'cuda' else False)

        H = int(args.max_horizon)
        alpha = float(args.cp_alpha)
        norm = str(args.cp_norm)

        # Collect scores on calibration set
        def _collect_scores(loader) -> np.ndarray:
            all_scores = []
            model.eval()
            with torch.no_grad():
                for batch in loader:
                    input_frames = batch["input_frames"].to(device)
                    B = input_frames.shape[0]
                    T_in = input_frames.shape[1]
                    target_offset = int(batch.get('target_offset', torch.tensor(1))[0].item()) if isinstance(batch.get('target_offset', 1), torch.Tensor) else int(batch.get('target_offset', 1))
                    start_idx = target_offset - 1
                    if start_idx < 0 or start_idx >= T_in:
                        continue

                    # Actions
                    actions_seq = None
                    if "actions" in batch and model.action_dim > 0:
                        actions_seq = batch["actions"].to(device)

                    # Optionally fetch GT frames+actions from NPZ for horizons beyond window
                    gt_frames_full = None
                    gt_actions_full = None
                    gt_steps_avail = 0
                    if bool(args.gt_from_npz):
                        ds = loader.dataset.dataset if isinstance(loader.dataset, torch.utils.data.Subset) else loader.dataset
                        # Need global indices to locate; TrajectoryDataset returns global_idx
                        if "global_idx" in batch:
                            idxs = batch["global_idx"].tolist()
                            gt_list = []
                            act_list = []
                            for gi in idxs:
                                file_idx, local_start = ds._locate(int(gi))
                                f = ds.files[file_idx]
                                s0 = int(local_start)
                                frames = f["frames"]
                                actions = f["actions"]
                                avail_future = int(frames.shape[0] - (s0 + start_idx + 1))
                                take = max(0, min(H, avail_future))
                                if take > 0:
                                    gt_u8 = frames[(s0 + start_idx + 1):(s0 + start_idx + 1 + take)]
                                    gt_f = torch.from_numpy(gt_u8.astype(np.float32)) / 255.0
                                else:
                                    gt_f = torch.zeros((0, *frames.shape[1:]), dtype=torch.float32)
                                gt_list.append(gt_f)
                                if model.action_dim > 0:
                                    if take > 0:
                                        raw_a = actions[(s0 + start_idx):(s0 + start_idx + take)]
                                        a = torch.from_numpy(raw_a.astype(np.float32))
                                        mean = torch.from_numpy(ds.act_mean.astype(np.float32))
                                        std = torch.from_numpy(ds.act_std.astype(np.float32))
                                        a = (a - mean) / std
                                        a = torch.clamp(a, -3.0, 3.0) / 3.0
                                    else:
                                        a = torch.zeros((0, int(actions.shape[1])), dtype=torch.float32)
                                    act_list.append(a)
                            gt_steps_avail = min([g.shape[0] for g in gt_list]) if gt_list else 0
                            if gt_steps_avail > 0:
                                gt_frames_full = torch.stack([g[:gt_steps_avail] for g in gt_list], dim=0).to(device)
                                if model.action_dim > 0:
                                    gt_actions_full = torch.stack([a[:gt_steps_avail] for a in act_list], dim=0).to(device)

                    # Start latent
                    first_frame = input_frames[:, start_idx, ...]
                    mu_start, _ = model.encode(first_frame)
                    z_cur = mu_start

                    # Encode GT latents (mu) for horizons where available
                    if gt_frames_full is not None:
                        T_cmp = int(gt_steps_avail)
                        gt_flat = gt_frames_full.reshape(B*T_cmp, *gt_frames_full.shape[2:])
                        mu_target, _ = model.encode(gt_flat)
                        z_target = mu_target.reshape(B, T_cmp, *mu_target.shape[1:]) if model.vae_encoder is not None else mu_target.reshape(B, T_cmp, -1)
                    else:
                        target_frames = batch["target_frames"].to(device)
                        T_cmp = int(target_frames.shape[1])
                        tf_flat = target_frames.reshape(B*T_cmp, *target_frames.shape[2:])
                        mu_target, _ = model.encode(tf_flat)
                        z_target = mu_target.reshape(B, T_cmp, *mu_target.shape[1:]) if model.vae_encoder is not None else mu_target.reshape(B, T_cmp, -1)

                    # Rollout and score
                    scores_row = torch.full((B, H), float("nan"), device=device)
                    for t in range(1, H + 1):
                        z_in = z_cur.unsqueeze(1)
                        a_step = None
                        if gt_actions_full is not None:
                            if gt_actions_full.size(1) >= t:
                                a_step = gt_actions_full[:, t-1:t, :]
                            else:
                                a_step = gt_actions_full[:, -1:, :]
                        elif actions_seq is not None:
                            action_idx = min(start_idx + t - 1, actions_seq.shape[1] - 1)
                            a_step = actions_seq[:, action_idx:action_idx+1, :]
                        z_next = model.predict(z_in, a_step).squeeze(1)
                        if t <= T_cmp:
                            s = nonconformity_scores(z_next, z_target[:, t-1, ...], norm=norm)
                            scores_row[:, t-1] = s
                        z_cur = z_next
                    all_scores.append(scores_row.detach().cpu().numpy())
            if not all_scores:
                return np.zeros((0, H), dtype=np.float64)
            return np.concatenate(all_scores, axis=0)

        q_obj: Optional[CPQuantiles] = None
        if args.cp_calibrate:
            calib_scores = _collect_scores(calib_loader)
            q = quantiles_per_horizon(calib_scores, alpha=alpha)
            q_obj = CPQuantiles(alpha=alpha, norm=norm, q=q)
            with open(cp_quant_path, "w", encoding="utf-8") as f:
                json.dump(q_obj.to_dict(), f, indent=2)
            print(f"\n[CP] Saved quantiles to {cp_quant_path}")
            print(f"[CP] q summary: {set_size_summary(q)}")

        if args.cp_eval:
            if q_obj is None:
                with open(cp_quant_path, "r", encoding="utf-8") as f:
                    q_obj = CPQuantiles.from_dict(json.load(f))
                print(f"\n[CP] Loaded quantiles from {cp_quant_path}")
            test_scores = _collect_scores(test_loader)
            cov = coverage_per_horizon(test_scores, q_obj.q)
            cp_results = {
                "alpha": q_obj.alpha,
                "norm": q_obj.norm,
                "calib_size": calib_n,
                "test_size": max(0, total_n - calib_n),
                "q": q_obj.q.tolist(),
                "coverage": cov.tolist(),
                "q_summary": set_size_summary(q_obj.q),
                "coverage_mean": float(np.nanmean(cov)) if np.isfinite(cov).any() else float("nan"),
            }
            print(f"\n[CP] Mean coverage across horizons: {cp_results['coverage_mean']:.4f} (target={1.0-alpha:.4f})")
            # Export coverage CSV
            cp_csv = os.path.join(args.save_dir, "cp_coverage.csv")
            with open(cp_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["horizon", "q", "coverage"])
                for t in range(1, H + 1):
                    w.writerow([t, q_obj.q[t-1], cov[t-1]])
            print(f"[CP] Saved CP coverage to {cp_csv}")

        # ---- CP Visualization ----
        # Always try to write plots when CP is requested (matplotlib already used in this script).
        try:
            # Plot q_t vs horizon
            q_plot = os.path.join(args.save_dir, "cp_quantiles.png")
            plt.figure(figsize=(10, 4))
            plt.plot(list(range(1, H + 1)), list(q_obj.q), linewidth=2)
            plt.xlabel("Horizon (steps)")
            plt.ylabel("q_t (radius)")
            plt.title(f"Conformal quantiles per horizon (alpha={alpha}, norm={norm})")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(q_plot, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[CP] Saved plot: {q_plot}")

            # Plot coverage vs horizon if available
            if args.cp_eval and cp_results is not None:
                cov_plot = os.path.join(args.save_dir, "cp_coverage.png")
                plt.figure(figsize=(10, 4))
                xs = list(range(1, H + 1))
                ys = [float(x) if np.isfinite(x) else float("nan") for x in cov]
                plt.plot(xs, ys, linewidth=2, label="empirical coverage")
                plt.axhline(1.0 - alpha, linestyle="--", linewidth=1.5, label=f"target {1.0-alpha:.2f}")
                plt.ylim(0.0, 1.0)
                plt.xlabel("Horizon (steps)")
                plt.ylabel("Coverage")
                plt.title("Conformal coverage vs horizon")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(cov_plot, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"[CP] Saved plot: {cov_plot}")
        except Exception as e:
            print(f"[CP] Warning: failed to generate CP plots: {e}")

    # Single-trajectory CP band visualization (needs saved quantiles)
    if args.cp_traj_plot:
        try:
            visualize_cp_trajectory_band(
                model=model,
                dataset=dataset,
                device=device,
                cp_quantiles_path=cp_quant_path,
                sample_idx=int(args.cp_traj_sample_idx),
                horizon=int(args.cp_traj_horizon),
                pca_fit_samples=int(args.cp_traj_pca_samples),
                seed=int(args.cp_seed),
                gt_from_npz=bool(args.gt_from_npz),
                save_dir=args.save_dir,
                manual_scale=float(args.cp_traj_scale),
            )
        except Exception as e:
            print(f"[CP] Warning: failed to generate CP trajectory band plot: {e}")

    if args.cp_boundary_plot:
        try:
            visualize_cp_boundary_decode(
                model=model,
                dataset=dataset,
                device=device,
                cp_quantiles_path=cp_quant_path,
                sample_idx=int(args.cp_traj_sample_idx),
                step_t=int(args.cp_boundary_step),
                num_boundary_samples=int(args.cp_boundary_num),
                seed=int(args.cp_seed),
                gt_from_npz=bool(args.gt_from_npz),
                save_dir=args.save_dir,
            )
        except Exception as e:
            print(f"[CP] Warning: failed to generate CP boundary decode plot: {e}")
    
    # Check 3: Visualization
    if run_vis:
        visualize_predictions(model, dataloader, device, 
                              num_samples=args.num_vis_samples, 
                              rollout_steps=args.max_horizon,
                              save_dir=args.save_dir)
    
    # Additional: 30-step rollout image visualization
    if run_vis and args.max_horizon >= 30:
        print("\n" + "="*70)
        print("Creating 30-step Rollout Image Visualization")
        print("="*70)
        rollout_img_path = os.path.join(args.save_dir, 'rollout_30step.png')
        visualize_rollout_images(model, dataset, device, rollout_img_path, 
                                rollout_steps=30, sample_idx=0)
    
    # Plot rollout MSE vs horizon
    if run_rollout_plot and rollout_results is not None:
        print("\n" + "="*70)
        print("Plotting Rollout MSE vs Horizon")
        print("="*70)
        curves = rollout_results.get("curves", {})
        plt.figure(figsize=(10, 6))
        # If linear degenerates to identity, they can overlap perfectly; draw linear AFTER identity so it's visible.
        plot_order = ["identity", "linear", "lstm"]
        style = {
            "lstm": dict(linestyle="-", marker="o", linewidth=2, markersize=5, zorder=3),
            "identity": dict(linestyle="--", marker="o", linewidth=2, markersize=4, alpha=0.7, zorder=1),
            "linear": dict(linestyle=":", marker="o", linewidth=2, markersize=4, zorder=2),
        }
        for name in plot_order:
            if name not in curves:
                continue
            c = curves[name]
            latent_curve = c.get("latent_mse", {})
            horizons = sorted([h for h, v in latent_curve.items() if v is not None])
            mses = [latent_curve[h] for h in horizons]
            if horizons:
                plt.plot(horizons, mses, label=name, **style.get(name, {}))
        plt.xlabel('Horizon (steps)', fontsize=12)
        plt.ylabel('MSE', fontsize=12)
        plt.title('Multi-step Open-loop Rollout (Latent): MSE vs Horizon', fontsize=14)
        plt.grid(True, alpha=0.3)
        if curves:
            plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(args.save_dir, 'rollout_mse_vs_horizon.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {plot_path}")
    
    # Save results (optional)
    results = {
        'baseline': baseline_results,
        'rollout': rollout_results
    }
    
    # ---- Structured exports (JSON + CSV) for papers/reports ----
    # When running with --only_cp / --skip_check2, rollout_results can be None.
    curves = (rollout_results or {}).get("curves", {})
    thresholds = {
        "psnr_threshold": float(args.psnr_threshold),
        "ssim_threshold": float(args.ssim_threshold),
        "mse_threshold": float(args.mse_threshold),
    }
    effective = {}
    for name, c in curves.items():
        effective[name] = {
            "psnr": effective_horizon_from_curve(c.get("psnr", {}), lambda x: x >= thresholds["psnr_threshold"]),
            "ssim": effective_horizon_from_curve(c.get("ssim", {}), lambda x: x >= thresholds["ssim_threshold"]),
            "img_mse": effective_horizon_from_curve(c.get("img_mse", {}), lambda x: x <= thresholds["mse_threshold"]),
        }

    export = {
        "args": vars(args),
        "thresholds": thresholds,
        "effective_horizon": effective,
        "baseline": baseline_results,
        "rollout": rollout_results,
        "cp": cp_results,
    }

    if run_exports:
        results_path = os.path.join(args.save_dir, 'eval_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {results_path}")

    if run_exports:
        export_path = os.path.join(args.save_dir, "rollout_metrics.json")
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2)
        print(f"Saved rollout metrics to {export_path}")

    # Curves CSVs: horizon,lstm,identity,linear for each metric
    def _write_curve_csv(metric_key: str, out_name: str):
        if not curves:
            return
        all_h = set()
        for c in curves.values():
            all_h.update([int(h) for h in c.get(metric_key, {}).keys()])
        horizons = sorted(all_h)
        out_path = os.path.join(args.save_dir, out_name)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["horizon", "lstm", "identity", "linear"])
            for h in horizons:
                row = [h]
                for name in ["lstm", "identity", "linear"]:
                    v = curves.get(name, {}).get(metric_key, {}).get(h, None)
                    row.append("" if v is None else float(v))
                w.writerow(row)
        print(f"Saved CSV: {out_path}")

    if run_exports and rollout_results is not None:
        _write_curve_csv("latent_mse", "rollout_latent_mse.csv")
        _write_curve_csv("img_mse", "rollout_img_mse.csv")
        _write_curve_csv("psnr", "rollout_psnr.csv")
        _write_curve_csv("ssim", "rollout_ssim.csv")

    # Effective horizon CSV
    if run_exports and rollout_results is not None:
        eff_path = os.path.join(args.save_dir, "effective_horizon.csv")
        with open(eff_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["method", "psnr>=thr", "ssim>=thr", "img_mse<=thr"])
            for name in ["lstm", "identity", "linear"]:
                e = effective.get(name, {})
                w.writerow([name, e.get("psnr", ""), e.get("ssim", ""), e.get("img_mse", "")])
        print(f"Saved CSV: {eff_path}")
    
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
