#!/usr/bin/env python3
"""
Core evaluation metrics: baseline vs LSTM, multi-step rollout
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict
from torch.utils.data import DataLoader

from predictor.core.vae_predictor import VAEPredictor
from predictor.evaluation.eval_utils import to_01, psnr_from_mse, ssim


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
            target_offset = int(batch.get('target_offset', torch.tensor(1))[0].item()) if isinstance(batch.get('target_offset', 1), torch.Tensor) else int(batch.get('target_offset', 1))
            if target_offset != 1 or target_frames.shape[1] != T:
                # This check is specifically designed for next-step teacher forcing alignment.
                if batch_idx == 0:
                    print(f"[Skip Check1] baseline-vs-lstm assumes target_offset=1 and target_length==input_length. "
                          f"Got target_offset={target_offset}, input_length={T}, target_length={target_frames.shape[1]}.")
                continue
            
            # Get actions if available
            actions_seq = None
            if 'actions' in batch and model.action_dim > 0:
                actions_full = batch['actions'].to(device)
                actions_seq = actions_full[:, 0:T, :]  # align with input sequence length
            
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
    
    if num_samples == 0:
        print("\n[WARNING] No valid samples found for baseline vs LSTM comparison!")
        print("  Possible reasons:")
        print("  1. Dataset is empty")
        print("  2. target_offset configuration doesn't match data")
        print("  3. All batches were skipped due to misalignment")
        return {
            'baseline_mse': float('nan'),
            'lstm_mse': float('nan'),
            'improvement': float('nan')
        }
    
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
                               max_horizon: int = 15, max_batches: int = None,
                               psnr_threshold: float = 20.0, ssim_threshold: float = 0.5,
                               mc_samples: int = 1,
                               gt_from_npz: bool = False,
                               mse_threshold: float = 0.01) -> Dict:
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

    methods = ["lstm", "identity", "linear"]

    def _init_hdict():
        return {h: [] for h in range(1, max_horizon + 1)}

    horizon_latent_mse = {m: _init_hdict() for m in methods}
    horizon_img_mse = {m: _init_hdict() for m in methods}
    horizon_psnr = {m: _init_hdict() for m in methods}
    horizon_ssim = {m: _init_hdict() for m in methods}
    # Only meaningful for the learned model
    horizon_latent_std = _init_hdict()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            input_frames = batch['input_frames'].to(device)  # (B, 15, 3, 64, 64)
            target_frames = batch['target_frames'].to(device)  # (B, 15, 3, 64, 64)
            B = input_frames.shape[0]
            T_in = input_frames.shape[1]
            T = target_frames.shape[1]
            target_offset = int(batch.get('target_offset', torch.tensor(1))[0].item()) if isinstance(batch.get('target_offset', 1), torch.Tensor) else int(batch.get('target_offset', 1))
            start_idx = target_offset - 1
            if start_idx < 0 or start_idx >= T_in:
                raise ValueError(f"Bad alignment: need 0 <= target_offset-1 < input_length. "
                                 f"Got target_offset={target_offset}, input_length={T_in}.")
            
            # Get actions if available
            actions_seq = None
            if 'actions' in batch and model.action_dim > 0:
                actions_full = batch['actions'].to(device)  # (B, L, action_dim)
                actions_seq = actions_full  # keep full window; we'll index with start_idx + step - 1

            # Optionally pull GT (and actions) directly from the underlying NPZ memmap,
            # so we can evaluate horizons beyond the sampled window even if sequence_length is small.
            gt_frames_full = None  # (B, K, 3, H, W) float32 in [0,1]
            gt_actions_full = None  # (B, K, A) normalized like dataset, aligned to transitions (step k uses index k-1)
            gt_steps_avail = 0
            if gt_from_npz:
                if not hasattr(dataloader, "dataset") or not hasattr(dataloader.dataset, "_locate"):
                    raise ValueError("gt_from_npz requires dataloader.dataset to be TrajectoryDataset")
                if "global_idx" not in batch:
                    raise ValueError("gt_from_npz requires TrajectoryDataset to return 'global_idx'")
                ds = dataloader.dataset
                idxs = batch["global_idx"].tolist()
                gt_list = []
                act_list = []
                for gi in idxs:
                    file_idx, local_start = ds._locate(int(gi))
                    f = ds.files[file_idx]
                    s0 = int(local_start)
                    # We start rollout from frame (s0 + start_idx). Step 1 compares to frame (s0 + start_idx + 1).
                    frames = f["frames"]  # (T,3,H,W) uint8
                    actions = f["actions"]  # (T,A) float32
                    max_take = int(max_horizon)
                    # How many future GT frames can we fetch within this file?
                    # We compare step 1..K to frames (s0+start_idx+1 .. s0+start_idx+K)
                    avail_future = int(frames.shape[0] - (s0 + start_idx + 1))
                    take = max(0, min(max_take, avail_future))
                    # Fetch frames for steps 1..take: indices (s0+start_idx+1) .. (s0+start_idx+take)
                    if take > 0:
                        gt_u8 = frames[(s0 + start_idx + 1):(s0 + start_idx + 1 + take)]
                        gt_f = torch.from_numpy(gt_u8.astype(np.float32)) / 255.0
                    else:
                        gt_f = torch.zeros((0, *frames.shape[1:]), dtype=torch.float32)
                    gt_list.append(gt_f)

                    if model.action_dim > 0:
                        # Actions aligned to transitions: for step k (1-based), use action at index (s0+start_idx+k-1)
                        if take > 0:
                            raw_a = actions[(s0 + start_idx):(s0 + start_idx + take)]  # (take, A)
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
                    gt_frames_full = torch.stack([g[:gt_steps_avail] for g in gt_list], dim=0).to(device)  # (B,K,3,H,W)
                    if model.action_dim > 0:
                        gt_actions_full = torch.stack([a[:gt_steps_avail] for a in act_list], dim=0).to(device)  # (B,K,A)
                else:
                    gt_frames_full = None
                    gt_actions_full = None
            
            # Encode first frame for rollout start
            first_frame = input_frames[:, start_idx, ...]  # (B, 3, 64, 64)
            mu_start, _ = model.encode(first_frame)
            
            # Encode target frames for comparison (either from window, or from NPZ future frames)
            if gt_frames_full is not None:
                T_cmp = int(gt_steps_avail)
                gt_flat = gt_frames_full.reshape(B*T_cmp, *gt_frames_full.shape[2:])
                mu_target, _ = model.encode(gt_flat)
            else:
                T_cmp = int(T)
                target_flat = target_frames.reshape(B*T_cmp, *target_frames.shape[2:])
                mu_target, _ = model.encode(target_flat)
            
            if model.vae_encoder is not None:
                z_start = mu_start  # (B, C, 4, 4)
                z_target = mu_target.reshape(B, T_cmp, *mu_target.shape[1:])  # (B, T, C, 4, 4)
            else:
                z_start = mu_start  # (B, D)
                z_target = mu_target.reshape(B, T_cmp, -1)  # (B, T, D)
            
            # Rollout states for each method
            z_cur_lstm = z_start
            z_cur_id = z_start
            z_cur_lin = z_start
            # z_{t-1} for linear extrap. If we don't have a previous observed frame (start_idx==0),
            # linear extrap degenerates to identity (and will overlap).
            if start_idx > 0:
                prev_frame = input_frames[:, start_idx - 1, ...]
                mu_prev, _ = model.encode(prev_frame)
                z_prev_lin = mu_prev
            else:
                z_prev_lin = z_start
            
            for step in range(1, max_horizon + 1):
                # Prepare input for learned model single-step prediction
                z_current_expanded = z_cur_lstm.unsqueeze(1)
                
                # Get action for this step if available
                a_step = None
                if gt_actions_full is not None:
                    # Use the real future actions aligned to GT-from-NPZ horizons.
                    # step is 1-based, so use index step-1.
                    if gt_actions_full.size(1) >= step:
                        a_step = gt_actions_full[:, (step - 1):(step - 1) + 1, :]
                    else:
                        # If we ran out of actions (near end-of-file), repeat the last available one.
                        a_step = gt_actions_full[:, -1:, :]
                elif actions_seq is not None:
                    # Fallback: use window actions; for horizons beyond window we repeat last action.
                    action_idx = min(start_idx + step - 1, actions_seq.shape[1] - 1)
                    a_step = actions_seq[:, action_idx:action_idx+1, :]  # (B, 1, action_dim)
                
                # Predict next latent (optionally MC dropout for epistemic uncertainty)
                if int(mc_samples) > 1 and hasattr(model, "predict_mc"):
                    mc = model.predict_mc(z_current_expanded, a_step, mc_samples=int(mc_samples), enable_dropout=True)
                    z_next = mc["mean"].squeeze(1)
                    z_std = mc["std"].squeeze(1)
                else:
                    z_next = model.predict(z_current_expanded, a_step).squeeze(1)
                    z_std = None

                # Baselines:
                # 1) Identity: z_{t+1} = z_t
                z_next_id = z_cur_id
                # 2) Linear extrapolation: z_{t+1} = z_t + (z_t - z_{t-1})
                # For step==1, fall back to identity (no z_{t-1}).
                if step == 1:
                    z_next_lin = z_cur_lin
                else:
                    z_next_lin = z_cur_lin + (z_cur_lin - z_prev_lin)
                
                # Compare with target only if we have target frames (step <= T)
                if step <= T_cmp:
                    z_target_step = z_target[:, step-1, ...]  # (B, C, 4, 4) or (B, D)
                    
                    def _latent_mse(zp: torch.Tensor) -> float:
                        if model.vae_encoder is not None:
                            return F.mse_loss(zp.reshape(B, -1), z_target_step.reshape(B, -1), reduction="mean").item()
                        return F.mse_loss(zp, z_target_step, reduction="mean").item()

                    horizon_latent_mse["lstm"][step].append(_latent_mse(z_next))
                    horizon_latent_mse["identity"][step].append(_latent_mse(z_next_id))
                    horizon_latent_mse["linear"][step].append(_latent_mse(z_next_lin))

                    if z_std is not None:
                        # summarize latent std magnitude for this horizon
                        if model.vae_encoder is not None:
                            s = z_std.reshape(B, -1).mean(dim=1).mean().item()
                        else:
                            s = z_std.mean(dim=1).mean().item()
                        horizon_latent_std[step].append(float(s))

                    # Image-space metrics (decode without skip features to avoid leakage)
                    try:
                        gt_img = to_01(gt_frames_full[:, step-1, ...]) if gt_frames_full is not None else to_01(target_frames[:, step-1, ...])

                        pred_img = to_01(model.decode_images(z_next))
                        im_mse = F.mse_loss(pred_img, gt_img, reduction='mean').item()
                        horizon_img_mse["lstm"][step].append(im_mse)
                        horizon_psnr["lstm"][step].append(psnr_from_mse(im_mse))
                        horizon_ssim["lstm"][step].append(ssim(pred_img, gt_img))

                        pred_img_id = to_01(model.decode_images(z_next_id))
                        im_mse_id = F.mse_loss(pred_img_id, gt_img, reduction='mean').item()
                        horizon_img_mse["identity"][step].append(im_mse_id)
                        horizon_psnr["identity"][step].append(psnr_from_mse(im_mse_id))
                        horizon_ssim["identity"][step].append(ssim(pred_img_id, gt_img))

                        pred_img_lin = to_01(model.decode_images(z_next_lin))
                        im_mse_lin = F.mse_loss(pred_img_lin, gt_img, reduction='mean').item()
                        horizon_img_mse["linear"][step].append(im_mse_lin)
                        horizon_psnr["linear"][step].append(psnr_from_mse(im_mse_lin))
                        horizon_ssim["linear"][step].append(ssim(pred_img_lin, gt_img))
                    except Exception:
                        # If decoder is unavailable or decode fails, skip image metrics.
                        pass
                
                # Advance rollouts
                z_cur_lstm = z_next
                z_cur_id = z_next_id
                z_prev_lin = z_cur_lin
                z_cur_lin = z_next_lin
            
            if batch_idx % 10 == 0 or (batch_idx + 1) == total_batches:
                progress = (batch_idx + 1) / total_batches * 100
                print(f"Batch {batch_idx+1}/{total_batches} ({progress:.1f}%): Rollout completed")
    
    def _avg_curve(hdict):
        out = {}
        for h in range(1, max_horizon + 1):
            out[h] = float(np.mean(hdict[h])) if hdict[h] else None
        return out

    curves = {}
    for m in methods:
        curves[m] = {
            "latent_mse": _avg_curve(horizon_latent_mse[m]),
            "img_mse": _avg_curve(horizon_img_mse[m]),
            "psnr": _avg_curve(horizon_psnr[m]),
            "ssim": _avg_curve(horizon_ssim[m]),
        }

    avg_latent_std = _avg_curve(horizon_latent_std)

    # If linear is identical to identity (common when target_offset==1 so there's no previous frame),
    # warn the user so they don't think it's missing.
    try:
        diffs = []
        for h in range(1, max_horizon + 1):
            a = curves["identity"]["latent_mse"].get(h, None)
            b = curves["linear"]["latent_mse"].get(h, None)
            if a is None or b is None:
                continue
            diffs.append(abs(float(a) - float(b)))
        if diffs and max(diffs) < 1e-12:
            print("\nNote: 'linear' baseline overlaps 'identity' here (degenerate case).")
            print("  Reason: linear extrap needs two past frames; with target_offset=1 the rollout starts at frame 0, so we can't estimate velocity.")
            print("  Fix: use a start index > 0 (e.g., set target_offset=2 and target_length=14 with sequence_length=16), or evaluate a later start point.")
    except Exception:
        pass
    
    for m in methods:
        print(f"\nResults ({m}): latent MSE vs horizon")
        for h in range(1, max_horizon + 1):
            v = curves[m]["latent_mse"][h]
            if v is not None:
                print(f"  Horizon {h:2d}: {v:.6f}")

        if any(v is not None for v in curves[m]["img_mse"].values()):
            print(f"\nImage-space metrics ({m}):")
            for h in range(1, max_horizon + 1):
                im = curves[m]["img_mse"][h]
                if im is None:
                    continue
                print(f"  Horizon {h:2d}: img_mse={im:.6f}  psnr={curves[m]['psnr'][h]:.2f}  ssim={curves[m]['ssim'][h]:.3f}")

            # Effective horizon by thresholds
            from predictor.evaluation.eval_utils import effective_horizon_from_curve
            eff_psnr = effective_horizon_from_curve(curves[m]["psnr"], lambda x: x >= float(psnr_threshold))
            eff_ssim = effective_horizon_from_curve(curves[m]["ssim"], lambda x: x >= float(ssim_threshold))
            eff_mse = effective_horizon_from_curve(curves[m]["img_mse"], lambda x: x <= float(mse_threshold))
            print(f"\nEffective horizon ({m}): PSNR≥{psnr_threshold} => {eff_psnr} steps; "
                  f"SSIM≥{ssim_threshold} => {eff_ssim} steps; "
                  f"imgMSE≤{mse_threshold} => {eff_mse} steps")

    if any(v is not None for v in avg_latent_std.values()):
        print("\nUncertainty (MC-dropout latent std, mean over dims+batch) [lstm only]:")
        for h in range(1, max_horizon + 1):
            if avg_latent_std[h] is None:
                continue
            print(f"  Horizon {h:2d}: latent_std={avg_latent_std[h]:.6f}")

    return {
        # Back-compat: keep top-level keys as the learned model
        "latent_mse": curves["lstm"]["latent_mse"],
        "img_mse": curves["lstm"]["img_mse"],
        "psnr": curves["lstm"]["psnr"],
        "ssim": curves["lstm"]["ssim"],
        # New: all methods
        "curves": curves,
        "latent_std_mc": avg_latent_std,
    }
