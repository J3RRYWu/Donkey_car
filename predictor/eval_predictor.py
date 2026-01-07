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
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib import patches
import csv
import json

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
from conformal import CPQuantiles, nonconformity_scores, quantiles_per_horizon, coverage_per_horizon, set_size_summary


def _fit_pca_2d(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a simple PCA (2 components) using SVD.
    Returns (mean: (D,), components: (2,D)) such that proj = (x-mean) @ components.T.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError(f"Need X shape (N,D) with N>=2, got {X.shape}")
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    # SVD on centered data
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    comps = vt[:2, :]
    return mean.reshape(-1), comps


def _pca_project(X: np.ndarray, mean: np.ndarray, comps: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return (X - mean.reshape(1, -1)) @ comps.T


def visualize_cp_trajectory_band(
    model: VAEPredictor,
    dataset: TrajectoryDataset,
    device: torch.device,
    cp_quantiles_path: str,
    sample_idx: int,
    horizon: int,
    pca_fit_samples: int = 500,
    seed: int = 42,
    gt_from_npz: bool = True,
    save_dir: str = "./eval_results",
):
    """Visualize a single rollout in 2D PCA latent space with CP band (q_t) around predictions.

    We project high-D latents to 2D with PCA fit on a random subset of dataset windows.
    Then plot:
      - predicted latent trajectory (open-loop)
      - GT latent trajectory
      - per-step CP band as circles (L2) or squares (Linf) with radius/halfwidth q_t
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load CP quantiles
    with open(cp_quantiles_path, "r", encoding="utf-8") as f:
        q_obj = CPQuantiles.from_dict(json.load(f))
    q = np.asarray(q_obj.q, dtype=np.float64)
    H = int(min(int(horizon), int(q.shape[0])))
    if H <= 0:
        raise ValueError("Bad horizon or empty quantiles.")

    # Fit PCA on random subset of samples (use start frame latent + first target latent)
    N = len(dataset)
    k = min(int(pca_fit_samples), N)
    gen = torch.Generator().manual_seed(int(seed))
    idxs = torch.randperm(N, generator=gen)[:k].tolist()
    X_list = []
    model.eval()
    with torch.no_grad():
        for gi in idxs:
            s = dataset[int(gi)]
            inp = s["input_frames"].unsqueeze(0).to(device)
            tgt = s["target_frames"].unsqueeze(0).to(device)
            target_offset = int(s.get("target_offset", torch.tensor(1)).item()) if isinstance(s.get("target_offset", 1), torch.Tensor) else int(s.get("target_offset", 1))
            start_idx = target_offset - 1
            start_idx = max(0, min(start_idx, inp.shape[1] - 1))
            mu0, _ = model.encode(inp[:, start_idx, ...])
            X_list.append(mu0.reshape(1, -1).detach().cpu().numpy())
            if tgt.shape[1] > 0:
                mu1, _ = model.encode(tgt[:, 0, ...])
                X_list.append(mu1.reshape(1, -1).detach().cpu().numpy())
    X = np.concatenate(X_list, axis=0) if X_list else None
    if X is None or X.shape[0] < 2:
        raise ValueError("Not enough latents to fit PCA.")
    mean, comps = _fit_pca_2d(X)

    # Get sample window and (optionally) pull longer GT/actions from NPZ
    sample = dataset[int(sample_idx)]
    input_frames = sample["input_frames"].unsqueeze(0).to(device)
    actions_win = sample.get("actions", None)
    actions_win = actions_win.unsqueeze(0).to(device) if (actions_win is not None and getattr(model, "action_dim", 0) > 0) else None
    target_offset = int(sample.get("target_offset", torch.tensor(1)).item()) if isinstance(sample.get("target_offset", 1), torch.Tensor) else int(sample.get("target_offset", 1))
    start_idx = target_offset - 1
    if start_idx < 0 or start_idx >= input_frames.shape[1]:
        raise ValueError(f"Bad target_offset for this sample: target_offset={target_offset}, input_len={input_frames.shape[1]}")

    gt_frames = None
    gt_actions = None
    gt_steps = 0
    if gt_from_npz:
        # Locate in underlying NPZ and fetch future frames/actions
        file_idx, local_start = dataset._locate(int(sample_idx))
        f = dataset.files[file_idx]
        s0 = int(local_start)
        frames = f["frames"]  # uint8
        actions = f["actions"]  # float32
        avail_future = int(frames.shape[0] - (s0 + start_idx + 1))
        gt_steps = max(0, min(H, avail_future))
        if gt_steps > 0:
            gt_u8 = frames[(s0 + start_idx + 1):(s0 + start_idx + 1 + gt_steps)]
            gt_frames = torch.from_numpy(gt_u8.astype(np.float32)).unsqueeze(0).to(device) / 255.0  # (1,gt_steps,3,H,W)
            if getattr(model, "action_dim", 0) > 0:
                raw_a = actions[(s0 + start_idx):(s0 + start_idx + gt_steps)]  # (gt_steps,A)
                a = torch.from_numpy(raw_a.astype(np.float32))
                mean_a = torch.from_numpy(dataset.act_mean.astype(np.float32))
                std_a = torch.from_numpy(dataset.act_std.astype(np.float32))
                a = (a - mean_a) / std_a
                a = torch.clamp(a, -3.0, 3.0) / 3.0
                gt_actions = a.unsqueeze(0).to(device)  # (1,gt_steps,A)
    else:
        tgt = sample["target_frames"].unsqueeze(0).to(device)
        gt_steps = min(H, int(tgt.shape[1]))
        gt_frames = tgt[:, :gt_steps, ...] if gt_steps > 0 else None
        gt_actions = actions_win[:, start_idx:start_idx + gt_steps, :] if (actions_win is not None and gt_steps > 0) else None

    if gt_steps <= 0 or gt_frames is None:
        raise ValueError("No GT frames available for the requested horizon; enable --gt_from_npz or reduce horizon.")

    # Encode start latent + GT latents
    with torch.no_grad():
        mu_start, _ = model.encode(input_frames[:, start_idx, ...])
        z_cur = mu_start
        # GT latents as mu (consistent with CP scoring)
        B1, Tgt = gt_frames.shape[:2]
        gt_flat = gt_frames.reshape(B1*Tgt, *gt_frames.shape[2:])
        mu_gt, _ = model.encode(gt_flat)
        z_gt = mu_gt.reshape(1, gt_steps, *mu_gt.shape[1:]) if model.vae_encoder is not None else mu_gt.reshape(1, gt_steps, -1)

        z_pred_seq = []
        for t in range(1, gt_steps + 1):
            a_step = None
            if gt_actions is not None:
                a_step = gt_actions[:, t-1:t, :]
            elif actions_win is not None:
                # fallback to window action, repeat last
                idx = min(start_idx + t - 1, actions_win.shape[1] - 1)
                a_step = actions_win[:, idx:idx+1, :]
            z_next = model.predict(z_cur.unsqueeze(1), a_step).squeeze(1)
            z_pred_seq.append(z_next)
            z_cur = z_next

    z_pred = torch.stack(z_pred_seq, dim=1)  # (1,gt_steps,...)
    zp = z_pred.reshape(gt_steps, -1).detach().cpu().numpy()
    zg = z_gt.reshape(gt_steps, -1).detach().cpu().numpy()

    Pp = _pca_project(zp, mean, comps)  # (T,2)
    Pg = _pca_project(zg, mean, comps)  # (T,2)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(Pp[:, 0], Pp[:, 1], "-o", linewidth=2, markersize=4, label="Pred (LSTM)")
    ax.plot(Pg[:, 0], Pg[:, 1], "-o", linewidth=2, markersize=4, label="GT")

    # Confidence band (q_t)
    for t in range(gt_steps):
        qt = float(q[t])
        if not np.isfinite(qt):
            continue
        if q_obj.norm.lower() == "linf":
            rect = patches.Rectangle((Pp[t, 0] - qt, Pp[t, 1] - qt), 2*qt, 2*qt, linewidth=0, alpha=0.10)
            ax.add_patch(rect)
        else:
            circ = patches.Circle((Pp[t, 0], Pp[t, 1]), radius=qt, linewidth=0, alpha=0.10)
            ax.add_patch(circ)

    ax.set_title(f"CP band in PCA latent space (alpha={q_obj.alpha}, norm={q_obj.norm})\nSample {sample_idx}, steps={gt_steps}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(save_dir, f"cp_band_traj_sample_{sample_idx}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[CP] Saved trajectory band plot: {out_path}")


def visualize_cp_boundary_decode(
    model: VAEPredictor,
    dataset: TrajectoryDataset,
    device: torch.device,
    cp_quantiles_path: str,
    sample_idx: int,
    step_t: int,
    num_boundary_samples: int = 4,
    seed: int = 42,
    gt_from_npz: bool = True,
    save_dir: str = "./eval_results",
):
    """Lock a horizon t, sample a few points on the CP ball boundary around z_hat_t, and decode to images.

    - Compute z_hat_t by open-loop rollout for a chosen sample.
    - Sample K directions u_k on the L2 unit sphere (in flattened latent space),
      and form z_k = z_hat_t + q_t * u_k (boundary points).
    - Decode {z_hat_t, z_k} to images using the VAE decoder (no skip features).
    """
    os.makedirs(save_dir, exist_ok=True)
    with open(cp_quantiles_path, "r", encoding="utf-8") as f:
        q_obj = CPQuantiles.from_dict(json.load(f))
    q = np.asarray(q_obj.q, dtype=np.float64)
    t = int(step_t)
    if t <= 0 or t > int(q.shape[0]):
        raise ValueError(f"step_t must be in [1,{int(q.shape[0])}], got {t}")
    qt = float(q[t - 1])
    if not np.isfinite(qt) or qt <= 0:
        raise ValueError(f"Bad q_t at t={t}: {qt}")
    if q_obj.norm.lower() != "l2":
        raise ValueError("Boundary sampling currently supports only L2 CP quantiles (cp_norm=l2).")

    sample = dataset[int(sample_idx)]
    input_frames = sample["input_frames"].unsqueeze(0).to(device)
    target_offset = int(sample.get("target_offset", torch.tensor(1)).item()) if isinstance(sample.get("target_offset", 1), torch.Tensor) else int(sample.get("target_offset", 1))
    start_idx = target_offset - 1
    if start_idx < 0 or start_idx >= input_frames.shape[1]:
        raise ValueError(f"Bad target_offset for this sample: target_offset={target_offset}, input_len={input_frames.shape[1]}")

    # Pull actions aligned to rollout if available
    actions_win = sample.get("actions", None)
    actions_win = actions_win.unsqueeze(0).to(device) if (actions_win is not None and getattr(model, "action_dim", 0) > 0) else None
    gt_actions = None
    if gt_from_npz and getattr(model, "action_dim", 0) > 0:
        file_idx, local_start = dataset._locate(int(sample_idx))
        f = dataset.files[file_idx]
        s0 = int(local_start)
        raw_actions = f["actions"]
        avail = int(raw_actions.shape[0] - (s0 + start_idx))
        take = max(0, min(t, avail))
        if take > 0:
            raw_a = raw_actions[(s0 + start_idx):(s0 + start_idx + take)]
            a = torch.from_numpy(raw_a.astype(np.float32))
            mean_a = torch.from_numpy(dataset.act_mean.astype(np.float32))
            std_a = torch.from_numpy(dataset.act_std.astype(np.float32))
            a = (a - mean_a) / std_a
            a = torch.clamp(a, -3.0, 3.0) / 3.0
            gt_actions = a.unsqueeze(0).to(device)  # (1,take,A)

    # Compute z_hat_t by rolling out t steps
    model.eval()
    with torch.no_grad():
        mu_start, _ = model.encode(input_frames[:, start_idx, ...])
        z_cur = mu_start
        for k in range(1, t + 1):
            a_step = None
            if gt_actions is not None:
                if gt_actions.size(1) >= k:
                    a_step = gt_actions[:, k - 1:k, :]
                else:
                    a_step = gt_actions[:, -1:, :]
            elif actions_win is not None:
                idx = min(start_idx + k - 1, actions_win.shape[1] - 1)
                a_step = actions_win[:, idx:idx + 1, :]
            z_cur = model.predict(z_cur.unsqueeze(1), a_step).squeeze(1)
        z_hat = z_cur  # (1, C,4,4) or (1,D)

    # Sample boundary points in flattened space
    torch.manual_seed(int(seed))
    z0 = z_hat[0].detach().cpu()
    z0_flat = z0.reshape(-1)
    D = int(z0_flat.numel())
    K = int(max(1, num_boundary_samples))
    dirs = torch.randn(K, D)
    dirs = dirs / (torch.linalg.vector_norm(dirs, ord=2, dim=1, keepdim=True) + 1e-12)
    z_samples_flat = z0_flat.unsqueeze(0) + float(qt) * dirs  # (K,D)

    # Stack center + boundary samples
    all_flat = torch.cat([z0_flat.unsqueeze(0), z_samples_flat], dim=0)  # (K+1,D)
    if z_hat.dim() == 4:  # conv latent
        C, Hs, Ws = z_hat.shape[1:]
        all_z = all_flat.view(K + 1, C, Hs, Ws).to(device)
    else:
        all_z = all_flat.to(device)

    # Decode to images
    with torch.no_grad():
        imgs = model.decode_images(all_z)  # (K+1,3,64,64)
        imgs = _to_01(imgs).detach().cpu().numpy().transpose(0, 2, 3, 1)  # (K+1,H,W,3)

    # Plot as a single row
    fig, axes = plt.subplots(1, K + 1, figsize=((K + 1) * 3, 3))
    if K + 1 == 1:
        axes = [axes]
    for i in range(K + 1):
        axes[i].imshow(imgs[i])
        axes[i].axis("off")
        if i == 0:
            axes[i].set_title(f"z_hat@{t}", fontsize=10)
        else:
            axes[i].set_title(f"+q_t dir{i}", fontsize=10)
    fig.suptitle(f"CP boundary decode (alpha={q_obj.alpha}, norm={q_obj.norm})  t={t}, q_t={qt:.3f}", fontsize=12)
    fig.tight_layout()

    out_path = os.path.join(save_dir, f"cp_boundary_decode_t{t}_sample_{sample_idx}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[CP] Saved boundary decode plot: {out_path}")


def _to_01(x: torch.Tensor) -> torch.Tensor:
    """Convert model outputs to [0,1] range if they look like tanh outputs."""
    if x.min() < 0:
        x = (x + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0)


def _psnr_from_mse(mse: float, data_range: float = 1.0, eps: float = 1e-12) -> float:
    mse = float(max(mse, eps))
    return float(10.0 * np.log10((data_range ** 2) / mse))


def _gaussian_window(window_size: int = 11, sigma: float = 1.5, device=None, dtype=None) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)  # (1,1,ws,ws)
    return w


def _ssim(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0, window_size: int = 11, sigma: float = 1.5) -> float:
    """SSIM over batch, computed on grayscale for speed. Expects img in [0,1], shape (B,3,H,W)."""
    img1 = _to_01(img1)
    img2 = _to_01(img2)
    # grayscale
    if img1.size(1) == 3:
        w_rgb = img1.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        img1g = (img1 * w_rgb).sum(dim=1, keepdim=True)
        img2g = (img2 * w_rgb).sum(dim=1, keepdim=True)
    else:
        img1g = img1
        img2g = img2

    window = _gaussian_window(window_size, sigma, device=img1.device, dtype=img1.dtype)
    pad = window_size // 2
    mu1 = F.conv2d(img1g, window, padding=pad)
    mu2 = F.conv2d(img2g, window, padding=pad)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(img1g * img1g, window, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(img2g * img2g, window, padding=pad) - mu2_sq
    sigma12 = F.conv2d(img1g * img2g, window, padding=pad) - mu12

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-12)
    return float(ssim_map.mean().item())


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
                        gt_img = _to_01(gt_frames_full[:, step-1, ...]) if gt_frames_full is not None else _to_01(target_frames[:, step-1, ...])

                        pred_img = _to_01(model.decode_images(z_next))
                        im_mse = F.mse_loss(pred_img, gt_img, reduction='mean').item()
                        horizon_img_mse["lstm"][step].append(im_mse)
                        horizon_psnr["lstm"][step].append(_psnr_from_mse(im_mse))
                        horizon_ssim["lstm"][step].append(_ssim(pred_img, gt_img))

                        pred_img_id = _to_01(model.decode_images(z_next_id))
                        im_mse_id = F.mse_loss(pred_img_id, gt_img, reduction='mean').item()
                        horizon_img_mse["identity"][step].append(im_mse_id)
                        horizon_psnr["identity"][step].append(_psnr_from_mse(im_mse_id))
                        horizon_ssim["identity"][step].append(_ssim(pred_img_id, gt_img))

                        pred_img_lin = _to_01(model.decode_images(z_next_lin))
                        im_mse_lin = F.mse_loss(pred_img_lin, gt_img, reduction='mean').item()
                        horizon_img_mse["linear"][step].append(im_mse_lin)
                        horizon_psnr["linear"][step].append(_psnr_from_mse(im_mse_lin))
                        horizon_ssim["linear"][step].append(_ssim(pred_img_lin, gt_img))
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
            eff_psnr = 0
            eff_ssim = 0
            eff_mse = 0
            for h in range(1, max_horizon + 1):
                ps = curves[m]["psnr"][h]
                ss = curves[m]["ssim"][h]
                im = curves[m]["img_mse"][h]
                if ps is not None and ps >= float(psnr_threshold):
                    eff_psnr = h
                if ss is not None and ss >= float(ssim_threshold):
                    eff_ssim = h
                if im is not None and im <= float(mse_threshold):
                    eff_mse = h
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
        T_in = input_frames.shape[1]
        target_offset = int(batch.get('target_offset', torch.tensor(1))[0].item()) if isinstance(batch.get('target_offset', 1), torch.Tensor) else int(batch.get('target_offset', 1))
        start_idx = target_offset - 1
        if start_idx < 0 or start_idx >= T_in:
            raise ValueError(f"Bad alignment: need 0 <= target_offset-1 < input_length. "
                             f"Got target_offset={target_offset}, input_length={T_in}.")
        
        # Get actions if available
        actions_seq = None
        if 'actions' in batch and model.action_dim > 0:
            actions_full = batch['actions'].to(device)
            actions_seq = actions_full[:, 0:T_in, :]  # align with rollout start
        
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
            first_frame = input_seq[:, start_idx, ...]  # (1, 3, 64, 64)
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
                    # Use action aligned to transition (start_idx + step) -> (start_idx + step + 1)
                    action_idx = min(start_idx + step, a_seq.shape[1] - 1)
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
    # Only use actions if the predictor was trained/configured with action_dim > 0.
    if actions is not None and getattr(model, "action_dim", 0) > 0:
        actions = actions.unsqueeze(0).to(device)  # (1, 15, action_dim)
    else:
        actions = None
    
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
    if actions is not None and getattr(model, "action_dim", 0) > 0:
        actions = actions.unsqueeze(0).to(device)  # (1, 16, action_dim) (dataset returns length=sequence_length)
    else:
        actions = None

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
                       help='Conformal alpha (e.g., 0.05 for 95% target coverage).')
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
    parser.add_argument('--cp_traj_plot', action='store_true',
                       help='If set: plot a single test trajectory in 2D PCA latent space with CP band (q_t) around predictions.')
    parser.add_argument('--cp_traj_sample_idx', type=int, default=0,
                       help='Dataset index for CP trajectory band visualization.')
    parser.add_argument('--cp_traj_horizon', type=int, default=50,
                       help='Horizon (steps) for CP trajectory band visualization.')
    parser.add_argument('--cp_traj_pca_samples', type=int, default=500,
                       help='Number of random windows to fit PCA for CP trajectory plot.')
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
                       help='If set: write per-step latent/image deltas to *_debug.json to diagnose \"frozen\" videos')

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
    def _effective_horizon_from_curve(curve: Dict[int, float], predicate) -> int:
        """Largest horizon h such that predicate(curve[h]) is True. curve values may be None."""
        best = 0
        for h in sorted(curve.keys()):
            v = curve[h]
            if v is None:
                continue
            try:
                if predicate(float(v)):
                    best = int(h)
            except Exception:
                continue
        return best

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
            "psnr": _effective_horizon_from_curve(c.get("psnr", {}), lambda x: x >= thresholds["psnr_threshold"]),
            "ssim": _effective_horizon_from_curve(c.get("ssim", {}), lambda x: x >= thresholds["ssim_threshold"]),
            "img_mse": _effective_horizon_from_curve(c.get("img_mse", {}), lambda x: x <= thresholds["mse_threshold"]),
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

