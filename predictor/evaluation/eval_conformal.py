#!/usr/bin/env python3
"""
Conformal Prediction (CP) evaluation and visualizations
"""

import torch
import numpy as np
import os
import json
from typing import Tuple
from matplotlib import patches
import matplotlib.pyplot as plt

from predictor.core.vae_predictor import VAEPredictor, TrajectoryDataset
from predictor.conformal.conformal import CPQuantiles
from predictor.evaluation.eval_utils import to_01


def fit_pca_2d(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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


def pca_project(X: np.ndarray, mean: np.ndarray, comps: np.ndarray) -> np.ndarray:
    """Project latents X onto 2D PCA space."""
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
    manual_scale: float = 0.01,
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
    mean, comps = fit_pca_2d(X)

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

    Pp = pca_project(zp, mean, comps)  # (T,2)
    Pg = pca_project(zg, mean, comps)  # (T,2)

    # Compute automatic scaling for reference
    dists_2d = np.linalg.norm(Pp - Pg, axis=1)  # (T,)
    dists_highd = np.linalg.norm(zp - zg, axis=1)  # (T,)
    scale_factors = np.where(dists_highd > 1e-6, dists_2d / dists_highd, 1.0)
    scale_auto = float(np.median(scale_factors[np.isfinite(scale_factors)]))
    
    # Use manual scale (user can adjust via --cp_traj_scale)
    scale = float(manual_scale)
    
    trajectory_spread = np.std(Pp, axis=0).max()
    print(f"[CP Vis] Scaling: manual={scale:.6f}, auto={scale_auto:.6f}, trajectory_spread={trajectory_spread:.2f}")
    print(f"[CP Vis] Using manual scale: {scale:.6f} (adjust with --cp_traj_scale)")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(Pp[:, 0], Pp[:, 1], "-o", linewidth=2, markersize=4, label="Pred (LSTM)")
    ax.plot(Pg[:, 0], Pg[:, 1], "-o", linewidth=2, markersize=4, label="GT")

    # Confidence band (q_t) - SCALED to 2D space
    for t in range(gt_steps):
        qt = float(q[t]) * scale  # Apply scaling
        if not np.isfinite(qt):
            continue
        if q_obj.norm.lower() == "linf":
            rect = patches.Rectangle((Pp[t, 0] - qt, Pp[t, 1] - qt), 2*qt, 2*qt, linewidth=0, alpha=0.10)
            ax.add_patch(rect)
        else:
            circ = patches.Circle((Pp[t, 0], Pp[t, 1]), radius=qt, linewidth=0, alpha=0.10)
            ax.add_patch(circ)

    ax.set_title(f"CP band in PCA latent space (alpha={q_obj.alpha}, norm={q_obj.norm})\nSample {sample_idx}, steps={gt_steps}, scale={scale:.4f} (64D→2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal', adjustable='box')  # Make sure circles look circular
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
        imgs = to_01(imgs).detach().cpu().numpy().transpose(0, 2, 3, 1)  # (K+1,H,W,3)

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
