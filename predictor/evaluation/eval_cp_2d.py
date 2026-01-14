#!/usr/bin/env python3
"""
Conformal Prediction in 2D PCA space (theoretically correct visualization)

Instead of:
1. Compute CP in 64D -> get q_t
2. Project to 2D -> draw circles with scaled q_t (WRONG!)

We do:
1. Project all data to 2D first
2. Compute CP in 2D -> get q_t_2d
3. Draw circles with q_t_2d (CORRECT!)
"""

import torch
import numpy as np
from typing import Tuple, List
from sklearn.decomposition import PCA

from predictor.conformal.conformal import predictor.conformal.conformal_quantile, CPQuantiles


def compute_cp_in_2d_pca_space(
    model,
    dataset,
    device: torch.device,
    calib_indices: List[int],
    test_indices: List[int],
    max_horizon: int = 50,
    alpha: float = 0.05,
    pca_components: int = 2,
    pca_fit_samples: int = 500,
    seed: int = 42,
    gt_from_npz: bool = True,
) -> Tuple[CPQuantiles, PCA, np.ndarray, np.ndarray]:
    """
    Compute CP quantiles in 2D PCA space (theoretically correct).
    
    Returns:
        cp_quantiles: CPQuantiles object with q computed in 2D space
        pca: Fitted PCA object
        mean: Mean vector for PCA centering
        components: PCA components
    """
    print("\n" + "="*70)
    print("Computing CP in 2D PCA Space (Theoretically Correct Method)")
    print("="*70)
    
    # Step 1: Fit PCA on a subset
    print(f"Step 1: Fitting PCA ({pca_components}D) on {min(pca_fit_samples, len(dataset))} samples...")
    torch.manual_seed(seed)
    pca_idx = torch.randperm(len(dataset))[:pca_fit_samples].tolist()
    
    latents_for_pca = []
    model.eval()
    with torch.no_grad():
        for idx in pca_idx[:min(pca_fit_samples, len(dataset))]:
            sample = dataset[idx]
            input_frames = sample["input_frames"].unsqueeze(0).to(device)
            target_frames = sample["target_frames"].unsqueeze(0).to(device)
            target_offset = sample.get("target_offset", 1)
            if isinstance(target_offset, torch.Tensor):
                target_offset = int(target_offset.item())
            start_idx = target_offset - 1
            
            if 0 <= start_idx < input_frames.shape[1]:
                mu_start, _ = model.encode(input_frames[:, start_idx, ...])
                latents_for_pca.append(mu_start.reshape(-1).cpu().numpy())
                
                if target_frames.shape[1] > 0:
                    mu_target, _ = model.encode(target_frames[:, 0, ...])
                    latents_for_pca.append(mu_target.reshape(-1).cpu().numpy())
    
    X = np.stack(latents_for_pca)  # (N, 64)
    pca = PCA(n_components=pca_components)
    pca.fit(X)
    explained_var = pca.explained_variance_ratio_
    print(f"   PCA explained variance: {explained_var}")
    print(f"   Total: {explained_var.sum():.4f}")
    
    # Step 2: Collect scores in 2D space on calibration set
    print(f"\nStep 2: Collecting calibration scores in 2D space ({len(calib_indices)} samples)...")
    
    def _collect_scores_2d(indices):
        all_scores = []
        for idx in indices[:min(500, len(indices))]:  # limit for speed
            sample = dataset[idx]
            input_frames = sample["input_frames"].unsqueeze(0).to(device)
            target_frames = sample["target_frames"].unsqueeze(0).to(device)
            target_offset = sample.get("target_offset", 1)
            if isinstance(target_offset, torch.Tensor):
                target_offset = int(target_offset.item())
            start_idx = target_offset - 1
            
            if not (0 <= start_idx < input_frames.shape[1]):
                continue
            
            actions = sample.get("actions", None)
            if actions is not None and model.action_dim > 0:
                actions = actions.unsqueeze(0).to(device)
            
            with torch.no_grad():
                # Encode start
                mu_start, _ = model.encode(input_frames[:, start_idx, ...])
                z_cur = mu_start
                
                # Encode targets and project to 2D
                T = min(max_horizon, target_frames.shape[1])
                if T == 0:
                    continue
                    
                target_flat = target_frames[:, :T, ...].reshape(T, *target_frames.shape[2:])
                mu_target, _ = model.encode(target_flat)
                z_target_64d = mu_target.reshape(T, -1).cpu().numpy()  # (T, 64)
                z_target_2d = pca.transform(z_target_64d)  # (T, 2)
                
                # Rollout and project predictions to 2D
                scores_row = np.full(max_horizon, np.nan)
                for t in range(1, min(T+1, max_horizon+1)):
                    a_step = None
                    if actions is not None and actions.shape[1] > 0:
                        action_idx = min(start_idx + t - 1, actions.shape[1] - 1)
                        a_step = actions[:, action_idx:action_idx+1, :]
                    
                    z_next = model.predict(z_cur.unsqueeze(1), a_step).squeeze(1)
                    z_cur = z_next
                    
                    # Project prediction to 2D
                    z_pred_64d = z_next.reshape(-1).cpu().numpy()  # (64,)
                    z_pred_2d = pca.transform(z_pred_64d.reshape(1, -1))[0]  # (2,)
                    
                    # Compute L2 distance in 2D space
                    if t <= T:
                        dist_2d = np.linalg.norm(z_pred_2d - z_target_2d[t-1])
                        scores_row[t-1] = dist_2d
                
                all_scores.append(scores_row)
        
        return np.stack(all_scores) if all_scores else np.zeros((0, max_horizon))
    
    calib_scores = _collect_scores_2d(calib_indices)
    print(f"   Collected {calib_scores.shape[0]} calibration samples")
    
    # Step 3: Compute quantiles in 2D
    print(f"\nStep 3: Computing quantiles in 2D space (alpha={alpha})...")
    q_2d = np.full(max_horizon, np.nan)
    for t in range(max_horizon):
        scores_t = calib_scores[:, t]
        scores_t = scores_t[np.isfinite(scores_t)]
        if len(scores_t) > 0:
            q_2d[t] = conformal_quantile(scores_t, alpha=alpha)
    
    print(f"   q_2d range: [{np.nanmin(q_2d):.3f}, {np.nanmax(q_2d):.3f}]")
    print(f"   q_2d median: {np.nanmedian(q_2d):.3f}")
    
    # Step 4: Evaluate coverage on test set
    print(f"\nStep 4: Evaluating coverage on test set ({len(test_indices)} samples)...")
    test_scores = _collect_scores_2d(test_indices)
    
    coverage = []
    for t in range(max_horizon):
        scores_t = test_scores[:, t]
        scores_t = scores_t[np.isfinite(scores_t)]
        if len(scores_t) > 0 and np.isfinite(q_2d[t]):
            cov = np.mean(scores_t <= q_2d[t])
            coverage.append(cov)
        else:
            coverage.append(np.nan)
    
    coverage = np.array(coverage)
    mean_cov = np.nanmean(coverage)
    print(f"   Mean coverage: {mean_cov:.4f} (target: {1-alpha:.4f})")
    
    cp_quantiles = CPQuantiles(alpha=alpha, norm="l2_in_2d_pca", q=q_2d)
    
    return cp_quantiles, pca, pca.mean_, pca.components_


def visualize_cp_2d_correct(
    model,
    dataset,
    device: torch.device,
    cp_quantiles_2d: CPQuantiles,
    pca: PCA,
    sample_idx: int,
    horizon: int,
    save_path: str,
    gt_from_npz: bool = True,
):
    """
    Visualize CP in 2D with theoretically correct quantiles.
    """
    import matplotlib.pyplot as plt
    from matplotlib import patches
    
    sample = dataset[sample_idx]
    input_frames = sample["input_frames"].unsqueeze(0).to(device)
    target_frames = sample["target_frames"].unsqueeze(0).to(device)
    target_offset = sample.get("target_offset", 1)
    if isinstance(target_offset, torch.Tensor):
        target_offset = int(target_offset.item())
    start_idx = target_offset - 1
    
    actions = sample.get("actions", None)
    if actions is not None and model.action_dim > 0:
        actions = actions.unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        # Encode start
        mu_start, _ = model.encode(input_frames[:, start_idx, ...])
        z_cur = mu_start
        
        # Encode GT and project
        T = min(horizon, target_frames.shape[1])
        target_flat = target_frames[:, :T, ...].reshape(T, *target_frames.shape[2:])
        mu_target, _ = model.encode(target_flat)
        z_gt_64d = mu_target.reshape(T, -1).cpu().numpy()
        z_gt_2d = pca.transform(z_gt_64d)
        
        # Rollout and project
        z_pred_2d_list = []
        for t in range(1, T+1):
            a_step = None
            if actions is not None and actions.shape[1] > 0:
                action_idx = min(start_idx + t - 1, actions.shape[1] - 1)
                a_step = actions[:, action_idx:action_idx+1, :]
            
            z_next = model.predict(z_cur.unsqueeze(1), a_step).squeeze(1)
            z_cur = z_next
            
            z_pred_64d = z_next.reshape(-1).cpu().numpy()
            z_pred_2d = pca.transform(z_pred_64d.reshape(1, -1))[0]
            z_pred_2d_list.append(z_pred_2d)
    
    z_pred_2d = np.stack(z_pred_2d_list)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(z_pred_2d[:, 0], z_pred_2d[:, 1], "-o", linewidth=2, markersize=4, label="Pred (LSTM)", zorder=3)
    ax.plot(z_gt_2d[:, 0], z_gt_2d[:, 1], "-o", linewidth=2, markersize=4, label="GT", zorder=3)
    
    # Draw CP circles with theoretically correct quantiles
    for t in range(T):
        qt_2d = float(cp_quantiles_2d.q[t])
        if np.isfinite(qt_2d):
            circ = patches.Circle(
                (z_pred_2d[t, 0], z_pred_2d[t, 1]),
                radius=qt_2d,
                linewidth=0,
                alpha=0.15,
                zorder=1
            )
            ax.add_patch(circ)
    
    ax.set_title(f"CP in 2D PCA space (THEORETICALLY CORRECT)\n"
                 f"alpha={cp_quantiles_2d.alpha}, sample={sample_idx}, steps={T}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved theoretically correct CP visualization: {save_path}")
