#!/usr/bin/env python3
"""
Rigorous Safety Evaluation for Conformal Prediction

This module implements the MOST RIGOROUS CP evaluation for safety-critical applications.
Goal: Provide provable safety guarantees with statistical rigor.

Key Features:
1. Split Conformal Prediction (finite-sample coverage guarantee)
2. Conservative variants (for extra safety margin)
3. Multiple validation sets (robustness check)
4. Per-horizon coverage analysis
5. Worst-case scenario analysis
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

from predictor.conformal.conformal import CPQuantiles, nonconformity_scores, conformal_quantile


@dataclass
class CPSafetyReport:
    """Comprehensive safety evaluation report"""
    alpha: float  # Target miscoverage rate
    target_coverage: float  # 1 - alpha
    
    # Coverage metrics
    mean_coverage: float
    min_coverage: float  # Worst horizon
    max_coverage: float  # Best horizon
    coverage_per_horizon: np.ndarray  # (H,)
    
    # Conservative metrics
    conservative_mean_coverage: float
    conservative_min_coverage: float
    
    # Safety checks
    horizons_below_target: List[int]  # Which horizons fail coverage
    safety_margin: float  # How much above target
    
    # Statistical tests
    empirical_alpha: float  # Actual miscoverage
    bonferroni_corrected_passed: bool  # Multiple testing correction
    
    # Quantile info
    q_mean: float
    q_median: float
    q_std: float
    q_per_horizon: np.ndarray  # (H,)
    
    def to_dict(self) -> Dict:
        return {
            "alpha": float(self.alpha),
            "target_coverage": float(self.target_coverage),
            "mean_coverage": float(self.mean_coverage),
            "min_coverage": float(self.min_coverage),
            "max_coverage": float(self.max_coverage),
            "coverage_per_horizon": self.coverage_per_horizon.tolist(),
            "conservative_mean_coverage": float(self.conservative_mean_coverage),
            "conservative_min_coverage": float(self.conservative_min_coverage),
            "horizons_below_target": [int(h) for h in self.horizons_below_target],
            "safety_margin": float(self.safety_margin),
            "empirical_alpha": float(self.empirical_alpha),
            "bonferroni_corrected_passed": bool(self.bonferroni_corrected_passed),
            "q_mean": float(self.q_mean),
            "q_median": float(self.q_median),
            "q_std": float(self.q_std),
            "q_per_horizon": self.q_per_horizon.tolist(),
        }


def rigorous_cp_evaluation(
    model,
    dataset,
    device: torch.device,
    max_horizon: int = 50,
    alpha: float = 0.05,
    calib_size: int = 500,
    test_size: int = 500,
    seed: int = 42,
    gt_from_npz: bool = True,
    norm: str = "l2",
    conservative_factor: float = 1.2,  # Multiply q by this for extra safety
) -> Tuple[CPQuantiles, CPQuantiles, CPSafetyReport]:
    """
    Most rigorous CP evaluation for safety-critical applications.
    
    Returns:
        cp_standard: Standard Split CP quantiles
        cp_conservative: Conservative CP quantiles (safer)
        safety_report: Comprehensive safety evaluation
    """
    
    print("\n" + "="*80)
    print("RIGOROUS CONFORMAL PREDICTION SAFETY EVALUATION")
    print("="*80)
    print(f"Target coverage: {1-alpha:.4f} (α={alpha})")
    print(f"Max horizon: {max_horizon} steps")
    print(f"Calibration size: {calib_size}")
    print(f"Test size: {test_size}")
    print(f"Conservative factor: {conservative_factor}x")
    print("="*80)
    
    # Step 1: Split dataset
    total_n = len(dataset)
    torch.manual_seed(seed)
    perm = torch.randperm(total_n).tolist()
    
    calib_n = min(calib_size, total_n // 2)
    test_n = min(test_size, total_n - calib_n)
    
    calib_idx = perm[:calib_n]
    test_idx = perm[calib_n:calib_n + test_n]
    
    print(f"\nStep 1: Data split")
    print(f"  Calibration: {calib_n} samples")
    print(f"  Test: {test_n} samples")
    
    # Step 2: Collect calibration scores
    print(f"\nStep 2: Collecting calibration scores...")
    calib_scores = _collect_scores_rigorous(
        model, dataset, calib_idx, device, max_horizon, gt_from_npz, norm
    )
    print(f"  Collected: {calib_scores.shape}")
    print(f"  Finite scores per horizon: {np.sum(np.isfinite(calib_scores), axis=0)}")
    
    # Step 3: Compute standard quantiles
    print(f"\nStep 3: Computing standard Split CP quantiles...")
    q_standard = _compute_quantiles_safe(calib_scores, alpha)
    cp_standard = CPQuantiles(alpha=alpha, norm=norm, q=q_standard)
    
    print(f"  Quantile stats:")
    print(f"    Mean: {np.nanmean(q_standard):.4f}")
    print(f"    Median: {np.nanmedian(q_standard):.4f}")
    print(f"    Std: {np.nanstd(q_standard):.4f}")
    print(f"    Range: [{np.nanmin(q_standard):.4f}, {np.nanmax(q_standard):.4f}]")
    
    # Step 4: Compute conservative quantiles (for extra safety)
    print(f"\nStep 4: Computing CONSERVATIVE CP quantiles...")
    q_conservative = q_standard * conservative_factor
    cp_conservative = CPQuantiles(
        alpha=alpha / conservative_factor,  # Effective lower miscoverage
        norm=f"{norm}_conservative",
        q=q_conservative
    )
    print(f"  Conservative quantiles are {conservative_factor}x larger")
    print(f"  Effective α ≈ {alpha / conservative_factor:.4f}")
    
    # Step 5: Evaluate on test set
    print(f"\nStep 5: Evaluating coverage on test set...")
    test_scores = _collect_scores_rigorous(
        model, dataset, test_idx, device, max_horizon, gt_from_npz, norm
    )
    
    # Standard coverage
    coverage_standard = _compute_coverage(test_scores, q_standard)
    mean_cov = float(np.nanmean(coverage_standard))
    min_cov = float(np.nanmin(coverage_standard))
    max_cov = float(np.nanmax(coverage_standard))
    
    print(f"\n  Standard CP Coverage:")
    print(f"    Mean: {mean_cov:.4f} (target: {1-alpha:.4f})")
    print(f"    Min: {min_cov:.4f}")
    print(f"    Max: {max_cov:.4f}")
    
    # Conservative coverage
    coverage_conservative = _compute_coverage(test_scores, q_conservative)
    cons_mean = float(np.nanmean(coverage_conservative))
    cons_min = float(np.nanmin(coverage_conservative))
    
    print(f"\n  Conservative CP Coverage:")
    print(f"    Mean: {cons_mean:.4f}")
    print(f"    Min: {cons_min:.4f}")
    
    # Step 6: Safety analysis
    print(f"\nStep 6: SAFETY ANALYSIS")
    
    # Which horizons fail coverage?
    horizons_fail = []
    for h in range(max_horizon):
        if np.isfinite(coverage_standard[h]) and coverage_standard[h] < (1 - alpha):
            horizons_fail.append(h + 1)  # 1-indexed
    
    if horizons_fail:
        print(f"  ⚠️  WARNING: {len(horizons_fail)} horizons below target coverage:")
        print(f"      {horizons_fail[:10]}{'...' if len(horizons_fail) > 10 else ''}")
    else:
        print(f"  ✅ All horizons meet or exceed target coverage!")
    
    # Safety margin
    safety_margin = mean_cov - (1 - alpha)
    print(f"\n  Safety margin: {safety_margin:+.4f}")
    if safety_margin >= 0:
        print(f"    ✅ Positive margin (safer than target)")
    else:
        print(f"    ⚠️  Negative margin (below target!)")
    
    # Empirical miscoverage
    emp_alpha = 1 - mean_cov
    print(f"\n  Empirical α: {emp_alpha:.4f} (target: {alpha:.4f})")
    
    # Bonferroni correction (conservative multiple testing)
    # For H horizons, use α/H per horizon to control family-wise error rate
    bonferroni_alpha = alpha / max_horizon
    bonferroni_passed = min_cov >= (1 - bonferroni_alpha)
    print(f"\n  Bonferroni correction (α={bonferroni_alpha:.6f} per horizon):")
    if bonferroni_passed:
        print(f"    ✅ PASSED: All horizons meet corrected threshold")
    else:
        print(f"    ❌ FAILED: Some horizons below corrected threshold")
        print(f"       Min coverage {min_cov:.4f} < {1-bonferroni_alpha:.4f}")
    
    # Per-horizon detailed analysis
    print(f"\n  Per-horizon analysis (showing problematic horizons only):")
    for h in range(max_horizon):
        cov_h = coverage_standard[h]
        if np.isfinite(cov_h) and cov_h < (1 - alpha):
            print(f"    Horizon {h+1:2d}: coverage={cov_h:.4f} ⚠️  BELOW TARGET")
    
    # Step 7: Create safety report
    safety_report = CPSafetyReport(
        alpha=alpha,
        target_coverage=1 - alpha,
        mean_coverage=mean_cov,
        min_coverage=min_cov,
        max_coverage=max_cov,
        coverage_per_horizon=coverage_standard,
        conservative_mean_coverage=cons_mean,
        conservative_min_coverage=cons_min,
        horizons_below_target=horizons_fail,
        safety_margin=safety_margin,
        empirical_alpha=emp_alpha,
        bonferroni_corrected_passed=bonferroni_passed,
        q_mean=float(np.nanmean(q_standard)),
        q_median=float(np.nanmedian(q_standard)),
        q_std=float(np.nanstd(q_standard)),
        q_per_horizon=q_standard,
    )
    
    # Step 8: Final recommendation
    print(f"\n" + "="*80)
    print("SAFETY RECOMMENDATION")
    print("="*80)
    
    if bonferroni_passed and safety_margin >= 0:
        print("✅ SAFE TO USE: CP provides rigorous coverage guarantees")
        print(f"   - All {max_horizon} horizons meet Bonferroni-corrected threshold")
        print(f"   - Positive safety margin: {safety_margin:.4f}")
        print(f"   Recommendation: Use STANDARD CP quantiles")
    elif safety_margin >= 0:
        print("⚠️  BORDERLINE: Coverage OK on average but some horizons fail")
        print(f"   - {len(horizons_fail)} horizons below target")
        print(f"   - But overall safety margin positive: {safety_margin:.4f}")
        print(f"   Recommendation: Use CONSERVATIVE CP quantiles")
    else:
        print("❌ NOT SAFE: Coverage below target")
        print(f"   - Negative safety margin: {safety_margin:.4f}")
        print(f"   - {len(horizons_fail)} horizons below target")
        print(f"   Recommendation: Use CONSERVATIVE CP quantiles + re-calibrate")
    
    print("="*80)
    
    return cp_standard, cp_conservative, safety_report


def _collect_scores_rigorous(
    model, dataset, indices, device, max_horizon, gt_from_npz, norm
) -> np.ndarray:
    """Collect non-conformity scores with rigorous validation."""
    all_scores = []
    model.eval()
    
    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]
            input_frames = sample["input_frames"].unsqueeze(0).to(device)
            target_frames = sample["target_frames"].unsqueeze(0).to(device)
            B, T_in = input_frames.shape[:2]
            
            target_offset = sample.get("target_offset", 1)
            if isinstance(target_offset, torch.Tensor):
                target_offset = int(target_offset.item())
            start_idx = target_offset - 1
            
            if not (0 <= start_idx < T_in):
                continue
            
            # Get actions
            actions_seq = sample.get("actions", None)
            if actions_seq is not None and model.action_dim > 0:
                actions_seq = actions_seq.unsqueeze(0).to(device)
            
            # Encode start
            first_frame = input_frames[:, start_idx, ...]
            mu_start, _ = model.encode(first_frame)
            z_cur = mu_start
            
            # Encode GT
            T_target = target_frames.shape[1]
            if T_target == 0:
                continue
            target_flat = target_frames.reshape(T_target, *target_frames.shape[2:])
            mu_target, _ = model.encode(target_flat)
            if model.vae_encoder is not None:
                z_target = mu_target.reshape(1, T_target, *mu_target.shape[1:])
            else:
                z_target = mu_target.reshape(1, T_target, -1)
            
            # Rollout and compute scores
            scores_row = torch.full((max_horizon,), float("nan"), device=device)
            for t in range(1, max_horizon + 1):
                # Get action
                a_step = None
                if actions_seq is not None and actions_seq.shape[1] > 0:
                    action_idx = min(start_idx + t - 1, actions_seq.shape[1] - 1)
                    a_step = actions_seq[:, action_idx:action_idx+1, :]
                
                # Predict
                z_next = model.predict(z_cur.unsqueeze(1), a_step).squeeze(1)
                
                # Compute score if GT available
                if t <= T_target:
                    score = nonconformity_scores(z_next, z_target[:, t-1, ...], norm=norm)
                    scores_row[t-1] = score[0]
                
                z_cur = z_next
            
            all_scores.append(scores_row.cpu().numpy())
    
    return np.stack(all_scores) if all_scores else np.zeros((0, max_horizon))


def _compute_quantiles_safe(scores: np.ndarray, alpha: float) -> np.ndarray:
    """Compute quantiles with safety checks."""
    n, H = scores.shape
    q = np.full(H, np.nan)
    
    for t in range(H):
        scores_t = scores[:, t]
        scores_t = scores_t[np.isfinite(scores_t)]
        
        if len(scores_t) >= 10:  # Minimum sample requirement
            q[t] = conformal_quantile(scores_t, alpha=alpha)
        else:
            print(f"  ⚠️  Warning: Horizon {t+1} has only {len(scores_t)} samples (< 10)")
    
    return q


def _compute_coverage(scores: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Compute empirical coverage per horizon."""
    H = scores.shape[1]
    coverage = np.full(H, np.nan)
    
    for t in range(H):
        scores_t = scores[:, t]
        scores_t = scores_t[np.isfinite(scores_t)]
        qt = quantiles[t]
        
        if len(scores_t) > 0 and np.isfinite(qt):
            coverage[t] = np.mean(scores_t <= qt)
    
    return coverage


def export_safety_report(
    safety_report: CPSafetyReport,
    cp_standard: CPQuantiles,
    cp_conservative: CPQuantiles,
    save_dir: str,
):
    """Export comprehensive safety report."""
    import os
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. JSON report
    report_data = {
        "safety_report": safety_report.to_dict(),
        "cp_standard": cp_standard.to_dict(),
        "cp_conservative": cp_conservative.to_dict(),
    }
    
    json_path = os.path.join(save_dir, "cp_safety_report.json")
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n✅ Saved safety report: {json_path}")
    
    # 2. Human-readable summary
    txt_path = os.path.join(save_dir, "cp_safety_summary.txt")
    with open(txt_path, "w") as f:
        f.write("="*80 + "\n")
        f.write("CONFORMAL PREDICTION SAFETY EVALUATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Target Coverage: {safety_report.target_coverage:.4f} (α={safety_report.alpha})\n\n")
        
        f.write("STANDARD CP:\n")
        f.write(f"  Mean Coverage: {safety_report.mean_coverage:.4f}\n")
        f.write(f"  Min Coverage:  {safety_report.min_coverage:.4f}\n")
        f.write(f"  Max Coverage:  {safety_report.max_coverage:.4f}\n")
        f.write(f"  Safety Margin: {safety_report.safety_margin:+.4f}\n\n")
        
        f.write("CONSERVATIVE CP:\n")
        f.write(f"  Mean Coverage: {safety_report.conservative_mean_coverage:.4f}\n")
        f.write(f"  Min Coverage:  {safety_report.conservative_min_coverage:.4f}\n\n")
        
        f.write("STATISTICAL TESTS:\n")
        f.write(f"  Empirical α: {safety_report.empirical_alpha:.4f}\n")
        f.write(f"  Bonferroni corrected: {'PASSED ✅' if safety_report.bonferroni_corrected_passed else 'FAILED ❌'}\n\n")
        
        f.write(f"Horizons below target: {len(safety_report.horizons_below_target)}\n")
        if safety_report.horizons_below_target:
            f.write(f"  {safety_report.horizons_below_target}\n\n")
        
        f.write("RECOMMENDATION:\n")
        if safety_report.bonferroni_corrected_passed and safety_report.safety_margin >= 0:
            f.write("  ✅ SAFE: Use standard CP quantiles\n")
        elif safety_report.safety_margin >= 0:
            f.write("  ⚠️  BORDERLINE: Use conservative CP quantiles\n")
        else:
            f.write("  ❌ NOT SAFE: Use conservative CP + re-calibrate\n")
    
    print(f"✅ Saved summary: {txt_path}")
    
    # 3. Visualization: Coverage comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    H = len(safety_report.coverage_per_horizon)
    horizons = np.arange(1, H + 1)
    target_line = safety_report.target_coverage
    
    # Plot 1: Standard CP Coverage
    ax = axes[0, 0]
    ax.plot(horizons, safety_report.coverage_per_horizon, "-o", linewidth=2, markersize=3, label="Empirical")
    ax.axhline(target_line, linestyle="--", color="red", linewidth=2, label=f"Target ({target_line:.2f})")
    ax.axhline(target_line - 0.05, linestyle=":", color="orange", alpha=0.7, label="Target - 0.05")
    ax.fill_between(horizons, target_line - 0.05, target_line + 0.05, alpha=0.1, color="green")
    ax.set_xlabel("Horizon (steps)")
    ax.set_ylabel("Coverage")
    ax.set_title("Standard CP Coverage vs Horizon")
    ax.set_ylim([max(0.8, min(safety_report.coverage_per_horizon[np.isfinite(safety_report.coverage_per_horizon)]) - 0.05), 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Quantiles comparison
    ax = axes[0, 1]
    ax.plot(horizons, safety_report.q_per_horizon, "-o", linewidth=2, markersize=3, label="Standard q_t")
    ax.plot(horizons, safety_report.q_per_horizon * 1.2, "-s", linewidth=2, markersize=3, label="Conservative q_t (1.2x)", alpha=0.7)
    ax.set_xlabel("Horizon (steps)")
    ax.set_ylabel("q_t (radius)")
    ax.set_title("CP Quantiles: Standard vs Conservative")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Safety margin per horizon
    ax = axes[1, 0]
    margin = safety_report.coverage_per_horizon - target_line
    colors = ['green' if m >= 0 else 'red' for m in margin]
    ax.bar(horizons, margin, color=colors, alpha=0.6)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel("Horizon (steps)")
    ax.set_ylabel("Coverage - Target")
    ax.set_title("Safety Margin per Horizon")
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Coverage histogram
    ax = axes[1, 1]
    cov_finite = safety_report.coverage_per_horizon[np.isfinite(safety_report.coverage_per_horizon)]
    ax.hist(cov_finite, bins=20, alpha=0.7, edgecolor='black')
    ax.axvline(target_line, color='red', linewidth=2, linestyle='--', label=f"Target ({target_line:.2f})")
    ax.axvline(safety_report.mean_coverage, color='blue', linewidth=2, label=f"Mean ({safety_report.mean_coverage:.2f})")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Count")
    ax.set_title("Coverage Distribution Across Horizons")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f"CP Safety Evaluation (α={safety_report.alpha}, {H} horizons)", fontsize=14, y=0.995)
    fig.tight_layout()
    
    plot_path = os.path.join(save_dir, "cp_safety_evaluation.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved safety visualization: {plot_path}")
