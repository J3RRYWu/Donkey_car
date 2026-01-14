# Evaluation Modules - Refactored Structure

The evaluation code has been refactored into multiple modules for better maintainability and organization.

## Module Structure

```
predictor/
├── eval_predictor.py          # Main entry point (command-line interface)
├── eval_metrics.py             # Core evaluation metrics
├── eval_visualization.py       # Standard visualizations
├── eval_conformal.py           # Conformal Prediction (CP) evaluation
├── eval_utils.py               # Utility functions
├── vae_predictor.py            # Model definitions
└── conformal.py                # CP core logic (dataclasses, scoring)
```

## Module Descriptions

### `eval_predictor.py` (Main Entry Point)
- **Purpose**: Command-line interface and workflow coordination
- **Contains**:
  - Argument parsing (all CLI options)
  - Dataset loading and dataloader setup
  - CP calibration/evaluation coordination
  - Result aggregation and export (JSON, CSV)
  - Calls functions from other modules

### `eval_metrics.py` (Core Evaluation)
- **Purpose**: Quantitative evaluation metrics
- **Functions**:
  - `compute_baseline_vs_lstm()`: One-step prediction comparison
  - `compute_multi_step_rollout()`: Open-loop multi-step rollout with:
    - LSTM, Identity, Linear extrapolation baselines
    - Latent-space MSE
    - Image-space metrics (MSE, PSNR, SSIM)
    - Effective horizon calculation
    - MC-dropout uncertainty (optional)

### `eval_visualization.py` (Standard Visualizations)
- **Purpose**: Image and video generation
- **Functions**:
  - `visualize_predictions()`: GT vs VAE Recon vs LSTM grid
  - `visualize_rollout_images()`: 30-step rollout visualization
  - `generate_prediction_video()`: Long-term prediction video (with optional GT side-by-side)

### `eval_conformal.py` (Conformal Prediction)
- **Purpose**: CP-specific visualizations
- **Functions**:
  - `fit_pca_2d()`, `pca_project()`: PCA utilities for 2D projection
  - `visualize_cp_trajectory_band()`: PCA latent space with CP confidence band
  - `visualize_cp_boundary_decode()`: Sample and decode points on CP boundary sphere

### `eval_utils.py` (Utility Functions)
- **Purpose**: Reusable helper functions
- **Functions**:
  - `to_01()`: Normalize images to [0,1]
  - `psnr_from_mse()`: Calculate PSNR from MSE
  - `gaussian_window()`, `ssim()`: SSIM calculation
  - `overlay_sbs_labels()`: Add labels to side-by-side images
  - `effective_horizon_from_curve()`: Find effective horizon from metrics

### `conformal.py` (CP Core Logic)
- **Purpose**: CP algorithms (existing file, not refactored)
- **Contains**: `CPQuantiles`, scoring functions, quantile calculation, coverage metrics

### `vae_predictor.py` (Model & Dataset)
- **Purpose**: Model architecture and data loading (existing file, not refactored)
- **Contains**: `VAEPredictor`, `TrajectoryDataset`, `load_model()`

## Usage

The refactored code is **100% backward compatible**. All existing commands work exactly as before:

```bash
# Standard evaluation
py -3.11 predictor/eval_predictor.py \
    --model_path checkpoints/best_model.pt \
    --vae_model_path ../checkpoints_64x64/vae_epoch_300.pth \
    --data_dir ../npz_transfer \
    --npz_files traj1.npz traj2.npz \
    --max_horizon 50 \
    --gt_from_npz

# CP evaluation
py -3.11 predictor/eval_predictor.py \
    --model_path checkpoints/best_model.pt \
    --vae_model_path ../checkpoints_64x64/vae_epoch_300.pth \
    --data_dir ../npz_transfer \
    --npz_files traj1.npz \
    --cp_calibrate \
    --cp_eval \
    --cp_boundary_plot \
    --cp_boundary_step 20 \
    --max_horizon 50 \
    --gt_from_npz
```

## Benefits of Refactoring

1. **Maintainability**: Each module has a single, clear responsibility
2. **Readability**: Smaller files (200-400 lines each) vs. 2000-line monolith
3. **Testability**: Individual modules can be tested in isolation
4. **Reusability**: Functions can be imported and used in other scripts
5. **Debugging**: Easier to locate and fix issues in specific modules
6. **Extensibility**: New features can be added to appropriate modules without cluttering main entry point

## File Sizes (Approximate)

- `eval_predictor.py`: ~650 lines (was 1984)
- `eval_metrics.py`: ~430 lines
- `eval_visualization.py`: ~460 lines
- `eval_conformal.py`: ~280 lines
- `eval_utils.py`: ~130 lines
- **Total**: ~1950 lines (similar total, but modular!)

## Migration Notes

No code changes required! The refactored code:
- Uses the same imports
- Has the same CLI interface
- Produces identical outputs
- Only reorganizes internal structure
