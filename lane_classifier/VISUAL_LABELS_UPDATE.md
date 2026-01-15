# Visual Labels Update

## Problem

The original CTE-based labels were inconsistent with visual observations:
- Images showing car on **left** were labeled as **Right**
- Only ~50% of samples were labeled as **Left**, but driving behavior indicated **~80% Left** (inner track)

## Solution

Implemented **visual-based label generation** using red lane line detection:

### Method (`dataset_visual.py`)

1. **HSV Color Detection**: Identify red pixels in image
2. **Horizontal Center of Mass**: Calculate x-coordinate of red lane line
3. **Label Assignment**:
   - Red line on **RIGHT** (x > 0.5) → Car is **LEFT** (label=0) 
   - Red line on **LEFT** (x < 0.5) → Car is **RIGHT** (label=1)

### Results

- **79.6% Left, 20.4% Right** - Matches expected distribution ✓
- Visual verification confirmed label correctness
- More reliable than CTE-based labels

## Usage

Training with visual labels:
```bash
python -m lane_classifier.train \
  --data_dir npz_data \
  --npz_files traj1_64x64.npz traj2_64x64.npz \
  --epochs 30 \
  --balance_classes \
  --save_dir lane_classifier/checkpoints_visual
```

## Files

- `dataset_visual.py`: New dataset class with visual label generation
- `checkpoints_visual/`: Models trained with visual labels
- All `eval_*` scripts compatible with both label types
