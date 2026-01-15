# Final Evaluation Summary

## Overview

We trained a CNN binary classifier to determine if a car is on the LEFT or RIGHT side of a lane using **visual labels** (red lane line position detection) instead of unreliable CTE values.

## 📊 Results

### 1. CNN Model Performance (on Real Images)

**Model**: `checkpoints_visual/best_model.pt`  
**Training**: 30 epochs with visual labels

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **96.45%** |
| **Left Precision** | 98.05% |
| **Left Recall** | 94.86% |
| **Right Precision** | 94.93% |
| **Right Recall** | 98.08% |
| **ECE** | **0.0304** |

✅ **Excellent performance on real images!**

---

### 2. End-to-End System Performance (LSTM→VAE→CNN)

**Pipeline**: LSTM predicts latent → VAE decodes → CNN classifies

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **27.40%** |
| **ECE** | **0.7260** |

❌ **Poor performance - CNN predictions all collapse to one class (Left)**

---

## 🔍 Root Cause Analysis

### Problem: Domain Shift

1. **CNN Training**: Trained on **real images** with clear red lane lines
2. **End-to-End Eval**: CNN receives **LSTM-predicted images** which are:
   - Blurry and low quality
   - Missing or unclear red lane lines
   - Insufficient detail for lane position classification

### Evidence

From prediction samples (`eval_e2e_visual_final/e2e_prediction_samples.png`):
- **All CNN predictions**: Left (100% confidence)
- **All true labels**: Right
- **Predicted images**: Visually degraded, red lines barely visible

The CNN was never trained on such degraded images, so it defaults to predicting the majority class (Left, which was 79.6% in training).

---

## 💡 Conclusions

### What Works ✅

1. **Visual Label Generation**: Much more reliable than CTE-based labels
   - 79.6% Left / 20.4% Right matches expected distribution
   - Labels verified visually

2. **CNN on Real Images**: Excellent performance
   - 96.45% accuracy
   - Well-calibrated (ECE = 0.0304)
   - Balanced performance on both classes

### What Doesn't Work ❌

1. **End-to-End Pipeline**: LSTM prediction quality insufficient
   - Predicted images too degraded for CNN classification
   - Need better VAE or different latent space representation

---

## 🎯 Recommendations

### Option 1: Improve LSTM+VAE Quality
- Train VAE with stronger reconstruction loss
- Use skip connections in VAE
- Increase latent dimension
- Train LSTM longer with better regularization

### Option 2: Train CNN on Predicted Images
- Create training dataset from LSTM-predicted images
- Train CNN to be robust to image degradation
- Use data augmentation (blur, noise) during CNN training

### Option 3: Use Latent Space Classification
- Instead of decoding to images, classify directly in latent space
- Train a classifier: `latent_z → {Left, Right}`
- Avoids image reconstruction bottleneck

---

## 📁 Files

### Models
- `lane_classifier/checkpoints_visual/best_model.pt` - CNN trained with visual labels
- `vae_recon/best_model.pt` - VAE model
- `predictor/checkpoints/Donkey_car_checkpoints_best_model.pt` - LSTM predictor

### Evaluation Results
- `lane_classifier/eval_visual_results/` - CNN evaluation on real images
- `lane_classifier/eval_e2e_visual_final/` - End-to-end pipeline evaluation

### Code
- `lane_classifier/dataset_visual.py` - Visual label generation dataset
- `lane_classifier/train.py` - CNN training script
- `lane_classifier/eval_end_to_end.py` - End-to-end evaluation script

---

## Summary

The **visual label approach** successfully addressed the CTE reliability issue, achieving **96.45%** accuracy on real images. However, the **end-to-end pipeline** fails due to poor LSTM prediction quality - the generated images are too degraded for the CNN to classify accurately. Future work should focus on improving image generation quality or exploring latent space classification to bypass the reconstruction bottleneck.
