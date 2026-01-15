"""
Proper evaluation script - only evaluate on validation set
只在验证集上评估（模型训练时没见过的数据）
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

from cnn_model import get_model
from dataset import LaneDataset


def evaluate_model(model, dataset, device, batch_size=32):
    """Evaluate model on dataset"""
    model.eval()
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_ctes = []
    
    with torch.no_grad():
        for images, labels, ctes in dataloader:
            images = images.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probs.cpu().numpy())
            all_ctes.extend(ctes.numpy())
    
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    probabilities = np.array(all_probabilities)
    ctes = np.array(all_ctes)
    
    return predictions, labels, probabilities, ctes


def calculate_ece(confidences, predictions, true_labels, n_bins=10):
    """Calculate Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_count = in_bin.sum()
        
        if bin_count > 0:
            accuracy_in_bin = (predictions[in_bin] == true_labels[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += (bin_count / len(confidences)) * np.abs(avg_confidence_in_bin - accuracy_in_bin)
    
    return ece


def plot_results(predictions, labels, probabilities, ctes, save_dir):
    """Plot all evaluation results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Confusion Matrix
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Left', 'Right'],
                yticklabels=['Left', 'Right'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Validation Set ONLY)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_proper.png'), dpi=150)
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Validation Set ONLY)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curve_proper.png'), dpi=150)
    plt.close()
    
    # CTE Distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Overall CTE distribution
    axes[0, 0].hist(ctes, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='Center')
    axes[0, 0].set_xlabel('CTE')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Overall CTE Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # CTE by True Label
    left_ctes = ctes[labels == 0]
    right_ctes = ctes[labels == 1]
    axes[0, 1].hist([left_ctes, right_ctes], bins=30, alpha=0.7,
                    label=['Left (0)', 'Right (1)'], color=['blue', 'orange'])
    axes[0, 1].set_xlabel('CTE')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('CTE by True Label')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # CTE by Prediction
    left_pred_ctes = ctes[predictions == 0]
    right_pred_ctes = ctes[predictions == 1]
    axes[1, 0].hist([left_pred_ctes, right_pred_ctes], bins=30, alpha=0.7,
                    label=['Pred Left (0)', 'Pred Right (1)'], color=['cyan', 'magenta'])
    axes[1, 0].set_xlabel('CTE')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('CTE by Prediction')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Correct vs Incorrect predictions
    correct_mask = predictions == labels
    correct_ctes = ctes[correct_mask]
    incorrect_ctes = ctes[~correct_mask]
    axes[1, 1].hist([correct_ctes, incorrect_ctes], bins=30, alpha=0.7,
                    label=['Correct', 'Incorrect'], color=['green', 'red'])
    axes[1, 1].set_xlabel('CTE')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('CTE: Correct vs Incorrect Predictions')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'cte_distribution_proper.png'), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Proper Evaluation - Validation Set Only')
    
    parser.add_argument('--model_path', type=str,
                        default='lane_classifier/checkpoints_corrected/best_model.pt',
                        help='Path to model')
    parser.add_argument('--data_dir', type=str, default='npz_data',
                        help='Data directory')
    parser.add_argument('--npz_files', nargs='+',
                        default=['traj1_64x64.npz', 'traj2_64x64.npz'],
                        help='NPZ files')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio (must match training)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (must match training)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device')
    parser.add_argument('--output_dir', type=str,
                        default='lane_classifier/eval_results_proper',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("="*60)
    print("PROPER Evaluation - Validation Set ONLY")
    print("="*60)
    print(f"Device: {device}")
    print(f"Random seed: {args.seed} (CRITICAL: Must match training!)")
    print(f"Val split: {args.val_split} (CRITICAL: Must match training!)")
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'args' in checkpoint:
        model_type = checkpoint['args'].get('model_type', 'standard')
        dropout = checkpoint['args'].get('dropout', 0.5)
    else:
        model_type = 'standard'
        dropout = 0.5
    
    model = get_model(model_type=model_type, dropout_rate=dropout)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"  Model type: {model_type}")
    
    # Load FULL dataset
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.abspath(os.path.join(script_dir, '..', data_dir))
    
    npz_paths = [os.path.join(data_dir, f) for f in args.npz_files]
    print(f"\nLoading data from: {npz_paths}")
    
    full_dataset = LaneDataset(
        npz_paths=npz_paths,
        normalize=True,
        target_size=64,
        balance_classes=False  # Don't balance - use original distribution
    )
    
    print(f"Full dataset size: {len(full_dataset)}")
    
    # Split same way as training
    total_size = len(full_dataset)
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size
    
    # CRITICAL: Use same seed to get same split
    generator = torch.Generator().manual_seed(args.seed)
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator
    )
    
    print(f"\n{'='*60}")
    print(f"CRITICAL: Using VALIDATION SET ONLY")
    print(f"{'='*60}")
    print(f"Train set size: {train_size} (NOT used for evaluation)")
    print(f"Val set size: {val_size} (ONLY this is evaluated)")
    print(f"\nThis is the TRUE test of generalization!")
    
    # Evaluate ONLY on validation set
    print(f"\nEvaluating model on validation set...")
    predictions, labels, probabilities, ctes = evaluate_model(
        model, val_dataset, device, args.batch_size
    )
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    
    # Per-class metrics
    left_mask = labels == 0
    right_mask = labels == 1
    left_acc = (predictions[left_mask] == labels[left_mask]).mean() if left_mask.sum() > 0 else 0
    right_acc = (predictions[right_mask] == labels[right_mask]).mean() if right_mask.sum() > 0 else 0
    
    # ECE
    confidences = np.max(probabilities, axis=1)
    ece = calculate_ece(confidences, predictions, labels, n_bins=10)
    
    # ROC AUC
    fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Print results
    print("\n" + "="*60)
    print("PROPER Evaluation Results (Validation Set ONLY)")
    print("="*60)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  ECE: {ece:.4f}")
    
    print(f"\nPer-Class Accuracy:")
    print(f"  Left (0): {left_acc*100:.2f}%")
    print(f"  Right (1): {right_acc*100:.2f}%")
    
    print(f"\nClass Distribution:")
    print(f"  Left (0): {left_mask.sum()} ({left_mask.sum()/len(labels)*100:.1f}%)")
    print(f"  Right (1): {right_mask.sum()} ({right_mask.sum()/len(labels)*100:.1f}%)")
    
    print(f"\nConfidence Statistics:")
    print(f"  Mean: {confidences.mean():.4f}")
    print(f"  Median: {np.median(confidences):.4f}")
    print(f"  Min: {confidences.min():.4f}")
    print(f"  Max: {confidences.max():.4f}")
    
    # Confusion matrix details
    cm = confusion_matrix(labels, predictions)
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"               Left  Right")
    print(f"  True  Left   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"        Right  {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'metrics_proper.txt')
    with open(metrics_path, 'w') as f:
        f.write("PROPER Evaluation Results (Validation Set ONLY)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Data: {args.npz_files}\n")
        f.write(f"Val split: {args.val_split}\n")
        f.write(f"Random seed: {args.seed}\n")
        f.write(f"Val set size: {val_size}\n\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"  Accuracy: {accuracy*100:.2f}%\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall: {recall:.4f}\n")
        f.write(f"  F1 Score: {f1:.4f}\n")
        f.write(f"  ROC AUC: {roc_auc:.4f}\n")
        f.write(f"  ECE: {ece:.4f}\n\n")
        f.write(f"Per-Class Accuracy:\n")
        f.write(f"  Left (0): {left_acc*100:.2f}%\n")
        f.write(f"  Right (1): {right_acc*100:.2f}%\n")
    
    print(f"\nSaved metrics to {metrics_path}")
    
    # Plot results
    print(f"\nGenerating plots...")
    plot_results(predictions, labels, probabilities, ctes, args.output_dir)
    
    print(f"\n[*] Proper evaluation complete!")
    print(f"\nThese are the REAL performance numbers!")
    print(f"Compare with eval_results/ to see the difference.")


if __name__ == '__main__':
    main()
