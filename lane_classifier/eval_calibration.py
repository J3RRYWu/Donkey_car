"""
Calculate ECE and calibration metrics for Lane Classifier
计算车道分类器的ECE和校准指标
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from cnn_model import get_model
from dataset import LaneDataset


def calculate_ece(confidences, predictions, true_labels, n_bins=10):
    """
    Calculate Expected Calibration Error
    
    Args:
        confidences: max probabilities (confidence scores)
        predictions: predicted labels
        true_labels: true labels
        n_bins: number of bins for ECE calculation
    
    Returns:
        ece, accuracies, confidences_list, bin_counts, bin_boundaries
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    accuracies = []
    confidences_list = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_count = in_bin.sum()
        
        if bin_count > 0:
            accuracy_in_bin = (predictions[in_bin] == true_labels[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            
            ece += (bin_count / len(confidences)) * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
            accuracies.append(accuracy_in_bin)
            confidences_list.append(avg_confidence_in_bin)
            bin_counts.append(bin_count)
        else:
            accuracies.append(None)
            confidences_list.append(None)
            bin_counts.append(0)
    
    return ece, accuracies, confidences_list, bin_counts, bin_boundaries


def plot_calibration_curve(ece, accuracies, confidences_list, bin_counts, bin_boundaries, save_path):
    """Plot reliability diagram"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Filter valid bins
    valid_bins = [i for i in range(len(bin_counts)) if bin_counts[i] > 0]
    valid_confidences = [confidences_list[i] for i in valid_bins]
    valid_accuracies = [accuracies[i] for i in valid_bins]
    valid_counts = [bin_counts[i] for i in valid_bins]
    
    # Plot 1: Reliability Diagram
    axes[0].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    axes[0].bar(valid_confidences, valid_accuracies, 
                width=0.08, alpha=0.7, label='Model', 
                edgecolor='black', linewidth=1.5, color='steelblue')
    axes[0].set_xlabel('Confidence', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title(f'Reliability Diagram (ECE={ece:.4f})', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    
    # Plot 2: Sample Distribution
    bin_centers = [(bin_boundaries[i] + bin_boundaries[i+1])/2 for i in valid_bins]
    axes[1].bar(bin_centers, valid_counts, width=0.08, alpha=0.7, 
                edgecolor='black', linewidth=1.5, color='coral')
    axes[1].set_xlabel('Confidence', fontsize=12)
    axes[1].set_ylabel('Number of Samples', fontsize=12)
    axes[1].set_title('Sample Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved calibration curve to {save_path}")


def evaluate_calibration(model, dataset, device, batch_size=32):
    """Evaluate model calibration"""
    model.eval()
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, ctes in dataloader:
            images = images.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    predictions = np.array(all_predictions)
    probabilities = np.array(all_probabilities)
    labels = np.array(all_labels)
    
    # Get confidence (max probability)
    confidences = np.max(probabilities, axis=1)
    
    return predictions, probabilities, confidences, labels


def main():
    parser = argparse.ArgumentParser(description='Evaluate Calibration and Calculate ECE')
    
    parser.add_argument('--model_path', type=str, 
                        default='lane_classifier/checkpoints_corrected/best_model.pt',
                        help='Path to model')
    parser.add_argument('--data_dir', type=str, default='npz_data',
                        help='Data directory')
    parser.add_argument('--npz_files', nargs='+',
                        default=['traj1_64x64.npz', 'traj2_64x64.npz'],
                        help='NPZ files')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--n_bins', type=int, default=10,
                        help='Number of bins for ECE')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device')
    parser.add_argument('--output_dir', type=str, 
                        default='lane_classifier/eval_results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("="*60)
    print("Calibration Evaluation and ECE Calculation")
    print("="*60)
    print(f"Device: {device}")
    
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
    
    # Load data
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.abspath(os.path.join(script_dir, '..', data_dir))
    
    npz_paths = [os.path.join(data_dir, f) for f in args.npz_files]
    print(f"\nLoading data from: {npz_paths}")
    
    dataset = LaneDataset(
        npz_paths=npz_paths,
        normalize=True,
        target_size=64,
        balance_classes=False
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Evaluate
    print("\nEvaluating model...")
    predictions, probabilities, confidences, labels = evaluate_calibration(
        model, dataset, device, args.batch_size
    )
    
    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # Calculate ECE
    print("\nCalculating ECE...")
    ece, accuracies, confidences_list, bin_counts, bin_boundaries = calculate_ece(
        confidences, predictions, labels, n_bins=args.n_bins
    )
    
    # Print results
    print("\n" + "="*60)
    print("Calibration Results")
    print("="*60)
    print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
    print(f"ECE: {ece:.4f}")
    
    if ece < 0.05:
        print("  -> Excellent calibration!")
    elif ece < 0.10:
        print("  -> Good calibration")
    elif ece < 0.20:
        print("  -> Fair calibration, may need improvement")
    else:
        print("  -> Poor calibration, needs recalibration")
    
    print(f"\nPer-bin statistics (n_bins={args.n_bins}):")
    print(f"{'Bin':<15} {'Count':<10} {'Confidence':<15} {'Accuracy':<15} {'Gap':<10}")
    print("-" * 70)
    
    for i, (lower, upper) in enumerate(zip(bin_boundaries[:-1], bin_boundaries[1:])):
        if bin_counts[i] > 0:
            gap = abs(confidences_list[i] - accuracies[i])
            print(f"{lower:.2f}-{upper:.2f}      {bin_counts[i]:<10} "
                  f"{confidences_list[i]:<15.4f} {accuracies[i]:<15.4f} {gap:<10.4f}")
    
    # Confidence distribution
    print(f"\nConfidence distribution:")
    print(f"  Mean confidence: {confidences.mean():.4f}")
    print(f"  Median confidence: {np.median(confidences):.4f}")
    print(f"  Min confidence: {confidences.min():.4f}")
    print(f"  Max confidence: {confidences.max():.4f}")
    
    conf_ranges = [
        ("<50%", confidences < 0.5),
        ("50-90%", (confidences >= 0.5) & (confidences < 0.9)),
        ("90-95%", (confidences >= 0.9) & (confidences < 0.95)),
        ("95-99%", (confidences >= 0.95) & (confidences < 0.99)),
        (">=99%", confidences >= 0.99)
    ]
    
    print(f"\nSamples by confidence range:")
    for range_name, mask in conf_ranges:
        count = mask.sum()
        pct = count / len(confidences) * 100
        if count > 0:
            acc = (predictions[mask] == labels[mask]).mean() * 100
            print(f"  {range_name:<10} {count:>6} ({pct:>5.1f}%)  Accuracy: {acc:.2f}%")
    
    # Plot calibration curve
    calib_path = os.path.join(args.output_dir, 'calibration_curve.png')
    plot_calibration_curve(ece, accuracies, confidences_list, bin_counts,
                          bin_boundaries, calib_path)
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'calibration_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Calibration Evaluation Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Data: {args.npz_files}\n\n")
        f.write(f"Overall Accuracy: {accuracy*100:.2f}%\n")
        f.write(f"ECE ({args.n_bins} bins): {ece:.4f}\n\n")
        f.write(f"Mean Confidence: {confidences.mean():.4f}\n")
        f.write(f"Median Confidence: {np.median(confidences):.4f}\n\n")
        f.write("Per-bin statistics:\n")
        f.write(f"{'Bin':<15} {'Count':<10} {'Confidence':<15} {'Accuracy':<15} {'Gap':<10}\n")
        f.write("-" * 70 + "\n")
        for i, (lower, upper) in enumerate(zip(bin_boundaries[:-1], bin_boundaries[1:])):
            if bin_counts[i] > 0:
                gap = abs(confidences_list[i] - accuracies[i])
                f.write(f"{lower:.2f}-{upper:.2f}      {bin_counts[i]:<10} "
                       f"{confidences_list[i]:<15.4f} {accuracies[i]:<15.4f} {gap:<10.4f}\n")
    
    print(f"\nSaved calibration metrics to {metrics_path}")
    print(f"\n[*] Calibration evaluation complete!")


if __name__ == '__main__':
    main()
