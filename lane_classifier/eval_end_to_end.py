"""
End-to-End Evaluation: LSTM -> VAE -> CNN -> ECE
评估流程：
1. LSTM 预测未来 latent: z_pred
2. VAE 解码: z_pred → 预测图像
3. CNN 分类: 预测图像 → P(左), P(右)
4. 对比真实标签: 真实图像 → 真实位置
5. 计算 ECE: 置信度 vs 准确率
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from typing import Tuple, List
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lane_classifier.cnn_model import get_model as get_lane_model
from vae_recon.vae_model_64x64 import SimpleVAE64x64, load_model_64x64
from predictor.core.vae_predictor import VAEPredictor


def load_models(vae_path, predictor_path, cnn_path, device):
    """
    加载三个模型：VAE, LSTM Predictor, CNN Classifier
    """
    print("="*60)
    print("Loading Models...")
    print("="*60)
    
    # 1. Load VAE (64x64)
    print(f"\n1. Loading VAE from {vae_path}")
    vae_model = load_model_64x64(vae_path, device)
    vae_model.eval()
    
    # 2. Load LSTM Predictor
    print(f"\n2. Loading LSTM Predictor from {predictor_path}")
    checkpoint = torch.load(predictor_path, map_location=device)
    
    # Get predictor args from checkpoint
    if 'args' in checkpoint:
        args = checkpoint['args']
        latent_dim = args.get('latent_dim', 64)
        hidden_size = args.get('hidden_size', 256)
        action_dim = args.get('action_dim', 2)  # Default to 2 if not in checkpoint
    else:
        latent_dim = 64
        hidden_size = 256
        action_dim = 2  # Inferred from weight shapes
    
    predictor = VAEPredictor(
        latent_dim=latent_dim,
        image_size=64,
        channels=3,
        action_dim=action_dim,
        predictor_type='lstm',
        hidden_size=hidden_size,
        vae_model_path=vae_path,
        freeze_vae=True
    )
    predictor.load_state_dict(checkpoint['model_state_dict'])
    predictor.to(device)
    predictor.eval()
    print(f"  Latent dim: {latent_dim}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Action dim: {action_dim}")
    
    # 3. Load CNN Lane Classifier
    print(f"\n3. Loading CNN Lane Classifier from {cnn_path}")
    cnn_checkpoint = torch.load(cnn_path, map_location=device)
    
    if 'args' in cnn_checkpoint:
        model_type = cnn_checkpoint['args'].get('model_type', 'standard')
        dropout = cnn_checkpoint['args'].get('dropout', 0.5)
    else:
        model_type = 'standard'
        dropout = 0.5
    
    cnn_model = get_lane_model(model_type=model_type, dropout_rate=dropout)
    cnn_model.load_state_dict(cnn_checkpoint['model_state_dict'])
    cnn_model.to(device)
    cnn_model.eval()
    print(f"  Model type: {model_type}")
    
    return vae_model, predictor, cnn_model


def load_sequence_data(npz_paths, sequence_length=16, prediction_horizon=1):
    """
    加载序列数据用于预测
    Returns:
        sequences: List of (frames, cte) tuples
    """
    print("\n" + "="*60)
    print("Loading Sequence Data...")
    print("="*60)
    
    all_frames = []
    all_cte = []
    
    for npz_path in npz_paths:
        if not os.path.exists(npz_path):
            print(f"Warning: {npz_path} not found, skipping")
            continue
        
        print(f"\nLoading {npz_path}...")
        data = np.load(npz_path)
        
        frames = data['frame']  # (T, 3, 64, 64)
        cte = data['cte']  # (T,)
        
        print(f"  Frames: {frames.shape}")
        print(f"  CTE: {cte.shape}")
        
        all_frames.append(frames)
        all_cte.append(cte)
    
    # Concatenate
    all_frames = np.concatenate(all_frames, axis=0)
    all_cte = np.concatenate(all_cte, axis=0)
    
    # IMPORTANT: Invert CTE sign to match training data definition
    # (Same as in lane_classifier/dataset.py)
    all_cte = -all_cte
    
    # Filter NaN
    valid_mask = ~np.isnan(all_cte)
    all_frames = all_frames[valid_mask]
    all_cte = all_cte[valid_mask]
    
    print(f"\nTotal valid samples: {len(all_frames)}")
    
    # Create sequences
    sequences = []
    total_length = sequence_length + prediction_horizon
    
    for i in range(len(all_frames) - total_length + 1):
        input_frames = all_frames[i:i+sequence_length]  # Past frames
        target_frame = all_frames[i+sequence_length+prediction_horizon-1]  # Future frame
        target_cte = all_cte[i+sequence_length+prediction_horizon-1]  # Future CTE
        
        sequences.append({
            'input_frames': input_frames,
            'target_frame': target_frame,
            'target_cte': target_cte
        })
    
    print(f"Created {len(sequences)} sequences")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Prediction horizon: {prediction_horizon}")
    
    return sequences, np.median(all_cte)


def predict_and_classify(vae_model, predictor, cnn_model, sequences, cte_threshold, device, batch_size=32):
    """
    端到端预测和分类
    """
    print("\n" + "="*60)
    print("Running End-to-End Prediction...")
    print("="*60)
    
    all_predictions = []
    all_probabilities = []
    all_true_labels = []
    all_predicted_images = []
    all_target_images = []
    
    num_batches = (len(sequences) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processing batch {batch_idx+1}/{num_batches}...")
            
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(sequences))
            batch_sequences = sequences[start_idx:end_idx]
            
            # Prepare batch
            batch_input_frames = []
            batch_target_frames = []
            batch_target_cte = []
            
            for seq in batch_sequences:
                batch_input_frames.append(seq['input_frames'])
                batch_target_frames.append(seq['target_frame'])
                batch_target_cte.append(seq['target_cte'])
            
            batch_input_frames = np.array(batch_input_frames).astype(np.float32) / 255.0
            batch_target_frames = np.array(batch_target_frames).astype(np.float32) / 255.0
            batch_target_cte = np.array(batch_target_cte)
            
            # Convert to tensors
            input_frames_t = torch.from_numpy(batch_input_frames).to(device)  # (B, T, 3, 64, 64)
            target_frames_t = torch.from_numpy(batch_target_frames).to(device)  # (B, 3, 64, 64)
            
            # Step 1: LSTM predicts future latent
            # Encode input sequence to latents
            B, T, C, H, W = input_frames_t.shape
            input_flat = input_frames_t.view(B*T, C, H, W)
            
            # Use VAE to encode
            mu, logvar, _ = vae_model.encode(input_flat)
            z_input = mu  # Use mean as latent (B*T, latent_dim, 4, 4)
            
            # Reshape to sequence
            z_seq = z_input.view(B, T, *z_input.shape[1:])  # (B, T, latent_dim, 4, 4)
            
            # Prepare actions if needed
            actions_seq = None
            if predictor.action_dim > 0:
                actions_seq = torch.zeros(B, T, predictor.action_dim).to(device)
            
            # Use the CORRECT prediction method (rollout from context)
            # Predict 1 step into the future
            try:
                z_pred = predictor.rollout_from_context(
                    z_context=z_seq,
                    steps=1,
                    a_full=actions_seq,
                    context_action_len=T-1,
                    start_action_index=T-1
                )  # Returns (B, 1, latent_dim, 4, 4)
                
                z_pred = z_pred[:, 0, ...]  # (B, latent_dim, 4, 4)
            except:
                # Fallback: use predict method
                if actions_seq is not None:
                    z_pred = predictor.predict(z_seq, actions_seq)
                else:
                    z_pred = predictor.predict(z_seq)
                
                # If returns sequence, take last
                if z_pred.dim() == 5:
                    z_pred = z_pred[:, -1, ...]  # (B, latent_dim, 4, 4)
            
            # Step 2: VAE decodes predicted latent to image
            predicted_images = vae_model.decode(z_pred, skip_features=None)  # (B, 3, 64, 64)
            predicted_images = torch.clamp(predicted_images, 0, 1)
            
            # Step 3: CNN classifies predicted images
            cnn_outputs = cnn_model(predicted_images)  # (B, 2)
            probabilities = torch.softmax(cnn_outputs, dim=1)  # (B, 2)
            predictions = cnn_outputs.argmax(1)  # (B,)
            
            # Step 4: Get true labels from target CTE
            # Match training definition: CTE < threshold → Right (1)
            true_labels = (batch_target_cte < cte_threshold).astype(np.int64)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_true_labels.extend(true_labels)
            
            # Store sample images for visualization
            if batch_idx < 5:  # Only first few batches
                all_predicted_images.extend(predicted_images.cpu().numpy())
                all_target_images.extend(batch_target_frames)
    
    print(f"\nProcessed {len(all_predictions)} predictions")
    
    return (np.array(all_predictions), 
            np.array(all_probabilities),
            np.array(all_true_labels),
            all_predicted_images[:50],  # Limit to 50 samples
            all_target_images[:50])


def calculate_ece(probabilities, predictions, true_labels, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE)
    ECE measures the difference between confidence and accuracy
    """
    print("\n" + "="*60)
    print("Calculating Expected Calibration Error (ECE)...")
    print("="*60)
    
    # Get confidence (max probability)
    confidences = np.max(probabilities, axis=1)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    accuracies = []
    confidences_list = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_count = in_bin.sum()
        
        if bin_count > 0:
            # Accuracy in this bin
            accuracy_in_bin = (predictions[in_bin] == true_labels[in_bin]).mean()
            # Average confidence in this bin
            avg_confidence_in_bin = confidences[in_bin].mean()
            
            # Contribution to ECE
            ece += (bin_count / len(confidences)) * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
            accuracies.append(accuracy_in_bin)
            confidences_list.append(avg_confidence_in_bin)
            bin_counts.append(bin_count)
        else:
            accuracies.append(None)
            confidences_list.append(None)
            bin_counts.append(0)
    
    print(f"\nECE: {ece:.4f}")
    print(f"\nPer-bin statistics:")
    print(f"{'Bin':<10} {'Count':<10} {'Confidence':<15} {'Accuracy':<15} {'Gap':<10}")
    print("-" * 65)
    
    for i, (lower, upper) in enumerate(zip(bin_lowers, bin_uppers)):
        if bin_counts[i] > 0:
            gap = abs(confidences_list[i] - accuracies[i])
            print(f"{lower:.2f}-{upper:.2f}  {bin_counts[i]:<10} "
                  f"{confidences_list[i]:<15.4f} {accuracies[i]:<15.4f} {gap:<10.4f}")
    
    return ece, accuracies, confidences_list, bin_counts, bin_boundaries


def plot_calibration_curve(ece, accuracies, confidences_list, bin_counts, bin_boundaries, save_path):
    """Plot reliability diagram (calibration curve)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Filter out empty bins
    valid_bins = [i for i in range(len(bin_counts)) if bin_counts[i] > 0]
    valid_confidences = [confidences_list[i] for i in valid_bins]
    valid_accuracies = [accuracies[i] for i in valid_bins]
    valid_counts = [bin_counts[i] for i in valid_bins]
    
    # Plot 1: Reliability Diagram
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    ax1.bar(valid_confidences, valid_accuracies, 
            width=0.08, alpha=0.7, label='Model', 
            edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Confidence', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title(f'Reliability Diagram (ECE={ece:.4f})', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Plot 2: Sample distribution
    bin_centers = [(bin_boundaries[i] + bin_boundaries[i+1])/2 for i in valid_bins]
    ax2.bar(bin_centers, valid_counts, width=0.08, alpha=0.7, 
            edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Confidence', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Sample Distribution', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved calibration curve to {save_path}")


def plot_results(predictions, probabilities, true_labels, 
                predicted_images, target_images, save_dir):
    """Plot evaluation results"""
    print("\n" + "="*60)
    print("Generating Visualizations...")
    print("="*60)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Right', 'Left'],
                yticklabels=['Right', 'Left'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (End-to-End)')
    plt.tight_layout()
    cm_path = os.path.join(save_dir, 'e2e_confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")
    
    # 2. Confidence Distribution
    confidences = np.max(probabilities, axis=1)
    correct = (predictions == true_labels)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist([confidences[correct], confidences[~correct]], 
                 bins=30, label=['Correct', 'Incorrect'], 
                 alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Confidence')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Confidence Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy vs Confidence
    conf_bins = np.linspace(0, 1, 11)
    bin_accs = []
    bin_centers = []
    
    for i in range(len(conf_bins) - 1):
        mask = (confidences >= conf_bins[i]) & (confidences < conf_bins[i+1])
        if mask.sum() > 0:
            bin_accs.append((predictions[mask] == true_labels[mask]).mean() * 100)
            bin_centers.append((conf_bins[i] + conf_bins[i+1]) / 2)
    
    axes[1].plot(bin_centers, bin_accs, 'o-', linewidth=2, markersize=8)
    axes[1].plot([0, 1], [0, 100], 'k--', label='Perfect Calibration')
    axes[1].set_xlabel('Confidence')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy vs Confidence')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    conf_path = os.path.join(save_dir, 'e2e_confidence_analysis.png')
    plt.savefig(conf_path, dpi=150)
    plt.close()
    print(f"Saved confidence analysis to {conf_path}")
    
    # 3. Sample Predictions
    num_samples = min(16, len(predicted_images))
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_samples):
        pred_img = predicted_images[i].transpose(1, 2, 0)  # (64, 64, 3)
        target_img = target_images[i].transpose(1, 2, 0)
        
        # Concatenate predicted and target
        combined = np.concatenate([pred_img, target_img], axis=1)  # (64, 128, 3)
        
        axes[i].imshow(combined)
        axes[i].axis('off')
        
        pred_label = 'Right' if predictions[i] == 0 else 'Left'
        true_label = 'Right' if true_labels[i] == 0 else 'Left'
        conf = probabilities[i, predictions[i]]
        color = 'green' if predictions[i] == true_labels[i] else 'red'
        
        title = f'Pred: {pred_label} ({conf:.2f})\nTrue: {true_label}'
        axes[i].set_title(title, fontsize=8, color=color)
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Predicted (Left) vs Target (Right) Images', fontsize=14)
    plt.tight_layout()
    samples_path = os.path.join(save_dir, 'e2e_prediction_samples.png')
    plt.savefig(samples_path, dpi=150)
    plt.close()
    print(f"Saved prediction samples to {samples_path}")


def main():
    parser = argparse.ArgumentParser(description='End-to-End Evaluation: LSTM->VAE->CNN->ECE')
    
    # Model paths
    parser.add_argument('--vae_path', type=str, default='vae_recon/best_model.pt',
                        help='Path to VAE model')
    parser.add_argument('--predictor_path', type=str, default='predictor/checkpoints/best_model.pt',
                        help='Path to LSTM predictor model')
    parser.add_argument('--cnn_path', type=str, default='lane_classifier/checkpoints/best_model.pt',
                        help='Path to CNN lane classifier')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='npz_data',
                        help='Directory containing NPZ files')
    parser.add_argument('--npz_files', nargs='+',
                        default=['traj1_64x64.npz', 'traj2_64x64.npz'],
                        help='NPZ files to evaluate')
    parser.add_argument('--sequence_length', type=int, default=16,
                        help='Input sequence length')
    parser.add_argument('--prediction_horizon', type=int, default=1,
                        help='Prediction horizon (frames ahead)')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--ece_bins', type=int, default=10,
                        help='Number of bins for ECE calculation')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cpu/cuda)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='lane_classifier/eval_e2e',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("="*60)
    print("End-to-End Evaluation: LSTM -> VAE -> CNN -> ECE")
    print("="*60)
    print(f"Device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup data paths
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.abspath(os.path.join(script_dir, '..', data_dir))
    
    npz_paths = [os.path.join(data_dir, f) for f in args.npz_files]
    
    # Load models
    vae_model, predictor, cnn_model = load_models(
        args.vae_path, args.predictor_path, args.cnn_path, device
    )
    
    # Load data
    sequences, cte_threshold = load_sequence_data(
        npz_paths, args.sequence_length, args.prediction_horizon
    )
    print(f"\nCTE threshold (median): {cte_threshold:.4f}")
    
    # Run end-to-end prediction
    predictions, probabilities, true_labels, predicted_images, target_images = predict_and_classify(
        vae_model, predictor, cnn_model, sequences, cte_threshold, device, args.batch_size
    )
    
    # Calculate metrics
    print("\n" + "="*60)
    print("Overall Results")
    print("="*60)
    
    accuracy = (predictions == true_labels).mean() * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, 
                                target_names=['Right', 'Left'],
                                digits=4))
    
    # Calculate ECE
    ece, accuracies, confidences_list, bin_counts, bin_boundaries = calculate_ece(
        probabilities, predictions, true_labels, n_bins=args.ece_bins
    )
    
    # Plot calibration curve
    calib_path = os.path.join(args.output_dir, 'e2e_calibration_curve.png')
    plot_calibration_curve(ece, accuracies, confidences_list, bin_counts, 
                          bin_boundaries, calib_path)
    
    # Plot other results
    plot_results(predictions, probabilities, true_labels,
                predicted_images, target_images, args.output_dir)
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'e2e_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("End-to-End Evaluation Results\n")
        f.write("="*60 + "\n\n")
        f.write(f"Models:\n")
        f.write(f"  VAE: {args.vae_path}\n")
        f.write(f"  Predictor: {args.predictor_path}\n")
        f.write(f"  CNN: {args.cnn_path}\n\n")
        f.write(f"Data: {args.npz_files}\n")
        f.write(f"Sequence Length: {args.sequence_length}\n")
        f.write(f"Prediction Horizon: {args.prediction_horizon}\n\n")
        f.write(f"Overall Accuracy: {accuracy:.2f}%\n")
        f.write(f"ECE: {ece:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(true_labels, predictions,
                                     target_names=['Right', 'Left'],
                                     digits=4))
    
    print(f"\nSaved metrics to {metrics_path}")
    print(f"\n[*] End-to-end evaluation complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
