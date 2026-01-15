"""
端到端评估 - 潜在空间分类方案
LSTM → Predicted Latent → Latent Classifier → {Left, Right}
跳过VAE解码，直接在潜在空间分类
"""
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
# from tqdm import tqdm  # Optional

# 导入模型
from vae_recon.vae_model_64x64 import SimpleVAE64x64
from predictor.core.vae_predictor import VAEPredictor
from lane_classifier.latent_classifier import get_latent_classifier
from lane_classifier.dataset_visual import detect_red_line_position, get_visual_label


class SequenceDataset(Dataset):
    """序列数据集用于评估"""
    def __init__(self, npz_files, context_length=10):
        self.frames_list = []
        self.context_length = context_length
        
        for npz_file in npz_files:
            data = np.load(npz_file)
            frames = data['frame']  # (N, 3, 64, 64)
            self.frames_list.append(frames)
        
        # 计算可用序列
        self.sequences = []
        for traj_idx, frames in enumerate(self.frames_list):
            num_frames = len(frames)
            for start_idx in range(0, num_frames - context_length - 10, 5):
                self.sequences.append((traj_idx, start_idx))
        
        print(f"Created {len(self.sequences)} test sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        traj_idx, start_idx = self.sequences[idx]
        frames = self.frames_list[traj_idx]
        
        # Context + target
        context = frames[start_idx:start_idx + self.context_length]
        target = frames[start_idx + self.context_length]
        
        return (torch.from_numpy(context).float(),
                torch.from_numpy(target).float())


def predict_and_classify_latent(vae, predictor, classifier, test_loader, device, num_samples=20):
    """
    使用LSTM预测latent，然后在latent空间分类
    """
    vae.eval()
    predictor.eval()
    classifier.eval()
    
    all_predictions = []
    all_true_labels = []
    all_confidences = []
    
    # 用于可视化的样本
    sample_contexts = []
    sample_targets = []
    sample_preds = []
    sample_labels = []
    sample_confs = []
    sample_pred_latents = []
    
    with torch.no_grad():
        for batch_idx, (batch_context, batch_target) in enumerate(test_loader):
            if batch_idx % 10 == 0:
                print(f"  Processing batch {batch_idx}/{len(test_loader)}...")
            batch_context = batch_context.to(device)  # (B, T, 3, 64, 64)
            batch_target = batch_target.to(device)    # (B, 3, 64, 64)
            
            B = batch_context.size(0)
            
            # Step 1: Encode context frames
            context_latents = []
            for t in range(batch_context.size(1)):
                frame_t = batch_context[:, t]  # (B, 3, 64, 64)
                z_t, _, _ = vae.encode(frame_t)
                context_latents.append(z_t)
            context_latents = torch.stack(context_latents, dim=1)  # (B, T, C, H, W)
            
            # Step 2: LSTM predicts next latent
            # Flatten latents and add dummy actions
            B, T, C, H, W = context_latents.shape
            latent_flat = context_latents.view(B, T, -1)  # (B, T, C*H*W=1024)
            
            # Add dummy actions (zeros) to match LSTM input dimension
            dummy_actions = torch.zeros(B, T, 2, device=device)
            lstm_input = torch.cat([latent_flat, dummy_actions], dim=-1)  # (B, T, 1026)
            
            # LSTM forward
            out, _ = predictor.lstm(lstm_input)  # (B, T, hidden)
            out = predictor.lstm_out(out[:, -1])  # Last timestep -> (B, C*H*W)
            
            # Reshape back to spatial
            pred_latent = out.view(B, C, H, W)  # (B, C, H, W)
            
            # Step 3: Classify in latent space
            logits = classifier(pred_latent)  # (B, 2)
            probs = F.softmax(logits, dim=1)
            predictions = logits.argmax(1)
            confidences = probs.max(1)[0]
            
            # Step 4: Get true labels from target frames (visual method)
            batch_true_labels = []
            for i in range(B):
                target_img = batch_target[i].cpu().numpy()  # (3, 64, 64)
                img_transposed = np.transpose(target_img, (1, 2, 0))
                img_uint8 = (img_transposed * 255).astype(np.uint8)
                red_x = detect_red_line_position(img_uint8)
                label = get_visual_label(red_x)
                batch_true_labels.append(label)
            true_labels = torch.tensor(batch_true_labels, device=device)
            
            # 收集结果
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(true_labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            
            # Collect samples for visualization
            if len(sample_contexts) < num_samples:
                for i in range(min(B, num_samples - len(sample_contexts))):
                    sample_contexts.append(batch_context[i, -1].cpu())
                    sample_targets.append(batch_target[i].cpu())
                    sample_preds.append(predictions[i].item())
                    sample_labels.append(true_labels[i].item())
                    sample_confs.append(confidences[i].item())
                    sample_pred_latents.append(pred_latent[i].cpu())
    
    return (np.array(all_predictions), np.array(all_true_labels), np.array(all_confidences),
            sample_contexts, sample_targets, sample_preds, sample_labels, sample_confs,
            sample_pred_latents)


def calculate_ece(confidences, predictions, true_labels, num_bins=10):
    """计算Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    bin_data = []
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = in_bin.sum()
        
        if bin_size > 0:
            bin_confidences = confidences[in_bin]
            bin_predictions = predictions[in_bin]
            bin_true_labels = true_labels[in_bin]
            
            bin_accuracy = (bin_predictions == bin_true_labels).mean()
            bin_confidence = bin_confidences.mean()
            
            ece += (bin_size / len(confidences)) * abs(bin_accuracy - bin_confidence)
            
            bin_data.append({
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'bin_size': bin_size,
                'accuracy': bin_accuracy,
                'confidence': bin_confidence
            })
        else:
            bin_data.append({
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'bin_size': 0,
                'accuracy': 0,
                'confidence': 0
            })
    
    return ece, bin_data


def plot_results(predictions, true_labels, confidences, 
                 sample_contexts, sample_targets, sample_preds, sample_labels, sample_confs,
                 sample_pred_latents, vae_decoder,
                 output_dir):
    """Generate all evaluation visualizations with decoded images"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Left', 'Right'],
                yticklabels=['Left', 'Right'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Latent E2E')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    
    # 2. Calibration curve
    ece, bin_data = calculate_ece(confidences, predictions, true_labels, num_bins=10)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    bin_centers = [(b['bin_lower'] + b['bin_upper']) / 2 for b in bin_data]
    bin_accs = [b['accuracy'] for b in bin_data]
    bin_confs = [b['confidence'] for b in bin_data]
    bin_sizes = [b['bin_size'] for b in bin_data]
    
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax1.bar(bin_centers, bin_accs, width=0.08, alpha=0.5, label='Accuracy')
    ax1.plot(bin_centers, bin_confs, 'ro-', label='Confidence')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(f'Calibration Curve (ECE={ece:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(bin_centers, bin_sizes, width=0.08)
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax2.set_title('Prediction Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'calibration_curve.png'), dpi=150)
    plt.close()
    
    # 3. Prediction samples with decoded images
    num_show = min(12, len(sample_contexts))
    fig, axes = plt.subplots(4, 6, figsize=(18, 12))
    
    # Decode predicted latents to images
    device = next(vae_decoder.parameters()).device
    with torch.no_grad():
        for i in range(num_show):
            row = i // 3
            col_base = (i % 3) * 2
            
            # Left column: Real target image
            ax_real = axes[row, col_base]
            img_real = sample_targets[i].numpy().transpose(1, 2, 0)
            ax_real.imshow(img_real)
            ax_real.set_title('Real Target', fontsize=9)
            ax_real.axis('off')
            
            # Right column: Decoded predicted image
            ax_pred = axes[row, col_base + 1]
            pred_latent = sample_pred_latents[i].unsqueeze(0).to(device)  # (1, C, H, W)
            decoded_img = vae_decoder.decode(pred_latent)
            decoded_img = torch.clamp(decoded_img, 0, 1)
            img_decoded = decoded_img.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
            ax_pred.imshow(img_decoded)
            
            pred = sample_preds[i]
            true = sample_labels[i]
            conf = sample_confs[i]
            
            pred_str = 'Left' if pred == 0 else 'Right'
            true_str = 'Left' if true == 0 else 'Right'
            
            color = 'green' if pred == true else 'red'
            ax_pred.set_title(f'Pred: {pred_str} ({conf:.2f})\nTrue: {true_str}',
                             color=color, fontweight='bold', fontsize=9)
            ax_pred.axis('off')
    
    plt.suptitle('Latent E2E: Real vs LSTM Predicted (decoded)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_samples.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[*] Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='End-to-End Latent Space Evaluation')
    parser.add_argument('--vae_path', type=str, default='vae_recon/best_model.pt',
                        help='VAE model path')
    parser.add_argument('--predictor_path', type=str, default='predictor/checkpoints/Donkey_car_checkpoints_best_model.pt',
                        help='LSTM predictor path')
    parser.add_argument('--classifier_path', type=str, default='lane_classifier/latent_classifier_checkpoints/best_model.pt',
                        help='Latent classifier path')
    parser.add_argument('--npz_files', nargs='+', 
                        default=['npz_data/traj1_64x64.npz', 'npz_data/traj2_64x64.npz'],
                        help='NPZ data files')
    parser.add_argument('--context_length', type=int, default=10,
                        help='Context length for LSTM')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--output_dir', type=str, default='lane_classifier/eval_latent_e2e',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("End-to-End Latent Space Evaluation")
    print("="*70)
    print(f"Device: {device}")
    
    # 1. 加载VAE
    print(f"\n[1/4] Loading VAE from {args.vae_path}...")
    vae = SimpleVAE64x64(latent_dim=64)
    vae_checkpoint = torch.load(args.vae_path, map_location=device, weights_only=False)
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    vae = vae.to(device)
    vae.eval()
    
    # 2. 加载LSTM Predictor
    print(f"[2/4] Loading LSTM Predictor from {args.predictor_path}...")
    predictor_checkpoint = torch.load(args.predictor_path, map_location=device, weights_only=False)
    
    predictor = VAEPredictor(
        latent_dim=64,
        image_size=64,
        channels=3,
        action_dim=2,
        predictor_type='lstm',
        hidden_size=predictor_checkpoint.get('hidden_size', 256),
        residual_prediction=True,
        vae_model_path=args.vae_path,
        freeze_vae=True
    )
    # Load predictor weights (LSTM components)
    # The checkpoint contains the state dict for lstm, lstm_out, etc.
    predictor.load_state_dict(predictor_checkpoint['model_state_dict'], strict=False)
    predictor = predictor.to(device)
    predictor.eval()
    
    # 3. 加载Latent Classifier
    print(f"[3/4] Loading Latent Classifier from {args.classifier_path}...")
    classifier_checkpoint = torch.load(args.classifier_path, map_location=device, weights_only=False)
    model_type = classifier_checkpoint['args'].get('model_type', 'conv')
    
    classifier = get_latent_classifier(
        model_type=model_type,
        latent_dim=64,
        latent_spatial_size=4,
        dropout_rate=0.0  # No dropout in eval
    )
    classifier.load_state_dict(classifier_checkpoint['model_state_dict'])
    classifier = classifier.to(device)
    classifier.eval()
    
    print(f"Classifier type: {model_type}")
    
    # 4. 准备数据
    print(f"[4/4] Loading test data...")
    test_dataset = SequenceDataset(args.npz_files, context_length=args.context_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 5. 评估
    print(f"\n{'='*70}")
    print("Running Evaluation...")
    print("="*70)
    
    (predictions, true_labels, confidences,
     sample_contexts, sample_targets, sample_preds, sample_labels, sample_confs,
     sample_pred_latents) = \
        predict_and_classify_latent(vae, predictor, classifier, test_loader, device, num_samples=12)
    
    # 6. 计算指标
    accuracy = (predictions == true_labels).mean()
    ece, _ = calculate_ece(confidences, predictions, true_labels)
    
    print(f"\n{'='*70}")
    print("Results:")
    print("="*70)
    print(f"Total samples: {len(predictions)}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(true_labels, predictions, 
                                target_names=['Left', 'Right'],
                                digits=4))
    
    # 7. 可视化
    print(f"\n{'='*70}")
    print("Generating Visualizations...")
    print("="*70)
    plot_results(predictions, true_labels, confidences,
                 sample_contexts, sample_targets, sample_preds, sample_labels, sample_confs,
                 sample_pred_latents, vae,
                 args.output_dir)
    
    # 8. 保存指标
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write("Latent Space End-to-End Evaluation\n")
        f.write("="*50 + "\n\n")
        f.write(f"Accuracy: {accuracy*100:.2f}%\n")
        f.write(f"ECE: {ece:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(true_labels, predictions,
                                      target_names=['Left', 'Right'],
                                      digits=4))
    
    print(f"\n{'='*70}")
    print("Evaluation Complete!")
    print("="*70)
    print(f"All results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
