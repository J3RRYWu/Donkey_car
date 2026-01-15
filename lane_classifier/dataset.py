"""
Dataset loader for Lane Classification based on CTE
Loads images and CTE values from NPZ files, generates binary labels
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class LaneDataset(Dataset):
    """
    Dataset for lane position classification
    Loads images and CTE values from NPZ files
    Generates binary labels: 0 for left, 1 for right
    """
    
    def __init__(self, npz_paths, normalize=True, target_size=64, 
                 cte_threshold=None, balance_classes=True):
        """
        Args:
            npz_paths: list of paths to NPZ files
            normalize: whether to normalize images to [0, 1]
            target_size: resize images to this size (default 64x64)
            cte_threshold: threshold for splitting left/right. If None, uses median
            balance_classes: whether to balance left/right samples
        """
        self.normalize = normalize
        self.target_size = target_size
        
        # Load all images and CTE values from NPZ files
        all_images = []
        all_cte = []
        
        for npz_path in npz_paths:
            if not os.path.exists(npz_path):
                print(f"Warning: {npz_path} does not exist, skipping")
                continue
                
            print(f"Loading {npz_path}...")
            data = np.load(npz_path, allow_pickle=True)
            
            # Check available keys
            print(f"  Available keys: {list(data.keys())}")
            
            # Load frames
            if 'frame' in data:
                frames = data['frame']  # (T, 3, H, W) or (T, H, W, 3)
                print(f"  Frames shape: {frames.shape}")
            else:
                print(f"  Warning: 'frame' key not found in {npz_path}")
                continue
            
            # Load CTE values
            if 'cte' in data:
                cte = data['cte']  # (T,)
                print(f"  CTE shape: {cte.shape}")
                print(f"  CTE range: [{cte.min():.4f}, {cte.max():.4f}]")
            else:
                print(f"  Warning: 'cte' key not found in {npz_path}")
                continue
            
            # Ensure same length
            min_len = min(len(frames), len(cte))
            frames = frames[:min_len]
            cte = cte[:min_len]
            
            all_images.append(frames)
            all_cte.append(cte)
            print(f"  Loaded {len(frames)} samples")
        
        if not all_images:
            raise ValueError("No data loaded from NPZ files")
        
        # Concatenate all data
        self.images = np.concatenate(all_images, axis=0)
        self.cte = np.concatenate(all_cte, axis=0)
        
        # IMPORTANT: Invert CTE sign because data was collected with reversed definition
        # Negative CTE in data = actually LEFT side (inner track)
        # Positive CTE in data = actually RIGHT side (outer track)
        self.cte = -self.cte
        print(f"\n[!] CTE signs inverted to match physical meaning")
        
        print(f"\nTotal samples before filtering: {len(self.images)}")
        
        # Filter out NaN CTE values
        valid_mask = ~np.isnan(self.cte)
        n_invalid = (~valid_mask).sum()
        if n_invalid > 0:
            print(f"  Filtering out {n_invalid} samples with NaN CTE values")
            self.images = self.images[valid_mask]
            self.cte = self.cte[valid_mask]
        
        print(f"\nTotal samples after filtering: {len(self.images)}")
        print(f"CTE statistics:")
        print(f"  Mean: {self.cte.mean():.4f}")
        print(f"  Std: {self.cte.std():.4f}")
        print(f"  Min: {self.cte.min():.4f}")
        print(f"  Max: {self.cte.max():.4f}")
        print(f"  Median: {np.median(self.cte):.4f}")
        
        # Determine CTE threshold for left/right split
        if cte_threshold is None:
            # Use median as threshold
            self.cte_threshold = np.median(self.cte)
            print(f"\nUsing median CTE as threshold: {self.cte_threshold:.4f}")
        else:
            self.cte_threshold = cte_threshold
            print(f"\nUsing provided CTE threshold: {self.cte_threshold:.4f}")
        
        # Generate binary labels based on CTE (after sign inversion above)
        # After inversion: positive CTE = left side (inner), negative CTE = right side (outer)
        # CTE >= threshold -> left (0), CTE < threshold -> right (1)
        self.labels = (self.cte < self.cte_threshold).astype(np.int64)
        
        # Count samples per class
        n_left = np.sum(self.labels == 0)
        n_right = np.sum(self.labels == 1)
        print(f"\nClass distribution:")
        print(f"  Left (0): {n_left} ({n_left/len(self.labels)*100:.1f}%)")
        print(f"  Right (1): {n_right} ({n_right/len(self.labels)*100:.1f}%)")
        
        # Balance classes if requested
        if balance_classes and n_left != n_right:
            print(f"\nBalancing classes...")
            self._balance_classes()
        
        # Check if images need resizing
        self.need_resize = False
        if len(self.images) > 0:
            sample_shape = self.images[0].shape
            # Handle both (3, H, W) and (H, W, 3) formats
            if len(sample_shape) == 3:
                if sample_shape[0] == 3:  # (3, H, W)
                    h, w = sample_shape[1], sample_shape[2]
                else:  # (H, W, 3)
                    h, w = sample_shape[0], sample_shape[1]
                
                if h != target_size or w != target_size:
                    print(f"\nImages will be resized from {h}x{w} to {target_size}x{target_size}")
                    self.need_resize = True
        
        print(f"\nFinal dataset size: {len(self.images)} samples")
    
    def _balance_classes(self):
        """Balance the dataset by undersampling the majority class"""
        left_indices = np.where(self.labels == 0)[0]
        right_indices = np.where(self.labels == 1)[0]
        
        n_left = len(left_indices)
        n_right = len(right_indices)
        
        # Undersample the majority class
        if n_left > n_right:
            # Randomly select n_right samples from left
            selected_left = np.random.choice(left_indices, n_right, replace=False)
            selected_indices = np.concatenate([selected_left, right_indices])
        else:
            # Randomly select n_left samples from right
            selected_right = np.random.choice(right_indices, n_left, replace=False)
            selected_indices = np.concatenate([left_indices, selected_right])
        
        # Shuffle indices
        np.random.shuffle(selected_indices)
        
        # Update dataset
        self.images = self.images[selected_indices]
        self.cte = self.cte[selected_indices]
        self.labels = self.labels[selected_indices]
        
        print(f"  Balanced to {len(self.images)} samples per class")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: (3, 64, 64) normalized tensor
            label: 0 (left) or 1 (right)
            cte: original CTE value
        """
        # Get image
        image = self.images[idx]
        
        # Handle different image formats
        if len(image.shape) == 3:
            if image.shape[0] == 3:  # (3, H, W)
                # Already in correct format
                pass
            else:  # (H, W, 3)
                image = image.transpose(2, 0, 1)  # -> (3, H, W)
        
        # Resize if needed
        if self.need_resize:
            # Convert to (H, W, 3) for PIL
            image_hwc = image.transpose(1, 2, 0)
            img_pil = Image.fromarray(image_hwc.astype(np.uint8))
            img_resized = img_pil.resize((self.target_size, self.target_size), 
                                         Image.Resampling.LANCZOS)
            image = np.array(img_resized).transpose(2, 0, 1)  # -> (3, H, W)
        
        # Convert to float32
        image = image.astype(np.float32)
        
        # Normalize to [0, 1]
        if self.normalize:
            image = image / 255.0
        
        # Convert to tensor
        image_t = torch.from_numpy(image)
        
        # Get label and CTE
        label = self.labels[idx]
        cte = self.cte[idx]
        
        return image_t, label, cte
    
    def get_class_weights(self):
        """
        Calculate class weights for weighted loss
        Returns:
            weights: tensor of shape (2,) for [left, right] classes
        """
        n_left = np.sum(self.labels == 0)
        n_right = np.sum(self.labels == 1)
        total = len(self.labels)
        
        # Inverse frequency weighting
        weight_left = total / (2 * n_left) if n_left > 0 else 1.0
        weight_right = total / (2 * n_right) if n_right > 0 else 1.0
        
        weights = torch.tensor([weight_left, weight_right], dtype=torch.float32)
        return weights


def create_dataloaders(npz_paths, batch_size=32, val_split=0.2, 
                       target_size=64, balance_classes=True, num_workers=0):
    """
    Create train and validation dataloaders
    Args:
        npz_paths: list of paths to NPZ files
        batch_size: batch size
        val_split: fraction of data for validation
        target_size: image size
        balance_classes: whether to balance classes
        num_workers: number of workers for dataloader
    Returns:
        train_loader, val_loader, class_weights
    """
    # Create full dataset
    full_dataset = LaneDataset(
        npz_paths=npz_paths,
        normalize=True,
        target_size=target_size,
        balance_classes=balance_classes
    )
    
    # Split into train and validation
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {train_size} samples")
    print(f"  Val: {val_size} samples")
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Get class weights
    class_weights = full_dataset.get_class_weights()
    
    return train_loader, val_loader, class_weights


if __name__ == '__main__':
    # Test dataset
    import sys
    
    # Test with local npz files
    npz_paths = [
        'npz_data/traj1_64x64.npz',
        'npz_data/traj2_64x64.npz'
    ]
    
    print("Testing LaneDataset...")
    print("="*60)
    
    try:
        dataset = LaneDataset(
            npz_paths=npz_paths,
            normalize=True,
            target_size=64,
            balance_classes=True
        )
        
        print("\nTesting __getitem__...")
        for i in range(min(3, len(dataset))):
            image, label, cte = dataset[i]
            print(f"\nSample {i}:")
            print(f"  Image shape: {image.shape}")
            print(f"  Image dtype: {image.dtype}")
            print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
            print(f"  Label: {label} ({'left' if label == 0 else 'right'})")
            print(f"  CTE: {cte:.4f}")
        
        print("\nClass weights:")
        weights = dataset.get_class_weights()
        print(f"  Left (0): {weights[0]:.4f}")
        print(f"  Right (1): {weights[1]:.4f}")
        
        print("\n" + "="*60)
        print("Testing create_dataloaders...")
        train_loader, val_loader, class_weights = create_dataloaders(
            npz_paths=npz_paths,
            batch_size=8,
            val_split=0.2,
            target_size=64,
            balance_classes=True
        )
        
        print(f"\nTrain loader: {len(train_loader)} batches")
        print(f"Val loader: {len(val_loader)} batches")
        
        # Test one batch
        for images, labels, ctes in train_loader:
            print(f"\nBatch:")
            print(f"  Images shape: {images.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  CTEs shape: {ctes.shape}")
            print(f"  Labels: {labels}")
            break
        
        print("\n[*] Dataset test passed!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
