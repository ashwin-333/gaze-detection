import os
# Required packages:
# pip install torch numpy scikit-learn tqdm
# or activate conda environment with these packages
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import time
from sklearn.metrics import f1_score, precision_score, recall_score
import argparse
import math
import torch.nn.functional as F
from tqdm import tqdm  # Add tqdm for progress bars
from sklearn.model_selection import StratifiedKFold
from collections import Counter

# Setting paths to the new data location
base_dir = Path("/home/sheo1/nvme1/GazeVI-ML")
# Define subjects to use - updated to include all S1 through S6
subjects = ["S1-01", "S1-02", "S1-03", "S1-04", "S1-05", "S1-06"]
# Define lens types if needed
lens_types = ["FT", "FW", "LT", "LW"]  # You can add "FW" if needed

# Configure input modalities and parameters
parser = argparse.ArgumentParser(description='Gaze classification with transformer model')
parser.add_argument('--use_left_gaze', action='store_true', default=True, help='Use left gaze data')
parser.add_argument('--use_right_gaze', action='store_true', default=True, help='Use right gaze data')
parser.add_argument('--use_head_pose', action='store_true', default=True, help='Use head pose data')
parser.add_argument('--auto_select_eye', action='store_true', help='Automatically select eye with lower NaN rate')
parser.add_argument('--nan_threshold', type=float, default=0.2, help='Maximum fraction of NaNs allowed')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--model_dim', type=int, default=64, help='Transformer model dimension')
parser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
parser.add_argument('--weight_method', type=str, default='inverse', choices=['inverse', 'balanced', 'effective_samples', 'none'], 
                   help='Method for computing class weights')
parser.add_argument('--focal_loss', action='store_true', help='Use focal loss instead of cross entropy')
parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma parameter for focal loss')
parser.add_argument('--weight_factor', type=float, default=1.0, help='Factor to adjust weight intensity (higher = stronger weighting)')
parser.add_argument('--use_weighted_loss', action='store_true', default=True, help='Use weighted cross entropy loss')
parser.add_argument('--add_velocity', action='store_true', default=True, 
                   help='Add velocity features for gaze data')
parser.add_argument('--add_acceleration', action='store_true', default=True, 
                   help='Add acceleration features for gaze data')
parser.add_argument('--normalize_features', action='store_true', default=True, 
                   help='Normalize gaze features using z-score normalization')
parser.add_argument('--mixup_alpha', type=float, default=0.2, 
                   help='Alpha parameter for mixup augmentation (0 to disable)')
parser.add_argument('--use_lr_scheduler', action='store_true', default=True, 
                    help='Use learning rate scheduler')
parser.add_argument('--scheduler_type', type=str, default='cosine', 
                    choices=['cosine', 'step', 'plateau'], 
                    help='Type of learning rate scheduler')
parser.add_argument('--label_smoothing', type=float, default=0.1, 
                    help='Label smoothing factor (0 to disable)')
parser.add_argument('--optimizer', type=str, default='adamw', 
                    choices=['adam', 'adamw', 'sgd'], 
                    help='Optimizer type')
parser.add_argument('--use_kfold', action='store_true', default=True, 
                    help='Use stratified k-fold cross-validation')
parser.add_argument('--n_folds', type=int, default=5, 
                    help='Number of folds for cross-validation')
parser.add_argument('--use_ensemble', action='store_true', default=True, 
                    help='Use model ensemble')
parser.add_argument('--balance_dataset', action='store_true', default=True, 
                    help='Balance dataset classes by undersampling majority class')
parser.add_argument('--temporal_smoothing', action='store_true', default=True, 
                    help='Apply temporal smoothing to predictions')
parser.add_argument('--attention_dropout', type=float, default=0.2, 
                    help='Dropout rate for attention layers')

# Default to using both left and right gaze if no specific option is selected
args = parser.parse_args()

print(f"Configuration:")
print(f"  Use left gaze: {args.use_left_gaze}")
print(f"  Use right gaze: {args.use_right_gaze}")
print(f"  Use head pose: {args.use_head_pose}")
print(f"  Add velocity features: {args.add_velocity}")
print(f"  Add acceleration features: {args.add_acceleration}")
print(f"  Normalize features: {args.normalize_features}")
print(f"  Using weighted loss: {args.use_weighted_loss}")

# Load the dataset
def load_pkl(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

# Collect all meta files
all_data = {}
print("Loading data files...")
for subject in tqdm(subjects, desc="Processing subjects"):
    for lens_type in lens_types:
        meta_path = base_dir / subject / lens_type / "meta.pkl"
        if meta_path.exists():
            print(f"Loading {meta_path}")
            try:
                data = load_pkl(meta_path)
                all_data[f"{subject}_{lens_type}"] = data
                print(f"Successfully loaded {len(data)} segments from {meta_path}")
            except Exception as e:
                print(f"Error loading {meta_path}: {e}")
        else:
            print(f"File not found: {meta_path}")

if not all_data:
    raise ValueError("No data was loaded! Check the file paths.")

# Calculate the input dimension based on selected modalities
def calculate_input_dim(add_velocity=True, add_acceleration=True):
    """Calculate the input dimension based on selected data sources."""
    # Base dimensions for position data
    base_dim = 0
    if args.use_left_gaze:
        base_dim += 2  # x, y coordinates
    if args.use_right_gaze:
        base_dim += 2  # x, y coordinates
    if args.use_head_pose:
        base_dim += 3  # euler angles (x, y, z)
    
    # Apply feature multipliers for velocity and acceleration
    feature_multiplier = 1
    if add_velocity:
        feature_multiplier += 1  # Add velocity features
    if add_acceleration:
        feature_multiplier += 1  # Add acceleration features
    
    # Calculate total dimension - only apply multiplier to gaze features, not head pose
    gaze_dim = 0
    if args.use_left_gaze:
        gaze_dim += 2  # x, y coordinates
    if args.use_right_gaze:
        gaze_dim += 2  # x, y coordinates
        
    total_dim = (gaze_dim * feature_multiplier) + (3 if args.use_head_pose else 0)
    
    # Ensure at least 2D input
    return max(total_dim, 2)

# Calculate input dim with velocity and acceleration
input_dim = calculate_input_dim(args.add_velocity, args.add_acceleration)
print(f"Input dimension: {input_dim}")

# Enhanced GazeDataset to handle the new data format
class GazeDataset(Dataset):
    def __init__(self, data_dict, exclude_non_reading_for_training=False, add_velocity=True, add_acceleration=True, normalize_features=True):
        # Updated label mapping
        self.label_map = {"reading": 0, "scanning": 1, "non reading": 2}
        self.add_velocity = add_velocity
        self.add_acceleration = add_acceleration
        self.normalize_features = normalize_features
        
        # Process all data into a usable format
        self.all_samples = []
        self.file_names = []
        
        print("Processing dataset...")
        processed_count = 0
        skipped_count = 0
        
        # Collect feature statistics for normalization
        if normalize_features:
            print("Computing feature statistics for normalization...")
            all_gaze_data = []
            all_head_data = []
            for file_name, data in tqdm(data_dict.items(), desc="Collecting statistics"):
                for frame_num, frame_data in data.items():
                    if args.use_left_gaze and "left_gaze_screen" in frame_data:
                        all_gaze_data.extend(frame_data["left_gaze_screen"])
                    if args.use_right_gaze and "right_gaze_screen" in frame_data:
                        all_gaze_data.extend(frame_data["right_gaze_screen"])
                    if args.use_head_pose and "head_rot_euler" in frame_data and not all(x == 0 for x in frame_data["head_rot_euler"]):
                        all_head_data.append(frame_data["head_rot_euler"])
            
            # Compute mean and std for gaze data
            all_gaze_array = np.array(all_gaze_data)
            all_gaze_array = all_gaze_array[~np.isnan(all_gaze_array).any(axis=1)]
            self.gaze_mean = np.mean(all_gaze_array, axis=0)
            self.gaze_std = np.std(all_gaze_array, axis=0)
            print(f"Gaze mean: {self.gaze_mean}, std: {self.gaze_std}")
            
            # Compute mean and std for head pose data
            if all_head_data:
                all_head_array = np.array(all_head_data)
                all_head_array = all_head_array[~np.isnan(all_head_array).any(axis=1)]
                self.head_mean = np.mean(all_head_array, axis=0)
                self.head_std = np.std(all_head_array, axis=0)
                print(f"Head pose mean: {self.head_mean}, std: {self.head_std}")
            else:
                self.head_mean = np.zeros(3)
                self.head_std = np.ones(3)
        
        for file_name, data in tqdm(data_dict.items(), desc="Processing files"):
            for frame_num, frame_data in tqdm(data.items(), desc=f"Processing frames in {file_name}", leave=False):
                # Check for required fields
                if not all(k in frame_data for k in ["reading_label"]):
                    skipped_count += 1
                    continue
                    
                # Get required data - only using regular gaze_screen, not comp_gaze
                if args.use_left_gaze and "left_gaze_screen" not in frame_data:
                    skipped_count += 1
                    continue
                if args.use_right_gaze and "right_gaze_screen" not in frame_data:
                    skipped_count += 1
                    continue
                if args.use_head_pose and "head_rot_euler" not in frame_data:
                    skipped_count += 1
                    continue
                
                # Store data for this sample
                sample = {
                    "file_name": file_name,
                    "frame_num": frame_num,
                    "reading_label": frame_data["reading_label"]
                }
                
                # Only use regular gaze_screen data, not comp_gaze
                if args.use_left_gaze:
                    left_gaze = np.array(frame_data["left_gaze_screen"])
                    sample["left_gaze"] = left_gaze
                    if "left_gaze_mask" in frame_data:
                        sample["left_mask"] = np.array(frame_data["left_gaze_mask"])
                    else:
                        # If no mask provided, create one (non-zero values are valid)
                        sample["left_mask"] = ~np.all(left_gaze == 0, axis=1)
                
                # Only use regular gaze_screen data, not comp_gaze
                if args.use_right_gaze:
                    right_gaze = np.array(frame_data["right_gaze_screen"])
                    sample["right_gaze"] = right_gaze
                    if "right_gaze_mask" in frame_data:
                        sample["right_mask"] = np.array(frame_data["right_gaze_mask"])
                    else:
                        # If no mask provided, create one (non-zero values are valid)
                        sample["right_mask"] = ~np.all(right_gaze == 0, axis=1)
                
                # Process head pose
                if args.use_head_pose:
                    sample["head_pose"] = np.array(frame_data["head_rot_euler"])
                    # Don't normalize here - will normalize in __getitem__
                
                # Only add if labels are valid
                valid_labels = []
                if isinstance(sample["reading_label"], list):
                    for label in sample["reading_label"]:
                        norm_label = self._normalize_label(label)
                        if norm_label in self.label_map:
                            valid_labels.append(norm_label)
                else:
                    norm_label = self._normalize_label(sample["reading_label"])
                    if norm_label in self.label_map:
                        valid_labels.append(norm_label)
                
                if not valid_labels:
                    continue
                    
                sample["normalized_labels"] = valid_labels
                
                # Skip non-reading samples if requested
                if exclude_non_reading_for_training and all(label == "non reading" for label in valid_labels):
                    skipped_count += 1
                    continue
                    
                self.all_samples.append(sample)
                self.file_names.append(f"{file_name}_{frame_num}")
                processed_count += 1
        
        # Count labels for reporting
        self.label_counts = {"reading": 0, "scanning": 0, "non reading": 0}
        for sample in self.all_samples:
            for label in sample["normalized_labels"]:
                if label in self.label_counts:
                    self.label_counts[label] += 1
        
        print(f"Dataset created with {len(self.all_samples)} samples (processed: {processed_count}, skipped: {skipped_count})")
        print("Dataset label distribution:")
        for label, count in self.label_counts.items():
            print(f"  {label}: {count} ({count/sum(self.label_counts.values())*100:.1f}%)")
    
    def _normalize_label(self, label):
        """Normalize label string."""
        if not label:
            return "unknown"
        label_str = str(label).lower().strip()
        if label_str == "line_changing":
            label_str = "scanning"
        elif label_str == "resting":
            label_str = "non reading"
        return label_str
        
    def _normalize_head_data(self, data):
        """Z-score normalize the head pose data with safety checks."""
        if np.all(self.head_std == 0):
            # Avoid division by zero
            return np.zeros_like(data)
        
        # Clip extreme values to avoid numerical issues
        clipped_data = np.clip(data, 
                              self.head_mean - 5 * self.head_std, 
                              self.head_mean + 5 * self.head_std)
        
        # Normalize
        normalized = (clipped_data - self.head_mean) / (self.head_std + 1e-6)
        
        # Clip again to avoid extreme values
        normalized = np.clip(normalized, -5.0, 5.0)
        
        # Replace any NaNs
        normalized = np.nan_to_num(normalized, nan=0.0)
        
        return normalized
        
    def _calculate_velocity(self, positions):
        """Calculate velocity features from positions."""
        # Calculate velocity (dx, dy)
        velocity = np.zeros_like(positions)
        # Only calculate for valid positions (non-zero and non-NaN)
        valid_mask = ~np.isnan(positions).any(axis=1) & ~(positions == 0).all(axis=1)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 1:
            for i in range(1, len(valid_indices)):
                prev_idx = valid_indices[i-1]
                curr_idx = valid_indices[i]
                if curr_idx - prev_idx == 1:  # Only calculate if consecutive
                    velocity[curr_idx] = positions[curr_idx] - positions[prev_idx]
        
        # Replace any NaNs that might have occurred
        velocity = np.nan_to_num(velocity, nan=0.0)
        return velocity
        
    def _calculate_acceleration(self, velocity):
        """Calculate acceleration features from velocity."""
        # Calculate acceleration (dvx, dvy)
        acceleration = np.zeros_like(velocity)
        # Only calculate for valid velocities (non-zero and non-NaN)
        valid_mask = ~np.isnan(velocity).any(axis=1) & ~(velocity == 0).all(axis=1)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 1:
            for i in range(1, len(valid_indices)):
                prev_idx = valid_indices[i-1]
                curr_idx = valid_indices[i]
                if curr_idx - prev_idx == 1:  # Only calculate if consecutive
                    acceleration[curr_idx] = velocity[curr_idx] - velocity[prev_idx]
        
        # Replace any NaNs that might have occurred
        acceleration = np.nan_to_num(acceleration, nan=0.0)
        return acceleration
        
    def _normalize_data(self, data):
        """Z-score normalize the data with safety checks."""
        if np.all(self.gaze_std == 0):
            # Avoid division by zero
            return np.zeros_like(data)
        
        # Clip extreme values to avoid numerical issues
        clipped_data = np.clip(data, 
                              self.gaze_mean - 5 * self.gaze_std, 
                              self.gaze_mean + 5 * self.gaze_std)
        
        # Normalize
        normalized = (clipped_data - self.gaze_mean) / (self.gaze_std + 1e-6)
        
        # Clip again to avoid extreme values
        normalized = np.clip(normalized, -5.0, 5.0)
        
        # Replace any NaNs
        normalized = np.nan_to_num(normalized, nan=0.0)
        
        return normalized

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        sample = self.all_samples[idx]
        features = []
        attention_mask = []
        
        # Process left gaze
        if args.use_left_gaze and "left_gaze" in sample:
            left_gaze = sample["left_gaze"].copy()
            mask = sample["left_mask"]
            
            # Apply normalization if requested
            if self.normalize_features:
                # Only normalize valid data points
                valid_indices = np.where(mask)[0]
                if len(valid_indices) > 0:
                    left_gaze[valid_indices] = self._normalize_data(left_gaze[valid_indices])
                
            # Add original gaze positions
            features.append(left_gaze)
            attention_mask.append(mask)
            
            # Add velocity features if requested
            if self.add_velocity and left_gaze.shape[0] > 1:
                velocity = self._calculate_velocity(left_gaze)
                features.append(velocity)
                attention_mask.append(mask)  # Use same mask as positions
                
                # Add acceleration features if requested
                if self.add_acceleration and left_gaze.shape[0] > 2:
                    acceleration = self._calculate_acceleration(velocity)
                    features.append(acceleration)
                    attention_mask.append(mask)  # Use same mask as positions
        
        # Process right gaze
        if args.use_right_gaze and "right_gaze" in sample:
            right_gaze = sample["right_gaze"].copy()
            mask = sample["right_mask"]
            
            # Apply normalization if requested
            if self.normalize_features:
                # Only normalize valid data points
                valid_indices = np.where(mask)[0]
                if len(valid_indices) > 0:
                    right_gaze[valid_indices] = self._normalize_data(right_gaze[valid_indices])
                
            # Add original gaze positions
            features.append(right_gaze)
            attention_mask.append(mask)
            
            # Add velocity features if requested
            if self.add_velocity and right_gaze.shape[0] > 1:
                velocity = self._calculate_velocity(right_gaze)
                features.append(velocity)
                attention_mask.append(mask)  # Use same mask as positions
                
                # Add acceleration features if requested
                if self.add_acceleration and right_gaze.shape[0] > 2:
                    acceleration = self._calculate_acceleration(velocity)
                    features.append(acceleration)
                    attention_mask.append(mask)  # Use same mask as positions
        
        # Process head pose - repeat to match sequence length
        if args.use_head_pose and "head_pose" in sample:
            head_pose = sample["head_pose"].copy()
            # If head pose is a single vector, repeat it to match sequence length
            if len(head_pose.shape) == 1:
                seq_len = features[0].shape[0] if features else 1
                head_pose = np.tile(head_pose, (seq_len, 1))
            
            # Apply normalization to head pose data if requested
            if self.normalize_features and not np.all(head_pose == 0):
                head_pose = self._normalize_head_data(head_pose)
            
            features.append(head_pose)
            # Assume head pose is valid where gaze is valid
            if attention_mask:
                mask = attention_mask[0].copy()  # Use same mask as first feature
            else:
                mask = np.ones(head_pose.shape[0], dtype=bool)
            attention_mask.append(mask)
        
        # Combine features
        if features:
            # Align sequence length (use shortest sequence)
            min_length = min(f.shape[0] for f in features)
            features = [f[:min_length] for f in features]
            attention_mask = [m[:min_length] for m in attention_mask]
            
            # Ensure all attention masks have the same shape before combining
            if not all(m.shape == attention_mask[0].shape for m in attention_mask):
                # Reshape masks to match the first one
                for i in range(1, len(attention_mask)):
                    if attention_mask[i].shape != attention_mask[0].shape:
                        # Either truncate or extend the mask
                        if len(attention_mask[i]) > len(attention_mask[0]):
                            attention_mask[i] = attention_mask[i][:len(attention_mask[0])]
                        else:
                            # Pad with False
                            pad_length = len(attention_mask[0]) - len(attention_mask[i])
                            attention_mask[i] = np.concatenate([attention_mask[i], 
                                                               np.zeros(pad_length, dtype=bool)])
            
            # Concatenate features along feature dimension
            combined_features = np.concatenate(features, axis=1)
            # Combine masks with error checking
            try:
                combined_mask = np.all(attention_mask, axis=0)
            except ValueError:
                # If there's an issue with mask shapes, use the first mask as fallback
                combined_mask = attention_mask[0]
                print(f"Warning: Mask shape mismatch in sample {idx}. Using fallback mask.")
        else:
            raise ValueError("No features were extracted")
        
        # Get label (use first label for now)
        label_str = sample["normalized_labels"][0]
        label_idx = self.label_map[label_str]
        
        # Convert data to tensors
        gaze_tensor = torch.tensor(combined_features, dtype=torch.float32)
        mask_tensor = torch.tensor(combined_mask, dtype=torch.bool)
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        
        return gaze_tensor, mask_tensor, label_tensor, idx

# Updated collate function with attention mask handling
def collate_gaze(batch):
    gaze_list, mask_list, label_list, idx_list = zip(*batch)
    
    # Find max sequence length
    max_length = max(g.shape[0] for g in gaze_list)
    
    # Pad gaze data and attention masks
    padded_gaze = []
    padded_masks = []
    
    for g, m in zip(gaze_list, mask_list):
        # Current shape and padding needed
        curr_length, feat_dim = g.shape
        pad_length = max_length - curr_length
        
        # Pad gaze features
        if pad_length > 0:
            gaze_pad = torch.zeros(pad_length, feat_dim, dtype=g.dtype)
            g_padded = torch.cat([g, gaze_pad], dim=0)
        else:
            g_padded = g
            
        # Pad attention mask (pad with False)
        if pad_length > 0:
            mask_pad = torch.zeros(pad_length, dtype=torch.bool)
            m_padded = torch.cat([m, mask_pad], dim=0)
        else:
            m_padded = m
            
        padded_gaze.append(g_padded)
        padded_masks.append(m_padded)
    
    # Stack tensors into batches
    batch_gaze = torch.stack(padded_gaze)     # [batch, seq_len, feat_dim]
    batch_masks = torch.stack(padded_masks)   # [batch, seq_len]
    batch_labels = torch.tensor(label_list, dtype=torch.long)
    
    return batch_gaze, batch_masks, batch_labels, idx_list

# Improved Transformer with advanced features but more stability
class GazeTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=128, num_classes=3, num_layers=3, nhead=4, dropout=0.1, activation='relu'):
        super(GazeTransformer, self).__init__()
        
        self.model_dim = model_dim
        
        # Input projection with layer normalization
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.LayerNorm(model_dim)
        )
        
        # Positional encoding for sequence data
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        
        # Transformer layers with self-attention
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=model_dim * 2,  # Smaller feedforward dimension
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output layers with simpler architecture
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_classes)
        )

    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, input_dim]
        # mask: [batch_size, seq_len] - True for valid positions, False for padding/invalid
        
        # Project input to model dimension
        x = self.input_projection(x)  # [batch_size, seq_len, model_dim]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create attention mask for transformer
        if mask is not None:
            # Check if any batch has all masked values and fix
            has_all_masked = (~mask).all(dim=1)
            if has_all_masked.any():
                # Fix masks with all False by setting at least one position to True
                for i in range(mask.size(0)):
                    if has_all_masked[i]:
                        if mask.size(1) > 0:
                            # Set at least first position to valid (True)
                            mask[i, 0] = True
        
            # Invert mask: False -> positions to attend, True -> positions to mask
            attn_mask = ~mask
        else:
            attn_mask = None
        
        # Apply transformer with masking
        x = self.transformer_encoder(x, src_key_padding_mask=attn_mask)
        
        # Global average pooling over valid positions
        if mask is not None:
            # Use broadcasting to mask out padding
            mask_expanded = mask.unsqueeze(-1).to(x.dtype)  # [batch_size, seq_len, 1]
            x = x * mask_expanded  # Zero out padded positions
            
            # Sum and divide by number of valid positions
            seq_lengths = mask.sum(dim=1, keepdim=True)  # [batch_size, 1]
            seq_lengths = torch.clamp(seq_lengths, min=1)  # Avoid division by zero
            x = x.sum(dim=1) / seq_lengths  # [batch_size, model_dim]
        else:
            # Simple average if no mask
            x = x.mean(dim=1)  # [batch_size, model_dim]
        
        # Classification
        logits = self.classifier(x)  # [batch_size, num_classes]
        return logits

# Positional encoding for sequential data
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Create the dataset and DataLoader with all segments
print("Creating dataset...")
# Create training dataset without non-reading samples
train_dataset = GazeDataset(all_data, exclude_non_reading_for_training=True)
# Create test dataset also without non-reading samples
test_dataset = GazeDataset(all_data, exclude_non_reading_for_training=True)

batch_size = args.batch_size
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_gaze)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_gaze)
print(f"Training dataset size: {len(train_dataset)} samples")
print(f"Testing dataset size: {len(test_dataset)} samples")

# Instantiate the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GazeTransformer(
    input_dim=input_dim,
    model_dim=args.model_dim,
    num_classes=3,  # Keep model output as 3 classes for compatibility
    num_layers=args.num_layers,
    nhead=args.nhead
).to(device)

# Define loss and optimizer with enhanced class weights to handle imbalance
class_counts = np.array([train_dataset.label_counts["reading"], 
                        train_dataset.label_counts["scanning"], 
                        train_dataset.label_counts["non reading"]])
total_samples = sum(class_counts)
num_classes = len(class_counts)

# Calculate class weights based on selected method
if args.weight_method == 'none':
    class_weights = torch.ones(num_classes, dtype=torch.float32)
elif args.weight_method == 'inverse':
    # Inverse frequency weighting (more weight to rare classes)
    class_weights = torch.tensor(1.0 / (class_counts + 1e-8), dtype=torch.float32)
    # Apply weighting factor
    class_weights = torch.pow(class_weights, args.weight_factor)
elif args.weight_method == 'balanced':
    # Balanced weighting (inverse of normalized frequency)
    class_weights = torch.tensor(total_samples / (class_counts * num_classes + 1e-8), dtype=torch.float32)
    # Apply weighting factor
    class_weights = torch.pow(class_weights, args.weight_factor)
elif args.weight_method == 'effective_samples':
    # Effective number of samples (reduces overfitting to minority classes)
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * num_classes
    class_weights = torch.tensor(weights, dtype=torch.float32)
    # Apply weighting factor
    class_weights = torch.pow(class_weights, args.weight_factor)

# Normalize weights to sum to num_classes
class_weights = class_weights * (num_classes / torch.sum(class_weights))
class_weights = class_weights.to(device)

print(f"Class weights ({args.weight_method}):")
for i, (name, weight) in enumerate(zip(["reading", "scanning", "non reading"], class_weights)):
    print(f"  {name}: {weight:.4f}")

# Define focal loss if requested
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

# Choose loss function based on arguments
if args.focal_loss:
    print(f"Using Focal Loss (gamma={args.focal_gamma}) with {args.weight_method} weighting")
    criterion = FocalLoss(weight=class_weights, gamma=args.focal_gamma)
else:
    print(f"Using Cross Entropy Loss with {args.weight_method} weighting")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

# Train the model
num_epochs = args.epochs
print(f"Training model for {num_epochs} epochs on {device}...")
start_time = time.time()

for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_correct = 0
    epoch_total = 0
    epoch_loss = 0.0
    
    for i, (gaze_batch, mask_batch, label_batch, _) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}/{num_epochs}", leave=False)):
        gaze_batch = gaze_batch.to(device)
        mask_batch = mask_batch.to(device)
        label_batch = label_batch.to(device)
        
        # Forward pass with attention mask
        logits = model(gaze_batch, mask_batch)
        loss = criterion(logits, label_batch)
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == label_batch).sum().item()
        epoch_correct += correct
        epoch_total += len(label_batch)
        epoch_loss += loss.item() * len(label_batch)
        
        # Update model
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # Print epoch accuracy and loss
    epoch_accuracy = 100 * epoch_correct / epoch_total
    avg_loss = epoch_loss / epoch_total
    print(f"Epoch {epoch}/{num_epochs} - Accuracy: {epoch_accuracy:.2f}%, Loss: {avg_loss:.4f}")

train_time = time.time() - start_time
print(f"Training completed in {train_time:.2f} seconds")

# Evaluate the model
print("Generating predictions and evaluating model...")
idx_to_label = {0: "reading", 1: "scanning", 2: "non reading"}
active_classes = ["reading", "scanning"]  # Only consider these classes for evaluation
model.eval()

all_results = []
total_correct = 0
total_samples = 0
all_true_labels = []
all_predictions = []
class_correct = {"reading": 0, "scanning": 0}
class_total = {"reading": 0, "scanning": 0}

with torch.no_grad():
    for gaze_batch, mask_batch, label_batch, idx_batch in tqdm(test_dataloader, desc="Evaluating", leave=False):
        gaze_batch = gaze_batch.to(device)
        mask_batch = mask_batch.to(device)
        label_batch = label_batch.to(device)
        
        # Forward pass with attention mask
        outputs = model(gaze_batch, mask_batch)
        predictions = torch.argmax(outputs, dim=1)
        
        # Store labels and predictions for F1 calculation
        all_true_labels.extend(label_batch.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
        
        # Count correct predictions
        correct = (predictions == label_batch).cpu().numpy()
        total_correct += correct.sum()
        total_samples += len(label_batch)
        
        # Track class-wise accuracy
        for i in range(len(label_batch)):
            true_label = idx_to_label[label_batch[i].item()]
            pred_label = idx_to_label[predictions[i].item()]
            if true_label in active_classes:
                class_total[true_label] += 1
                if correct[i]:
                    class_correct[true_label] += 1
        
        # Save individual results
        for i, idx in enumerate(idx_batch):
            file_path = test_dataset.file_names[idx]
            pred_label = idx_to_label[predictions[i].item()]
            true_label = idx_to_label[label_batch[i].item()]
            
            result = {
                "file_path": file_path,
                "prediction": pred_label,
                "actual": true_label,
                "correct": bool(correct[i])
            }
            all_results.append(result)

# Calculate final accuracy
final_accuracy = 100 * total_correct / total_samples
print(f"Overall accuracy: {final_accuracy:.2f}%")

# Calculate class-wise accuracy
print("\nAccuracy by class:")
for label in active_classes:
    if class_total[label] > 0:
        class_accuracy = 100 * class_correct[label] / class_total[label]
        print(f"  {label}: {class_accuracy:.2f}% ({class_correct[label]}/{class_total[label]})")
    else:
        print(f"  {label}: No samples")

# Calculate F1 scores
f1_macro = f1_score(all_true_labels, all_predictions, average='macro') * 100
f1_weighted = f1_score(all_true_labels, all_predictions, average='weighted') * 100
f1_per_class = f1_score(all_true_labels, all_predictions, average=None) * 100

# Calculate precision and recall
precision_macro = precision_score(all_true_labels, all_predictions, average='macro') * 100
recall_macro = recall_score(all_true_labels, all_predictions, average='macro') * 100

print(f"F1 Score (macro): {f1_macro:.2f}%")
print(f"F1 Score (weighted): {f1_weighted:.2f}%")
print(f"Precision (macro): {precision_macro:.2f}%")
print(f"Recall (macro): {recall_macro:.2f}%")

print("F1 Score per class:")
for i, class_name in idx_to_label.items():
    if i < len(f1_per_class):
        print(f"  {class_name}: {f1_per_class[i]:.2f}%")

# Create confusion matrix
confusion = {
    "reading": {"reading": 0, "scanning": 0},
    "scanning": {"reading": 0, "scanning": 0}
}

for result in all_results:
    actual = result["actual"]
    pred = result["prediction"]
    # Only consider reading and scanning classes
    if actual in active_classes and pred in active_classes:
        confusion[actual][pred] += 1

print("\nConfusion Matrix:")
print(f"                   Predicted")
print(f"                 | Reading | Scanning")
print(f"-------------------|---------|----------")
for actual in active_classes:
    r = confusion[actual]["reading"]
    s = confusion[actual]["scanning"]
    total = r + s
    r_percent = r / total * 100 if total > 0 else 0
    s_percent = s / total * 100 if total > 0 else 0
    
    print(f"Actual {actual:12s} | {r:3d} ({r_percent:4.1f}%) | {s:3d} ({s_percent:4.1f}%)")

# Create results summary
config = {
    "use_left_gaze": args.use_left_gaze,
    "use_right_gaze": args.use_right_gaze,
    "use_head_pose": args.use_head_pose,
    "nan_threshold": args.nan_threshold,
    "input_dim": input_dim,
    "model_dim": args.model_dim,
    "num_layers": args.num_layers,
    "nhead": args.nhead,
    "epochs": num_epochs,
    "batch_size": batch_size,
    "learning_rate": args.lr,
    "weight_method": args.weight_method,
    "focal_loss": args.focal_loss,
    "focal_gamma": args.focal_gamma,
    "weight_factor": args.weight_factor,
    "use_weighted_loss": args.use_weighted_loss,
    "add_velocity": args.add_velocity,
    "add_acceleration": args.add_acceleration,
    "normalize_features": args.normalize_features
}

results_summary = {
    "config": config,
    "num_samples": total_samples,
    "class_distribution": {
        "reading": train_dataset.label_counts["reading"],
        "scanning": train_dataset.label_counts["scanning"]
    },
    "accuracy": float(f"{final_accuracy:.2f}"),
    "f1_macro": float(f"{f1_macro:.2f}"),
    "f1_weighted": float(f"{f1_weighted:.2f}"),
    "precision_macro": float(f"{precision_macro:.2f}"),
    "recall_macro": float(f"{recall_macro:.2f}"),
    "f1_per_class": {
        idx_to_label[i]: float(f"{score:.2f}") 
        for i, score in enumerate(f1_per_class) 
        if i < len(f1_per_class) and idx_to_label[i] in active_classes
    },
    "class_accuracy": {
        label: float(f"{100 * class_correct[label] / class_total[label]:.2f}") 
        if class_total[label] > 0 else 0
        for label in active_classes
    },
    "confusion_matrix": confusion,
    "training_time_seconds": train_time,
    "sample_results": all_results
}

# Save to JSON file with configuration in the filename
modalities = []
if args.use_left_gaze: modalities.append("left")
if args.use_right_gaze: modalities.append("right")
if args.use_head_pose: modalities.append("head")

modality_str = "_".join(modalities) if modalities else "default"
output_file = f"gaze_transformer_S1-S6_{modality_str}_results.json"
with open(output_file, "w") as f:
    json.dump(results_summary, f, indent=2)

print(f"Results saved to {output_file}")

# Add a stratified sampler to balance the dataset
class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dataset
        if labels is None:
            # Extract labels from dataset
            self.labels = np.array([dataset.all_samples[i]["normalized_labels"][0] for i in range(len(dataset))])
        
        # Count samples per class
        self.count = Counter(self.labels)
        
        # Get indices of each class
        self.class_indices = {}
        for class_name in set(self.labels):
            self.class_indices[class_name] = np.where(self.labels == class_name)[0]
        
        # Determine number of samples per class
        self.min_class_size = min([len(indices) for indices in self.class_indices.values()])
        
        # Calculate number of batches
        self.n_batches = int(np.ceil(2 * self.min_class_size * len(self.class_indices) / args.batch_size))
        
    def __iter__(self):
        # Create balanced batches
        indices = []
        
        # Sample the same number of examples from each class
        for class_name, class_indices in self.class_indices.items():
            indices.extend(np.random.choice(class_indices, self.min_class_size, replace=False))
        
        # Shuffle indices
        np.random.shuffle(indices)
        
        # Yield batches
        for i in range(0, len(indices), args.batch_size):
            yield indices[i:i + args.batch_size]
    
    def __len__(self):
        return self.n_batches

# Enhancement: Time-based splitting instead of random
def time_based_split(dataset, test_ratio=0.2):
    """Split the dataset based on timestamp to preserve temporal ordering."""
    # Sort samples by timestamp if available, otherwise use frame_num as proxy
    sorted_indices = []
    for idx, sample in enumerate(dataset.all_samples):
        timestamp = sample.get("timestamp", sample["frame_num"])
        subject = sample["file_name"].split('_')[0]  # Extract subject ID
        sorted_indices.append((idx, subject, timestamp))
    
    # Sort by subject and timestamp
    sorted_indices.sort(key=lambda x: (x[1], x[2]))
    
    # Extract just the indices
    indices = [x[0] for x in sorted_indices]
    
    # Split maintaining temporal order within each subject
    subjects = set([x[1] for x in sorted_indices])
    train_indices = []
    test_indices = []
    
    for subject in subjects:
        subject_indices = [idx for idx, subj, _ in sorted_indices if subj == subject]
        split_point = int(len(subject_indices) * (1 - test_ratio))
        train_indices.extend(subject_indices[:split_point])
        test_indices.extend(subject_indices[split_point:])
    
    return train_indices, test_indices

# Enhance GazeTransformer with residual connections for better gradient flow
class EnhancedGazeTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=128, num_classes=3, num_layers=3, nhead=4, 
                 dropout=0.1, attention_dropout=0.2, activation='relu'):
        super(EnhancedGazeTransformer, self).__init__()
        
        self.model_dim = model_dim
        
        # Input projection with layer normalization
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.LayerNorm(model_dim)
        )
        
        # Positional encoding for sequence data
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        
        # Pre-LayerNorm Transformer architecture (more stable)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=model_dim * 2,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,  # Apply normalization first (Pre-LN)
            dropout_rate=dropout,
            attention_dropout_rate=attention_dropout  # Separate attention dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Multi-stage classifier with residual connections
        self.classifier_1 = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier_2 = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.LayerNorm(model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, num_classes)
        )
        
        # Skip connection from input to classifier
        self.skip_connection = nn.Linear(model_dim, model_dim)

    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, input_dim]
        # mask: [batch_size, seq_len] - True for valid positions, False for padding/invalid
        
        # Project input to model dimension
        x = self.input_projection(x)  # [batch_size, seq_len, model_dim]
        
        # Add positional encoding
        encoded = self.pos_encoder(x)
        
        # Create attention mask for transformer
        if mask is not None:
            # Check if any batch has all masked values and fix
            has_all_masked = (~mask).all(dim=1)
            if has_all_masked.any():
                # Fix masks with all False by setting at least one position to True
                for i in range(mask.size(0)):
                    if has_all_masked[i]:
                        if mask.size(1) > 0:
                            # Set at least first position to valid (True)
                            mask[i, 0] = True
            
            # Invert mask: False -> positions to attend, True -> positions to mask
            attn_mask = ~mask
        else:
            attn_mask = None
        
        # Apply transformer with masking
        transformer_output = self.transformer_encoder(encoded, src_key_padding_mask=attn_mask)
        
        # Global average pooling over valid positions
        if mask is not None:
            # Use broadcasting to mask out padding
            mask_expanded = mask.unsqueeze(-1).to(transformer_output.dtype)
            transformer_output = transformer_output * mask_expanded
            
            # Sum and divide by number of valid positions
            seq_lengths = mask.sum(dim=1, keepdim=True)
            seq_lengths = torch.clamp(seq_lengths, min=1)
            pooled = transformer_output.sum(dim=1) / seq_lengths
        else:
            # Simple average if no mask
            pooled = transformer_output.mean(dim=1)
        
        # Skip connection from input to final classifier
        skip = self.skip_connection(x.mean(dim=1))
        
        # First classifier stage
        cls1_output = self.classifier_1(pooled)
        
        # Add skip connection
        combined = cls1_output + skip
        
        # Final classification
        logits = self.classifier_2(combined)
        
        return logits

# Extend with ensemble capabilities
class ModelEnsemble:
    def __init__(self, models):
        self.models = models
    
    def predict(self, gaze_batch, mask_batch, device):
        """Make ensemble prediction with voting."""
        all_predictions = []
        with torch.no_grad():
            for model in self.models:
                model.eval()
                logits = model(gaze_batch.to(device), mask_batch.to(device))
                predictions = torch.softmax(logits, dim=1)
                all_predictions.append(predictions.cpu())
        
        # Average predictions from all models
        avg_predictions = torch.stack(all_predictions).mean(dim=0)
        return torch.argmax(avg_predictions, dim=1)
        
# Function to apply temporal smoothing to predictions
def apply_temporal_smoothing(predictions, sample_ids, window_size=5):
    """Apply temporal smoothing to predictions based on file_name and frame_num."""
    sample_data = {}
    for pred, sample_id in zip(predictions, sample_ids):
        file_name, frame_num = sample_id.rsplit('_', 1)
        frame_num = int(frame_num)
        
        if file_name not in sample_data:
            sample_data[file_name] = []
        
        sample_data[file_name].append((frame_num, pred))
    
    # Sort by frame number
    smoothed_predictions = []
    for file_name, data in sample_data.items():
        data.sort(key=lambda x: x[0])
        preds = [x[1] for x in data]
        
        # Apply smoothing
        smoothed_preds = []
        for i in range(len(preds)):
            window_start = max(0, i - window_size // 2)
            window_end = min(len(preds), i + window_size // 2 + 1)
            window = preds[window_start:window_end]
            # Majority vote in window
            counts = Counter(window)
            smoothed_preds.append(counts.most_common(1)[0][0])
            
        # Add back to results
        smoothed_predictions.extend(smoothed_preds)
    
    return smoothed_predictions

# Update the experiment runner to use K-fold CV and ensemble
def run_advanced_experiment(config):
    print("\n" + "="*50)
    print(f"Running advanced experiment with configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*50 + "\n")
    
    # Set args based on config
    for key, value in config.items():
        setattr(args, key, value)
    
    # Recalculate input_dim based on new configuration
    input_dim = calculate_input_dim(args.add_velocity, args.add_acceleration)
    print(f"Input dimension: {input_dim}")
    
    # Create the dataset
    full_dataset = GazeDataset(all_data, exclude_non_reading_for_training=False,
                               add_velocity=args.add_velocity,
                               add_acceleration=args.add_acceleration,
                               normalize_features=args.normalize_features)
    
    # Use time-based or stratified split
    if args.use_kfold:
        # Prepare for k-fold cross validation
        labels = np.array([full_dataset.all_samples[i]["normalized_labels"][0] for i in range(len(full_dataset))])
        kfold = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
        
        # Track results across folds
        all_fold_results = []
        ensemble_models = []
        
        for fold, (train_indices, test_indices) in enumerate(kfold.split(np.zeros(len(labels)), labels)):
            print(f"\nTraining fold {fold+1}/{args.n_folds}")
            
            # Create train and test subsets
            train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
            test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
            
            # Balance dataset if specified
            if args.balance_dataset:
                train_loader = DataLoader(
                    train_dataset, 
                    batch_sampler=BalancedBatchSampler(train_dataset, labels[train_indices]),
                    collate_fn=collate_gaze
                )
            else:
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=args.batch_size, 
                    shuffle=True, 
                    collate_fn=collate_gaze
                )
                
            test_loader = DataLoader(
                test_dataset, 
                batch_size=args.batch_size, 
                shuffle=False, 
                collate_fn=collate_gaze
            )
            
            # Use enhanced model architecture
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = EnhancedGazeTransformer(
                input_dim=input_dim,
                model_dim=args.model_dim,
                num_classes=3,
                num_layers=args.num_layers,
                nhead=args.nhead,
                attention_dropout=args.attention_dropout
            ).to(device)
            
            # Compute class weights for this fold
            fold_labels = [labels[i] for i in train_indices]
            class_counts = Counter(fold_labels)
            class_weights = compute_class_weights(class_counts, args.weight_method)
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
            
            # Setup loss, optimizer, and scheduler
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
            optimizer = get_optimizer(model, args)
            scheduler = get_lr_scheduler(optimizer, args, train_loader)
            
            # Train the model
            best_model_state = train_model(
                model, train_loader, criterion, optimizer, scheduler, 
                device, args.epochs, f"Fold {fold+1}"
            )
            
            # Load best model for evaluation
            model.load_state_dict(best_model_state)
            
            # Save model for ensemble
            if args.use_ensemble:
                ensemble_models.append(model.cpu())
            
            # Evaluate on this fold's test set
            test_results = evaluate_model(model, test_loader, device)
            all_fold_results.append(test_results)
            
            print(f"Fold {fold+1} results:")
            print(f"  Accuracy: {test_results['accuracy']:.2f}%")
            print(f"  F1 Score: {test_results['f1_macro']:.2f}%")
            
        # Compute average results across folds
        avg_accuracy = sum(r['accuracy'] for r in all_fold_results) / len(all_fold_results)
        avg_f1 = sum(r['f1_macro'] for r in all_fold_results) / len(all_fold_results)
        
        print(f"\nAverage across {args.n_folds} folds:")
        print(f"  Accuracy: {avg_accuracy:.2f}%")
        print(f"  F1 Score: {avg_f1:.2f}%")
        
        # Ensemble prediction if enabled
        if args.use_ensemble:
            print("\nRunning ensemble prediction...")
            ensemble = ModelEnsemble([m.to(device) for m in ensemble_models])
            
            # Create a single test loader for final evaluation
            test_indices = list(range(len(full_dataset)))
            np.random.shuffle(test_indices)
            test_indices = test_indices[:5000]  # Use a subset for efficiency
            test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_gaze)
            
            # Evaluate ensemble
            ensemble_results = evaluate_ensemble(ensemble, test_loader, device)
            
            print(f"Ensemble results:")
            print(f"  Accuracy: {ensemble_results['accuracy']:.2f}%")
            print(f"  F1 Score: {ensemble_results['f1_macro']:.2f}%")
            
            return {
                "accuracy": ensemble_results['accuracy'],
                "f1_macro": ensemble_results['f1_macro'],
                "avg_fold_accuracy": avg_accuracy,
                "avg_fold_f1": avg_f1,
                "individual_folds": [r['accuracy'] for r in all_fold_results],
                "config": config
            }
        
        return {
            "accuracy": avg_accuracy,
            "f1_macro": avg_f1,
            "individual_folds": [r['accuracy'] for r in all_fold_results],
            "config": config
        }
    else:
        # Traditional train/test split
        train_indices, test_indices = time_based_split(full_dataset)
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
        
        # Rest of the standard experiment runner...
        
        # Use standard training and return results
        return run_experiment(config)

# Helper functions for the advanced experiment runner
def compute_class_weights(class_counts, method='inverse'):
    """Compute class weights based on class distribution."""
    total = sum(class_counts.values())
    num_classes = len(class_counts)
    
    if method == 'inverse':
        weights = {cls: total / (count * num_classes) for cls, count in class_counts.items()}
    elif method == 'balanced':
        weights = {cls: 1.0 for cls in class_counts}
    else:
        weights = {cls: 1.0 for cls in class_counts}
    
    # Make sure all classes are covered
    for cls in ['reading', 'scanning', 'non reading']:
        if cls not in weights:
            weights[cls] = 1.0
    
    # Convert to list in proper order
    return [weights['reading'], weights['scanning'], weights['non reading']]

def train_model(model, train_loader, criterion, optimizer, scheduler, device, epochs, prefix=""):
    """Train the model and return best model state."""
    model.train()
    best_accuracy = 0
    best_model_state = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_correct = 0
        epoch_total = 0
        epoch_loss = 0.0
        
        for i, (gaze_batch, mask_batch, label_batch, _) in enumerate(tqdm(train_loader, 
                                                                        desc=f"{prefix} Epoch {epoch}/{epochs}", 
                                                                        leave=False)):
            # Handle NaN values
            if torch.isnan(gaze_batch).any() or torch.isinf(gaze_batch).any():
                gaze_batch = torch.nan_to_num(gaze_batch, nan=0.0, posinf=0.0, neginf=0.0)
                
            gaze_batch = gaze_batch.to(device)
            mask_batch = mask_batch.to(device)
            label_batch = label_batch.to(device)
            
            # Forward pass
            logits = model(gaze_batch, mask_batch)
            
            # Handle numerical issues
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"Warning: NaN or Inf in model output at batch {i}. Skipping batch.")
                continue
                
            loss = criterion(logits, label_batch)
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf loss at batch {i}. Skipping batch.")
                continue
                
            # Compute accuracy
            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == label_batch).sum().item()
            epoch_correct += correct
            epoch_total += len(label_batch)
            epoch_loss += loss.item() * len(label_batch)
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            # Update LR scheduler if it's step-based
            if scheduler is not None and args.scheduler_type == 'cosine':
                scheduler.step()
        
        # Print epoch stats
        if epoch_total > 0:
            epoch_accuracy = 100 * epoch_correct / epoch_total
            avg_loss = epoch_loss / epoch_total
            print(f"{prefix} Epoch {epoch}/{epochs} - Accuracy: {epoch_accuracy:.2f}%, "
                  f"Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Update best model
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        
        # Update LR scheduler if it's epoch-based
        if scheduler is not None and args.scheduler_type in ['step', 'plateau']:
            if args.scheduler_type == 'plateau':
                scheduler.step(epoch_accuracy)
            else:
                scheduler.step()
    
    return best_model_state

def evaluate_model(model, test_loader, device):
    """Evaluate model on test data."""
    model.eval()
    correct = 0
    total = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for gaze_batch, mask_batch, label_batch, _ in tqdm(test_loader, desc="Evaluating", leave=False):
            if torch.isnan(gaze_batch).any() or torch.isinf(gaze_batch).any():
                gaze_batch = torch.nan_to_num(gaze_batch, nan=0.0, posinf=0.0, neginf=0.0)
                
            gaze_batch = gaze_batch.to(device)
            mask_batch = mask_batch.to(device)
            label_batch = label_batch.to(device)
            
            # Forward pass
            logits = model(gaze_batch, mask_batch)
            pred = torch.argmax(logits, dim=1)
            
            # Store results
            correct += (pred == label_batch).sum().item()
            total += len(label_batch)
            predictions.extend(pred.cpu().numpy())
            targets.extend(label_batch.cpu().numpy())
    
    accuracy = 100 * correct / total
    f1 = f1_score(targets, predictions, average='macro') * 100
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1,
        'predictions': predictions,
        'targets': targets
    }

def evaluate_ensemble(ensemble, test_loader, device):
    """Evaluate ensemble model on test data."""
    correct = 0
    total = 0
    predictions = []
    targets = []
    sample_ids = []
    
    for gaze_batch, mask_batch, label_batch, idx_batch in tqdm(test_loader, desc="Evaluating Ensemble", leave=False):
        if torch.isnan(gaze_batch).any() or torch.isinf(gaze_batch).any():
            gaze_batch = torch.nan_to_num(gaze_batch, nan=0.0, posinf=0.0, neginf=0.0)
                
        # Get ensemble predictions
        pred = ensemble.predict(gaze_batch, mask_batch, device)
        
        # Get sample IDs for temporal smoothing
        batch_sample_ids = [full_dataset.file_names[i] for i in idx_batch]
        
        # Store results
        correct += (pred == label_batch).sum().item()
        total += len(label_batch)
        predictions.extend(pred.numpy())
        targets.extend(label_batch.numpy())
        sample_ids.extend(batch_sample_ids)
    
    # Apply temporal smoothing if enabled
    if args.temporal_smoothing:
        print("Applying temporal smoothing...")
        predictions = apply_temporal_smoothing(predictions, sample_ids)
    
    accuracy = 100 * sum(p == t for p, t in zip(predictions, targets)) / total
    f1 = f1_score(targets, predictions, average='macro') * 100
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1,
        'predictions': predictions,
        'targets': targets
    }
