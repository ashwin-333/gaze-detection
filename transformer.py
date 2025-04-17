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

# Setting paths to the new data location
base_dir = Path("/home/sheo1/nvme1/GazeVI-ML")
# Define subjects to use - updated to include all S1 through S6
subjects = ["S1-01", "S1-02", "S1-03", "S1-04", "S1-05", "S1-06"]
# Define lens types if needed
lens_types = ["FT"]  # You can add "FW" if needed

# Configure input modalities and parameters
parser = argparse.ArgumentParser(description='Gaze classification with transformer model')
parser.add_argument('--use_left_gaze', action='store_true', help='Use left gaze data')
parser.add_argument('--use_right_gaze', action='store_true', help='Use right gaze data')
parser.add_argument('--use_head_pose', action='store_true', help='Use head pose data')
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

# Default to using both left and right gaze if no specific option is selected
args = parser.parse_args()
if not (args.use_left_gaze or args.use_right_gaze or args.use_head_pose):
    # Default to using both left and right gaze as requested
    args.use_left_gaze = True
    args.use_right_gaze = True

print(f"Configuration:")
print(f"  Use left gaze: {args.use_left_gaze}")
print(f"  Use right gaze: {args.use_right_gaze}")
print(f"  Use head pose: {args.use_head_pose}")
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
def calculate_input_dim():
    """Calculate the input dimension based on selected data sources."""
    dim = 0
    if args.use_left_gaze:
        dim += 2  # x, y coordinates
    if args.use_right_gaze:
        dim += 2  # x, y coordinates
    if args.use_head_pose:
        dim += 3  # euler angles (x, y, z)
    
    # Ensure at least 2D input
    return max(dim, 2)

input_dim = calculate_input_dim()
print(f"Input dimension: {input_dim}")

# Enhanced GazeDataset to handle the new data format
class GazeDataset(Dataset):
    def __init__(self, data_dict, exclude_non_reading_for_training=False):
        # Updated label mapping
        self.label_map = {"reading": 0, "scanning": 1, "non reading": 2}
        
        # Process all data into a usable format
        self.all_samples = []
        self.file_names = []
        
        print("Processing dataset...")
        processed_count = 0
        skipped_count = 0
        
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
                    sample["left_gaze"] = np.array(frame_data["left_gaze_screen"])
                    if "left_gaze_mask" in frame_data:
                        sample["left_mask"] = np.array(frame_data["left_gaze_mask"])
                    else:
                        # If no mask provided, create one (non-zero values are valid)
                        sample["left_mask"] = ~np.all(sample["left_gaze"] == 0, axis=1)
                
                # Only use regular gaze_screen data, not comp_gaze
                if args.use_right_gaze:
                    sample["right_gaze"] = np.array(frame_data["right_gaze_screen"])
                    if "right_gaze_mask" in frame_data:
                        sample["right_mask"] = np.array(frame_data["right_gaze_mask"])
                    else:
                        # If no mask provided, create one (non-zero values are valid)
                        sample["right_mask"] = ~np.all(sample["right_gaze"] == 0, axis=1)
                
                # Process head pose
                if args.use_head_pose:
                    sample["head_pose"] = np.array(frame_data["head_rot_euler"])
                
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

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        sample = self.all_samples[idx]
        features = []
        attention_mask = []
        
        # Process left gaze
        if args.use_left_gaze and "left_gaze" in sample:
            left_gaze = sample["left_gaze"]
            features.append(left_gaze)
            mask = sample["left_mask"]
            attention_mask.append(mask)
        
        # Process right gaze
        if args.use_right_gaze and "right_gaze" in sample:
            right_gaze = sample["right_gaze"]
            features.append(right_gaze)
            mask = sample["right_mask"]
            attention_mask.append(mask)
        
        # Process head pose - repeat to match sequence length
        if args.use_head_pose and "head_pose" in sample:
            head_pose = sample["head_pose"]
            # If head pose is a single vector, repeat it to match sequence length
            if len(head_pose.shape) == 1:
                seq_len = features[0].shape[0] if features else 1
                head_pose = np.tile(head_pose, (seq_len, 1))
            features.append(head_pose)
            # Assume head pose is valid where gaze is valid
            if attention_mask:
                mask = attention_mask[0]  # Use same mask as first feature
            else:
                mask = np.ones(head_pose.shape[0], dtype=bool)
            attention_mask.append(mask)
        
        # Combine features
        if features:
            # Align sequence length (use shortest sequence)
            min_length = min(f.shape[0] for f in features)
            features = [f[:min_length] for f in features]
            attention_mask = [m[:min_length] for m in attention_mask]
            
            # Concatenate features along feature dimension
            combined_features = np.concatenate(features, axis=1)
            # Combine masks
            combined_mask = np.all(attention_mask, axis=0)
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

# Improved Transformer with masking and positional encoding
class GazeTransformer(nn.Module):
    def __init__(self, input_dim, model_dim=64, num_classes=3, num_layers=2, nhead=4, dropout=0.1):
        super(GazeTransformer, self).__init__()
        
        self.model_dim = model_dim
        self.input_projection = nn.Linear(input_dim, model_dim)
        
        # Positional encoding for sequence data
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        
        # Transformer layers with self-attention
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True  # Use batch_first for simplicity
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim),
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
        # In PyTorch transformer, mask is True for positions to MASK OUT
        if mask is not None:
            # Invert mask: False -> positions to attend, True -> positions to mask
            attn_mask = ~mask
            
            # For src_key_padding_mask, we need a 2D [batch_size, seq_len] mask
            # where True values are positions to be masked out
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
    "use_weighted_loss": args.use_weighted_loss
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

# Let's run the experiment with different configurations
# First with left gaze only
def run_experiment(config):
    print("\n" + "="*50)
    print(f"Running experiment with configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*50 + "\n")
    
    # Set args based on config
    for key, value in config.items():
        setattr(args, key, value)
    
    # Recalculate input_dim based on new configuration
    input_dim = calculate_input_dim()
    print(f"Input dimension: {input_dim}")
    
    # Create the dataset and DataLoader with the config
    train_dataset = GazeDataset(all_data, exclude_non_reading_for_training=True)
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
    
    # Define loss and optimizer with class weights to handle imbalance
    class_counts = np.array([train_dataset.label_counts["reading"], 
                          train_dataset.label_counts["scanning"], 
                          train_dataset.label_counts["non reading"]])
    total_samples = sum(class_counts)
    num_classes = len(class_counts)
    
    # Calculate class weights based on selected method
    if args.weight_method == 'none' or not args.use_weighted_loss:
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
    
    # Choose loss function
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
        "training_time_seconds": train_time
    }
    
    return results_summary

# Run experiments with different configurations
experiment_configs = [
    # Experiment 1: Using both left and right gaze
    {
        "use_left_gaze": True,
        "use_right_gaze": True,
        "use_head_pose": False,
        "use_weighted_loss": True,
        "weight_method": "inverse",
        "epochs": args.epochs
    }
]

# Run all experiments and collect results
all_experiment_results = {}
for i, config in enumerate(tqdm(experiment_configs, desc="Running experiments")):
    config_name = f"exp_{i+1}"
    if config["use_head_pose"]:
        config_name += "_with_head"
    else:
        config_name += "_gaze_only"
    
    print(f"\nRunning experiment {i+1}/{len(experiment_configs)}: {config_name}")
    all_experiment_results[config_name] = run_experiment(config)

# Save all results to a single JSON file
output_file = f"gaze_transformer_S1-S6_results.json"
with open(output_file, "w") as f:
    json.dump(all_experiment_results, f, indent=2)

print(f"All experiment results saved to {output_file}")
