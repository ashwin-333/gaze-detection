import os
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

# --------------------------------------
# 1. Paths & Data Gathering (SAME AS BEFORE)
# --------------------------------------
FT_dir = Path("./S1-31/FT/")
FW_dir = Path("./S1-31/FW/")
data_dirs = [FT_dir, FW_dir]

def load_pkl(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def get_meta_files(data_dir):
    return sorted(list(data_dir.glob("*_meta.pkl")))

meta_files = []
for d in data_dirs:
    meta_files += get_meta_files(d)

print(f"Found {len(meta_files)} total meta .pkl files")

def get_left_gaze(segment):
    return segment.get("left_gaze_screen", None)

def get_right_gaze(segment):
    return segment.get("right_gaze_screen", None)

def get_reading_label(segment):
    if "reading_label" in segment:
        return segment["reading_label"]
    elif "label" in segment:
        return segment["label"]
    else:
        return None

def is_valid_segment(segment, missing_threshold=0.2):
    # We'll just check left gaze for missingness, or you can combine left+right logic
    gaze = get_left_gaze(segment)
    if gaze is None or len(gaze) == 0:
        return False
    gaze = np.array(gaze)
    missing_ratio = np.isnan(gaze).mean()
    return missing_ratio < missing_threshold

# --------------------------------------
# 2. Load All Valid Segments (SAME IDEA)
# --------------------------------------
all_segments = []
skipped_count = 0
for file_path in meta_files:
    try:
        seg = load_pkl(file_path)
        if is_valid_segment(seg):
            label_val = get_reading_label(seg)
            if label_val is None:
                skipped_count += 1
                continue
            if isinstance(label_val, list):
                label_str = label_val[0]
            else:
                label_str = label_val
            label_str = label_str.lower().strip()

            if label_str == "line_changing":
                label_str = "scanning"
            elif label_str == "resting":
                label_str = "non reading"

            if label_str in ["reading", "scanning", "non reading"]:
                seg["file_path"] = str(file_path)
                seg["label_str"] = label_str
                all_segments.append(seg)
            else:
                skipped_count += 1
        else:
            skipped_count += 1
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        skipped_count += 1

print(f"Loaded {len(all_segments)} valid segments")
print(f"Skipped {skipped_count} segments")

# --------------------------------------
# 3. Dataset with NaN Replacement & Mask
# --------------------------------------
class GazeDataset(Dataset):
    """
    This version:
      - Combines left & right gaze into a 4D vector [xL, yL, xR, yR]
      - Replaces NaNs with 0 and builds a mask (True where invalid)
    """
    def __init__(self, segments):
        self.segments = segments
        self.label_map = {"reading": 0, "scanning": 1, "non reading": 2}
        self.file_paths = [seg["file_path"] for seg in segments]

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        seg = self.segments[idx]
        
        # 1) Gaze: combine left and right gaze data
        left_gaze = get_left_gaze(seg)
        right_gaze = get_right_gaze(seg)
        if left_gaze is None:
            left_gaze = []
        if right_gaze is None:
            right_gaze = []
        left_gaze = np.array(left_gaze, dtype=np.float32)   # shape (T,2)
        right_gaze = np.array(right_gaze, dtype=np.float32) # shape (T,2)
        
        # Ensure both have the same length:
        T = max(left_gaze.shape[0], right_gaze.shape[0])
        if left_gaze.shape[0] < T:
            pad_left = np.zeros((T - left_gaze.shape[0], 2), dtype=np.float32)
            left_gaze = np.concatenate((left_gaze, pad_left), axis=0)
        if right_gaze.shape[0] < T:
            pad_right = np.zeros((T - right_gaze.shape[0], 2), dtype=np.float32)
            right_gaze = np.concatenate((right_gaze, pad_right), axis=0)
        # Combined gaze data shape: (T, 4)
        gaze_data = np.concatenate((left_gaze, right_gaze), axis=1)
        
        # 2) Build a nan_mask and replace NaNs with 0
        nan_mask = np.isnan(gaze_data).any(axis=1)  # Boolean mask: True if any channel is NaN
        gaze_data = np.nan_to_num(gaze_data, nan=0.0)
        
        # 3) Process Label
        label_str = seg["label_str"]
        label_idx = self.label_map[label_str]
        
        gaze_tensor = torch.tensor(gaze_data, dtype=torch.float32)
        mask_tensor = torch.tensor(nan_mask, dtype=torch.bool)
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        
        return gaze_tensor, mask_tensor, label_tensor, idx

# --------------------------------------
# 4. Collate Function: Pad Sequences & Build Final Mask
# --------------------------------------
def collate_gaze(batch):
    """
    Pads variable-length sequences and creates a final source key padding mask.
    """
    gaze_list, nan_mask_list, label_list, idx_list = zip(*batch)
    max_length = max(g.shape[0] for g in gaze_list)
    padded_gaze = []
    padded_nan_mask = []
    
    for g, nm in zip(gaze_list, nan_mask_list):
        T = g.shape[0]
        pad_length = max_length - T
        if pad_length > 0:
            pad_gaze = torch.zeros(pad_length, g.shape[1], dtype=g.dtype)
            g_padded = torch.cat([g, pad_gaze], dim=0)
            
            pad_mask = torch.ones(pad_length, dtype=torch.bool)
            nm_padded = torch.cat([nm, pad_mask], dim=0)
        else:
            g_padded = g
            nm_padded = nm
        padded_gaze.append(g_padded)
        padded_nan_mask.append(nm_padded)
    
    batch_gaze = torch.stack(padded_gaze, dim=0)      # Shape: (batch, max_length, 4)
    batch_nan_mask = torch.stack(padded_nan_mask, dim=0)  # Shape: (batch, max_length)
    batch_labels = torch.tensor(label_list, dtype=torch.long)
    
    # Use the padded nan_mask as src_key_padding_mask (True = ignore)
    src_key_padding_mask = batch_nan_mask
    return batch_gaze, src_key_padding_mask, batch_labels, idx_list

# --------------------------------------
# 5. Baseline Transformer Model with Mask Support
# --------------------------------------
class BaselineTransformer(nn.Module):
    def __init__(self, input_dim=4, model_dim=64, num_classes=3, num_layers=2, nhead=4):
        super(BaselineTransformer, self).__init__()
        self.input_linear = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, x, src_key_padding_mask=None):
        # x: (batch_size, T, input_dim)
        x = self.input_linear(x)     # -> [batch, T, model_dim]
        x = x.transpose(0, 1)        # -> [T, batch, model_dim]
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = x.transpose(0, 1)        # -> [batch, T, model_dim]
        x = x.transpose(1, 2)        # -> [batch, model_dim, T]
        x = self.pool(x)             # -> [batch, model_dim, 1]
        x = x.squeeze(-1)            # -> [batch, model_dim]
        logits = self.classifier(x)  # -> [batch, num_classes]
        return logits

# --------------------------------------
# 6. Build Dataset, DataLoader, Train
# --------------------------------------
dataset = GazeDataset(all_segments)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_gaze)
test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_gaze)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BaselineTransformer(input_dim=4, model_dim=64, num_classes=3, num_layers=2, nhead=4).to(device)

# ------------------------
# Incorporate Cross Entropy Loss
# ------------------------
# For multi-class classification, CrossEntropyLoss is a standard choice.
# If data is imbalanced, you might want to compute class weights and pass them to CrossEntropyLoss.
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

num_epochs = 3
for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_correct = 0
    epoch_total = 0

    for gaze_batch, mask_batch, label_batch, idx_list in dataloader:
        gaze_batch = gaze_batch.to(device)   # (batch, T, 4)
        mask_batch = mask_batch.to(device)   # (batch, T)
        label_batch = label_batch.to(device)

        logits = model(gaze_batch, src_key_padding_mask=mask_batch)
        loss = criterion(logits, label_batch)

        preds = logits.argmax(dim=1)
        correct = (preds == label_batch).sum().item()
        epoch_correct += correct
        epoch_total += label_batch.size(0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    epoch_acc = 100.0 * epoch_correct / epoch_total
    print(f"Epoch {epoch} - Accuracy: {epoch_acc:.2f}%")

print("Training complete!")
