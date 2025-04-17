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

# Setting local paths instead of Google Drive
FT_dir = Path("./S1-31/FT/")
FW_dir = Path("./S1-31/FW/")
data_dirs = [FT_dir, FW_dir]

# In this case we work with meta files ending with *_meta.pkl
def load_pkl(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def get_meta_files(data_dir):
    return sorted(list(data_dir.glob("*_meta.pkl")))

# Collect all meta files from both folders
meta_files = []
for d in data_dirs:
    meta_files += get_meta_files(d)

print(f"Found {len(meta_files)} total meta .pkl files")

# Helper functions to extract gaze data and reading label.
def get_gaze(segment):
    if "left_gaze_screen" in segment:
        return segment["left_gaze_screen"]
    elif "right_gaze_screen" in segment:
        return segment["right_gaze_screen"]
    else:
        return None

def get_reading_label(segment):
    if "reading_label" in segment:
        return segment["reading_label"]
    elif "label" in segment:
        return segment["label"]
    else:
        return None

def is_valid_segment(segment, missing_threshold=0.2):
    gaze = get_gaze(segment)
    if gaze is None:
        return False
    gaze = np.array(gaze)
    if gaze.size == 0:
        return False
    missing_ratio = np.isnan(gaze).mean()
    return missing_ratio < missing_threshold

# Load all valid segments
print("Loading all valid segments...")
start_time = time.time()
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
            
            # Map labels
            if label_str == "line_changing":
                label_str = "scanning"
            elif label_str == "resting":
                label_str = "non reading"
            
            # Only add if it's one of our three categories
            if label_str in ["reading", "scanning", "non reading"]:
                # Store file path in segment for tracking
                seg["file_path"] = str(file_path)
                all_segments.append(seg)
            else:
                skipped_count += 1
        else:
            skipped_count += 1
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        skipped_count += 1

load_time = time.time() - start_time
print(f"Loaded {len(all_segments)} valid segments in {load_time:.2f} seconds")
print(f"Skipped {skipped_count} segments due to invalid data, missing labels, or other issues")

# 2. Custom Dataset for Raw Gaze Data (with Updated Label Mapping)
class GazeDataset(Dataset):
    def __init__(self, segments):
        # Updated label mapping with a third category:
        self.label_map = {"reading": 0, "scanning": 1, "non reading": 2}
        self.segments = segments
        self.file_paths = [seg["file_path"] for seg in segments]  # Keep track of file paths

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        seg = self.segments[idx]
        gaze_data = get_gaze(seg)
        label_val = get_reading_label(seg)
        
        # Process label
        if isinstance(label_val, list):
            label_str = label_val[0]
        else:
            label_str = label_val
        label_str = label_str.lower().strip()
        if label_str == "line_changing":
            label_str = "scanning"
        elif label_str == "resting":
            label_str = "non reading"
        label_idx = self.label_map[label_str]
        
        # Convert data to tensors
        gaze_tensor = torch.tensor(np.array(gaze_data), dtype=torch.float32)  # shape: [T, 2]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        return gaze_tensor, label_tensor, idx  # Return idx to track sample

# 3. Custom Collate Function for Gaze Data
def collate_gaze(batch):
    gaze_list, label_list, idx_list = zip(*batch)
    max_length = max(g.shape[0] for g in gaze_list)
    padded_gaze = []
    for g in gaze_list:
        pad_length = max_length - g.shape[0]
        if pad_length > 0:
            pad = torch.zeros(pad_length, g.shape[1], dtype=g.dtype)
            g_padded = torch.cat([g, pad], dim=0)
        else:
            g_padded = g
        padded_gaze.append(g_padded)
    batch_gaze = torch.stack(padded_gaze)   # shape: [batch, max_length, 2]
    batch_labels = torch.tensor(label_list, dtype=torch.long)
    return batch_gaze, batch_labels, idx_list

# 4. Baseline Transformer Model for Gaze Classification
class BaselineTransformer(nn.Module):
    def __init__(self, input_dim=2, model_dim=64, num_classes=3, num_layers=2, nhead=4):
        super(BaselineTransformer, self).__init__()
        self.input_linear = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        # x: [batch_size, T, input_dim]
        x = self.input_linear(x)               # -> [batch_size, T, model_dim]
        x = x.transpose(0, 1)                  # -> [T, batch_size, model_dim]
        x = self.transformer_encoder(x)        # -> [T, batch_size, model_dim]
        x = x.transpose(0, 1)                  # -> [batch_size, T, model_dim]
        x = x.transpose(1, 2)                  # -> [batch_size, model_dim, T]
        x = self.pool(x)                       # -> [batch_size, model_dim, 1]
        x = x.squeeze(-1)                      # -> [batch_size, model_dim]
        logits = self.classifier(x)            # -> [batch_size, num_classes]
        return logits

# Create the dataset and DataLoader with all segments
print("Creating dataset...")
dataset = GazeDataset(all_segments)
batch_size = 32  # Larger batch size for faster processing
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_gaze)
test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_gaze)
print(f"Dataset size: {len(dataset)} samples")

# Instantiate the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BaselineTransformer(input_dim=2, model_dim=64, num_classes=3, num_layers=2, nhead=4).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Train for 3 epochs
num_epochs = 3
print(f"Training model for {num_epochs} epochs on {device}...")
start_time = time.time()

for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_correct = 0
    epoch_total = 0
    
    for i, (gaze_batch, label_batch, _) in enumerate(dataloader):
        gaze_batch = gaze_batch.to(device)
        label_batch = label_batch.to(device)
        
        logits = model(gaze_batch)
        loss = criterion(logits, label_batch)
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=1)
        correct = (predictions == label_batch).sum().item()
        epoch_correct += correct
        epoch_total += len(label_batch)
        
        # Update model
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
    # Print epoch accuracy
    epoch_accuracy = 100 * epoch_correct / epoch_total
    print(f"Epoch {epoch}/{num_epochs} - Accuracy: {epoch_accuracy:.2f}%")

train_time = time.time() - start_time
print(f"Training completed in {train_time:.2f} seconds")

# Make predictions on the entire dataset and save to JSON
print("Generating predictions for all samples...")
idx_to_label = {0: "reading", 1: "scanning", 2: "non reading"}
model.eval()

all_results = []
total_correct = 0
total_samples = 0
all_true_labels = []
all_predictions = []

with torch.no_grad():
    for gaze_batch, label_batch, idx_batch in test_dataloader:
        gaze_batch = gaze_batch.to(device)
        label_batch = label_batch.to(device)
        
        outputs = model(gaze_batch)
        predictions = torch.argmax(outputs, dim=1)
        
        # Store labels and predictions for F1 calculation
        all_true_labels.extend(label_batch.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
        
        # Count correct predictions
        correct = (predictions == label_batch).cpu().numpy()
        total_correct += correct.sum()
        total_samples += len(label_batch)
        
        # Save individual results
        for i, idx in enumerate(idx_batch):
            file_path = dataset.file_paths[idx]
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
print(f"Overall accuracy on all {total_samples} samples: {final_accuracy:.2f}%")

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
    print(f"  {class_name}: {f1_per_class[i]:.2f}%")

# Create results summary
results_summary = {
    "num_samples": total_samples,
    "accuracy": float(f"{final_accuracy:.2f}"),
    "f1_macro": float(f"{f1_macro:.2f}"),
    "f1_weighted": float(f"{f1_weighted:.2f}"),
    "precision_macro": float(f"{precision_macro:.2f}"),
    "recall_macro": float(f"{recall_macro:.2f}"),
    "f1_per_class": {
        idx_to_label[i]: float(f"{score:.2f}") for i, score in enumerate(f1_per_class)
    },
    "sample_results": all_results
}

# Save to JSON file
output_file = "gaze_classification_results.json"
with open(output_file, "w") as f:
    json.dump(results_summary, f, indent=2)

print(f"Results saved to {output_file}")
