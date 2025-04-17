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

# In this case we work with meta files ending with *_meta.pkl
def load_pkl(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def get_meta_files(data_dir):
    return sorted(list(data_dir.glob("*_meta.pkl")))

# Get meta files separately from FT and FW folders
ft_meta_files = get_meta_files(FT_dir)
fw_meta_files = get_meta_files(FW_dir)

print(f"Found {len(ft_meta_files)} files in FT directory")
print(f"Found {len(fw_meta_files)} files in FW directory")

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

# Function to load and process segments
def load_segments(file_paths):
    segments = []
    skipped_count = 0
    
    for file_path in file_paths:
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
                    segments.append(seg)
                else:
                    skipped_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            skipped_count += 1
    
    return segments, skipped_count

# Load training data (FT) and testing data (FW)
print("Loading training data (FT)...")
start_time = time.time()
train_segments, train_skipped = load_segments(ft_meta_files)
train_load_time = time.time() - start_time
print(f"Loaded {len(train_segments)} valid training segments in {train_load_time:.2f} seconds")
print(f"Skipped {train_skipped} training segments")

print("\nLoading testing data (FW)...")
start_time = time.time()
test_segments, test_skipped = load_segments(fw_meta_files)
test_load_time = time.time() - start_time
print(f"Loaded {len(test_segments)} valid testing segments in {test_load_time:.2f} seconds")
print(f"Skipped {test_skipped} testing segments")

# Count labels in both datasets
def count_labels(segments):
    label_counts = {"reading": 0, "scanning": 0, "non reading": 0}
    for seg in segments:
        label_val = get_reading_label(seg)
        if isinstance(label_val, list):
            label_str = label_val[0]
        else:
            label_str = label_val
        label_str = label_str.lower().strip()
        if label_str == "line_changing":
            label_str = "scanning"
        elif label_str == "resting":
            label_str = "non reading"
        
        if label_str in label_counts:
            label_counts[label_str] += 1
    return label_counts

train_label_counts = count_labels(train_segments)
test_label_counts = count_labels(test_segments)

print("\nTraining set label distribution:")
for label, count in train_label_counts.items():
    print(f"  {label}: {count} samples ({count/len(train_segments)*100:.1f}%)")

print("\nTesting set label distribution:")
for label, count in test_label_counts.items():
    print(f"  {label}: {count} samples ({count/len(test_segments)*100:.1f}%)")

# 2. Custom Dataset for Raw Gaze Data
class GazeDataset(Dataset):
    def __init__(self, segments):
        self.label_map = {"reading": 0, "scanning": 1, "non reading": 2}
        self.segments = segments
        self.file_paths = [seg["file_path"] for seg in segments]

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
        return gaze_tensor, label_tensor, idx

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

# Create datasets and DataLoaders
print("\nCreating datasets...")
train_dataset = GazeDataset(train_segments)
test_dataset = GazeDataset(test_segments)

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_gaze)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_gaze)

print(f"Training dataset: {len(train_dataset)} samples")
print(f"Testing dataset: {len(test_dataset)} samples")

# Instantiate the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BaselineTransformer(input_dim=2, model_dim=64, num_classes=3, num_layers=2, nhead=4).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Train for 3 epochs on FT data
num_epochs = 3
print(f"\nTraining model on FT data for {num_epochs} epochs using {device}...")
start_time = time.time()

for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_correct = 0
    epoch_total = 0
    
    for i, (gaze_batch, label_batch, _) in enumerate(train_dataloader):
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
    
    # Print epoch accuracy on training data
    epoch_accuracy = 100 * epoch_correct / epoch_total
    print(f"Epoch {epoch}/{num_epochs} - Training Accuracy: {epoch_accuracy:.2f}%")

train_time = time.time() - start_time
print(f"Training completed in {train_time:.2f} seconds")

# Evaluate on FW test data
print("\nEvaluating model on FW data...")
idx_to_label = {0: "reading", 1: "scanning", 2: "non reading"}
model.eval()

all_results = []
total_correct = 0
total_samples = 0
class_correct = {"reading": 0, "scanning": 0, "non reading": 0}
class_total = {"reading": 0, "scanning": 0, "non reading": 0}
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
        
        # Track class-wise accuracy
        for i in range(len(label_batch)):
            true_label = idx_to_label[label_batch[i].item()]
            pred_correct = correct[i]
            class_total[true_label] += 1
            if pred_correct:
                class_correct[true_label] += 1
        
        # Save individual results
        for i, idx in enumerate(idx_batch):
            file_path = test_dataset.file_paths[idx]
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
print(f"Overall accuracy on FW test data: {final_accuracy:.2f}%")

# Calculate F1 scores
f1_macro = f1_score(all_true_labels, all_predictions, average='macro', zero_division=0) * 100
f1_weighted = f1_score(all_true_labels, all_predictions, average='weighted', zero_division=0) * 100
f1_per_class = f1_score(all_true_labels, all_predictions, average=None, zero_division=0) * 100

# Calculate precision and recall
precision_macro = precision_score(all_true_labels, all_predictions, average='macro', zero_division=0) * 100
recall_macro = recall_score(all_true_labels, all_predictions, average='macro', zero_division=0) * 100

print(f"F1 Score (macro): {f1_macro:.2f}%")
print(f"F1 Score (weighted): {f1_weighted:.2f}%")
print(f"Precision (macro): {precision_macro:.2f}%")
print(f"Recall (macro): {recall_macro:.2f}%")

print("F1 Score per class:")
for i, class_name in idx_to_label.items():
    if i < len(f1_per_class):
        print(f"  {class_name}: {f1_per_class[i]:.2f}%")
    else:
        print(f"  {class_name}: N/A (no samples)")

# Calculate per-class accuracy
print("\nAccuracy by class:")
for label in ["reading", "scanning", "non reading"]:
    if class_total[label] > 0:
        class_accuracy = 100 * class_correct[label] / class_total[label]
        print(f"  {label}: {class_accuracy:.2f}% ({class_correct[label]}/{class_total[label]})")
    else:
        print(f"  {label}: No samples")

# Create confusion matrix
confusion = {
    "reading": {"reading": 0, "scanning": 0, "non reading": 0},
    "scanning": {"reading": 0, "scanning": 0, "non reading": 0},
    "non reading": {"reading": 0, "scanning": 0, "non reading": 0}
}

for result in all_results:
    actual = result["actual"]
    pred = result["prediction"]
    confusion[actual][pred] += 1

print("\nConfusion Matrix:")
print(f"                   Predicted")
print(f"                 | Reading | Scanning | Non-reading")
print(f"-------------------|---------|----------|------------")
for actual in ["reading", "scanning", "non reading"]:
    r = confusion[actual]["reading"]
    s = confusion[actual]["scanning"]
    n = confusion[actual]["non reading"]
    total = r + s + n
    r_percent = r / total * 100 if total > 0 else 0
    s_percent = s / total * 100 if total > 0 else 0
    n_percent = n / total * 100 if total > 0 else 0
    
    print(f"Actual {actual:12s} | {r:3d} ({r_percent:4.1f}%) | {s:3d} ({s_percent:4.1f}%) | {n:3d} ({n_percent:4.1f}%)")

# Create results summary
results_summary = {
    "train_samples": len(train_dataset),
    "test_samples": len(test_dataset),
    "accuracy": float(f"{final_accuracy:.2f}"),
    "f1_macro": float(f"{f1_macro:.2f}"),
    "f1_weighted": float(f"{f1_weighted:.2f}"),
    "precision_macro": float(f"{precision_macro:.2f}"),
    "recall_macro": float(f"{recall_macro:.2f}"),
    "f1_per_class": {
        idx_to_label[i]: float(f"{score:.2f}") if i < len(f1_per_class) else 0
        for i, score in enumerate(f1_per_class)
    },
    "class_accuracy": {
        label: float(f"{100 * class_correct[label] / class_total[label]:.2f}") 
        if class_total[label] > 0 else 0
        for label in ["reading", "scanning", "non reading"]
    },
    "confusion_matrix": confusion,
    "sample_results": all_results
}

# Save to JSON file
output_file = "train_ft_test_fw_results.json"
with open(output_file, "w") as f:
    json.dump(results_summary, f, indent=2)

print(f"\nResults saved to {output_file}") 