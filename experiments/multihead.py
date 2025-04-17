import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json, time, math
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn.functional as F
import argparse
from scipy.spatial.transform import Rotation as R

#
# 1) CONFIG & ARGPARSE
#
parser = argparse.ArgumentParser(description='Gaze classification with transformer + noise augmentation')
parser.add_argument('--use_left_gaze',      action='store_true')
parser.add_argument('--use_right_gaze',     action='store_true')
parser.add_argument('--use_left_comp_gaze', action='store_true')
parser.add_argument('--use_right_comp_gaze',action='store_true')
parser.add_argument('--use_head_pose',      action='store_true')
parser.add_argument('--auto_select_eye',    action='store_true')
parser.add_argument('--nan_threshold', type=float, default=0.2)
parser.add_argument('--batch_size',    type=int,   default=32)
parser.add_argument('--epochs',        type=int,   default=10)
parser.add_argument('--lr',            type=float, default=1e-4)
parser.add_argument('--model_dim',     type=int,   default=64)
parser.add_argument('--num_layers',    type=int,   default=3)
parser.add_argument('--nhead',         type=int,   default=4)
parser.add_argument('--weight_method', type=str,   default='inverse',
                    choices=['inverse','balanced','effective_samples','none'])
parser.add_argument('--focal_loss',    action='store_true')
parser.add_argument('--focal_gamma',   type=float, default=2.0)
parser.add_argument('--weight_factor', type=float, default=1.0)
parser.add_argument('--input_noise_std', type=float, default=0.0,
                    help='stddev of Gaussian noise added to inputs during training')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

# If no gaze flags, default to all four
if not any([args.use_left_gaze, args.use_right_gaze,
            args.use_left_comp_gaze, args.use_right_comp_gaze,
            args.use_head_pose, args.auto_select_eye]):
    args.use_left_gaze = args.use_right_gaze = True
    args.use_left_comp_gaze = args.use_right_comp_gaze = True

print("Configuration:")
for k,v in vars(args).items():
    print(f"  {k}: {v}")

#
# 2) UTILITIES: loading & NaN handling
#
data_root = Path("/media/nvme1/sheo1/GazeVI-ML")
subjects  = [f"S1-{i:02d}" for i in range(1,7)]
lens_types= ["FT","FW","LT","LW"]

# Load head pose data cache
head_pose_cache = {}

def load_head_pose_data(subject, lens_type):
    """Load head pose data from head.pkl file if not already in cache"""
    key = (subject, lens_type)
    if key in head_pose_cache:
        return head_pose_cache[key]
    
    head_file = data_root / subject / lens_type / "head.pkl"
    if not head_file.exists():
        head_pose_cache[key] = None
        return None
    
    try:
        with open(head_file, "rb") as f:
            head_data = pickle.load(f)
        head_pose_cache[key] = head_data
        return head_data
    except Exception as e:
        print(f"Error loading head data for {subject}/{lens_type}: {e}")
        head_pose_cache[key] = None
        return None

def get_meta_files():
    metas = []
    for subj in subjects:
        for lt in lens_types:
            p = data_root/subj/lt/"meta.pkl"
            if p.exists():
                metas.append(p)
    return metas

def load_pkl(fp):
    with open(fp,"rb") as f:
        return pickle.load(f)

def get_nan_rate(data):
    if data is None: return 1.0
    return np.mean(np.isnan(np.array(data)))

def normalize_label(seg):
    lbl = seg.get("reading_label", seg.get("label", None))
    if lbl is None: return None
    if isinstance(lbl, list):
        lbl = max(set(lbl), key=lbl.count)
    s = lbl.lower().strip()
    if s=="line_changing": s="scanning"
    if s=="resting":      s="non reading"
    return s

def is_valid_segment(seg, thresh=args.nan_threshold):
    # must have at least one gaze source under NaN threshold
    for key in ("left_gaze_screen","right_gaze_screen",
                "left_comp_gaze_screen","right_comp_gaze_screen"):
        if key in seg:
            data = seg[key]
            if data is not None and get_nan_rate(data)<=thresh:
                return True
    return False

def extract_head_pose_for_segment(seg):
    """Extract head pose data for a given segment from head.pkl"""
    # Parse file path to get subject and lens type
    file_path = Path(seg["file_path"])
    subject = file_path.parent.parent.name
    lens_type = file_path.parent.name
    
    # Get segment frame indices
    start_idx = seg.get("start_idx", 0)
    end_idx = seg.get("end_idx", 0)
    
    # Load head pose data
    head_data = load_head_pose_data(subject, lens_type)
    if head_data is None:
        return None
        
    # Extract head pose for this segment's frames
    head_pose_frames = []
    for idx in range(start_idx, end_idx + 1):
        if idx in head_data:
            frame_data = head_data[idx]
            if "head_rot_matrix" in frame_data:
                # Convert rotation matrix to Euler angles (pitch, yaw, roll)
                rot_matrix = np.array(frame_data["head_rot_matrix"])
                try:
                    r = R.from_matrix(rot_matrix)
                    euler = r.as_euler('xyz', degrees=True)  # [pitch, yaw, roll]
                    head_pose_frames.append(euler.tolist())  # Convert to list for consistency
                except Exception:
                    head_pose_frames.append([np.nan, np.nan, np.nan])
            else:
                head_pose_frames.append([np.nan, np.nan, np.nan])
        else:
            head_pose_frames.append([np.nan, np.nan, np.nan])
    
    return head_pose_frames

#
# 3) LOAD & FILTER all_segments
#
print("Loading segments...")
meta_files = get_meta_files()
all_segs = []
for mf in meta_files:
    data = load_pkl(mf)
    for _,seg in data.items():
        if not is_valid_segment(seg): continue
        lbl = normalize_label(seg)
        if lbl not in ("reading","scanning","non reading"): continue
        seg["file_path"] = str(mf)
        seg["_label_str"] = lbl
        
        # Add head pose data if requested
        if args.use_head_pose:
            head_pose = extract_head_pose_for_segment(seg)
            if head_pose is not None:
                seg["head_pose_euler"] = head_pose
                
        all_segs.append(seg)

print(f"  total valid segments: {len(all_segs)}")
if args.use_head_pose:
    with_head = sum(1 for s in all_segs if "head_pose_euler" in s)
    print(f"  segments with head pose data: {with_head} ({100*with_head/len(all_segs):.1f}%)")

#
# 4) SPLIT reading & scanning only
#
# we drop "non reading" entirely here
rs_segs = [s for s in all_segs if s["_label_str"] in ("reading","scanning")]
rs_labels = [0 if s["_label_str"]=="reading" else 1 for s in rs_segs]

train_segs, tmp_segs, y_train, y_tmp = train_test_split(
    rs_segs, rs_labels, test_size=0.30, stratify=rs_labels,
    random_state=args.seed
)
val_segs, test_segs, y_val, y_test = train_test_split(
    tmp_segs, y_tmp, test_size=0.50, stratify=y_tmp,
    random_state=args.seed
)

print(f"Split sizes → train: {len(train_segs)}, val: {len(val_segs)}, test: {len(test_segs)}")

#
# 5) DATASET & DATALOADER
#
def select_best_gaze_data(seg):
    sources = {
      "left":   seg.get("left_gaze_screen"),
      "right":  seg.get("right_gaze_screen"),
      "lcomp":  seg.get("left_comp_gaze_screen"),
      "rcomp":  seg.get("right_comp_gaze_screen")
    }
    rates = {k:get_nan_rate(v) for k,v in sources.items() if v is not None}
    if not rates: return None
    best = min(rates, key=rates.get)
    return sources[best]

class GazeDataset(Dataset):
    def __init__(self, segments, augment=False):
        self.segs    = segments
        self.augment = augment
        self.label_map = {"reading":0,"scanning":1}
    def __len__(self): return len(self.segs)
    def __getitem__(self, i):
        seg = self.segs[i]
        # gather features
        feats, masks = [], []
        def add(src):
            arr = np.array(src)
            # Handle different dimensions for head pose data
            if len(arr.shape) == 1:  # Single vector
                arr = arr.reshape(1, -1)
            elif len(arr.shape) == 2 and arr.shape[1] == 3 and "head_pose" in str(src):
                # This is head pose data with correct shape
                pass
            # Create mask for NaN values
            if len(arr.shape) == 1:
                masks.append(~np.isnan(arr))
            else:
                masks.append(~np.isnan(arr).any(axis=1))
            feats.append(arr)
            
        if args.use_left_gaze    and "left_gaze_screen" in seg: add(seg["left_gaze_screen"])
        if args.use_right_gaze   and "right_gaze_screen" in seg: add(seg["right_gaze_screen"])
        if args.use_left_comp_gaze and "left_comp_gaze_screen" in seg: add(seg["left_comp_gaze_screen"])
        if args.use_right_comp_gaze and "right_comp_gaze_screen" in seg:add(seg["right_comp_gaze_screen"])
        if args.auto_select_eye  and not feats:
            best = select_best_gaze_data(seg)
            if best is not None: add(best)
        # head pose
        if args.use_head_pose:
            if "head_pose_euler" in seg:
                add(seg["head_pose_euler"])
            elif "head_rot_euler" in seg:
                add(seg["head_rot_euler"])
            elif "head_rot_matrix" in seg:
                add(seg["head_rot_matrix"])
        
        # If no features available, return a dummy sample
        if not feats:
            # Create dummy data (3D vector of zeros)
            x = torch.zeros(1, 3)
            m = torch.ones(1, dtype=torch.bool)  # All valid
            y = torch.tensor(self.label_map[seg["_label_str"]], dtype=torch.long)
            return x, m, y
            
        # combine
        if len(feats)>1:
            L = min(f.shape[0] for f in feats)
            feats = [f[:L] for f in feats]
            masks = [m[:L] for m in masks]
            X = np.concatenate(feats,axis=1)
            M = np.logical_and.reduce(masks)
        else:
            X, M = feats[0], masks[0]
        X = np.nan_to_num(X,0.0)
        # noise aug
        if self.augment and args.input_noise_std>0:
            X = X + np.random.randn(*X.shape)*args.input_noise_std
        # to tensor
        x = torch.from_numpy(X).float()
        m = torch.from_numpy(M).bool()
        y = torch.tensor(self.label_map[seg["_label_str"]],dtype=torch.long)
        return x,m,y

def collate(batch):
    Xs, Ms, Ys = zip(*batch)
    B = len(Xs)
    L = max(x.shape[0] for x in Xs)
    D = Xs[0].shape[1]
    Xp = torch.zeros(B,L,D)
    Mp = torch.zeros(B,L, dtype=torch.bool)
    for i,(x,m,y) in enumerate(batch):
        l = x.shape[0]
        Xp[i,:l] = x
        Mp[i,:l] = m
    return Xp, Mp, torch.stack(Ys)

train_ds = GazeDataset(train_segs, augment=True)
val_ds   = GazeDataset(val_segs,   augment=False)
test_ds  = GazeDataset(test_segs,  augment=False)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,collate_fn=collate)
test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,collate_fn=collate)

#
# 6) MODEL
#
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout=0.1,max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len,d_model)
        pos = torch.arange(0,max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))
        self.drop = nn.Dropout(dropout)
    def forward(self,x):
        x = x + self.pe[:,:x.size(1)].to(x.device)
        return self.drop(x)

class GazeTransformer(nn.Module):
    def __init__(self,input_dim,model_dim,heads,layers,drop=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim,model_dim)
        self.pos  = PositionalEncoding(model_dim,drop)
        enc = nn.TransformerEncoderLayer(model_dim,nhead=heads,dim_feedforward=4*model_dim,
                                         dropout=drop,batch_first=True)
        self.enc = nn.TransformerEncoder(enc,num_layers=layers)
        self.cls = nn.Sequential(
            nn.Linear(model_dim,model_dim),
            nn.ReLU(),nn.Dropout(drop),
            nn.Linear(model_dim,2)
        )
    def forward(self,x,mask):
        x = self.proj(x)
        x = self.pos(x)
        # mask: True where valid → transformer wants True = mask out → invert
        key_mask = ~mask
        x = self.enc(x, src_key_padding_mask=key_mask)
        # pool
        m = mask.unsqueeze(-1).float()
        x = (x*m).sum(1)/(m.sum(1).clamp(min=1))
        return self.cls(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# infer input_dim from one batch
_sample_x,_,_ = next(iter(train_loader))
input_dim = _sample_x.shape[2]

model = GazeTransformer(
    input_dim, args.model_dim,
    args.nhead, args.num_layers,
    drop=0.1
).to(device)

#
# 7) LOSS, OPTIM, SCHEDULER
#
# class weights for two classes
counts = np.bincount(y_train)
total = counts.sum()
if args.weight_method=='none':
    cw = torch.ones(2, dtype=torch.float32)
elif args.weight_method=='inverse':
    cw = torch.tensor(1.0/(counts+1e-8), dtype=torch.float32)**args.weight_factor
elif args.weight_method=='balanced':
    cw = torch.tensor(total/(counts*2+1e-8), dtype=torch.float32)**args.weight_factor
else:  # effective_samples
    beta=0.9999
    en = 1.-beta**counts
    w = (1-beta)/en
    cw = torch.tensor(w/ w.sum()*2, dtype=torch.float32)**args.weight_factor

cw = (cw*2/cw.sum()).to(device)

if args.focal_loss:
    class FocalLoss(nn.Module):
        def __init__(self,weight,gamma=2):
            super().__init__()
            self.weight=weight; self.gamma=gamma
        def forward(self,x,y):
            ce = F.cross_entropy(x,y,weight=self.weight,reduction='none')
            pt = torch.exp(-ce)
            fl = ((1-pt)**self.gamma*ce).mean()
            return fl
    criterion = FocalLoss(cw,args.focal_gamma)
else:
    criterion = nn.CrossEntropyLoss(weight=cw)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

#
# 8) TRAIN / VAL
#
best_val_f1 = 0.0
best_path   = "best_model.pt"

for epoch in range(1, args.epochs+1):
    model.train()
    tbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
    total_loss, correct, total = 0,0,0

    for X,M,Y in tbar:
        X,M,Y = X.to(device), M.to(device), Y.to(device)
        logits = model(X,M)
        loss   = criterion(logits, Y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
    
        preds = logits.argmax(1)
        correct += (preds==Y).sum().item()
        total   += Y.size(0)
        total_loss += loss.item()*Y.size(0)
        tbar.set_postfix(loss=total_loss/total, acc=100*correct/total)

    scheduler.step()

    # VALIDATION
    model.eval()
    vs, vp, vt = [], [], []
    with torch.no_grad():
        for X,M,Y in tqdm(val_loader, desc="  Val   "):
            X,M,Y = X.to(device), M.to(device), Y.to(device)
            logits = model(X,M)
            preds  = logits.argmax(1)
            vs.extend(Y.cpu().tolist())
            vp.extend(preds.cpu().tolist())
            vt.append( (preds==Y).sum().item() / Y.size(0) )

    val_f1 = f1_score(vs, vp, average='macro')
    val_acc= np.mean(vt)
    print(f" → Val F1: {100*val_f1:.2f}%, Val Acc: {100*val_acc:.2f}%")

    if val_f1>best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), best_path)

#
# 9) FINAL TEST EVAL
#
model.load_state_dict(torch.load(best_path))
model.eval()

ys, ps = [], []
with torch.no_grad():
    for X,M,Y in tqdm(test_loader, desc="Test    "):
        X,M,Y = X.to(device), M.to(device), Y.to(device)
        out = model(X,M)
        preds = out.argmax(1)
        ys.extend(Y.cpu().tolist())
        ps.extend(preds.cpu().tolist())

# overall metrics
print("\n=== TEST RESULTS ===")
print(f"Overall Accuracy: {100*(np.array(ys)==np.array(ps)).mean():.2f}%")
print("Per-class Accuracy:")
cm = confusion_matrix(ys,ps,labels=[0,1])
for i,cls in enumerate(("reading","scanning")):
    acc = cm[i,i]/cm[i].sum()*100 if cm[i].sum()>0 else 0.0
    print(f"  {cls:8s}: {acc:5.2f}% ({cm[i,i]}/{cm[i].sum()})")

print("\nF1 / Precision / Recall (macro):")
print(f"  F1:        {100*f1_score(ys,ps,average='macro'):.2f}%")
print(f"  Precision: {100*precision_score(ys,ps,average='macro'):.2f}%")
print(f"  Recall:    {100*recall_score(ys,ps,average='macro'):.2f}%")

print("\nConfusion Matrix (rows=true, cols=pred):")
print("         Pred→  reading  scanning")
print(f"True reading   {cm[0,0]:8d} {cm[0,1]:10d}")
print(f"     scanning  {cm[1,0]:8d} {cm[1,1]:10d}") 