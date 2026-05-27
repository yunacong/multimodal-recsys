"""Train Enhanced DIN V2 - 1 epoch quick test"""
import sys, time, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from pathlib import Path

sys.path.insert(0, "src/din_pipeline")
from dataset_enhanced_v2 import EnhancedDINDatasetV2, enhanced_collate_fn_v2
from model_enhanced_v2 import EnhancedDINv2

DIN_DIR = Path("data/processed/din")
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

with open(DIN_DIR / "meta.json") as f:
    meta = json.load(f)

print("加载数据...")
t0 = time.time()
train_ds = EnhancedDINDatasetV2(DIN_DIR / "din_train_enhanced.npz")
val_ds = EnhancedDINDatasetV2(DIN_DIR / "din_val_enhanced.npz")
train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True, collate_fn=enhanced_collate_fn_v2)
val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False, collate_fn=enhanced_collate_fn_v2)
print(f"  train: {len(train_ds):,}, val: {len(val_ds):,}, load {time.time()-t0:.1f}s")

model = EnhancedDINv2(
    n_items=meta["n_items"]+1, n_sub_cats=meta["n_sub_cats"],
    n_brands=meta["n_brands"], n_users=meta["n_users"],
).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Params: {n_params:,}")
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# 1 epoch
print("\nEpoch 1 training...")
model.train()
losses = []
t0 = time.time()
for step, batch in enumerate(train_loader):
    labels = batch["label"].to(device)
    batch_dev = {k: v.to(device) for k, v in batch.items() if k != "label"}
    optimizer.zero_grad()
    logits = model(batch_dev)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    losses.append(loss.item())
    if (step+1) % 500 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (step+1) * (len(train_loader) - step - 1)
        print(f"  Step {step+1}/{len(train_loader)} | loss {np.mean(losses[-500:]):.4f} | {(step+1)*1024/elapsed:.0f} sps | elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m")

print(f"\nTrain done: avg loss {np.mean(losses):.4f}, time {(time.time()-t0)/60:.1f}m")

# Eval
print("\nEvaluating val...")
t0 = time.time()
model.eval()
all_logits = []
all_labels = []
with torch.no_grad():
    for batch in val_loader:
        labels = batch["label"].to(device)
        batch_dev = {k: v.to(device) for k, v in batch.items() if k != "label"}
        logits = model(batch_dev)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
all_logits = np.concatenate(all_logits)
all_labels = np.concatenate(all_labels)
val_auc = roc_auc_score(all_labels, all_logits)
print(f"Val AUC: {val_auc:.4f}, eval time {time.time()-t0:.0f}s")

print(f"\n对比:")
print(f"  LightGBM v3-mpnet:    0.8122")
print(f"  最简 DIN (1 epoch):    0.6827")
print(f"  Enhanced DIN (5 ep):  0.5446 (bug)")
print(f"  Enhanced DIN V2 (1):  {val_auc:.4f}")

torch.save({"model_state": model.state_dict(), "val_auc": val_auc}, "models/din_v2_epoch1.pt")
print("Saved: models/din_v2_epoch1.pt")
