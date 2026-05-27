"""快速测试 Enhanced DIN: 50K 样本 3 epoch"""
import sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score
from pathlib import Path

sys.path.insert(0, "src/din_pipeline")
from dataset_enhanced import EnhancedDINDataset, enhanced_collate_fn
from model_enhanced import EnhancedDIN

print("=" * 70)
print("Enhanced DIN Quick Test (50K samples, 3 epochs)")
print("=" * 70)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

DIN_DIR = Path("data/processed/din")
train_full = EnhancedDINDataset(DIN_DIR / "din_train_enhanced.npz")
val_full = EnhancedDINDataset(DIN_DIR / "din_val_enhanced.npz")

np.random.seed(42)
train_idx = np.random.choice(len(train_full), 50000, replace=False)
val_idx = np.random.choice(len(val_full), 5000, replace=False)
train_ds = Subset(train_full, train_idx.tolist())
val_ds = Subset(val_full, val_idx.tolist())
print(f"train: {len(train_ds)}, val: {len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=enhanced_collate_fn)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=enhanced_collate_fn)

model = EnhancedDIN(n_items=207386, n_sub_cats=32, n_brands=502).to(device)
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
n_params = sum(p.numel() for p in model.parameters())
print(f"Params: {n_params:,}")

for epoch in range(1, 4):
    print(f"\n--- Epoch {epoch} ---")
    
    model.train()
    train_logits = []
    train_labels = []
    losses = []
    t0 = time.time()
    for batch in train_loader:
        labels = batch["label"].to(device)
        batch_dev = {k: v.to(device) for k, v in batch.items() if k != "label"}
        optimizer.zero_grad()
        logits = model(batch_dev)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        train_logits.append(logits.detach().cpu().numpy())
        train_labels.append(labels.cpu().numpy())
    train_logits = np.concatenate(train_logits)
    train_labels = np.concatenate(train_labels)
    train_auc = roc_auc_score(train_labels, train_logits)
    print(f"  Train: loss {np.mean(losses):.4f}, AUC {train_auc:.4f}, time {time.time()-t0:.1f}s")
    
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
    print(f"  Val:   AUC {val_auc:.4f}, gap {train_auc - val_auc:+.4f}")

print("\n关键对比:")
print(f"  最简 DIN (50K mini): Val AUC ~0.51")
print(f"  Enhanced (50K mini): Val AUC {val_auc:.4f}")
print(f"  涨了 {val_auc - 0.51:+.4f}? 如果是, 全量训练 AUC 0.78+ 可期")
