"""DeepFM 训练 - 全量 + 早停"""
import sys, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from model import DeepFM

DEEPFM_DIR = Path("data/processed/deepfm")
OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 4096
NUM_EPOCHS = 5
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 2

print("=" * 70)
print("DeepFM 训练")
print("=" * 70)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")
print(f"Config: batch={BATCH_SIZE}, lr={LR}, epochs={NUM_EPOCHS}")

# 加载数据
print("\n加载数据...")
t0 = time.time()
train_data = np.load(DEEPFM_DIR / "train.npz")
val_data = np.load(DEEPFM_DIR / "val.npz")

train_dense = torch.from_numpy(train_data["dense"]).float()
train_cat = torch.from_numpy(train_data["cat"]).long()
train_label = torch.from_numpy(train_data["label"]).float()

val_dense = torch.from_numpy(val_data["dense"]).float()
val_cat = torch.from_numpy(val_data["cat"]).long()
val_label = torch.from_numpy(val_data["label"]).float()

train_ds = TensorDataset(train_dense, train_cat, train_label)
val_ds = TensorDataset(val_dense, val_cat, val_label)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)

print(f"  train: {len(train_ds):,}, batches: {len(train_loader):,}")
print(f"  val:   {len(val_ds):,}, batches: {len(val_loader):,}")
print(f"  加载耗时: {time.time()-t0:.1f}s")

# 模型
with open(DEEPFM_DIR / "meta.json") as f:
    meta = json.load(f)

model = DeepFM(
    n_dense=meta["n_dense"],
    cat_vocab_sizes=[meta["n_sub_cats"], meta["n_brands"], meta["n_text_clusters"]],
).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"\n模型参数: {n_params:,}")

optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


def evaluate():
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for dense, cat, label in val_loader:
            dense, cat = dense.to(device), cat.to(device)
            logits = model(dense, cat)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(label.numpy())
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    return roc_auc_score(all_labels, all_logits)


# 训练
history = []
best_val_auc = 0
patience_counter = 0

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\n{'='*70}")
    print(f"Epoch {epoch}/{NUM_EPOCHS}")
    print(f"{'='*70}")
    
    model.train()
    losses = []
    train_logits_sample, train_labels_sample = [], []
    t0 = time.time()
    
    for step, (dense, cat, label) in enumerate(train_loader):
        dense, cat, label = dense.to(device), cat.to(device), label.to(device)
        optimizer.zero_grad()
        logits = model(dense, cat)
        loss = F.binary_cross_entropy_with_logits(logits, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())
        
        if step % 50 == 0:
            train_logits_sample.append(logits.detach().cpu().numpy())
            train_labels_sample.append(label.cpu().numpy())
        
        if (step + 1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (step+1) * (len(train_loader) - step - 1)
            sps = (step+1) * BATCH_SIZE / elapsed
            print(f"  Step {step+1}/{len(train_loader)} | loss {np.mean(losses[-200:]):.4f} | {sps:.0f} sps | elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m")
    
    tl = np.concatenate(train_logits_sample)
    tlab = np.concatenate(train_labels_sample)
    train_auc = roc_auc_score(tlab, tl)
    print(f"\n  Train: loss {np.mean(losses):.4f}, AUC {train_auc:.4f}, time {(time.time()-t0)/60:.1f}m")
    
    val_auc = evaluate()
    print(f"  Val: AUC {val_auc:.4f}, gap {train_auc - val_auc:+.4f}")
    
    history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                    "train_auc": float(train_auc), "val_auc": float(val_auc)})
    
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_counter = 0
        torch.save({"model_state": model.state_dict(), "val_auc": val_auc, "epoch": epoch},
                   OUT_DIR / "deepfm_best.pt")
        print(f"  ✅ 新最佳! 保存模型")
    else:
        patience_counter += 1
        print(f"  ⚠️ 没提升, patience {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print("\n🛑 早停")
            break
    
    with open(OUT_DIR / "deepfm_history.json", "w") as f:
        json.dump(history, f, indent=2)

print(f"\n{'='*70}")
print(f"🎉 训练完成!")
print(f"{'='*70}")
print(f"\n  Best Val AUC: {best_val_auc:.4f}")
print(f"\n对比:")
print(f"  LightGBM v3-mpnet:   0.8122")
print(f"  Minimal DIN:          0.6827")
print(f"  DeepFM:               {best_val_auc:.4f}")
print(f"  Delta vs LightGBM:    {best_val_auc - 0.8122:+.4f}")

import gc
del model, optimizer, train_loader, val_loader, train_ds, val_ds
gc.collect()
if device == "mps":
    torch.mps.empty_cache()
print("\n✅ 内存释放")
