"""Enhanced DIN 全量训练 (独立脚本, 跑完自动释放内存)

使用:
    python src/din_pipeline/train_enhanced.py

跑完自动:
  1. 保存最佳模型到 models/din_enhanced_best.pt
  2. 保存训练日志到 models/din_enhanced_history.json
  3. 进程退出, 内存 100% 释放
"""
import sys
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from dataset_enhanced import EnhancedDINDataset, enhanced_collate_fn
from model_enhanced import EnhancedDIN

# ============================================
# 配置
# ============================================
DIN_DIR = Path("data/processed/din")
OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 1024
NUM_EPOCHS = 5
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 2
LOG_INTERVAL = 500

print("=" * 70)
print("Enhanced DIN 全量训练 (本机 MPS)")
print("=" * 70)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")
print(f"Config: batch={BATCH_SIZE}, lr={LR}, epochs={NUM_EPOCHS}, patience={PATIENCE}")

# ============================================
# 数据
# ============================================
print("\n加载数据...")
t0 = time.time()
train_ds = EnhancedDINDataset(DIN_DIR / "din_train_enhanced.npz")
val_ds = EnhancedDINDataset(DIN_DIR / "din_val_enhanced.npz")

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=enhanced_collate_fn, num_workers=0,
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
    collate_fn=enhanced_collate_fn, num_workers=0,
)
print(f"  train: {len(train_ds):,}, batches: {len(train_loader):,}")
print(f"  val:   {len(val_ds):,}, batches: {len(val_loader):,}")
print(f"  加载耗时: {time.time()-t0:.1f}s")

# ============================================
# 模型
# ============================================
with open(DIN_DIR / "meta.json") as f:
    meta = json.load(f)

model = EnhancedDIN(
    n_items=meta["n_items"] + 1,
    n_sub_cats=meta["n_sub_cats"],
    n_brands=meta["n_brands"],
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"\n模型参数: {n_params:,}")

optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


# ============================================
# 训练循环
# ============================================
def train_one_epoch(epoch_num):
    model.train()
    losses = []
    train_logits = []
    train_labels = []
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
        # 采样保存 train logits (节省内存)
        if step % 100 == 0:
            train_logits.append(logits.detach().cpu().numpy())
            train_labels.append(labels.cpu().numpy())
        
        if (step + 1) % LOG_INTERVAL == 0:
            elapsed = time.time() - t0
            eta = elapsed / (step + 1) * (len(train_loader) - step - 1)
            avg_loss = np.mean(losses[-LOG_INTERVAL:])
            sps = (step + 1) * BATCH_SIZE / elapsed
            print(f"  Epoch {epoch_num} | Step {step+1}/{len(train_loader)} | "
                  f"loss {avg_loss:.4f} | {sps:.0f} samples/s | "
                  f"elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m")
    
    train_logits = np.concatenate(train_logits)
    train_labels = np.concatenate(train_labels)
    train_auc = roc_auc_score(train_labels, train_logits)
    
    return {
        "loss": np.mean(losses),
        "auc": train_auc,
        "time": time.time() - t0,
    }


def evaluate():
    model.eval()
    all_logits = []
    all_labels = []
    losses = []
    t0 = time.time()
    
    with torch.no_grad():
        for batch in val_loader:
            labels = batch["label"].to(device)
            batch_dev = {k: v.to(device) for k, v in batch.items() if k != "label"}
            logits = model(batch_dev)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            losses.append(loss.item())
    
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    auc = roc_auc_score(all_labels, all_logits)
    
    return {
        "auc": auc,
        "loss": np.mean(losses),
        "time": time.time() - t0,
    }


# ============================================
# 主训练循环
# ============================================
history = []
best_val_auc = 0
patience_counter = 0

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\n{'='*70}")
    print(f"Epoch {epoch}/{NUM_EPOCHS}")
    print(f"{'='*70}")
    
    train_metrics = train_one_epoch(epoch)
    print(f"\n  Train: loss {train_metrics['loss']:.4f}, "
          f"AUC {train_metrics['auc']:.4f}, "
          f"time {train_metrics['time']/60:.1f}m")
    
    print(f"\n  Evaluating val...")
    val_metrics = evaluate()
    print(f"  Val: AUC {val_metrics['auc']:.4f}, "
          f"loss {val_metrics['loss']:.4f}, "
          f"time {val_metrics['time']:.0f}s")
    
    gap = train_metrics['auc'] - val_metrics['auc']
    print(f"  Gap: {gap:+.4f}")
    
    history.append({
        "epoch": epoch,
        "train_loss": train_metrics["loss"],
        "train_auc": train_metrics["auc"],
        "val_loss": val_metrics["loss"],
        "val_auc": val_metrics["auc"],
        "gap": gap,
        "train_time_min": train_metrics["time"] / 60,
    })
    
    if val_metrics["auc"] > best_val_auc:
        best_val_auc = val_metrics["auc"]
        patience_counter = 0
        torch.save({
            "model_state": model.state_dict(),
            "epoch": epoch,
            "val_auc": val_metrics["auc"],
            "config": {
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "epochs_trained": epoch,
            }
        }, OUT_DIR / "din_enhanced_best.pt")
        print(f"  ✅ 新最佳! Val AUC {best_val_auc:.4f} (保存模型)")
    else:
        patience_counter += 1
        print(f"  ⚠️ 没提升, patience {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print(f"\n🛑 早停")
            break
    
    # 保存历史 (每个 epoch)
    with open(OUT_DIR / "din_enhanced_history.json", "w") as f:
        json.dump(history, f, indent=2)


# ============================================
# 最终报告
# ============================================
total_time = sum(h["train_time_min"] for h in history)
print(f"\n{'='*70}")
print(f"🎉 训练完成!")
print(f"{'='*70}")
print(f"\n  Best Val AUC: {best_val_auc:.4f}")
print(f"  总训练时间: {total_time:.1f} 分钟")
print(f"  训练 epoch 数: {len(history)}")
print(f"\n对比基准:")
print(f"  LightGBM v3-mpnet: AUC 0.8122")
print(f"  Enhanced DIN:      AUC {best_val_auc:.4f}")
print(f"  Delta:             {best_val_auc - 0.8122:+.4f}")
print(f"\n模型 + 日志保存:")
print(f"  models/din_enhanced_best.pt")
print(f"  models/din_enhanced_history.json")

# ============================================
# 显式释放内存 (打印验证)
# ============================================
import gc
del model, optimizer, train_loader, val_loader, train_ds, val_ds
gc.collect()
if device == "mps":
    torch.mps.empty_cache()
print(f"\n✅ 内存已释放, 脚本即将退出")
