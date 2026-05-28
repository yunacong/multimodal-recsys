"""双塔召回训练 - in-batch negatives"""
import sys, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from model import TwoTowerModel

RECALL_DIR = Path("data/processed/recall")
OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 1024
NUM_EPOCHS = 5
LR = 1e-3
PATIENCE = 2

print("=" * 70)
print("双塔召回训练")
print("=" * 70)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# 加载
print("\n加载数据...")
t0 = time.time()
train_pairs = np.load(RECALL_DIR / "train_pairs.npz")["pairs"]
val_pairs = np.load(RECALL_DIR / "val_pairs.npz")["pairs"]
user_features = np.load(RECALL_DIR / "user_features.npz")["feat"]
item_data = np.load(RECALL_DIR / "item_features.npz")
item_dense = item_data["dense"]
item_cat = item_data["cat"]
with open(RECALL_DIR / "meta.json") as f:
    meta = json.load(f)

print(f"  train_pairs: {len(train_pairs):,}")
print(f"  val_pairs: {len(val_pairs):,}")
print(f"  user_features: {user_features.shape}")
print(f"  item_dense: {item_dense.shape}, item_cat: {item_cat.shape}")
print(f"  耗时: {time.time()-t0:.1f}s")


class PairDataset(Dataset):
    def __init__(self, pairs, user_features, item_dense, item_cat):
        self.pairs = pairs
        self.user_features = user_features
        self.item_dense = item_dense
        self.item_cat = item_cat
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        u_idx, i_idx = self.pairs[idx]
        return {
            "user_idx": int(u_idx),
            "user_dense": self.user_features[u_idx],
            "item_idx": int(i_idx),
            "item_dense": self.item_dense[i_idx],
            "item_cat": self.item_cat[i_idx],
        }


def collate(batch):
    return {
        "user_idx": torch.tensor([b["user_idx"] for b in batch], dtype=torch.long),
        "user_dense": torch.tensor(np.stack([b["user_dense"] for b in batch]), dtype=torch.float32),
        "item_idx": torch.tensor([b["item_idx"] for b in batch], dtype=torch.long),
        "item_dense": torch.tensor(np.stack([b["item_dense"] for b in batch]), dtype=torch.float32),
        "item_cat": torch.tensor(np.stack([b["item_cat"] for b in batch]), dtype=torch.long),
    }


train_ds = PairDataset(train_pairs, user_features, item_dense, item_cat)
val_ds = PairDataset(val_pairs, user_features, item_dense, item_cat)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, collate_fn=collate)

# 模型
model = TwoTowerModel(
    n_users=meta["n_users"], n_items=meta["n_items"],
    user_dense_dim=meta["user_dense_dim"], item_dense_dim=meta["item_dense_dim"],
    n_sub_cats=meta["n_sub_cats"], n_brands=meta["n_brands"], n_text_clusters=meta["n_text_clusters"],
).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"\n模型参数: {n_params:,}")
optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-5)


def evaluate():
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in val_loader:
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            user_emb, item_emb = model(
                batch_dev["user_idx"], batch_dev["user_dense"],
                batch_dev["item_idx"], batch_dev["item_dense"], batch_dev["item_cat"],
            )
            loss = model.in_batch_loss(user_emb, item_emb)
            losses.append(loss.item())
    return np.mean(losses)


history = []
best_val_loss = float("inf")
patience_counter = 0

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\n{'='*70}\nEpoch {epoch}/{NUM_EPOCHS}\n{'='*70}")
    model.train()
    losses = []
    t0 = time.time()
    
    for step, batch in enumerate(train_loader):
        batch_dev = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        user_emb, item_emb = model(
            batch_dev["user_idx"], batch_dev["user_dense"],
            batch_dev["item_idx"], batch_dev["item_dense"], batch_dev["item_cat"],
        )
        loss = model.in_batch_loss(user_emb, item_emb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        
        if (step + 1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (step+1) * (len(train_loader) - step - 1)
            sps = (step+1) * BATCH_SIZE / elapsed
            print(f"  Step {step+1}/{len(train_loader)} | loss {np.mean(losses[-200:]):.4f} | {sps:.0f} sps | ETA {eta/60:.1f}m")
    
    train_loss = np.mean(losses)
    val_loss = evaluate()
    print(f"\n  Train: loss {train_loss:.4f}, time {(time.time()-t0)/60:.1f}m")
    print(f"  Val:   loss {val_loss:.4f}")
    
    history.append({"epoch": epoch, "train_loss": float(train_loss), "val_loss": float(val_loss)})
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save({"model_state": model.state_dict(), "val_loss": val_loss, "epoch": epoch},
                   OUT_DIR / "twotower_best.pt")
        print(f"  ✅ 新最佳! 保存")
    else:
        patience_counter += 1
        print(f"  ⚠️ 没提升 {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print("\n🛑 早停")
            break
    
    with open(OUT_DIR / "twotower_history.json", "w") as f:
        json.dump(history, f, indent=2)

print(f"\n{'='*70}\n🎉 训练完成!Best val_loss: {best_val_loss:.4f}\n{'='*70}")
