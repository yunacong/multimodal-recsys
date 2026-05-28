"""Day 9 - 续训 5 epoch (从 epoch 5 checkpoint)"""
import sys, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path("src/recall")))
from model import TwoTowerModel

RECALL_DIR = Path("data/processed/recall")
OUT_DIR = Path("models")

BATCH_SIZE = 1024
NUM_EPOCHS = 5  # 续训 5 epoch
LR = 5e-4  # lr 降低一半 (因为续训)
PATIENCE = 2

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# 加载数据
train_pairs = np.load(RECALL_DIR / "train_pairs.npz")["pairs"]
val_pairs = np.load(RECALL_DIR / "val_pairs.npz")["pairs"]
user_features = np.load(RECALL_DIR / "user_features.npz")["feat"]
item_data = np.load(RECALL_DIR / "item_features.npz")
item_dense = item_data["dense"]
item_cat = item_data["cat"]
with open(RECALL_DIR / "meta.json") as f:
    meta = json.load(f)


class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        u, i = self.pairs[idx]
        return {
            "user_idx": int(u),
            "user_dense": user_features[u],
            "item_idx": int(i),
            "item_dense": item_dense[i],
            "item_cat": item_cat[i],
        }


def collate(batch):
    return {
        "user_idx": torch.tensor([b["user_idx"] for b in batch], dtype=torch.long),
        "user_dense": torch.tensor(np.stack([b["user_dense"] for b in batch]), dtype=torch.float32),
        "item_idx": torch.tensor([b["item_idx"] for b in batch], dtype=torch.long),
        "item_dense": torch.tensor(np.stack([b["item_dense"] for b in batch]), dtype=torch.float32),
        "item_cat": torch.tensor(np.stack([b["item_cat"] for b in batch]), dtype=torch.long),
    }


train_ds = PairDataset(train_pairs)
val_ds = PairDataset(val_pairs)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, collate_fn=collate)

# 模型 + 加载 checkpoint
model = TwoTowerModel(
    n_users=meta["n_users"], n_items=meta["n_items"],
    user_dense_dim=meta["user_dense_dim"], item_dense_dim=meta["item_dense_dim"],
    n_sub_cats=meta["n_sub_cats"], n_brands=meta["n_brands"], n_text_clusters=meta["n_text_clusters"],
).to(device)
ckpt = torch.load(OUT_DIR / "twotower_best.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state"])
start_epoch = ckpt["epoch"]
prev_val_loss = ckpt["val_loss"]
print(f"从 epoch {start_epoch} 续训, prev val_loss {prev_val_loss:.4f}")

optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-5)


def evaluate():
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in val_loader:
            bd = {k: v.to(device) for k, v in batch.items()}
            u_emb, i_emb = model(bd["user_idx"], bd["user_dense"], bd["item_idx"], bd["item_dense"], bd["item_cat"])
            loss = model.in_batch_loss(u_emb, i_emb)
            losses.append(loss.item())
    return float(np.mean(losses))


# 续训历史
with open(OUT_DIR / "twotower_history.json") as f:
    history = json.load(f)

best_val_loss = prev_val_loss
patience_counter = 0

for ep_rel in range(1, NUM_EPOCHS + 1):
    epoch = start_epoch + ep_rel
    print(f"\n{'='*70}\nEpoch {epoch} (续训 {ep_rel}/{NUM_EPOCHS})\n{'='*70}")
    model.train()
    losses = []
    t0 = time.time()
    
    for step, batch in enumerate(train_loader):
        bd = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        u_emb, i_emb = model(bd["user_idx"], bd["user_dense"], bd["item_idx"], bd["item_dense"], bd["item_cat"])
        loss = model.in_batch_loss(u_emb, i_emb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        
        if (step + 1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (step+1) * (len(train_loader) - step - 1)
            sps = (step+1) * BATCH_SIZE / elapsed
            print(f"  Step {step+1}/{len(train_loader)} | loss {np.mean(losses[-200:]):.4f} | {sps:.0f} sps | ETA {eta/60:.1f}m")
    
    train_loss = float(np.mean(losses))
    val_loss = evaluate()
    print(f"\n  Train: loss {train_loss:.4f}, time {(time.time()-t0)/60:.1f}m")
    print(f"  Val:   loss {val_loss:.4f}")
    
    history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save({"model_state": model.state_dict(), "val_loss": val_loss, "epoch": epoch},
                   OUT_DIR / "twotower_best.pt")
        print(f"  ✅ 新最佳!")
    else:
        patience_counter += 1
        print(f"  ⚠️ patience {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print("\n🛑 早停")
            break
    
    with open(OUT_DIR / "twotower_history.json", "w") as f:
        json.dump(history, f, indent=2)

print(f"\n{'='*70}\n🎉 续训完成!Best val_loss: {best_val_loss:.4f}\n{'='*70}")
