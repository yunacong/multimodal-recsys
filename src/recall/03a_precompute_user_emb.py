"""Day 9 - 预计算 user embeddings, e2e 只用 numpy + LightGBM (避开 torch 冲突)"""
import sys, time, json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path("src/recall")))
from model import TwoTowerModel

RECALL_DIR = Path("data/processed/recall")
OUT_DIR = Path("models")

device = "cpu"
print("预计算所有 val users 的 emb...")

with open(RECALL_DIR / "meta.json") as f:
    meta = json.load(f)
user_features = np.load(RECALL_DIR / "user_features.npz")["feat"]

model = TwoTowerModel(
    n_users=meta["n_users"], n_items=meta["n_items"],
    user_dense_dim=meta["user_dense_dim"], item_dense_dim=meta["item_dense_dim"],
    n_sub_cats=meta["n_sub_cats"], n_brands=meta["n_brands"], n_text_clusters=meta["n_text_clusters"],
).to(device)
ckpt = torch.load(OUT_DIR / "twotower_best.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()

val_pairs = np.load(RECALL_DIR / "val_pairs.npz")["pairs"]
val_users = np.unique(val_pairs[:, 0])[:500]  # 只算前 500 个用

t0 = time.time()
all_user_embs = []
bs = 256
with torch.no_grad():
    for i in range(0, len(val_users), bs):
        end = min(i + bs, len(val_users))
        u_idx = val_users[i:end]
        idx_t = torch.from_numpy(u_idx).long()
        dense_t = torch.from_numpy(user_features[u_idx]).float()
        emb = model.user_tower(idx_t, dense_t)
        all_user_embs.append(emb.numpy())
all_user_embs = np.concatenate(all_user_embs).astype(np.float32)

np.savez(OUT_DIR / "val_user_embs.npz", users=val_users, embs=all_user_embs)
print(f"✅ 保存 {all_user_embs.shape}, 耗时 {time.time()-t0:.1f}s")
