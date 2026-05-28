"""Day 9 - 改用 numpy matmul 替代 FAISS (avoid OMP segfault on Mac)"""
import sys, time, json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path("src/recall")))
from model import TwoTowerModel

RECALL_DIR = Path("data/processed/recall")
OUT_DIR = Path("models")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

with open(RECALL_DIR / "meta.json") as f:
    meta = json.load(f)
user_features = np.load(RECALL_DIR / "user_features.npz")["feat"]
item_data = np.load(RECALL_DIR / "item_features.npz")
item_dense = item_data["dense"]
item_cat = item_data["cat"]
val_pairs = np.load(RECALL_DIR / "val_pairs.npz")["pairs"]

model = TwoTowerModel(
    n_users=meta["n_users"], n_items=meta["n_items"],
    user_dense_dim=meta["user_dense_dim"], item_dense_dim=meta["item_dense_dim"],
    n_sub_cats=meta["n_sub_cats"], n_brands=meta["n_brands"], n_text_clusters=meta["n_text_clusters"],
).to(device)
ckpt = torch.load(OUT_DIR / "twotower_best.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()
ep = ckpt["epoch"]
vl = ckpt["val_loss"]
print(f"加载模型: epoch {ep}, val_loss {vl:.4f}")

# Step 1: 编码 items
print("\n[1/4] 编码所有 item...")
t0 = time.time()
all_item_embs = []
bs = 4096
with torch.no_grad():
    for i in range(0, len(item_dense), bs):
        end = min(i + bs, len(item_dense))
        idx_t = torch.arange(i, end, device=device)
        dense_t = torch.from_numpy(item_dense[i:end]).float().to(device)
        cat_t = torch.from_numpy(item_cat[i:end]).long().to(device)
        emb = model.item_tower(idx_t, dense_t, cat_t)
        all_item_embs.append(emb.cpu().numpy())
all_item_embs = np.concatenate(all_item_embs).astype(np.float32)
print(f"  shape: {all_item_embs.shape}, {time.time()-t0:.1f}s")

# Step 2: 编码 val users
print("\n[2/4] 编码 val users...")
t0 = time.time()
val_users = np.unique(val_pairs[:, 0])
user_to_true = {}
for u, i in val_pairs:
    user_to_true.setdefault(int(u), set()).add(int(i))

all_user_embs = []
with torch.no_grad():
    for i in range(0, len(val_users), bs):
        end = min(i + bs, len(val_users))
        u_idx = val_users[i:end]
        idx_t = torch.from_numpy(u_idx).long().to(device)
        dense_t = torch.from_numpy(user_features[u_idx]).float().to(device)
        emb = model.user_tower(idx_t, dense_t)
        all_user_embs.append(emb.cpu().numpy())
all_user_embs = np.concatenate(all_user_embs).astype(np.float32)
print(f"  shape: {all_user_embs.shape}, val users: {len(val_users):,}, {time.time()-t0:.1f}s")

# Step 3: numpy top-K (batched matmul, no FAISS)
print("\n[3/4] Numpy top-K (no FAISS)...")
t0 = time.time()
K_LIST = [10, 50, 100, 200, 500]
K_MAX = max(K_LIST)

# 用户分 batch (省内存)
n_users = len(all_user_embs)
all_top_indices = np.zeros((n_users, K_MAX), dtype=np.int32)
USER_BATCH = 1000

for i in range(0, n_users, USER_BATCH):
    end = min(i + USER_BATCH, n_users)
    # scores = (USER_BATCH, n_items)
    scores = all_user_embs[i:end] @ all_item_embs.T
    # top-K via argpartition (比 argsort 快很多)
    top_idx = np.argpartition(-scores, K_MAX, axis=1)[:, :K_MAX]
    # 在 top-K 内部再排序
    row_idx = np.arange(end - i)[:, None]
    sort_order = np.argsort(-scores[row_idx, top_idx], axis=1)
    all_top_indices[i:end] = top_idx[row_idx, sort_order]
    
    if (i // USER_BATCH + 1) % 20 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (i + USER_BATCH) * (n_users - i - USER_BATCH)
        print(f"  {i+USER_BATCH}/{n_users} ({(i+USER_BATCH)/n_users*100:.0f}%) | elapsed {elapsed:.0f}s | ETA {eta:.0f}s")

search_time = time.time() - t0
print(f"  search 总耗时: {search_time:.1f}s")
print(f"  avg latency: {search_time*1000/n_users:.2f} ms/query")

# Step 4: Recall@K
print("\n[4/4] Recall@K 评估")
recall_results = {}
for K in K_LIST:
    recalls = []
    for i, u_idx in enumerate(val_users):
        true_items = user_to_true[int(u_idx)]
        retrieved = set(all_top_indices[i, :K].tolist())
        hit = len(retrieved & true_items)
        recalls.append(hit / len(true_items))
    recall_at_k = float(np.mean(recalls))
    recall_results[f"Recall@{K}"] = recall_at_k
    print(f"  Recall@{K:>3}: {recall_at_k:.4f}")

# 保存
np.save(OUT_DIR / "item_embeddings.npy", all_item_embs)
with open(OUT_DIR / "recall_metrics.json", "w") as f:
    json.dump(recall_results, f, indent=2)

print("\n" + "=" * 70)
print("🎉 召回评估完成!")
print("=" * 70)
print(json.dumps(recall_results, indent=2))
