"""Day 9 Phase 3-4: FAISS 索引 + Recall@K 评估"""
import sys, time, json
from pathlib import Path
import numpy as np
import torch
import faiss

sys.path.insert(0, str(Path("src/recall")))
from model import TwoTowerModel

RECALL_DIR = Path("data/processed/recall")
OUT_DIR = Path("models")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# 加载 meta + 数据
with open(RECALL_DIR / "meta.json") as f:
    meta = json.load(f)
user_features = np.load(RECALL_DIR / "user_features.npz")["feat"]
item_data = np.load(RECALL_DIR / "item_features.npz")
item_dense = item_data["dense"]
item_cat = item_data["cat"]
val_pairs = np.load(RECALL_DIR / "val_pairs.npz")["pairs"]

# 加载模型
model = TwoTowerModel(
    n_users=meta["n_users"], n_items=meta["n_items"],
    user_dense_dim=meta["user_dense_dim"], item_dense_dim=meta["item_dense_dim"],
    n_sub_cats=meta["n_sub_cats"], n_brands=meta["n_brands"], n_text_clusters=meta["n_text_clusters"],
).to(device)
ckpt = torch.load(OUT_DIR / "twotower_best.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"加载模型: epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f}")

# Step 1: 编码所有 item (一次性, 用于 FAISS 索引)
print("\n[1/4] 编码所有 item...")
t0 = time.time()
all_item_embs = []
batch_size = 4096
with torch.no_grad():
    for i in range(0, len(item_dense), batch_size):
        end = min(i + batch_size, len(item_dense))
        idx_t = torch.arange(i, end, device=device)
        dense_t = torch.from_numpy(item_dense[i:end]).float().to(device)
        cat_t = torch.from_numpy(item_cat[i:end]).long().to(device)
        emb = model.item_tower(idx_t, dense_t, cat_t)
        all_item_embs.append(emb.cpu().numpy())
all_item_embs = np.concatenate(all_item_embs).astype(np.float32)
print(f"  item_embs shape: {all_item_embs.shape}, {time.time()-t0:.1f}s")

# Step 2: 建 FAISS 索引 (Inner Product, 因为 emb 已经 L2-normalize 过)
print("\n[2/4] 构建 FAISS 索引 (IndexFlatIP)...")
t0 = time.time()
emb_dim = all_item_embs.shape[1]
index = faiss.IndexFlatIP(emb_dim)
index.add(all_item_embs)
print(f"  Index ntotal: {index.ntotal}, {time.time()-t0:.1f}s")

# Step 3: 编码 val users (用 unique users from val_pairs)
print("\n[3/4] 编码 val users + 召回 top-K...")
t0 = time.time()
val_users = np.unique(val_pairs[:, 0])
print(f"  val unique users: {len(val_users):,}")

# 构建 user -> true items 的 ground truth
user_to_true_items = {}
for u, i in val_pairs:
    user_to_true_items.setdefault(int(u), set()).add(int(i))

# 编码 users
all_user_embs = []
with torch.no_grad():
    for i in range(0, len(val_users), batch_size):
        end = min(i + batch_size, len(val_users))
        u_idx = val_users[i:end]
        idx_t = torch.from_numpy(u_idx).long().to(device)
        dense_t = torch.from_numpy(user_features[u_idx]).float().to(device)
        emb = model.user_tower(idx_t, dense_t)
        all_user_embs.append(emb.cpu().numpy())
all_user_embs = np.concatenate(all_user_embs).astype(np.float32)
print(f"  user_embs: {all_user_embs.shape}, {time.time()-t0:.1f}s")

# Step 4: FAISS search + Recall@K
print("\n[4/4] FAISS 搜索 + Recall@K 评估...")
t0 = time.time()
K_LIST = [10, 50, 100, 200, 500]
K_MAX = max(K_LIST)

scores, indices = index.search(all_user_embs, K_MAX)
print(f"  search 耗时: {time.time()-t0:.1f}s ({len(val_users):,} queries)")
print(f"  avg latency: {(time.time()-t0)*1000/len(val_users):.2f} ms/query")

# 计算 Recall@K
print("\n计算 Recall@K...")
recall_results = {}
for K in K_LIST:
    recalls = []
    for i, u_idx in enumerate(val_users):
        true_items = user_to_true_items[int(u_idx)]
        if len(true_items) == 0:
            continue
        retrieved = set(indices[i, :K].tolist())
        hit = len(retrieved & true_items)
        recalls.append(hit / len(true_items))
    recall_at_k = np.mean(recalls)
    recall_results[f"Recall@{K}"] = float(recall_at_k)
    print(f"  Recall@{K:>3}: {recall_at_k:.4f}")

# 保存
np.save(OUT_DIR / "item_embeddings.npy", all_item_embs)
faiss.write_index(index, str(OUT_DIR / "item.faiss"))
with open(OUT_DIR / "recall_metrics.json", "w") as f:
    json.dump(recall_results, f, indent=2)

print("\n" + "=" * 70)
print("🎉 召回评估完成!")
print("=" * 70)
print(json.dumps(recall_results, indent=2))
print(f"\n保存:")
print(f"  models/item_embeddings.npy ({all_item_embs.nbytes/1e6:.0f} MB)")
print(f"  models/item.faiss")
print(f"  models/recall_metrics.json")
