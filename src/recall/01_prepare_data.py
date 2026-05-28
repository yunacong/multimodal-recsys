"""Day 9 - 双塔召回数据准备

双塔召回 vs CTR 的关键差异:
  CTR (DeepFM/LightGBM): 输入是 (user, item) 对, 输出概率
  Recall (双塔): User Tower → user_emb, Item Tower → item_emb
                 训练时: <user_emb, item_emb> 拉近正样本, 推远负样本
                 推理时: ANN (FAISS) 从 207K 商品找 top-K

数据格式:
  正样本: (user_id, item_id) 真实交互
  负样本: 每个正样本对应 N 个负样本 (in-batch 或 random)

特征:
  User Tower: user_id_emb + user_meta (interaction_count, avg_rating, avg_price)
  Item Tower: item_id_emb + item_meta (sub_cat, brand, price, text_cluster, image_cluster)
"""

import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PROC = Path("data/processed")
RECALL_DIR = DATA_PROC / "recall"
RECALL_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("Day 9 - 双塔召回数据准备")
print("=" * 70)

# Step 1: 加载完整训练数据 (含全部特征)
print("\n[1/5] 加载训练数据")
t0 = time.time()
df = pd.read_parquet(DATA_PROC / "train_with_all_features.parquet")
print(f"  shape: {df.shape}")
print(f"  columns: {list(df.columns)[:10]}...")
print(f"  耗时: {time.time()-t0:.1f}s")

# Step 2: 仅取正样本 (双塔召回训练用正样本 + 随机负采样)
print("\n[2/5] 取正样本")
pos_df = df[df["label"] == 1].copy()
print(f"  正样本: {len(pos_df):,}")
print(f"  unique users: {pos_df['user_id'].nunique():,}")
print(f"  unique items: {pos_df['parent_asin'].nunique():,}")

# Step 3: 构建 user 和 item vocab
print("\n[3/5] 构建 vocab")
t0 = time.time()
unique_users = pos_df['user_id'].unique()
unique_items = df["parent_asin"].unique()  # 用全量 items (包括没正样本的)
user2idx = {u: i for i, u in enumerate(sorted(unique_users))}
item2idx = {it: i for i, it in enumerate(sorted(unique_items))}
n_users = len(user2idx)
n_items = len(item2idx)
print(f"  users: {n_users:,}")
print(f"  items: {n_items:,}")
print(f"  耗时: {time.time()-t0:.1f}s")

# Step 4: 构建 user 和 item 特征矩阵
print("\n[4/5] 构建特征矩阵")
t0 = time.time()

# User 特征 (每个 user 取一行 - 用最早的)
user_features = df.groupby("user_id").agg({
    "user_interaction_count": "first",
    "user_avg_rating": "first",
    "user_last_timestamp": "first",
    "user_avg_price": "first",
}).reset_index()
user_features["user_idx"] = user_features["user_id"].map(user2idx)
user_features = user_features.dropna(subset=["user_idx"]).sort_values("user_idx").reset_index(drop=True)
user_features["user_idx"] = user_features["user_idx"].astype(np.int32)
print(f"  user_features: {user_features.shape}")

# Item 特征
item_features = df.groupby("parent_asin").agg({
    "item_interaction_count": "first",
    "item_avg_rating": "first",
    "item_last_timestamp": "first",
    "price": "first",
    "price_missing": "first",
    "title_length": "first",
    "n_categories": "first",
    "sub_category_id": "first",
    "brand_id": "first",
}).reset_index()
item_features["item_idx"] = item_features["parent_asin"].map(item2idx)
item_features = item_features.dropna(subset=["item_idx"]).sort_values("item_idx").reset_index(drop=True)
item_features["item_idx"] = item_features["item_idx"].astype(np.int32)
print(f"  item_features: {item_features.shape}")

# 加 text_cluster (mpnet)
text_clusters = pd.read_csv(DATA_PROC / "item_text_clusters_mpnet.csv")
item_features = item_features.merge(text_clusters, on="parent_asin", how="left")
item_features["text_cluster_id_mpnet"] = item_features["text_cluster_id_mpnet"].fillna(0).astype(np.int16)

print(f"  耗时: {time.time()-t0:.1f}s")

# Step 5: 生成 (user_idx, pos_item_idx) 对 + train/val split
print("\n[5/5] 生成训练对 + split")
t0 = time.time()
pos_df["user_idx"] = pos_df['user_id'].map(user2idx)
pos_df["item_idx"] = pos_df['parent_asin'].map(item2idx)
pos_df = pos_df.dropna(subset=["user_idx", "item_idx"])

# 简单 8:2 split (random)
train_pairs, val_pairs = train_test_split(
    pos_df[["user_idx", "item_idx"]].values.astype(np.int32),
    test_size=0.1, random_state=42,
)
print(f"  train: {len(train_pairs):,}, val: {len(val_pairs):,}")

# 保存
np.savez_compressed(RECALL_DIR / "train_pairs.npz", pairs=train_pairs)
np.savez_compressed(RECALL_DIR / "val_pairs.npz", pairs=val_pairs)

# User 特征
user_feat_cols = ["user_interaction_count", "user_avg_rating", "user_last_timestamp", "user_avg_price"]
user_feat_matrix = user_features[user_feat_cols].values.astype(np.float32)
# 标准化
user_mean = user_feat_matrix.mean(axis=0)
user_std = user_feat_matrix.std(axis=0) + 1e-9
user_feat_matrix = (user_feat_matrix - user_mean) / user_std

# Item 特征
item_dense_cols = ["item_interaction_count", "item_avg_rating", "item_last_timestamp",
                   "price", "price_missing", "title_length", "n_categories"]
item_cat_cols = ["sub_category_id", "brand_id", "text_cluster_id_mpnet"]

item_dense = item_features[item_dense_cols].values.astype(np.float32)
item_mean = item_dense.mean(axis=0)
item_std = item_dense.std(axis=0) + 1e-9
item_dense = (item_dense - item_mean) / item_std

item_cat = item_features[item_cat_cols].values.astype(np.int32)

np.savez_compressed(RECALL_DIR / "user_features.npz", feat=user_feat_matrix)
np.savez_compressed(RECALL_DIR / "item_features.npz", dense=item_dense, cat=item_cat)

# meta
meta = {
    "n_users": int(n_users),
    "n_items": int(n_items),
    "user_dense_dim": len(user_feat_cols),
    "item_dense_dim": len(item_dense_cols),
    "n_sub_cats": int(item_cat[:, 0].max()) + 1,
    "n_brands": int(item_cat[:, 1].max()) + 1,
    "n_text_clusters": int(item_cat[:, 2].max()) + 1,
    "n_train_pairs": len(train_pairs),
    "n_val_pairs": len(val_pairs),
}
with open(RECALL_DIR / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"  耗时: {time.time()-t0:.1f}s")
print()
print("=" * 70)
print("🎉 Day 9 召回数据准备完成!")
print("=" * 70)
print(json.dumps(meta, indent=2))
