"""DeepFM 数据准备 - 从现有 parquet 转 npz (省去重新负采样时间)"""
import time, json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_PROC = Path("data/processed")
DEEPFM_DIR = DATA_PROC / "deepfm"
DEEPFM_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("DeepFM 数据准备")
print("=" * 70)

# Step 1: 加载 + merge clusters (和 v3 一样)
print("\n[1/4] 加载 parquet + merge clusters")
t0 = time.time()
df = pd.read_parquet(DATA_PROC / "train_with_all_features.parquet")
print(f"  shape: {df.shape}")

text_clusters = pd.read_csv(DATA_PROC / "item_text_clusters_mpnet.csv",
                            dtype={"parent_asin": "str", "text_cluster_id_mpnet": "int16"})
df = df.merge(text_clusters, on="parent_asin", how="left")
df["text_cluster_id_mpnet"] = df["text_cluster_id_mpnet"].fillna(0).astype(np.int16)
print(f"  merged: {df.shape}")
print(f"  耗时: {time.time()-t0:.1f}s")

# Step 2: 切分 (sklearn random_state=42, 和 v3 一样)
print("\n[2/4] Train/Val split (sklearn random=42, 复用 v3 切分)")
t0 = time.time()
from sklearn.model_selection import train_test_split
train_idx, val_idx = train_test_split(
    df.index, test_size=0.2, random_state=42, stratify=df["label"],
)
print(f"  train: {len(train_idx):,}, val: {len(val_idx):,}")
print(f"  耗时: {time.time()-t0:.1f}s")

# Step 3: 提取特征
print("\n[3/4] 提取特征矩阵")
t0 = time.time()
DENSE_COLS = [
    "user_interaction_count", "user_avg_rating", "user_last_timestamp",
    "item_interaction_count", "item_avg_rating", "item_last_timestamp",
    "price", "price_missing", "title_length", "n_categories",
    "user_avg_price", "user_price_diff", "pop_x_activity",
]  # 13 维 dense

CAT_COLS = ["sub_category_id", "brand_id", "text_cluster_id_mpnet"]  # 3 维 categorical

# Dense: 标准化 (训练集统计量)
dense_train = df.loc[train_idx, DENSE_COLS].values.astype(np.float32)
dense_val = df.loc[val_idx, DENSE_COLS].values.astype(np.float32)

# 标准化
mean = dense_train.mean(axis=0)
std = dense_train.std(axis=0) + 1e-9
dense_train = (dense_train - mean) / std
dense_val = (dense_val - mean) / std

# Categorical: 直接 int
cat_train = df.loc[train_idx, CAT_COLS].values.astype(np.int32)
cat_val = df.loc[val_idx, CAT_COLS].values.astype(np.int32)

# Label
y_train = df.loc[train_idx, "label"].values.astype(np.float32)
y_val = df.loc[val_idx, "label"].values.astype(np.float32)

# Cat vocab sizes
n_sub_cats = int(df["sub_category_id"].max()) + 1
n_brands = int(df["brand_id"].max()) + 1
n_text_clusters = 50

print(f"  dense shape: train {dense_train.shape}, val {dense_val.shape}")
print(f"  cat shape: train {cat_train.shape}, val {cat_val.shape}")
print(f"  vocab: sub_cats={n_sub_cats}, brands={n_brands}, text_clusters={n_text_clusters}")
print(f"  耗时: {time.time()-t0:.1f}s")

# Step 4: 保存
print("\n[4/4] 保存 npz")
t0 = time.time()
np.savez_compressed(DEEPFM_DIR / "train.npz",
                    dense=dense_train, cat=cat_train, label=y_train)
np.savez_compressed(DEEPFM_DIR / "val.npz",
                    dense=dense_val, cat=cat_val, label=y_val)

meta = {
    "n_dense": len(DENSE_COLS),
    "n_cat": len(CAT_COLS),
    "dense_cols": DENSE_COLS,
    "cat_cols": CAT_COLS,
    "n_sub_cats": n_sub_cats,
    "n_brands": n_brands,
    "n_text_clusters": n_text_clusters,
    "n_train": len(train_idx),
    "n_val": len(val_idx),
    "dense_mean": mean.tolist(),
    "dense_std": std.tolist(),
}
with open(DEEPFM_DIR / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"  耗时: {time.time()-t0:.1f}s")
for name in ["train", "val"]:
    p = DEEPFM_DIR / f"{name}.npz"
    print(f"  {name}.npz: {p.stat().st_size/1e6:.0f} MB")

print()
print("=" * 70)
print("🎉 DeepFM 数据完成!")
print("=" * 70)
