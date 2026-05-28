"""
数据泄露修复 + v3-mpnet 重训

步骤:
  1. 用 cutoff 前的交互重新计算 val 用户特征 (无泄露)
  2. 把干净 user features 存为 data/processed/user_features_no_leak.parquet
  3. 用干净特征重构 train/val 数据集
  4. 重训 LightGBM v3-mpnet，输出真实 AUC / NDCG@10 / LogLoss

运行:
    python src/fix_data_leakage.py
"""

import json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
MODELS_DIR   = PROJECT_ROOT / "models"
RAW_DIR      = PROJECT_ROOT / "data" / "raw"

SPLIT_PERCENTILE = 80
FEATURE_COLS_V3 = [
    "user_interaction_count", "user_avg_rating", "user_last_timestamp",
    "item_interaction_count", "item_avg_rating", "item_last_timestamp",
    "price", "price_missing", "title_length", "n_categories",
    "sub_category_id", "brand_id", "user_avg_price",
    "user_price_diff", "pop_x_activity",
    "text_cluster_id_mpnet",
]


# ── 工具函数 ────────────────────────────────────────────────────────
def ndcg_at_k(y_true, y_score, k=10):
    order    = np.argsort(-y_score)
    y_sorted = np.asarray(y_true)[order]
    ak       = min(k, len(y_sorted))
    gains    = y_sorted[:ak] / np.log2(np.arange(2, ak + 2))
    dcg      = gains.sum()
    ideal    = np.sort(y_true)[::-1][:ak] / np.log2(np.arange(2, ak + 2))
    idcg     = ideal.sum()
    return float(dcg / idcg) if idcg > 0 else 0.0


def compute_user_ndcg(df_val, preds, k=10, max_users=50_000):
    df_val = df_val.copy()
    df_val["_pred"] = preds
    users = df_val["user_id"].unique()
    if len(users) > max_users:
        rng   = np.random.default_rng(42)
        users = rng.choice(users, size=max_users, replace=False)
    sub    = df_val[df_val["user_id"].isin(users)]
    scores = [
        ndcg_at_k(g["label"].values, g["_pred"].values, k)
        for _, g in sub.groupby("user_id")
        if g["label"].sum() > 0
    ]
    return float(np.mean(scores))


# ── 1. 加载原始交互 + 商品价格 ──────────────────────────────────────
print("=" * 60)
print("Step 1: 加载原始数据")
print("=" * 60)
t0 = time.time()

raw = pd.read_csv(RAW_DIR / "BPC_5core_train.csv")
# 列: user_id, parent_asin, rating, timestamp
print(f"  原始交互: {len(raw):,} 行")

meta_price = pd.read_csv(DATA_PROC / "item_meta_features.csv")[["parent_asin", "price", "price_missing"]]
raw = raw.merge(meta_price, on="parent_asin", how="left")
raw["price"] = raw["price"].fillna(0)
print(f"  价格合并后: {raw['price'].isna().sum()} 空值")

# 确定 cutoff
df_all = pd.read_parquet(DATA_PROC / "train_with_all_features.parquet")
cutoff  = np.percentile(df_all["user_last_timestamp"].values, SPLIT_PERCENTILE)
print(f"  cutoff: {pd.Timestamp(cutoff, unit='ms')}  ({cutoff:.0f} ms)")
print(f"  耗时 {time.time()-t0:.1f}s")


# ── 2. 计算无泄露 user features ─────────────────────────────────────
print("\nStep 2: 计算无泄露 user features")
t0 = time.time()

# 关键：先从 parquet 确定哪些 user 属于 train / val split
train_user_ids = set(df_all[df_all["user_last_timestamp"] <= cutoff]["user_id"].unique())
val_user_ids   = set(df_all[df_all["user_last_timestamp"] >  cutoff]["user_id"].unique())
print(f"  parquet split: {len(train_user_ids):,} train users, {len(val_user_ids):,} val users")

# train 用户 → 只用属于 train-split 的用户的全量交互（全部在 cutoff 前）
train_user_feat = (
    raw[raw["user_id"].isin(train_user_ids)]
    .groupby("user_id")
    .agg(
        user_interaction_count=("rating",   "count"),
        user_avg_rating        =("rating",   "mean"),
        user_last_timestamp    =("timestamp","max"),
        user_avg_price         =("price",    "mean"),
    )
    .reset_index()
    .astype({"user_interaction_count": np.int32,
             "user_avg_rating":        np.float32,
             "user_avg_price":         np.float32})
)
print(f"  train user features: {len(train_user_feat):,}")

# val 用户 → 只用属于 val-split 的用户的 pre-cutoff 交互
raw_val_before = raw[
    raw["user_id"].isin(val_user_ids) & (raw["timestamp"] <= cutoff)
]
val_user_feat_warm = (
    raw_val_before.groupby("user_id")
    .agg(
        user_interaction_count=("rating",   "count"),
        user_avg_rating        =("rating",   "mean"),
        user_last_timestamp    =("timestamp","max"),
        user_avg_price         =("price",    "mean"),
    )
    .reset_index()
    .astype({"user_interaction_count": np.int32,
             "user_avg_rating":        np.float32,
             "user_avg_price":         np.float32})
)
warm_users = set(val_user_feat_warm["user_id"])
cold_users = val_user_ids - warm_users
print(f"  val users warm (pre-cutoff history): {len(warm_users):,}")
print(f"  val users cold (no history):         {len(cold_users):,}")

# 冷启动用户: 用 train 用户均值填充
fill_count = int(train_user_feat["user_interaction_count"].mean())
fill_rating = float(train_user_feat["user_avg_rating"].mean())
fill_price  = float(train_user_feat["user_avg_price"].mean())
fill_ts     = int(cutoff)   # 假设最后活跃时刻=cutoff
cold_df = pd.DataFrame({
    "user_id":               list(cold_users),
    "user_interaction_count": fill_count,
    "user_avg_rating":         fill_rating,
    "user_last_timestamp":     fill_ts,
    "user_avg_price":          fill_price,
})
cold_df = cold_df.astype({"user_interaction_count": np.int32,
                           "user_avg_rating":        np.float32,
                           "user_avg_price":         np.float32})
print(f"  cold-start 填充值: count={fill_count}, rating={fill_rating:.2f}, price={fill_price:.2f}")

# 合并所有
user_feat_no_leak = pd.concat([train_user_feat, val_user_feat_warm, cold_df],
                               ignore_index=True)
print(f"  总 user features: {len(user_feat_no_leak):,}")

out_path = DATA_PROC / "user_features_no_leak.parquet"
user_feat_no_leak.to_parquet(out_path, index=False)
print(f"  ✅ 保存: {out_path}")
print(f"  耗时 {time.time()-t0:.1f}s")


# ── 3. 重构 train/val 数据集 ────────────────────────────────────────
print("\nStep 3: 重构数据集 (替换 user features)")
t0 = time.time()

# 基础列：item features + label + user_id (来自原始 parquet)
item_cols = [
    "user_id", "parent_asin", "label",
    "item_interaction_count", "item_avg_rating", "item_last_timestamp",
    "price", "price_missing", "title_length", "n_categories",
    "sub_category_id", "brand_id",
    "user_last_timestamp",   # 用于 split (保留原值，只替换特征)
]
df_base = df_all[item_cols].copy()

# 合并无泄露 user features (注意: user_last_timestamp 来自 user_feat_no_leak,
#   不是 split 标志—— split 用原始的 df_all['user_last_timestamp'])
user_feat_merge = user_feat_no_leak.rename(
    columns={"user_last_timestamp": "user_last_ts_feat"}
)
df_merged = df_base.merge(
    user_feat_merge[["user_id", "user_interaction_count", "user_avg_rating",
                     "user_last_ts_feat", "user_avg_price"]],
    on="user_id", how="left"
)

# 用无泄露的 user features (user_last_timestamp 作为特征用 no-leak 版本)
df_merged["user_last_timestamp_feat"] = df_merged["user_last_ts_feat"].fillna(cutoff).astype(np.int64)
df_merged["user_interaction_count"]   = df_merged["user_interaction_count"].fillna(fill_count).astype(np.int32)
df_merged["user_avg_rating"]          = df_merged["user_avg_rating"].fillna(fill_rating).astype(np.float32)
df_merged["user_avg_price"]           = df_merged["user_avg_price"].fillna(fill_price).astype(np.float32)

# 重新计算交叉特征
def safe_price_diff(row):
    try:
        p = float(row["price"]); avg = float(row["user_avg_price"])
        if math.isnan(p) or math.isnan(avg): return 0.0
        return abs(p - avg) / (avg + 1e-6)
    except: return 0.0

df_merged["user_price_diff"] = (
    (df_merged["price"] - df_merged["user_avg_price"]).abs()
    / (df_merged["user_avg_price"] + 1e-6)
).astype(np.float32)

df_merged["pop_x_activity"] = (
    np.log1p(df_merged["user_interaction_count"].astype(float))
    * np.log1p(df_merged["item_interaction_count"].astype(float))
).astype(np.float32)

# 合并文本聚类
text_cl = pd.read_csv(DATA_PROC / "item_text_clusters_mpnet.csv")
df_merged = df_merged.merge(text_cl[["parent_asin", "text_cluster_id_mpnet"]],
                             on="parent_asin", how="left")
df_merged["text_cluster_id_mpnet"] = df_merged["text_cluster_id_mpnet"].fillna(0).astype(np.int16)

print(f"  merged shape: {df_merged.shape}")
print(f"  null check: {df_merged[FEATURE_COLS_V3].isna().sum().sum()} nulls in feature cols")
print(f"  耗时 {time.time()-t0:.1f}s")


# ── 4. 训练/验证切分 ─────────────────────────────────────────────────
# 用原始 user_last_timestamp (split 标志，不是 feature) 切分
split_ts = df_merged["user_last_timestamp"]  # 原始值
val_mask  = split_ts > cutoff

df_train = df_merged[~val_mask].reset_index(drop=True)
df_val   = df_merged[ val_mask].reset_index(drop=True)
print(f"\nStep 4: 数据切分")
print(f"  train: {len(df_train):,}  val: {len(df_val):,}")

# user_last_timestamp 作为特征时用 no-leak 版
FEAT = [c if c != "user_last_timestamp" else "user_last_timestamp_feat"
        for c in FEATURE_COLS_V3]
# 重命名: 模型期望 user_last_timestamp，需要保证列名一致
# → 直接把 user_last_timestamp_feat 重命名后在 X 矩阵里传入
FEAT_FINAL = FEATURE_COLS_V3   # 保持模型期望的列名不变
df_train = df_train.rename(columns={"user_last_timestamp_feat": "_ts_feat"})
df_val   = df_val.rename(  columns={"user_last_timestamp_feat": "_ts_feat"})

# 构造特征矩阵（用 no-leak user_last_ts_feat 替代 user_last_timestamp）
def build_X(df):
    cols = []
    for c in FEATURE_COLS_V3:
        if c == "user_last_timestamp":
            cols.append(df["_ts_feat"].values)
        else:
            cols.append(df[c].values)
    return np.column_stack(cols).astype(np.float32)

X_train = build_X(df_train)
y_train = df_train["label"].values.astype(np.int8)
X_val   = build_X(df_val)
y_val   = df_val["label"].values.astype(np.int8)
print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}")


# ── 5. 重训 v3-mpnet ────────────────────────────────────────────────
print("\nStep 5: 重训 LightGBM v3-mpnet (无泄露)")
t0 = time.time()

lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS_V3,
                         categorical_feature=["sub_category_id","brand_id","price_missing",
                                              "text_cluster_id_mpnet"])
lgb_val   = lgb.Dataset(X_val,   label=y_val,   reference=lgb_train)

params = {
    "objective":       "binary",
    "metric":          ["auc", "binary_logloss"],
    "learning_rate":   0.05,
    "num_leaves":      63,
    "min_data_in_leaf":50,
    "feature_fraction":0.8,
    "bagging_fraction":0.8,
    "bagging_freq":    5,
    "verbose":        -1,
    "seed":            42,
}
callbacks = [lgb.log_evaluation(period=100), lgb.early_stopping(stopping_rounds=50)]
booster = lgb.train(
    params,
    lgb_train,
    num_boost_round=500,
    valid_sets=[lgb_val],
    callbacks=callbacks,
)
print(f"  训练完成  best_iteration={booster.best_iteration}  耗时 {time.time()-t0:.1f}s")


# ── 6. 评估 ─────────────────────────────────────────────────────────
print("\nStep 6: 评估")
preds   = booster.predict(X_val)
auc     = roc_auc_score(y_val, preds)
ll      = log_loss(y_val, preds)
ndcg10  = compute_user_ndcg(df_val[["user_id","label"]], preds, k=10)

print(f"\n  ╔══════════════════════════════════════════╗")
print(f"  ║  v3-mpnet 无泄露重测结果                 ║")
print(f"  ║  AUC:      {auc:.6f}                    ║")
print(f"  ║  NDCG@10:  {ndcg10:.6f}                    ║")
print(f"  ║  LogLoss:  {ll:.6f}                    ║")
print(f"  ╚══════════════════════════════════════════╝")

# 保存模型
model_path = MODELS_DIR / "lightgbm_v3_mpnet_noleak.txt"
booster.save_model(str(model_path))
print(f"\n  ✅ 模型保存: {model_path}")

# 保存结果 JSON
result = {
    "model":          "lightgbm_v3_mpnet_noleak",
    "description":    "v3-mpnet 重训，用户特征修复数据泄露（只用 cutoff 前交互）",
    "best_iteration": booster.best_iteration,
    "val_auc":        round(auc, 6),
    "ndcg_at_10":     round(ndcg10, 6),
    "val_logloss":    round(ll, 6),
    "cutoff":         int(cutoff),
    "cutoff_date":    str(pd.Timestamp(cutoff, unit="ms")),
    "cold_start_users":         len(cold_users),
    "warm_start_users":         len(warm_users),
    "cold_start_fill": {
        "user_interaction_count": fill_count,
        "user_avg_rating":        round(fill_rating, 4),
        "user_avg_price":         round(fill_price, 4),
    },
}
out_json = MODELS_DIR / "v3_noleak_result.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"  ✅ 结果保存: {out_json}")
