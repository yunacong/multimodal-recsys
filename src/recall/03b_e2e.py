"""Day 9 Phase 5 - 端到端 pipeline (纯 numpy + LightGBM, 不 import torch)"""
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

RECALL_DIR = Path("data/processed/recall")
DATA_PROC = Path("data/processed")
OUT_DIR = Path("models")

print("=" * 70)
print("Day 9 Phase 5 - E2E Recall + Rank Pipeline")
print("=" * 70)

# [1/5] 加载预计算的 embs + LightGBM
print("\n[1/5] 加载...")
t0 = time.time()
val_user_data = np.load(OUT_DIR / "val_user_embs.npz")
val_users = val_user_data["users"]
val_user_embs = val_user_data["embs"]
all_item_embs = np.load(OUT_DIR / "item_embeddings.npy")
print(f"  user_embs: {val_user_embs.shape}, item_embs: {all_item_embs.shape}")

booster = lgb.Booster(model_file=str(OUT_DIR / "lightgbm_v3_mpnet.txt"))
print(f"  LightGBM features: {booster.num_feature()}")
print(f"  耗时 {time.time()-t0:.1f}s")

# [2/5] 加载 vocabularies + 特征数据
print("\n[2/5] 加载特征数据...")
t0 = time.time()
df_all = pd.read_parquet(DATA_PROC / "train_with_all_features.parquet")
text_clusters = pd.read_csv(DATA_PROC / "item_text_clusters_mpnet.csv")
df_all = df_all.merge(text_clusters, on="parent_asin", how="left")
df_all["text_cluster_id_mpnet"] = df_all["text_cluster_id_mpnet"].fillna(0).astype(np.int16)

with open(RECALL_DIR / "meta.json") as f:
    meta = json.load(f)
val_pairs = np.load(RECALL_DIR / "val_pairs.npz")["pairs"]

# 反查 vocab
user_vocab = sorted(df_all["user_id"].unique())
item_vocab = sorted(df_all["parent_asin"].unique())

# 构造 feature lookup tables
user_feat_lookup = df_all.groupby("user_id").first()[[
    "user_interaction_count", "user_avg_rating", "user_last_timestamp", "user_avg_price"
]]
item_feat_lookup = df_all.groupby("parent_asin").first()[[
    "item_interaction_count", "item_avg_rating", "item_last_timestamp", "price", "price_missing",
    "title_length", "n_categories", "sub_category_id", "brand_id", "text_cluster_id_mpnet"
]]

# user -> true items
user_to_true = {}
for u, i in val_pairs:
    user_to_true.setdefault(int(u), set()).add(int(i))
print(f"  耗时 {time.time()-t0:.1f}s")

# [3/5] E2E pipeline (500 users)
print("\n[3/5] E2E (500 users)...")
t0 = time.time()

K = 200
recall10_list = []
recall50_list = []
recall_at_200_baseline = []
latencies = []

for ui in range(len(val_users)):
    t_start = time.time()
    u_idx = val_users[ui]
    u_emb = val_user_embs[ui:ui+1]
    
    # Recall: top-K via numpy
    scores = u_emb @ all_item_embs.T  # [1, 206K]
    top_idx = np.argpartition(-scores[0], K)[:K]
    top_idx = top_idx[np.argsort(-scores[0, top_idx])]
    
    # 构造 LightGBM 特征
    user_id = user_vocab[u_idx]
    if user_id not in user_feat_lookup.index:
        continue
    user_row = user_feat_lookup.loc[user_id]
    u_iact = user_row["user_interaction_count"]
    u_avg_r = user_row["user_avg_rating"]
    u_last_ts = user_row["user_last_timestamp"]
    u_avg_p = user_row["user_avg_price"]
    
    feats = []
    valid_indices = []
    for ci, item_idx in enumerate(top_idx):
        it = item_vocab[item_idx]
        if it not in item_feat_lookup.index:
            continue
        item_row = item_feat_lookup.loc[it]
        i_price = item_row["price"]
        feat = [
            u_iact, u_avg_r, u_last_ts,
            item_row["item_interaction_count"], item_row["item_avg_rating"],
            item_row["item_last_timestamp"],
            i_price, item_row["price_missing"], item_row["title_length"], item_row["n_categories"],
            u_avg_p,
            abs(u_avg_p - i_price) if pd.notna(u_avg_p) and pd.notna(i_price) else 0,
            item_row["item_interaction_count"] * u_iact,
            item_row["sub_category_id"], item_row["brand_id"], item_row["text_cluster_id_mpnet"],
        ]
        feats.append(feat)
        valid_indices.append(int(item_idx))
    
    if len(feats) > 0:
        feats_arr = np.array(feats, dtype=np.float32)
        rank_scores = booster.predict(feats_arr)
        sort_order = np.argsort(-rank_scores)
        final_top = [valid_indices[i] for i in sort_order]
    else:
        final_top = []
    
    latencies.append((time.time() - t_start) * 1000)
    
    true_items = user_to_true[int(u_idx)]
    if len(true_items) > 0:
        recall10_list.append(len(set(final_top[:10]) & true_items) / len(true_items))
        recall50_list.append(len(set(final_top[:50]) & true_items) / len(true_items))
        recall_at_200_baseline.append(len(set(top_idx.tolist()) & true_items) / len(true_items))
    
    if (ui + 1) % 100 == 0:
        print(f"  {ui+1}/{len(val_users)} | avg latency {np.mean(latencies):.1f}ms | "
              f"recall@10 so far {np.mean(recall10_list):.4f}")

# [4/5] 报告
print(f"\n[4/5] 结果")
print("=" * 70)
print(f"\n  测试样本: {len(latencies)} val users")
print(f"\n  延迟 (端到端: 召回 + 排序):")
print(f"    平均: {np.mean(latencies):.2f} ms")
print(f"    P50:  {np.percentile(latencies, 50):.2f} ms")
print(f"    P95:  {np.percentile(latencies, 95):.2f} ms")
print(f"    P99:  {np.percentile(latencies, 99):.2f} ms")
print(f"\n  质量 (random baseline = 0.0001):")
print(f"    Recall@10  (排序后): {np.mean(recall10_list):.4f}")
print(f"    Recall@50  (排序后): {np.mean(recall50_list):.4f}")
print(f"    Recall@200 (仅召回): {np.mean(recall_at_200_baseline):.4f}")
print(f"\n  Rank 价值:")
if np.mean(recall_at_200_baseline) > 0:
    rank_lift = (np.mean(recall10_list) / (np.mean(recall_at_200_baseline) * 10/200))
    print(f"    LightGBM 排序提升 (Recall@10 vs 随机选 10/200): {rank_lift:.2f}x")

# [5/5] 保存
results = {
    "n_test": len(latencies),
    "avg_latency_ms": float(np.mean(latencies)),
    "p50_latency_ms": float(np.percentile(latencies, 50)),
    "p95_latency_ms": float(np.percentile(latencies, 95)),
    "p99_latency_ms": float(np.percentile(latencies, 99)),
    "recall_at_10_after_rank": float(np.mean(recall10_list)),
    "recall_at_50_after_rank": float(np.mean(recall50_list)),
    "recall_at_200_recall_only": float(np.mean(recall_at_200_baseline)),
}
with open(OUT_DIR / "day9_pipeline_metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n✅ 保存: models/day9_pipeline_metrics.json")
print(json.dumps(results, indent=2))
