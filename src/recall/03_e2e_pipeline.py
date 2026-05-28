"""Day 9 Phase 5 - 召回 + 排序 端到端 pipeline"""
import sys, time, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb

sys.path.insert(0, str(Path("src/recall")))
from model import TwoTowerModel

RECALL_DIR = Path("data/processed/recall")
DATA_PROC = Path("data/processed")
OUT_DIR = Path("models")

device = "cpu"
print("=" * 70)
print("Day 9 Phase 5 - 召回 + 排序 端到端 pipeline")
print("=" * 70)
print(f"Device: {device}")

# Step 1: 加载双塔模型 + LightGBM
print("\n[1/5] 加载模型...")
t0 = time.time()
with open(RECALL_DIR / "meta.json") as f:
    meta = json.load(f)

model = TwoTowerModel(
    n_users=meta["n_users"], n_items=meta["n_items"],
    user_dense_dim=meta["user_dense_dim"], item_dense_dim=meta["item_dense_dim"],
    n_sub_cats=meta["n_sub_cats"], n_brands=meta["n_brands"], n_text_clusters=meta["n_text_clusters"],
).to(device)
ckpt = torch.load(OUT_DIR / "twotower_best.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()

# LightGBM v3-mpnet
lgb_path = None
for p in OUT_DIR.glob("*.txt"):
    if "v3" in p.name and "mpnet" in p.name:
        lgb_path = p
        break
if lgb_path is None:
    for p in OUT_DIR.glob("*v3*.txt"):
        lgb_path = p
        break
if lgb_path is None:
    print("  ⚠️ 找不到 v3 模型, 列出 models/")
    for p in OUT_DIR.iterdir():
        print(f"    {p.name}")
    sys.exit(1)
print(f"  LightGBM: {lgb_path.name}")
booster = lgb.Booster(model_file=str(lgb_path))
print(f"  耗时: {time.time()-t0:.1f}s")

# Step 2: 加载所有需要的数据 (item embs + LightGBM 特征)
print("\n[2/5] 准备数据...")
t0 = time.time()
all_item_embs = np.load(OUT_DIR / "item_embeddings.npy")
print(f"  item_embs: {all_item_embs.shape}")

user_features = np.load(RECALL_DIR / "user_features.npz")["feat"]
item_data = np.load(RECALL_DIR / "item_features.npz")
item_dense = item_data["dense"]
item_cat = item_data["cat"]
val_pairs = np.load(RECALL_DIR / "val_pairs.npz")["pairs"]

# 加载完整 train_with_all_features 用于 LightGBM 特征 (含 cross features)
df_all = pd.read_parquet(DATA_PROC / "train_with_all_features.parquet")
text_clusters = pd.read_csv(DATA_PROC / "item_text_clusters_mpnet.csv")
df_all = df_all.merge(text_clusters, on="parent_asin", how="left")
df_all["text_cluster_id_mpnet"] = df_all["text_cluster_id_mpnet"].fillna(0).astype(np.int16)
print(f"  df_all: {df_all.shape}")

# 用 vocab 反查 user_id / parent_asin
user_vocab = sorted(df_all["user_id"].unique())
item_vocab = sorted(df_all["parent_asin"].unique())
print(f"  耗时: {time.time()-t0:.1f}s")

# Step 3: 选 500 个 val users 做端到端测试
print("\n[3/5] 端到端 pipeline (500 个 val user)...")
t0 = time.time()

# 取 unique val users 中前 500 个
unique_val_users = np.unique(val_pairs[:, 0])[:500]
print(f"  test users: {len(unique_val_users)}")

# user -> true items (用于评估)
user_to_true = {}
for u, i in val_pairs:
    user_to_true.setdefault(int(u), set()).add(int(i))

K = 200
ndcg10_list = []
recall10_list = []
recall50_list = []
recall_at_200_baseline = []  # 召回的 Recall (没经过排序)
latencies = []

# LightGBM 特征列 (16 维, 和 v3 一样)
LGB_FEATURES = [
    "user_interaction_count", "user_avg_rating", "user_last_timestamp",
    "item_interaction_count", "item_avg_rating", "item_last_timestamp",
    "price", "price_missing", "title_length", "n_categories",
    "user_avg_price", "user_price_diff", "pop_x_activity",
    "sub_category_id", "brand_id", "text_cluster_id_mpnet",
]

# 为快速 lookup 构建 (user_id, item_id) -> 特征行
df_all["user_id_str"] = df_all["user_id"]
df_all["item_id_str"] = df_all["parent_asin"]

# 每个 user 取一行 user 特征
user_feat_lookup = df_all.groupby("user_id").first()[["user_interaction_count", "user_avg_rating",
                                                       "user_last_timestamp", "user_avg_price"]]
# 每个 item 取一行 item 特征
item_feat_lookup = df_all.groupby("parent_asin").first()[["item_interaction_count", "item_avg_rating",
                                                           "item_last_timestamp", "price", "price_missing",
                                                           "title_length", "n_categories",
                                                           "sub_category_id", "brand_id", "text_cluster_id_mpnet"]]
print(f"  特征 lookup 表构建完成, {time.time()-t0:.1f}s")

# 端到端测试
print("\n[4/5] 测试 500 user 的端到端...")
t0 = time.time()
for ui, u_idx in enumerate(unique_val_users):
    t_start = time.time()
    
    # 4.1: User Tower 编码
    with torch.no_grad():
        u_emb_t = model.user_tower(
            torch.tensor([u_idx], device=device, dtype=torch.long),
            torch.from_numpy(user_features[u_idx:u_idx+1]).float().to(device),
        )
        u_emb = u_emb_t.cpu().numpy()
    
    # 4.2: 召回 top-200
    scores = u_emb @ all_item_embs.T  # [1, 206K]
    top_idx = np.argpartition(-scores[0], K)[:K]
    top_idx = top_idx[np.argsort(-scores[0, top_idx])]
    
    # 4.3: 构造 LightGBM 特征
    user_id = user_vocab[u_idx]
    item_ids = [item_vocab[i] for i in top_idx]
    
    user_row = user_feat_lookup.loc[user_id]
    feats = []
    for it in item_ids:
        if it not in item_feat_lookup.index:
            feats.append(None)
            continue
        item_row = item_feat_lookup.loc[it]
        u_avg_p = user_row["user_avg_price"]
        i_price = item_row["price"]
        feat = [
            user_row["user_interaction_count"], user_row["user_avg_rating"],
            user_row["user_last_timestamp"],
            item_row["item_interaction_count"], item_row["item_avg_rating"],
            item_row["item_last_timestamp"],
            i_price, item_row["price_missing"], item_row["title_length"], item_row["n_categories"],
            u_avg_p, abs(u_avg_p - i_price) if pd.notna(u_avg_p) and pd.notna(i_price) else 0,
            item_row["item_interaction_count"] * user_row["user_interaction_count"],
            item_row["sub_category_id"], item_row["brand_id"], item_row["text_cluster_id_mpnet"],
        ]
        feats.append(feat)
    
    # 过滤 None
    valid_idx = [i for i, f in enumerate(feats) if f is not None]
    valid_top_idx = [top_idx[i] for i in valid_idx]
    valid_feats = np.array([feats[i] for i in valid_idx], dtype=np.float32)
    
    # 4.4: LightGBM 打分
    if len(valid_feats) > 0:
        rank_scores = booster.predict(valid_feats)
        # 排序
        sort_order = np.argsort(-rank_scores)
        final_top = [valid_top_idx[i] for i in sort_order]
    else:
        final_top = []
    
    latencies.append((time.time() - t_start) * 1000)
    
    # 评估
    true_items = user_to_true[int(u_idx)]
    if len(true_items) > 0:
        # Recall@10 (排序后)
        retrieved_10 = set(final_top[:10])
        recall10_list.append(len(retrieved_10 & true_items) / len(true_items))
        # Recall@50 (排序后)
        retrieved_50 = set(final_top[:50])
        recall50_list.append(len(retrieved_50 & true_items) / len(true_items))
        # Recall@200 (召回, 不排序)
        retrieved_200 = set(top_idx.tolist())
        recall_at_200_baseline.append(len(retrieved_200 & true_items) / len(true_items))
    
    if (ui + 1) % 100 == 0:
        print(f"  {ui+1}/500 | avg latency {np.mean(latencies):.1f}ms")

print(f"\n[5/5] 端到端测试完成, 总耗时 {time.time()-t0:.1f}s")

# 报告
print("\n" + "=" * 70)
print("Day 9 端到端 Recall + Rank Pipeline 结果")
print("=" * 70)
print(f"\n  测试样本: {len(latencies)} val users")
print(f"\n  延迟:")
print(f"    平均: {np.mean(latencies):.2f} ms/query")
print(f"    P50:  {np.percentile(latencies, 50):.2f} ms")
print(f"    P95:  {np.percentile(latencies, 95):.2f} ms")
print(f"    P99:  {np.percentile(latencies, 99):.2f} ms")
print(f"\n  质量 (vs random=0.0001):")
print(f"    Recall@10  (after rank): {np.mean(recall10_list):.4f}")
print(f"    Recall@50  (after rank): {np.mean(recall50_list):.4f}")
print(f"    Recall@200 (recall only): {np.mean(recall_at_200_baseline):.4f}")
print(f"\n  关键发现:")
print(f"    召回 + 排序的 Recall@10 vs 单独召回 Recall@200:")
print(f"    {np.mean(recall10_list):.4f} (10) vs {np.mean(recall_at_200_baseline):.4f} (200)")

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
print(f"\n保存: models/day9_pipeline_metrics.json")
