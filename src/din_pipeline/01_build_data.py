"""Day 8 - DIN 数据 Pipeline (Last-out 切分)

切分策略:
  按 user 分组 + 按 timestamp 排序
  最后 1 个交互 -> test
  倒数第 2 个 -> val
  其余 -> train
"""
import os, time, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/Users/wangfengyuan/projects/multimodal-recsys")
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROC = PROJECT_ROOT / "data" / "processed"
DIN_DIR = DATA_PROC / "din"
DIN_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 20
NEG_RATIO = 5
POS_THRESHOLD = 4
SEED = 42

print("=" * 70)
print("Day 8 - DIN 数据 Pipeline (Last-out 切分)")
print("=" * 70)

# Step 1: 加载
print("\nStep 1: 加载原始 train.csv")
t0 = time.time()
df_all = pd.read_csv(DATA_RAW / "BPC_5core_train.csv")
print(f"  原始 train: {len(df_all):,} 行")
print(f"  耗时: {time.time()-t0:.1f}s")

# Step 2: Last-out 切分
print("\nStep 2: Last-out 切分")
t0 = time.time()
df_all = df_all.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

# 用 cumcount + size 给每个 user 的交互编号 (从后往前)
df_all["interaction_rank_desc"] = df_all.groupby("user_id").cumcount(ascending=False)
# rank=0 是最后一个, rank=1 倒数第二

test_df = df_all[df_all["interaction_rank_desc"] == 0].copy()
val_df = df_all[df_all["interaction_rank_desc"] == 1].copy()
train_df = df_all[df_all["interaction_rank_desc"] >= 2].copy()

# 删除辅助列
for d in [test_df, val_df, train_df]:
    d.drop(columns=["interaction_rank_desc"], inplace=True)
    d.reset_index(drop=True, inplace=True)

print(f"  ✅ train: {len(train_df):,} ({len(train_df)/len(df_all)*100:.1f}%)")
print(f"  ✅ val:   {len(val_df):,} ({len(val_df)/len(df_all)*100:.1f}%)")
print(f"  ✅ test:  {len(test_df):,} ({len(test_df)/len(df_all)*100:.1f}%)")
print(f"  耗时: {time.time()-t0:.1f}s")

# 注意: 部分 user 只有 1-2 个交互, 这些用户的所有交互在 train (因为 rank >= 2 才进 train)
# 验证: train + val + test = 总数
assert len(train_df) + len(val_df) + len(test_df) == len(df_all)

# Step 3: vocab
print("\nStep 3: 构建 vocab")
t0 = time.time()
all_items = df_all["parent_asin"].unique()
item2idx = {item: i + 1 for i, item in enumerate(sorted(all_items))}
n_items = len(item2idx)
all_users = df_all["user_id"].unique()
user2idx = {user: i for i, user in enumerate(sorted(all_users))}
n_users = len(user2idx)
print(f"  items: {n_items:,}, users: {n_users:,}")
pd.DataFrame([{"item_id": k, "idx": v} for k, v in item2idx.items()]).to_csv(DIN_DIR / "item_vocab.csv", index=False)
pd.DataFrame([{"user_id": k, "idx": v} for k, v in user2idx.items()]).to_csv(DIN_DIR / "user_vocab.csv", index=False)
print(f"  耗时: {time.time()-t0:.1f}s")

# Step 4: 生成样本 (关键修改: train 含历史 + 负采样, val/test 用 train 的完整历史)
def build_samples(df, item2idx, user2idx, seq_len, with_neg, neg_ratio, name, history_source=None):
    """
    df: 当前要生成样本的数据 (train/val/test)
    history_source: 用于构建历史的数据 (val/test 应该用 train 的完整历史)
                    如果 None, 用 df 自己 (train 用自己)
    """
    print(f"\n  [{name}] 准备...")
    t0 = time.time()
    
    # 先用 history_source 构建用户历史 (按时间排序后扫一遍)
    if history_source is None:
        history_source = df
    
    print(f"  [{name}] 用 {len(history_source):,} 行构建用户历史...")
    hist_src = history_source.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    hist_src["item_idx"] = hist_src["parent_asin"].map(item2idx)
    hist_src = hist_src.dropna(subset=["item_idx"])
    hist_src["item_idx"] = hist_src["item_idx"].astype(np.int32)
    
    user_history = defaultdict(list)
    user_seen = defaultdict(set)
    src_users = hist_src["user_id"].values
    src_items = hist_src["item_idx"].values
    for j in range(len(hist_src)):
        u = src_users[j]
        i = src_items[j]
        user_history[u].append(i)
        user_seen[u].add(i)
    print(f"  [{name}] 历史构建完成, 耗时 {time.time()-t0:.1f}s")
    
    # 现在处理 df 本身
    t0 = time.time()
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    df["user_idx"] = df["user_id"].map(user2idx)
    df["target_idx"] = df["parent_asin"].map(item2idx)
    df = df.dropna(subset=["user_idx", "target_idx"])
    df["user_idx"] = df["user_idx"].astype(np.int32)
    df["target_idx"] = df["target_idx"].astype(np.int32)
    
    user_ids_arr = df["user_idx"].values
    target_ids_arr = df["target_idx"].values
    user_str_arr = df["user_id"].values
    ratings_arr = df["rating"].values
    timestamps_arr = df["timestamp"].values.astype(np.int64)
    n_orig = len(df)
    
    n_pos = int((ratings_arr >= POS_THRESHOLD).sum())
    n_neg = n_pos * neg_ratio if with_neg else 0
    total = n_orig + n_neg
    print(f"  [{name}] 输入 {n_orig:,}, 预计输出 {total:,}")
    
    out_user = np.zeros(total, dtype=np.int32)
    out_target = np.zeros(total, dtype=np.int32)
    out_history = np.zeros((total, seq_len), dtype=np.int32)
    out_hist_len = np.zeros(total, dtype=np.int8)
    out_label = np.zeros(total, dtype=np.int8)
    out_ts = np.zeros(total, dtype=np.int64)
    
    rng = np.random.default_rng(SEED)
    all_item_idxs = np.array(list(item2idx.values()), dtype=np.int32)
    
    print(f"  [{name}] 生成样本...")
    t0 = time.time()
    out_idx = 0
    log_interval = max(1, n_orig // 20)
    
    # 对于 train, 我们要在遍历过程中"动态构建历史" (因为 train 自己就是历史源)
    # 对于 val/test, 历史已经全部从 train 构建好了, 不需要更新
    update_history = (history_source is df) or (history_source is None and name == "train")
    
    for i in range(n_orig):
        u_idx = user_ids_arr[i]
        u_str = user_str_arr[i]
        i_idx = target_ids_arr[i]
        rating = ratings_arr[i]
        ts = timestamps_arr[i]
        
        hist = user_history[u_str][-seq_len:]
        hist_len = len(hist)
        
        out_user[out_idx] = u_idx
        out_target[out_idx] = i_idx
        if hist_len > 0:
            out_history[out_idx, :hist_len] = hist
        out_hist_len[out_idx] = hist_len
        out_label[out_idx] = 1 if rating >= POS_THRESHOLD else 0
        out_ts[out_idx] = ts
        out_idx += 1
        
        if with_neg and rating >= POS_THRESHOLD:
            neg_cands = rng.choice(all_item_idxs, size=neg_ratio * 2, replace=True)
            n_added = 0
            for neg in neg_cands:
                if neg not in user_seen[u_str]:
                    out_user[out_idx] = u_idx
                    out_target[out_idx] = neg
                    if hist_len > 0:
                        out_history[out_idx, :hist_len] = hist
                    out_hist_len[out_idx] = hist_len
                    out_label[out_idx] = 0
                    out_ts[out_idx] = ts
                    out_idx += 1
                    n_added += 1
                    if n_added >= neg_ratio:
                        break
        
        # 仅 train 需要在遍历中更新自己的历史
        if update_history:
            # 注意: 在 last-out 切分下, train 已经 sort 过, 这里再 append 会保持顺序
            # 但其实 train 的历史已经全部加进 user_history 了 (在准备阶段)
            # 所以这里不需要再 append
            pass
        
        if (i + 1) % log_interval == 0:
            pct = (i + 1) / n_orig * 100
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_orig - i - 1)
            print(f"    [{name}] {pct:>5.1f}% | 已用 {elapsed:.0f}s | ETA {eta:.0f}s | 输出 {out_idx:,}")
    
    print(f"  [{name}] ✅ 完成: {out_idx:,} 行 (耗时 {time.time()-t0:.1f}s)")
    return {
        "user_idx": out_user[:out_idx],
        "target": out_target[:out_idx],
        "history": out_history[:out_idx],
        "history_len": out_hist_len[:out_idx],
        "label": out_label[:out_idx],
        "timestamp": out_ts[:out_idx],
    }

# Step 5: 生成 (val/test 用 train 作为历史源!)
print("\nStep 4: 生成 train (历史源=train, 含 1:5 负采样)")
print("-" * 70)
din_train = build_samples(train_df, item2idx, user2idx, SEQ_LEN, True, NEG_RATIO, "train", history_source=train_df)

print("\nStep 5: 生成 val (历史源=train, 无负采样)")
print("-" * 70)
din_val = build_samples(val_df, item2idx, user2idx, SEQ_LEN, False, 0, "val", history_source=train_df)

print("\nStep 6: 生成 test (历史源=train+val, 无负采样)")
print("-" * 70)
train_and_val = pd.concat([train_df, val_df], ignore_index=True)
din_test = build_samples(test_df, item2idx, user2idx, SEQ_LEN, False, 0, "test", history_source=train_and_val)

# Step 7: 保存
print("\nStep 7: 保存")
t0 = time.time()
np.savez_compressed(DIN_DIR / "din_train.npz", **din_train)
np.savez_compressed(DIN_DIR / "din_val.npz", **din_val)
np.savez_compressed(DIN_DIR / "din_test.npz", **din_test)
print(f"  保存耗时: {time.time()-t0:.1f}s")

for name, d in [("train", din_train), ("val", din_val), ("test", din_test)]:
    p = DIN_DIR / f"din_{name}.npz"
    print(f"  din_{name}.npz: {len(d['label']):>10,} 行, {p.stat().st_size/1e6:.0f} MB, pos {d['label'].mean():.3f}")

meta = {
    "split_strategy": "last_out",
    "seq_len": SEQ_LEN, "neg_ratio": NEG_RATIO, "pos_threshold": POS_THRESHOLD,
    "n_items": n_items, "n_users": n_users,
    "n_train_orig": int(len(train_df)),
    "n_val_orig": int(len(val_df)),
    "n_test_orig": int(len(test_df)),
    "n_train_din": int(len(din_train["label"])),
    "n_val_din": int(len(din_val["label"])),
    "n_test_din": int(len(din_test["label"])),
    "train_pos_ratio": float(din_train["label"].mean()),
    "val_pos_ratio": float(din_val["label"].mean()),
}
with open(DIN_DIR / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\n" + "=" * 70)
print("🎉 DIN 数据 Pipeline 完成!")
print("=" * 70)
print(json.dumps(meta, indent=2))
