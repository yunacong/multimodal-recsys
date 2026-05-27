"""Quick 1 epoch test of Enhanced DIN V2"""
import sys, time, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from pathlib import Path

sys.path.insert(0, "src/din_pipeline")
from dataset_enhanced import EnhancedDINDataset, enhanced_collate_fn
from model_enhanced_v2 import EnhancedDINv2

DIN_DIR = Path("data/processed/din")
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# 关键: dataset 没有传 user_idx 给 collate! 需要修复
# 看 collate_fn 是否包含 user_idx
import dataset_enhanced
import inspect
src = inspect.getsource(dataset_enhanced.enhanced_collate_fn)
if "user_idx" not in src:
    print("⚠️ collate_fn 没传 user_idx, 修复中...")
    print("跑 cat src/din_pipeline/dataset_enhanced.py 看看")
else:
    print("✅ collate_fn 已包含 user_idx")

# 也要看 Dataset 类
data = np.load(DIN_DIR / "din_train_enhanced.npz")
print(f"npz 字段: {list(data.keys())}")
assert "user_idx" in data.files, "user_idx 缺失!"
