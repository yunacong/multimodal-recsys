"""DIN Dataset - 读取 npz 并 yield batch"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class DINDataset(Dataset):
    """
    DIN 数据集
    
    每个样本包含:
      user_idx:      int (单值)
      target:        int (单值, 目标商品 idx)
      history:       (L,) int  (历史 item idx, 0 是 padding)
      history_len:   int (真实历史长度, 0~L)
      label:         int 0/1
    """
    
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.user_idx = data["user_idx"]
        self.target = data["target"]
        self.history = data["history"]
        self.history_len = data["history_len"]
        self.label = data["label"]
        self.n = len(self.label)
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        return {
            "user_idx": self.user_idx[idx],
            "target": self.target[idx],
            "history": self.history[idx],
            "history_len": self.history_len[idx],
            "label": self.label[idx],
        }


def collate_fn(batch):
    """把 dict list 合并成 batch tensor"""
    return {
        "user_idx": torch.tensor([b["user_idx"] for b in batch], dtype=torch.long),
        "target": torch.tensor([b["target"] for b in batch], dtype=torch.long),
        "history": torch.tensor(np.stack([b["history"] for b in batch]), dtype=torch.long),
        "history_len": torch.tensor([b["history_len"] for b in batch], dtype=torch.long),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.float32),
    }


def get_loader(npz_path, batch_size=512, shuffle=True, num_workers=0):
    """快捷创建 DataLoader"""
    ds = DINDataset(npz_path)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )


if __name__ == "__main__":
    # 简单测试
    print("Testing DIN Dataset...")
    DIN_DIR = Path("data/processed/din")
    ds = DINDataset(DIN_DIR / "din_val.npz")
    print(f"  Dataset size: {len(ds):,}")
    print(f"  First sample:")
    sample = ds[0]
    for k, v in sample.items():
        print(f"    {k}: {v}")
    
    print("\nTesting DataLoader...")
    loader = get_loader(DIN_DIR / "din_val.npz", batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print(f"  Batch shapes:")
    for k, v in batch.items():
        print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
