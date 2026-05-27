"""Enhanced DIN Dataset V2 - 含 user_idx"""
import numpy as np
import torch
from torch.utils.data import Dataset


class EnhancedDINDatasetV2(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.user_idx = data["user_idx"]
        self.target = data["target"]
        self.target_sub_cat = data["target_sub_cat"]
        self.target_brand = data["target_brand"]
        self.target_price = data["target_price"]
        self.history = data["history"]
        self.history_sub_cat = data["history_sub_cat"]
        self.history_brand = data["history_brand"]
        self.history_price = data["history_price"]
        self.history_len = data["history_len"]
        self.label = data["label"]
        self.n = len(self.label)
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        return {
            "user_idx": self.user_idx[idx],
            "target": self.target[idx],
            "target_sub_cat": self.target_sub_cat[idx],
            "target_brand": self.target_brand[idx],
            "target_price": self.target_price[idx],
            "history": self.history[idx],
            "history_sub_cat": self.history_sub_cat[idx],
            "history_brand": self.history_brand[idx],
            "history_price": self.history_price[idx],
            "history_len": self.history_len[idx],
            "label": self.label[idx],
        }


def enhanced_collate_fn_v2(batch):
    return {
        "user_idx": torch.tensor([b["user_idx"] for b in batch], dtype=torch.long),
        "target": torch.tensor([b["target"] for b in batch], dtype=torch.long),
        "target_sub_cat": torch.tensor([b["target_sub_cat"] for b in batch], dtype=torch.long),
        "target_brand": torch.tensor([b["target_brand"] for b in batch], dtype=torch.long),
        "target_price": torch.tensor([b["target_price"] for b in batch], dtype=torch.float32),
        "history": torch.tensor(np.stack([b["history"] for b in batch]), dtype=torch.long),
        "history_sub_cat": torch.tensor(np.stack([b["history_sub_cat"] for b in batch]), dtype=torch.long),
        "history_brand": torch.tensor(np.stack([b["history_brand"] for b in batch]), dtype=torch.long),
        "history_price": torch.tensor(np.stack([b["history_price"] for b in batch]), dtype=torch.float32),
        "history_len": torch.tensor([b["history_len"] for b in batch], dtype=torch.long),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.float32),
    }
