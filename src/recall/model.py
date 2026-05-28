"""双塔召回模型 (User Tower + Item Tower)

In-batch negatives + temperature scaling
L2-normalized embeddings (cosine similarity)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    def __init__(self, n_users, user_dense_dim, user_id_emb_dim=32, dense_proj=16, out_dim=64):
        super().__init__()
        self.user_id_emb = nn.Embedding(n_users, user_id_emb_dim)
        self.dense_proj = nn.Linear(user_dense_dim, dense_proj)
        input_dim = user_id_emb_dim + dense_proj
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, out_dim),
        )
    
    def forward(self, user_idx, user_dense):
        uid_e = self.user_id_emb(user_idx)
        dense_e = F.relu(self.dense_proj(user_dense))
        x = torch.cat([uid_e, dense_e], dim=-1)
        emb = self.mlp(x)
        return F.normalize(emb, p=2, dim=-1)


class ItemTower(nn.Module):
    def __init__(self, n_items, item_dense_dim,
                 n_sub_cats, n_brands, n_text_clusters,
                 item_id_emb_dim=32, dense_proj=16,
                 sub_cat_dim=8, brand_dim=16, text_dim=8,
                 out_dim=64):
        super().__init__()
        self.item_id_emb = nn.Embedding(n_items, item_id_emb_dim)
        self.sub_cat_emb = nn.Embedding(n_sub_cats, sub_cat_dim)
        self.brand_emb = nn.Embedding(n_brands, brand_dim)
        self.text_emb = nn.Embedding(n_text_clusters, text_dim)
        self.dense_proj = nn.Linear(item_dense_dim, dense_proj)
        
        input_dim = item_id_emb_dim + sub_cat_dim + brand_dim + text_dim + dense_proj
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, out_dim),
        )
    
    def forward(self, item_idx, item_dense, item_cat):
        iid_e = self.item_id_emb(item_idx)
        sub_e = self.sub_cat_emb(item_cat[:, 0])
        br_e = self.brand_emb(item_cat[:, 1])
        tx_e = self.text_emb(item_cat[:, 2])
        dense_e = F.relu(self.dense_proj(item_dense))
        x = torch.cat([iid_e, sub_e, br_e, tx_e, dense_e], dim=-1)
        emb = self.mlp(x)
        return F.normalize(emb, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, n_users, n_items, user_dense_dim, item_dense_dim,
                 n_sub_cats, n_brands, n_text_clusters, temperature=0.1):
        super().__init__()
        self.user_tower = UserTower(n_users, user_dense_dim)
        self.item_tower = ItemTower(n_items, item_dense_dim,
                                    n_sub_cats, n_brands, n_text_clusters)
        self.temperature = temperature
    
    def forward(self, user_idx, user_dense, item_idx, item_dense, item_cat):
        user_emb = self.user_tower(user_idx, user_dense)
        item_emb = self.item_tower(item_idx, item_dense, item_cat)
        return user_emb, item_emb
    
    def in_batch_loss(self, user_emb, item_emb):
        """In-batch negatives + softmax"""
        # 每个 user 对 batch 内所有 item 算分
        logits = user_emb @ item_emb.T / self.temperature  # [B, B]
        labels = torch.arange(user_emb.size(0), device=user_emb.device)
        return F.cross_entropy(logits, labels)
