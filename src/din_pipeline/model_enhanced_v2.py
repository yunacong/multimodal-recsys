"""Enhanced DIN V2 - 加 user_emb"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_item_full_emb(item, sub_cat, brand, price,
                     item_emb, sub_cat_emb, brand_emb):
    item_e = item_emb(item)
    sub_e = sub_cat_emb(sub_cat)
    brand_e = brand_emb(brand)
    price_e = price.unsqueeze(-1)
    return torch.cat([item_e, sub_e, brand_e, price_e], dim=-1)


class EnhancedAttention(nn.Module):
    def __init__(self, full_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(full_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, target, history, mask):
        B, L, D = history.shape
        target_e = target.unsqueeze(1).expand(-1, L, -1)
        diff = target_e - history
        prod = target_e * history
        concat = torch.cat([target_e, history, diff, prod], dim=-1)
        scores = self.mlp(concat).squeeze(-1)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        return torch.sigmoid(scores) * mask.float()


class EnhancedDINv2(nn.Module):
    """Enhanced DIN V2: 加 user_id embedding"""
    def __init__(self, n_items=207386, n_sub_cats=32, n_brands=502, n_users=729576,
                 item_emb_dim=32, sub_cat_emb_dim=8, brand_emb_dim=16, user_emb_dim=32,
                 mlp_hidden=128, seq_len=20):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, item_emb_dim, padding_idx=0)
        self.sub_cat_emb = nn.Embedding(n_sub_cats, sub_cat_emb_dim)
        self.brand_emb = nn.Embedding(n_brands, brand_emb_dim)
        self.user_emb = nn.Embedding(n_users, user_emb_dim)  # 新加!
        
        self.item_full_dim = item_emb_dim + sub_cat_emb_dim + brand_emb_dim + 1  # 57
        self.attention = EnhancedAttention(full_dim=self.item_full_dim, hidden_dim=mlp_hidden)
        
        # MLP 输入: [target_item_full, user_interest, user_emb, target * user_interest]
        mlp_in = self.item_full_dim * 3 + user_emb_dim  # 57*3 + 32 = 203
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),  # 稍微加大 dropout
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.BatchNorm1d(mlp_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden // 2, 1),
        )
    
    def forward(self, batch):
        target_full = get_item_full_emb(
            batch["target"], batch["target_sub_cat"], batch["target_brand"], batch["target_price"],
            self.item_emb, self.sub_cat_emb, self.brand_emb,
        )
        history_full = get_item_full_emb(
            batch["history"], batch["history_sub_cat"], batch["history_brand"], batch["history_price"],
            self.item_emb, self.sub_cat_emb, self.brand_emb,
        )
        
        B, L = batch["history"].shape
        positions = torch.arange(L, device=target_full.device).unsqueeze(0).expand(B, -1)
        mask = (positions < batch["history_len"].unsqueeze(1)).float()
        
        attn = self.attention(target_full, history_full, mask)
        user_interest = (attn.unsqueeze(-1) * history_full).sum(dim=1)
        
        user_e = self.user_emb(batch["user_idx"])  # 新加!
        
        concat = torch.cat([
            target_full,
            user_interest,
            target_full * user_interest,
            user_e,
        ], dim=-1)
        return self.mlp(concat).squeeze(-1)
