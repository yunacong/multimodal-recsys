"""Enhanced DIN - 多特征 attention"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_full_emb(target_item, sub_cat, brand, price,
                 item_emb_layer, sub_cat_emb_layer, brand_emb_layer):
    """每个 item 的完整 emb: [item_emb, sub_cat_emb, brand_emb, price] = D + 8 + 16 + 1"""
    item_e = item_emb_layer(target_item)
    sub_e = sub_cat_emb_layer(sub_cat)
    brand_e = brand_emb_layer(brand)
    # price 增加一个 dim
    if price.dim() == 1:
        price_e = price.unsqueeze(-1)
    else:
        price_e = price.unsqueeze(-1)
    return torch.cat([item_e, sub_e, brand_e, price_e], dim=-1)


class EnhancedAttention(nn.Module):
    def __init__(self, full_dim, hidden_dim=64):
        super().__init__()
        # 输入是 [target, history, target-history, target*history] = 4 * full_dim
        self.mlp = nn.Sequential(
            nn.Linear(full_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, target, history, mask):
        B, L, D = history.shape
        target_expanded = target.unsqueeze(1).expand(-1, L, -1)
        diff = target_expanded - history
        prod = target_expanded * history
        concat = torch.cat([target_expanded, history, diff, prod], dim=-1)
        scores = self.mlp(concat).squeeze(-1)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.sigmoid(scores) * mask.float()
        return weights


class EnhancedDIN(nn.Module):
    def __init__(self, n_items=207386, n_sub_cats=32, n_brands=502,
                 item_emb_dim=32, sub_cat_emb_dim=8, brand_emb_dim=16,
                 mlp_hidden=128, seq_len=20):
        super().__init__()
        self.n_items = n_items
        self.seq_len = seq_len
        
        # Embedding 层
        self.item_emb = nn.Embedding(n_items, item_emb_dim, padding_idx=0)
        self.sub_cat_emb = nn.Embedding(n_sub_cats, sub_cat_emb_dim)
        self.brand_emb = nn.Embedding(n_brands, brand_emb_dim)
        
        # Full dim = item + sub_cat + brand + price(1) = 32 + 8 + 16 + 1 = 57
        self.full_dim = item_emb_dim + sub_cat_emb_dim + brand_emb_dim + 1
        
        # Attention
        self.attention = EnhancedAttention(full_dim=self.full_dim, hidden_dim=mlp_hidden)
        
        # Main MLP: [target_full, user_interest, target_full * user_interest] = 3 * full_dim = 171
        self.mlp = nn.Sequential(
            nn.Linear(self.full_dim * 3, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.BatchNorm1d(mlp_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(mlp_hidden // 2, 1),
        )
    
    def forward(self, batch):
        # Target full emb [B, D]
        target_full = get_full_emb(
            batch["target"],
            batch["target_sub_cat"],
            batch["target_brand"],
            batch["target_price"],
            self.item_emb, self.sub_cat_emb, self.brand_emb,
        )
        
        # History full emb [B, L, D]
        # 注意: history_price 已经是 (B, L), 需要传 (B, L) shape
        B, L = batch["history"].shape
        history_full = get_full_emb(
            batch["history"],
            batch["history_sub_cat"],
            batch["history_brand"],
            batch["history_price"],
            self.item_emb, self.sub_cat_emb, self.brand_emb,
        )
        # history_full shape: [B, L, D]
        
        # Mask
        positions = torch.arange(L, device=target_full.device).unsqueeze(0).expand(B, -1)
        mask = (positions < batch["history_len"].unsqueeze(1)).float()
        
        # Attention
        attn_weights = self.attention(target_full, history_full, mask)  # [B, L]
        
        # Weighted sum
        weighted = attn_weights.unsqueeze(-1) * history_full  # [B, L, D]
        user_interest = weighted.sum(dim=1)  # [B, D]
        
        # Concat + MLP
        concat = torch.cat([target_full, user_interest, target_full * user_interest], dim=-1)
        logits = self.mlp(concat).squeeze(-1)
        return logits
