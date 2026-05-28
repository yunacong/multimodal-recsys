"""DeepFM Model

DeepFM = Wide (FM) + Deep (MLP)
        Shared embedding for categorical features

Reference: "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction" (Guo et al, 2017)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepFM(nn.Module):
    def __init__(self,
                 n_dense=13,
                 cat_vocab_sizes=[31, 501, 50],  # sub_cats, brands, text_clusters
                 emb_dim=16,
                 mlp_hidden=(256, 128, 64),
                 dropout=0.3):
        super().__init__()
        self.n_dense = n_dense
        self.n_cat = len(cat_vocab_sizes)
        self.emb_dim = emb_dim
        
        # Categorical embedding (shared between Wide and Deep)
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, emb_dim) for vocab_size in cat_vocab_sizes
        ])
        # 1st order weight (FM linear part)
        self.cat_linear = nn.ModuleList([
            nn.Embedding(vocab_size, 1) for vocab_size in cat_vocab_sizes
        ])
        
        # Dense linear (FM linear part)
        self.dense_linear = nn.Linear(n_dense, 1)
        # Dense → emb 投影 (用于 Deep 部分)
        self.dense_proj = nn.Linear(n_dense, emb_dim)
        
        # Deep MLP
        deep_input_dim = (self.n_cat + 1) * emb_dim  # cat_embs + dense_proj
        layers = []
        prev_dim = deep_input_dim
        for h in mlp_hidden:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.deep = nn.Sequential(*layers)
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, dense, cat):
        """
        dense: [B, n_dense] float
        cat:   [B, n_cat]   long
        
        Returns: logits [B]
        """
        B = dense.size(0)
        
        # ===== FM (Wide) part =====
        # 1st order
        first_order = self.dense_linear(dense).squeeze(-1)  # [B]
        for i, emb_layer in enumerate(self.cat_linear):
            first_order = first_order + emb_layer(cat[:, i]).squeeze(-1)
        
        # 2nd order: cat embeddings pair-wise
        cat_embs = torch.stack([
            emb_layer(cat[:, i]) for i, emb_layer in enumerate(self.cat_embeddings)
        ], dim=1)  # [B, n_cat, emb_dim]
        
        # FM 2nd order: 0.5 * (sum^2 - sum_of_squares)
        sum_emb = cat_embs.sum(dim=1)  # [B, emb_dim]
        square_sum = (sum_emb ** 2).sum(dim=-1)  # [B]
        sum_square = (cat_embs ** 2).sum(dim=1).sum(dim=-1)  # [B]
        second_order = 0.5 * (square_sum - sum_square)  # [B]
        
        # ===== Deep part =====
        dense_emb = self.dense_proj(dense)  # [B, emb_dim]
        # Concat all embeddings
        deep_input = torch.cat([
            cat_embs.flatten(start_dim=1),  # [B, n_cat * emb_dim]
            dense_emb,                       # [B, emb_dim]
        ], dim=-1)
        deep_out = self.deep(deep_input).squeeze(-1)  # [B]
        
        # ===== Combine =====
        logits = first_order + second_order + deep_out + self.bias
        return logits
