"""DIN (Deep Interest Network) 模型
Reference: "Deep Interest Network for Click-Through Rate Prediction" (Zhou et al, 2018)

核心思想:
  传统模型: 用户历史 = 简单 average pooling (所有历史 item 平等对待)
  DIN:      attention 根据 target 商品动态加权用户历史
            "买礼物时" attention 历史中买过的礼物
            "买零食时" attention 历史中买过的零食
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionUnit(nn.Module):
    """
    DIN Attention Unit
    
    给定 target_emb 和 history_emb, 计算每个历史 item 对 target 的相关度
    
    Inputs:
      target:  [B, D]    单个 target 商品向量
      history: [B, L, D] 用户历史 item 向量
      mask:    [B, L]    padding mask (1=真实, 0=padding)
    
    Output:
      attention_weights: [B, L]
    """
    
    def __init__(self, emb_dim=32, hidden_dim=64):
        super().__init__()
        # DIN 论文做法: 输入是 [target, history, target-history, target*history] (4*D)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, target, history, mask):
        # target: [B, D] -> [B, 1, D] -> [B, L, D] (broadcast)
        B, L, D = history.shape
        target_expanded = target.unsqueeze(1).expand(-1, L, -1)  # [B, L, D]
        
        # 4 种交互特征
        diff = target_expanded - history          # [B, L, D]
        prod = target_expanded * history          # [B, L, D]
        concat = torch.cat([target_expanded, history, diff, prod], dim=-1)  # [B, L, 4*D]
        
        # MLP 算分
        scores = self.mlp(concat).squeeze(-1)     # [B, L]
        
        # 屏蔽 padding (设成 -inf, softmax 后 = 0)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        
        # DIN 论文: 不做 softmax (保留绝对值)
        # 但工业实现通常做 softmax 让权重 sum=1
        # 我们这里折中: sigmoid (允许多个 high weight)
        weights = torch.sigmoid(scores)            # [B, L]
        weights = weights * mask.float()           # padding 处归零
        
        return weights


class DIN(nn.Module):
    """
    Deep Interest Network
    
    Args:
      n_items: 总商品数 (vocab size, 含 padding=0)
      emb_dim: 商品 embedding 维度
      mlp_hidden: 主 MLP 隐藏层维度
      seq_len:  历史长度
    """
    
    def __init__(self, n_items=207386, emb_dim=32, mlp_hidden=64, seq_len=20):
        super().__init__()
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        
        # Item embedding (idx 0 是 padding)
        self.item_emb = nn.Embedding(n_items, emb_dim, padding_idx=0)
        
        # Attention unit
        self.attention = AttentionUnit(emb_dim=emb_dim, hidden_dim=mlp_hidden)
        
        # 主 MLP: 输入 [target, user_interest, target*user_interest] = 3*D
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 3, mlp_hidden),
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
        """
        batch (dict):
          target:      [B]    long
          history:     [B, L] long  
          history_len: [B]    long
        
        Returns:
          logits:      [B]   (未 sigmoid, 用 BCEWithLogitsLoss)
        """
        target = batch["target"]       # [B]
        history = batch["history"]     # [B, L]
        hist_len = batch["history_len"]  # [B]
        
        B = target.size(0)
        L = history.size(1)
        
        # 1. Item embedding
        target_emb = self.item_emb(target)        # [B, D]
        history_emb = self.item_emb(history)      # [B, L, D]
        
        # 2. Padding mask: [B, L], 1 = 真实 history, 0 = padding
        # 用 history_len 构造: position i 是 mask=1 if i < hist_len
        positions = torch.arange(L, device=target.device).unsqueeze(0).expand(B, -1)  # [B, L]
        mask = (positions < hist_len.unsqueeze(1)).float()  # [B, L]
        
        # 3. Attention: 计算每个历史 item 的权重
        attention_weights = self.attention(target_emb, history_emb, mask)  # [B, L]
        
        # 4. Weighted sum: 用户兴趣表示 = sum(weight * history_emb)
        weighted = attention_weights.unsqueeze(-1) * history_emb   # [B, L, D]
        user_interest = weighted.sum(dim=1)                          # [B, D]
        
        # 5. 拼接 + MLP
        # [target_emb, user_interest, target_emb * user_interest] = [B, 3D]
        concat = torch.cat([
            target_emb,
            user_interest,
            target_emb * user_interest,
        ], dim=-1)
        
        logits = self.mlp(concat).squeeze(-1)  # [B]
        return logits
    
    def predict_proba(self, batch):
        """返回概率 (sigmoid 后)"""
        logits = self.forward(batch)
        return torch.sigmoid(logits)


if __name__ == "__main__":
    print("Testing DIN model...")
    
    # 模拟 batch
    B, L = 4, 20
    n_items = 207386
    emb_dim = 32
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = DIN(n_items=n_items, emb_dim=emb_dim, mlp_hidden=64, seq_len=L).to(device)
    print(f"\nModel:")
    print(model)
    
    # 参数量
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params: {n_params:,}")
    print(f"Trainable: {n_trainable:,}")
    print(f"Item emb 占比: {model.item_emb.weight.numel() / n_params * 100:.1f}%")
    
    # 模拟 batch forward
    batch = {
        "target": torch.randint(1, n_items, (B,), device=device),
        "history": torch.randint(0, n_items, (B, L), device=device),
        "history_len": torch.randint(0, L+1, (B,), device=device),
    }
    
    print(f"\nForward test:")
    logits = model(batch)
    print(f"  Output shape: {logits.shape}")
    print(f"  Output range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    print(f"  Proba range: [{torch.sigmoid(logits).min().item():.3f}, {torch.sigmoid(logits).max().item():.3f}]")
    
    # Backward test
    print(f"\nBackward test:")
    labels = torch.randint(0, 2, (B,), dtype=torch.float32, device=device)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Item emb grad mean: {model.item_emb.weight.grad.abs().mean().item():.6f}")
    
    print("\n✅ Model 测试通过!")
