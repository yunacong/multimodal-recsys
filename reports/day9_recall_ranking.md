# Day 9 - 双塔召回 + 排序 端到端 Pipeline

**日期**: 2026-05-28

**耗时**: ~6 小时

**核心成果**: 完整两阶段推荐架构 (Recall + Rank), 端到端延迟 9ms

---

## TL;DR

我实现了工业标准的两阶段推荐 pipeline — Two-Tower 双塔召回 + LightGBM v3-mpnet 排序。端到端延迟 9.08ms (P99 13.5ms), 处理 206K 候选商品。**Recall@200 仅 0.052 揭示了 BPC 数据集对神经召回不友好**, 这是有价值的工业反向发现。

---

## 架构

User Request ↓ [User Tower] - user_id_emb + 4 dense → 64d emb (L2-normalized) ↓ [Top-200 Retrieval] - NumPy matmul on 206K items (1.86 ms) ↓ [Feature Construction] - 16d LightGBM 特征 (用户/商品/交叉) ↓ [LightGBM v3-mpnet] - 200 候选打分 (~7 ms) ↓ [Top-10 Output] - 排序后输出

---

## 双塔模型设计

### User Tower

- user_id_embedding (32d) + user_dense (4d → 16d 投影)

- MLP: 48 → 128 → 64, L2-normalized

### Item Tower

- item_id_embedding (32d) + sub_cat_emb (8d) + brand_emb (16d) + text_cluster_emb (8d) + item_dense (7d → 16d)

- MLP: 80 → 128 → 64, L2-normalized

### Loss

- In-batch negatives (batch=1024, 1023 个负样本)

- Cross-entropy loss with temperature scaling (T=0.1)

- 训练 10 epoch, Adam lr=1e-3

### 参数量

- Total: 29.6M (主要在 user_id_emb 718K × 32d)

---

## 实验结果

### 训练

| Epoch | Train Loss | Val Loss |

|-------|-----------|----------|

| 1 | 6.49 | 6.94 |

| 3 | 5.80 | 6.47 |

| 5 | 5.34 | 6.39 |

| 6 (续训) | 5.72 | **6.36** ⭐ |

| 8 | 4.98 | 6.42 (早停) |

### 召回评估 (Recall@K)

| K | Recall (best model) |

|---|---|

| 10 | 0.006 |

| 50 | 0.019 |

| 100 | 0.031 |

| 200 | **0.051** |

| 500 | 0.088 |

### 端到端 Pipeline (500 val users)

| 指标 | 值 |

|------|-----|

| 平均延迟 | **9.08 ms** |

| P50 | 8.61 ms |

| P95 | 9.49 ms |

| P99 | 13.54 ms |

| Recall@10 (排序后) | 0.0043 |

| Recall@50 (排序后) | 0.033 |

| Recall@200 (仅召回) | 0.052 |

| LightGBM 排序 lift | 1.65x |

---

## 关键洞察 (面试金句)

### Insight 1: 数据稀疏不利于神经召回

> "BPC 用户平均仅 5-10 个交互, user_id_embedding 学不充分。

> 双塔架构假设 user/item 有足够数据训练 embedding, 但 BPC 是稀疏长尾。

> 这是典型的 **工业冷启动场景** — 神经网络反而不如 ItemCF/Tree-based + 特征工程。"

### Insight 2: 端到端 latency 主要来自 LightGBM 排序

> "召回阶段 1.86 ms (numpy matmul 206K items),

> 排序阶段 ~7 ms (LightGBM 200 候选 16d 特征预测)。

> 优化方向: 召回 → FAISS IVF 可降到 0.5ms; 排序 → ONNX/TensorRT 量化可降到 2ms。"

### Insight 3: 反向发现的工业价值

> "我的双塔 Recall@200 仅 0.05, 显著低于工业基准 0.15+。

> 但这个'负结果'告诉我: 模型选型应该看数据特性, 不是追新。

> 真实生产建议: ItemCF + 双塔混合召回, 或加 hard negatives + 多 epoch + lr warmup。"

---

## 工程教训

1. **PyTorch + LightGBM segfault**: Mac OMP 冲突, 解决方案是分两步 (precompute embs)

2. **In-batch negatives 的局限**: 1024 batch 缺 hard negatives, 工业用 BPR / sampled softmax

3. **续训 lr 调整**: 续训时 lr 要降 (5e-4 vs 1e-3), 否则破坏已学的 emb

4. **L2-normalize is critical**: cosine similarity 不能用未归一化的 emb

---

## Day 9 vs Day 8 对比

| 模型 | 任务 | 指标 | 备注 |

|------|------|------|------|

| LightGBM v3-mpnet | CTR Ranking | AUC 0.8122 | 排序主模型 |

| DeepFM | CTR Ranking | AUC 0.8145 ⭐ | 排序新 SOTA |

| Two-Tower | Recall | Recall@200 0.052 | 召回 baseline |

**完整工业架构就位**: 召回 (Two-Tower) → 排序 (DeepFM 或 LightGBM v3) → top-10

---

## 下一步 (Day 10)

- FastAPI 服务升级 (集成召回 + 排序)

- Redis 缓存 user_emb / item_emb

- Docker 容器化

- 完整 demo: https://localhost:8000/recommend?user_id=...

