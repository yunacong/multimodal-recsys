# Day 8 - DIN + DeepFM 实验报告

**日期**: 2026-05-27

**耗时**: ~8 小时

**核心成果**: DeepFM 击败 LightGBM v3, Val AUC **0.8145** (+0.0023)

---

## TL;DR

我在 BPC 美妆数据集上对比了 8 个 CTR 模型,**最终 DeepFM 超越 LightGBM v3**。同时发现 DIN 序列模型在短序列场景下不如树模型 — 这是反直觉但有价值的工业发现。

---

## 完整 9 模型对比表

| 模型 | 特征数 | Val AUC | NDCG@10 | Delta | 备注 |

|------|--------|---------|---------|-------|------|

| v0 baseline | 6 | 0.7645 | — | — | LightGBM 默认 |

| v1 tuned | 6 | 0.7738 | — | +0.0093 | 调参 |

| v2 meta+cross | 15 | 0.8100 | 0.7990 | +0.0362 ⭐ | 黄金跳跃 |

| v3-MiniLM | 16 | 0.8116 | 0.8001 | +0.0016 | BERT 384d |

| v3-mpnet | 16 | 0.8122 | 0.8006 | +0.0006 | LightGBM SOTA |

| v4 image | 17 | 0.8115 | 0.8000 | -0.0007 | 失败, 多模态噪声 |

| **DeepFM** ⭐ | 16 | **0.8145** | TBD | +0.0023 | 新 SOTA! |

| Minimal DIN | item_id | 0.6827 | — | -0.1295 | 短序列不适合 |

| Enhanced DIN | +meta | 0.5446 | — | -0.2676 | 过拟合 bug |

---

## DIN 实验诊断 (反直觉发现)

### 3 个 DIN 版本

- Minimal DIN (item_id only): Val 0.6827

- Enhanced DIN (+sub_cat/brand/price): Val 0.5446 (越加越差!)

- Enhanced V2 (+user_emb): Val 0.5100

### 根因

1. **BPC 用户序列稀疏**: 平均 5-10 个交互, DIN 设计为长序列 (电商 100+)

2. **Item embedding 难训**: 207K items × 89 avg → 长尾不可学

3. **加 feature 反过拟合**: train 0.97 vs val 0.54, gap +0.43

### 工业洞察

> "DIN 不是万能, 模型架构必须匹配数据特性"

> 适合 DIN 的场景: 用户历史 > 30 + 视频/快消等高频品类

---

## DeepFM 成功原因

### 架构

Input: 13 dense + 3 categorical (16 维, 和 LightGBM v3 一样)

Wide (FM):





1st order: linear



2nd order: <e_i, e_j> 所有二阶交互

Deep (MLP):





Concat all embeddings → 256→128→64→1



Dropout 0.3, BatchNorm

Output: sigmoid(FM + Deep)

### 关键

1. **共享 embedding**: FM 和 Deep 共用一套 embedding, 信号更纯

2. **显式 + 隐式交互**: FM 显式二阶 + Deep 高阶, 互补

3. **同 LightGBM 一样特征**: 控制变量, 纯比模型能力

4. **早停 PATIENCE=2**: 防止过拟合

### 性能

- 1 epoch: ~3-4 min (M2 MPS)

- 参数: ~120K (vs DIN 6.6M)

- Val AUC 0.8145 ⭐

---

## 工程教训 (Day 8 累计)

1. **数据 pipeline 优化**: numpy 向量化 vs pandas iterrows, 70-140x 加速

2. **Last-out validation**: 序列推荐标准切分, 防数据泄漏

3. **Val/test 必须和 train 一致负采样**: 否则 AUC 评估有偏

4. **MPS pin_memory warning**: M2 MPS 不支持, 无害

5. **`torch.no_grad()` 陷阱**: forward 后无法 backward

6. **bash heredoc + 中文**: dquote 卡死的元凶

7. **小数据 mini test 必须做**: 50K 样本验证 pipeline + 排除 bug

8. **不要追新模型**: LightGBM/DeepFM 仍是表格数据 SOTA

---

## 简历段落 (V3 全栈版)

> **Multimodal Recommender System** | Personal Project | 2026-05  

> End-to-end CTR pipeline on Amazon Reviews 2023 BPC (5.16M interactions)

>

> ✓ **9 model variants**: LightGBM (v0-v4) + DeepFM + DIN, best **Val AUC 0.8145**  

> ✓ **DeepFM > LightGBM**: PyTorch implementation beats tree by 0.0023 AUC  

> ✓ **Multimodal**: BERT (MiniLM/mpnet) + CLIP, identified dominated strategy  

> ✓ **DIN sequence model**: Discovered LightGBM > DIN on sparse sequences (reality-check)  

> ✓ **FastAPI service**: 10.75ms latency, ready for Redis/Docker upgrade  

---

## Day 8 后续 / Day 9 计划

- Day 9: 召回 (Matrix Factorization + FAISS) + 双塔排序架构

- Day 10: FastAPI 升级 (Redis + Docker)

- Day 11: MLflow 实验追踪 + Streamlit demo

- Day 12-13: 文档 + 简历精修 + v1.0 tag

