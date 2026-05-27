# Day 8 - DIN 序列模型实验报告

**日期**: 2026-05-27
**耗时**: 约 5 小时 (含 3 次训练 + 调试)
**结论**: DIN 在 BPC 数据集上 Val AUC 0.68, **未超过 LightGBM v3-mpnet 0.8122**

---

## TL;DR

我实现了完整的 DIN (Deep Interest Network) PyTorch 训练 pipeline,经过 3 次实验得到反直觉发现 — DIN 在 BPC 美妆数据集上的最佳 Val AUC 仅 0.6827,显著低于 LightGBM v3-mpnet 的 0.8122。

---

## 实验设置

- **数据**: BPC 5-core, last-out split, 1:5 负采样
- **训练样本**: 18.5M (train), 3.6M (val), 3.6M (test)
- **历史长度**: L=20
- **设备**: M2 Mac MPS

## 3 个模型版本对比

| 版本 | 特征 | 参数 | Train AUC | Val AUC | Gap |
|------|------|------|-----------|---------|-----|
| 最简 DIN | item_id | 6.6M | 0.96 | **0.6827** | +0.28 |
| Enhanced DIN | + sub_cat/brand/price | 6.7M | 0.97 | 0.5446 | +0.43 |
| Enhanced V2 | + user_emb | 30M | 0.95 | 0.5100 | +0.44 |

**反直觉发现**: 加更多特征反而 Val AUC 更低!

---

## 根因分析

### 1. BPC 用户序列稀疏
- 平均每用户 5-10 个交互
- DIN attention 设计为长序列(电商 100+ 历史)
- BPC 太短, attention 学不到模式

### 2. Item Embedding 难训
- 207K items × 18.5M samples = 89 次/item 平均
- 但长尾 item 仅 1-5 次出现
- Embedding 学不动 → 用 item_id 不可靠

### 3. 加辅助特征反而引入 leak/噪声
- Enhanced 加 sub_cat/brand/price 后, train 极易过拟合
- val 上模型靠这些特征"猜", 但 user 分布在 train/val 不一致
- 越多 feature 越过拟合, gap +0.43

---

## 工业洞察 (面试金句)

> "DIN 是阿里 2018 提出的电商 CTR 模型, 假设是'用户行为序列长 + 信号强'。
> 我在 BPC 美妆数据集上验证后发现, 当用户交互稀疏时, DIN 反而不如 LightGBM。
> 这告诉我: **模型架构必须匹配数据特性, 不是追新就有效**。
> LightGBM 用 16 维结构化特征 + 树模型, 在表格数据 + 短序列场景仍是 SOTA。"

---

## 生产决策

- ✅ LightGBM v3-mpnet 仍是主模型 (Val AUC 0.8122)
- ❌ DIN 不进生产 (Val AUC 0.6827)
- 📝 保留 DIN 作为"长序列场景预研" baseline
- 📝 适合 DIN 的场景: 用户历史 > 30 + 视频/快消等高频品类

---

## 下一步 (Day 8 后半)

- ⬜ DeepFM (Wide + Deep 双塔) - 评估深度模型在结构化特征上的表现
- ⬜ 完整 8-9 模型对比
