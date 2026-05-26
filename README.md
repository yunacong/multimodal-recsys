# 多模态推荐系统 (Multimodal Recommender System)

基于 **Amazon Reviews 2023 美妆与个护**(BPC)数据集的端到端多模态推荐系统。
本项目为 2026 年算法工程师(推荐方向)求职准备,完整覆盖工业级推荐系统的核心流程。

**当前最佳 Val AUC: 0.8122**(LightGBM + BERT mpnet 文本特征,16 维)

---

## 🎯 项目亮点

- **8 个模型版本** 系统化消融实验(从 6 维特征 → 17 维特征)
- **多模态实验**: BERT(MiniLM + mpnet)文本 + CLIP 图片 embedding
- **生产级分析**: 发现并量化多模态特征组合的 "dominated strategy"(支配策略陷阱)
- **诚实记录负结果**: 文本 + 图片叠加在 64% 的类目上反而下降
- **工程严谨**: 207K 张商品图下载 99.91% 成功率,CLIP 推理 RTX 4090 上 619 张/秒

---

## 📊 模型演进总表

| 版本 | 特征数 | Val AUC | NDCG@10 | 核心创新 |
|------|--------|---------|---------|---------|
| v0 基线 | 6 | 0.7645 | — | LightGBM 默认参数 |
| v1 调参 | 6 | 0.7738 | — | 超参数优化(+0.0093)|
| v2 meta+交叉 | 15 | 0.8100 | 0.7990 | 商品元数据 + 交叉特征(+0.0362 ⭐ 黄金跳跃)|
| v3-MiniLM | 16 | 0.8116 | 0.8001 | BERT MiniLM 文本聚类(+0.0016)|
| **v3-mpnet** | **16** | **0.8122** | **0.8006** | BERT mpnet 文本聚类(目前最佳)|
| v4 + 图片 | 17 | 0.8115 | 0.8000 | CLIP 图片聚类(**-0.0007** ⚠️ 反向)|
| v4-a 仅图片 | 16 | 0.8117 | — | 反向消融:用图片替换文本 |
| DIN | TBD | TBD | TBD | Deep Interest Network(Day 8)|
| DeepFM | TBD | TBD | TBD | Wide + Deep 双塔(Day 9)|

---

## 🔬 三大研究发现

### 发现 1: 特征工程的边际递减曲线

7 个版本的迭代清晰展示了特征工程的真实规律:
- **黄金期** (v1 → v2): +0.0362(meta + 交叉特征大幅提升)
- **平台期** (v2 → v3): +0.0022(BERT 边际收益)
- **负收益** (v3 → v4): -0.0007(CLIP 反而下降)

### 发现 2: 多模态特征的 "Dominated Strategy"(支配陷阱)

CLIP 图片 + BERT 文本叠加(v4)在 28 个子类目中,**18 个(64%)反而比单独使用文本或图片更差**。

**机制分析**(通过 LightGBM gain decomposition 验证):
- 单独 BERT(v3):text_cluster gain 950K
- 叠加 BERT+CLIP(v4):text_cluster gain 832K(下降 12%)
- 模型把同质信号分散到两个特征上,每次分裂被 "稀释"

### 发现 3: 不同类目偏好不同模态

通过 28 个子类目分析:
- **CLIP 强势**: 小样本视觉化类目(+0.005 ~ +0.026 AUC)
- **BERT 强势**: 大样本文字密集型类目
- **生产建议**: 按类目动态选择 modality,而非盲目叠加

详见 [reports/day7_ablation_analysis.md](./reports/day7_ablation_analysis.md)

---

## 🏗️ 系统架构

原始数据 (Amazon Reviews 2023 BPC) → 数据 Pipeline(1:5 负采样, 1222x 加速) → 特征工程(17 维) → 多模型集成(LightGBM v3-mpnet + DIN + DeepFM) → 在线服务(FastAPI + Redis) → A/B 测试

完整架构文档见 [ARCHITECTURE.md](./ARCHITECTURE.md)

---

## 📁 项目结构

主要目录:
- `data/raw/` — 原始 Amazon Reviews 2023 BPC
- `data/processed/` — 特征、embedding、聚类结果
- `notebooks/` — 7 个 Jupyter notebooks(01_数据探索 → 06_图片消融)
- `src/data_pipeline/` — 负采样、特征工程
- `src/embedding/` — BERT、CLIP 提取器
- `src/image_pipeline/` — 207K 图片下载 + CLIP 推理
- `src/models/` — DIN、DeepFM(Day 8-9)
- `models/` — 训练好的模型(8 个版本)
- `reports/daily/` — 每日日志
- `reports/day7_ablation_analysis.md` — 多模态消融分析报告
- `serving/` — FastAPI 应用(Day 11)

---

## 🚀 快速开始

### 环境配置

    conda create -n recsys python=3.11 -y
    conda activate recsys
    pip install -r requirements.txt
    # Apple Silicon Mac 用户:
    brew install libomp

### 训练模型

    # 基线 LightGBM
    jupyter notebook notebooks/03_lightgbm_baseline.ipynb
    
    # 最佳模型 (v3-mpnet)
    jupyter notebook notebooks/05_text_embedding_v3.ipynb
    
    # 多模态实验 + 消融
    jupyter notebook notebooks/06_image_embedding_v4.ipynb

---

## 🎓 工程亮点

### 数据 Pipeline 优化
- **1222x 加速** 负采样(内存映射 + 向量化)
- **流式 JSON 解析** 处理 2.7GB 元数据(峰值内存 60MB vs 朴素方法 5-15GB)

### 多模态特征工程
- **文本**: Sentence-BERT(MiniLM 384 维 + mpnet 768 维)→ K-means 50 聚类
- **图片**: CLIP ViT-B/16(512 维)处理 207K 商品图 → K-means 50 聚类
- **生产权衡**: 选 MiniLM 替代 mpnet(5x 加速换 -0.0006 AUC)

### 云端 GPU 成本控制
- 第 1-2 周总成本: 约 ¥30(约 $4 美元)
- 207K 图片下载: 1.5 小时, RTX 4090
- CLIP 推理: 207K 图片仅 5.6 分钟(619 张/秒)
- BERT 推理: 207K 标题仅 22 秒(8184 句/秒)

---

## 📈 8 周开发路线图

- ✅ **第 1 周** (Day 1-7): 数据 + LightGBM 基线 + 多模态实验
- 🟢 **第 2 周** (Day 8-14): DIN + DeepFM + 在线服务 + Demo
- ⬜ **第 3-4 周**: 召回 + 排序架构 + A/B 测试框架
- ⬜ **第 5-6 周**: MLflow 实验追踪 + 完善文档
- ⬜ **第 7-8 周**: 简历优化 + 模拟面试 + 求职投递

---

## 🛠️ 技术栈

- **语言**: Python 3.11
- **机器学习**: LightGBM, PyTorch, Sentence-Transformers, CLIP
- **数据处理**: Pandas, NumPy, Parquet
- **云端**: AutoDL(RTX 4090)
- **服务**: FastAPI, Redis(Day 11)
- **追踪**: MLflow(Day 12)

---

## 👤 作者

**yunacong** — 2026 届算法工程师(推荐方向)校招/社招候选人

- 8 周冲刺计划: 2026-05-23 至 2026-07-18
- 每日日志: [reports/daily/](./reports/daily/)
- 项目反向消融报告: [reports/day7_ablation_analysis.md](./reports/day7_ablation_analysis.md)
