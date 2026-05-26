# 系统架构 (Architecture)

本文档描述多模态推荐系统的完整架构、数据流和关键设计决策。

---

## 🏗️ 系统全景图

完整数据流分 5 大模块:

### 模块 1: 数据层 (Data Layer)
- 输入: Amazon Reviews 2023 BPC 5-core (5.16M 交互, 729K 用户, 207K 商品)
- 处理: 流式 JSON 解析(峰值内存 60MB vs 朴素方法 5-15GB)
- 输出: parquet 格式训练数据

### 模块 2: 特征工程层 (Feature Engineering)
- 用户特征 (3 维): user_interaction_count, user_avg_rating, user_last_timestamp
- 商品特征 (3 维): item_interaction_count, item_avg_rating, item_last_timestamp
- 商品元数据 (6 维): price, price_missing, title_length, n_categories, sub_category_id, brand_id
- 交叉特征 (3 维): user_avg_price, user_price_diff, pop_x_activity
- 多模态聚类 (2 维): text_cluster_id_mpnet, image_cluster_id

### 模块 3: 模型层 (Model Layer)
- LightGBM v3-mpnet: 当前最佳,17 维特征,Val AUC 0.8122
- DIN (Day 8): 用户行为序列建模,Attention 机制
- DeepFM (Day 9): Wide & Deep 双塔

### 模块 4: 服务层 (Serving)
- FastAPI: 单用户实时预测 API
- Redis cache: 热门商品/用户特征缓存
- 异步处理: 批量预测加速

### 模块 5: 评估层 (Evaluation)
- 离线指标: AUC, NDCG@10, Precision@K, Recall@K
- 在线 A/B 框架: 流量分桶 + 指标监控
- MLflow: 实验追踪 + 模型版本管理

---

## 📐 关键设计决策

### 决策 1: 用 LightGBM 而非深度模型作为基线

理由:
- LightGBM 在表格数据上的强 baseline 经过工业验证
- 训练快(本机 14 分钟 vs 深度模型 1-2 小时云端)
- categorical 特征处理优秀,免去 one-hot 编码

### 决策 2: K-means 聚类而非直接使用 embedding

理由:
- LightGBM 不擅长高维稠密向量(384-768 维)
- 50 cluster 转为单一 categorical 特征,LightGBM 友好
- 节省内存 (207K × 768 维 = 637 MB → 207K × 1 维 = 1.6 MB)

### 决策 3: 负采样比例 1:5

理由:
- 工业推荐常用 1:5 到 1:10 之间
- 平衡正负样本梯度信号
- 控制训练数据总量(24.6M,M2 本机可处理)

### 决策 4: BERT mpnet 用于训练,MiniLM 用于生产

理由:
- mpnet: Val AUC 0.8122 (最佳),768 维
- MiniLM: Val AUC 0.8116 (略低),384 维,5x 推理速度
- 生产推荐 MiniLM(每个 batch 节省 80% 时间,质量差 0.0006 可接受)

### 决策 5: 多模态特征按类目动态选择

来源: Day 7 反向消融实验发现
- 28 个类目分析: CLIP 在 12 个类目胜出, BERT 在 13 个类目胜出
- 单纯叠加在 64% 类目反而下降
- 生产实现: 按 sub_category_id 路由不同特征组合

---

## 🔄 数据流详细描述

### 训练时数据流

step 1: 原始数据
- Amazon Reviews 2023 BPC jsonl 文件
- meta_Beauty_and_Personal_Care.jsonl (2.7 GB)
- BPC_5core_train.csv (286 MB), valid.csv (41 MB), test.csv (41 MB)

step 2: 负采样
- 每个正样本配 5 个负样本
- 50% 随机负样本 + 50% popularity-based (power=0.75)
- 1222x 加速优化(向量化 + memory mapping)

step 3: 特征工程
- 用户聚合特征(基于训练集 timestamp)
- 商品聚合特征
- 交叉特征生成

step 4: 多模态 embedding
- BERT mpnet: 207K 标题 → 768 维 → K-means 50 cluster
- CLIP ViT-B/16: 207K 图片 → 512 维 → K-means 50 cluster

step 5: LightGBM 训练
- 17 维特征
- 500 轮 boosting (early stopping=20)
- categorical_feature 显式指定

### 推理时数据流(在线服务)

step 1: 请求接收
- POST /predict {user_id: str, item_ids: [str]}

step 2: 特征查询
- Redis cache 查询用户聚合特征
- 商品特征 + meta 特征
- text_cluster_id + image_cluster_id (预计算)

step 3: 模型预测
- LightGBM 批量预测 (按用户分组)
- 返回 sigmoid score

step 4: 排序返回
- 按 score 降序排列 item_ids
- 返回 top-K

---

## 💾 存储设计

### 本地存储 (M2 Mac)
- data/raw/: 原始数据(不上传 GitHub)
- data/processed/train_with_all_features.parquet (1.6 GB)
- models/*.txt: LightGBM 模型 (~6 MB 每个)

### 云端存储 (AutoDL 4090)
- 数据盘 /root/autodl-tmp/ (50 GB)
- 包括 207K 图片 (9.5 GB) + CLIP embedding (424 MB) + BERT embedding (637 MB)

### GitHub 存储 (代码 + 元数据)
- 不存大数据/模型文件
- 存 notebook + 报告 + 配置

---

## 🎯 性能指标

### 离线指标(当前最佳 v3-mpnet)
- Val AUC: 0.8122
- NDCG@10: 0.8006
- Precision@10: 0.1788
- Recall@10: 0.9690

### 训练性能
- LightGBM v3 训练: 14-16 分钟 (M2 本机, 8 线程)
- BERT mpnet 提取: 22 秒 (4090, 207K 标题)
- CLIP 推理: 5.6 分钟 (4090, 207K 图片)
- 图片下载: 1.5 小时 (32 线程, 99.91% 成功率)

### 推理性能 (待 Day 11 实测)
- 目标: 单用户 100 商品排序 < 50ms
- 设计: Redis cache + 批量预测

---

## 🚀 未来扩展

### 短期 (Week 2-3)
- DIN 实现 (序列建模)
- DeepFM 实现 (高阶交叉)
- 模型 ensemble

### 中期 (Week 4-5)
- 召回阶段 (双塔模型 + ANN 检索)
- 排序 + 重排
- 多目标优化

### 长期 (Week 6-8)
- 在线 A/B 测试
- 模型监控 + 自动重训
- 完整 ML Ops pipeline
