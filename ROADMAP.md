# 8 周开发路线图 (Roadmap)

本文档跟踪项目从 2026-05-23 到 2026-07-18 的 8 周完整开发计划。

---

## 📅 整体时间线

- 开始: 2026-05-23 (周五)
- 第一次试投: 2026-06-04 (Day 13)
- 正式投递: 2026-07-18 (Day 56)
- 总工期: 8 周 = 56 天

---

## ✅ 第 1 周: 数据 + 基线 (Day 1-7, 已完成)

### Day 1 (2026-05-23): 项目启动
- ✅ 创建 GitHub repo
- ✅ Conda 环境 recsys + Python 3.11
- ✅ 8 周 PDF 计划制定
- ✅ Amazon Reviews 2023 BPC 数据下载

### Day 2 (2026-05-23): 数据探索
- ✅ EDA: 用户/商品分布, 评分分布
- ✅ 5-core 过滤逻辑
- ✅ 负采样设计 (1:5)

### Day 3 (2026-05-24): LightGBM v0 基线
- ✅ 6 维特征: 用户聚合 + 商品聚合
- ✅ Val AUC 0.7645
- ✅ commit 13cc488

### Day 4 (2026-05-24): 调参 + 错误分析
- ✅ v1: hyperparameter 优化 → AUC 0.7738 (+0.0093)
- ✅ 用户级 top-K 评估
- ✅ 3 个反直觉发现 (NDCG 样本量敏感, Recall 类目不均衡, 负样本 false negative 3000x)
- ✅ commit cd4bd85

### Day 5 (2026-05-25): meta + 交叉特征
- ✅ 6 维 item meta 特征 (price, sub_category, brand 等)
- ✅ 3 维 cross 特征 (user_avg_price, pop_x_activity)
- ✅ 1222x 优化的 meta JOIN
- ✅ Val AUC 0.8100 (+0.0362) ⭐ 黄金跳跃
- ✅ commit 1d3f703

### Day 6 (2026-05-25): BERT 文本聚类
- ✅ 207K 商品标题 Sentence-BERT 提取
- ✅ MiniLM (384 维) vs mpnet (768 维) 对比
- ✅ K-means 50 cluster
- ✅ v3-mpnet Val AUC 0.8122 (+0.0022) — 当前最佳
- ✅ commit ef13c30, de1cf12

### Day 7 (2026-05-26): CLIP 多模态 + 消融实验
- ✅ 207K 商品图下载 (99.91% 成功率)
- ✅ CLIP ViT-B/16 提取 (5.6 分钟, 619 img/秒)
- ✅ K-means 50 cluster
- ✅ v4 训练: AUC 0.8115 (-0.0007 ⚠️ 反向!)
- ✅ 反向消融实验: v4-a (仅 CLIP) AUC 0.8117
- ✅ 28 类目分析: CLIP 12 胜, BERT 13 胜, 平 5
- ✅ 发现 dominated strategy
- ✅ 撰写 day7_ablation_analysis.md
- ✅ commit 386fac2, d8de094

---

## 🟢 第 2 周: DIN + DeepFM + 在线服务 (Day 8-14)

### Day 8 (今晚 + 明天): 架构 + DIN 启动
- 🟢 README + ARCHITECTURE + ROADMAP (今晚)
- 🟢 FastAPI 框架搭建 (今晚)
- 🟢 LightGBM v3-mpnet 接入 API (今晚)
- 🟢 DIN 数据 pipeline 设计 (今晚)
- ⬜ DIN PyTorch 模型实现 (明天)
- ⬜ 云端启动 DIN 训练 (明天晚)

### Day 9: DIN 评估 + DeepFM
- ⬜ DIN 训练结果分析
- ⬜ 加入模型对比表 (v0-DIN, 9 模型)
- ⬜ DeepFM 实现 + 训练
- ⬜ 完整模型对比报告

### Day 10: 召回 + 排序双塔
- ⬜ 召回模型 (基于 user_history matrix factorization)
- ⬜ 排序模型 (DIN/DeepFM)
- ⬜ 完整推荐 pipeline

### Day 11: 在线服务
- ⬜ FastAPI 完整 API 实现
- ⬜ Redis cache 集成
- ⬜ 异步处理
- ⬜ 本地 demo 部署

### Day 12: A/B 测试 + MLflow
- ⬜ A/B 测试框架代码
- ⬜ MLflow 实验追踪集成
- ⬜ 完整离线评估系统

### Day 13: 文档 + Streamlit demo
- ⬜ Streamlit 公开 demo
- ⬜ 完整 API 文档
- ⬜ 简历项目章节优化

### Day 14: 第 2 周总结
- ⬜ 完整 9-10 模型对比报告
- ⬜ 第 2 周日志
- ⬜ 6/4 试投准备 (如选择投递)

---

## ⬜ 第 3-4 周: 模型扩展 + 优化 (Day 15-28)

- ⬜ Multi-task learning (CTR + CVR 联合预估)
- ⬜ 序列推荐增强 (BST, SASRec)
- ⬜ Negative sampling 高级技巧 (in-batch, hard negative)
- ⬜ 模型 ensemble (LightGBM + DIN + DeepFM)
- ⬜ Online learning 框架探索

---

## ⬜ 第 5-6 周: 系统化 + 工程 (Day 29-42)

- ⬜ 完整 ML Ops pipeline
- ⬜ Docker 容器化
- ⬜ Kubernetes 部署 (可选)
- ⬜ 监控 + 告警系统
- ⬜ 性能优化 (TensorRT / ONNX)

---

## ⬜ 第 7-8 周: 求职准备 (Day 43-56)

- ⬜ 简历精修 (3-5 版)
- ⬜ LeetCode 突击 (剩余目标 50 题)
- ⬜ 八股突击 (剩余目标 20 题)
- ⬜ SQL 突击 (剩余目标 30 题)
- ⬜ 模拟面试 (5-10 场)
- ⬜ 行为面试准备
- ⬜ 实际投递 (目标 30+ 公司)

---

## 📊 关键里程碑

| 里程碑 | 计划日期 | 当前状态 |
|--------|---------|---------|
| LightGBM Hello World | Day 7 (6/4) | ✅ 提前完成 (Day 3) |
| 多模态实验 | Day 14 | ✅ 提前完成 (Day 7) |
| 完整模型对比 | Day 14 | 🟢 进行中 |
| 在线 demo 部署 | Day 21 | 🟢 加速到 Day 11 |
| 第一次试投 | Day 13 | ⬜ 待决定 |
| 完整 ML Ops | Day 42 | ⬜ 未开始 |
| 正式投递 | Day 56 | ⬜ 未开始 |

---

## 📈 累计进度

- 项目完成度: 35% (Day 7/56)
- 提前完成: 多模态实验 (原计划 Day 14, 实际 Day 7)
- 落后: LeetCode/八股/SQL 基础部分 (下周补)
- 风险: 第 7-8 周基础题突击是否充分

---

## 🎯 当前焦点 (本周内)

实现 **V3 全栈系统版**:
- 9-10 个模型对比 (含 DIN + DeepFM)
- FastAPI + Redis 在线服务
- Streamlit 公开 demo
- MLflow 实验追踪
- 完整技术文档

目标: 7 天内做到, 每天 10 小时投入。
