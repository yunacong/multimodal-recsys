# 集成说明：作为 B 端选品推荐服务使用

本项目（多模态商品选品推荐系统）可独立部署为推荐服务，也可被 [内容电商商家 AI 运营助手](https://github.com/yunacong/content-ecommerce-ai) 通过 Tool Calling 调用，形成"经营诊断 → 选品推荐"的完整链路。

---

## 两个项目的定位

| | 项目 | 定位 |
|--|------|------|
| 项目一 | content-ecommerce-ai | AI 运营助手应用层：经营诊断、Agent 编排、行动计划 |
| 项目二 | multimodal-recsys（本项目） | 多模态推荐服务层：特征工程、召回排序、B 端 API |

---

## 启动推荐服务

```bash
cd serving
pip install -r requirements.txt
KMP_DUPLICATE_LIB_OK=TRUE uvicorn app.main:app --port 8000
```

服务就绪后访问 http://localhost:8000/docs 查看完整接口文档。

---

## B 端选品推荐接口示例

### 场景：商家 CVR 下降，需要推荐高潜力护肤品 SKU

```bash
curl -X POST http://localhost:8000/merchant/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "merchant_id": "m_001",
    "business_goal": "提升CVR",
    "problem_type": "CVR下降",
    "target_category": "护肤/面膜",
    "price_range": [50, 150],
    "top_k": 5
  }'
```

返回：
```json
{
  "merchant_id": "m_001",
  "business_goal": "提升CVR",
  "problem_type": "CVR下降",
  "recommended_skus": [
    {
      "sku_id": "B01DX1OEFO",
      "score": 0.872,
      "reason": "用户评论卖点：补水效果好、敏感肌友好；评分 4.1 分口碑稳定",
      "content_angle": "熬夜急救补水、敏感肌温和修护",
      "risk": "包装质感偏弱，不适合高端定位",
      "evidence": ["基于10条评论，核心卖点：补水效果好、敏感肌友好"]
    }
  ],
  "total_candidates": 5000,
  "inference_time_ms": 45.2
}
```

---

## 商品评论洞察接口示例

```bash
curl -X POST http://localhost:8000/merchant/product_insight \
  -H "Content-Type: application/json" \
  -d '{"sku_id": "SKU_023", "n_reviews": 10, "use_llm": true}'
```

---

## 与项目一的联动方式

**方式一：远程 API 调用（生产架构）**

在项目一中设置：
```bash
export USE_REMOTE_RECOMMENDER=true
export RECOMMENDER_API_URL=http://localhost:8000
```

项目一 Agent 的 `merchant_recommend_products` 工具自动切换到远程调用本项目的 `/merchant/recommend` 接口。

**方式二：本地模块直接引用（Demo 单体部署）**

项目一内置了 `project2_infra/` 目录，包含 LightGBM 排序、FAISS 检索等核心模块，Demo 场景无需启动独立服务。

---

## 推荐系统技术架构

```
商家经营目标输入
      ↓
候选商品过滤（品类 / 价格）
      ↓
Two-Tower 召回（ANN 检索 top-200）
      ↓
特征构造（16维：用户行为 + 商品元数据 + BERT 文本聚类）
      ↓
LightGBM 精排（按经营目标调整特征权重）
      ↓
SHAP 特征归因（解释推荐理由）
      ↓
评论洞察驱动（卖点 / 痛点 / 内容角度）
      ↓
返回 Top-K SKU + 推荐理由 + 内容方向 + 风险提示
```

---

## 离线评估说明

| 指标 | 数值 | 说明 |
|------|------|------|
| LightGBM Val AUC（含泄露） | 0.8122 | 含时序泄露，不代表真实泛化能力 |
| **LightGBM Val AUC（严格评估）** | **0.609** | 修复时序泄露后真实泛化 AUC ⭐ |
| Two-Tower Recall@200 | 0.052 | Amazon BPC 5-core 数据集 |

> 时序泄露诊断和修复过程详见 [`reports/ablation_analysis.md`](reports/ablation_analysis.md)
