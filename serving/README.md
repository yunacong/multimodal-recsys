# 多模态商品选品推荐系统 · 在线服务

基于 FastAPI 的推荐服务，支持 C 端个性化推荐（Two-Tower 召回 + LightGBM 排序）和 B 端商家选品推荐。

## 启动

```bash
cd serving
KMP_DUPLICATE_LIB_OK=TRUE uvicorn app.main:app --port 8000
```

访问 http://localhost:8000/docs 查看完整交互式 API 文档。

---

## API 端点

### GET /
健康检查

### GET /model_info
返回模型详细信息（特征数、商品数、用户数、embedding 规模）

---

### POST /predict
C 端兼容接口：给定 user_id + item_ids，返回排序分数。

请求示例：
```json
{
  "user_id": "AGKHLEW2SOWHNMFQIJGBECAF7INQ",
  "item_ids": ["B0BWJGQ32Y", "B00N4LMZZK", "B01DX1OEFO"],
  "top_k": 10
}
```

响应示例：
```json
{
  "user_id": "AGKHLEW2SOWHNMFQIJGBECAF7INQ",
  "n_candidates": 3,
  "n_scored": 3,
  "recommendations": [
    {"parent_asin": "B01DX1OEFO", "score": 0.823, "rank": 1}
  ],
  "inference_time_ms": 12.34
}
```

---

### POST /recommend
C 端召回+排序：只给 user_id，自动 Two-Tower 召回 top-200 再 LightGBM 精排。

请求示例：
```json
{
  "user_id": "AGKHLEW2SOWHNMFQIJGBECAF7INQ",
  "recall_k": 200,
  "top_k": 10
}
```

---

### POST /merchant/recommend
**B 端商家选品推荐接口**。根据商家经营目标和当前问题类型，从商品池中召回并排序高潜力 SKU，返回推荐理由、内容方向和风险提示。可被项目一 AI 运营助手通过 Tool Calling 调用。

请求示例：
```json
{
  "merchant_id": "m_001",
  "business_goal": "提升CVR",
  "problem_type": "CVR下降",
  "target_category": "护肤/面膜",
  "price_range": [50, 150],
  "top_k": 10
}
```

响应示例：
```json
{
  "merchant_id": "m_001",
  "business_goal": "提升CVR",
  "problem_type": "CVR下降",
  "recommended_skus": [
    {
      "sku_id": "SKU_023",
      "product_name": "补水修护面膜",
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

支持的经营目标：提升GMV / 提升CTR / 提升CVR / 清库存 / 新品冷启动

---

### POST /merchant/product_insight
商品评论洞察接口。基于评论文本提取核心卖点、用户痛点、目标人群、内容角度和转化风险。

请求示例：
```json
{
  "sku_id": "SKU_023",
  "n_reviews": 10,
  "use_llm": true
}
```

响应示例：
```json
{
  "sku_id": "SKU_023",
  "avg_rating": 4.1,
  "review_count": 10,
  "positive_aspects": ["补水效果好", "敏感肌友好", "成分安全"],
  "negative_aspects": ["精华液偏少", "包装一般"],
  "target_users": ["熬夜党", "敏感肌"],
  "content_angles": ["熬夜急救补水", "敏感肌温和修护"],
  "conversion_risks": ["包装质感弱，不适合高端定位"],
  "recommendation": "适合中低客单价放量，内容突出温和补水"
}
```

---

### GET /merchant/goals
查询当前支持的经营目标和问题类型列表。

---

## 性能说明

- **模型**: LightGBM v3-mpnet（16 维特征）
- **离线评估 AUC**: 0.8122（含时序泄露）/ **0.609**（修复泄露后，真实泛化能力）
  - 详见 [`reports/ablation_analysis.md`](../reports/ablation_analysis.md)
- **Two-Tower 召回**: Recall@200 = 0.052（Amazon BPC 5-core 数据）
- **推理延迟**: 缓存命中约 1.5ms；冷请求单机测试约 30–80ms
  - 详见 [`PERFORMANCE_REPORT.md`](PERFORMANCE_REPORT.md)
- **启动时间**: 约 30 秒（加载模型 + 207K 商品特征）
