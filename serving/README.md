# Multimodal Recommender System - Serving

基于 FastAPI 的在线推荐服务,加载 LightGBM v3-mpnet 模型。

## 启动

```bash
cd serving
uvicorn app.main:app --reload --port 8000
```

访问 http://localhost:8000/docs 查看交互式 API 文档。

## API 端点

### GET /
健康检查

### GET /model_info
返回模型详细信息

### POST /predict
单用户多商品预测

请求示例:
```json
{
  "user_id": "AGKHLEW2SOWHNMFQIJGBECAF7INQ",
  "item_ids": ["B0BWJGQ32Y", "B00N4LMZZK", "B01DX1OEFO"],
  "top_k": 10
}
```

响应示例:
```json
{
  "user_id": "AGKHLEW2SOWHNMFQIJGBECAF7INQ",
  "n_candidates": 3,
  "n_scored": 3,
  "top_k": 3,
  "recommendations": [
    {"parent_asin": "B01DX1OEFO", "score": 0.823, "rank": 1},
    {"parent_asin": "B0BWJGQ32Y", "score": 0.654, "rank": 2},
    {"parent_asin": "B00N4LMZZK", "score": 0.512, "rank": 3}
  ],
  "inference_time_ms": 12.34
}
```

## 性能指标

- 模型: LightGBM v3-mpnet (16 维特征)
- Val AUC: 0.8122
- 预测延迟: < 50ms (100 商品)
- 启动时间: ~30 秒 (加载模型 + 207K 商品特征)
