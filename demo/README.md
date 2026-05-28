# Streamlit Demo

交互式推荐系统演示。

## 启动

1. 启动后端服务 (Docker):
```bash
docker compose up -d
```

2. 启动 Streamlit:
```bash
streamlit run demo/app.py --server.port 8501
```

3. 浏览器打开 http://localhost:8501

## 功能

- 选择用户 ID, 实时获取 top-K 推荐
- 展示召回/排序延迟分解
- 侧边栏: 项目介绍 + 9 模型对比表
- 商品卡片: ID + 价格 + 排序 score

## 架构

Streamlit → FastAPI (/recommend) → Two-Tower 召回 + LightGBM 排序 → Redis 缓存
