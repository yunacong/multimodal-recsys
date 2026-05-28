# Day 11 - Streamlit 交互 Demo

**日期**: 2026-05-28
**核心成果**: 可交互推荐系统 demo, 完整产品闭环

---

## TL;DR

用 Streamlit 搭建了交互式推荐 demo, 面试官可直接选用户、看实时推荐 + 延迟分解。完整链路: Streamlit → FastAPI → 双塔召回 + LightGBM 排序 → Redis 缓存。

---

## Demo 功能

1. **用户选择**: 100 个示例 user_id 下拉选择
2. **实时推荐**: 调用 /recommend, 展示 top-K 商品卡片
3. **延迟可视化**: 总延迟 / 召回 / 排序 / 候选数 实时显示
4. **缓存指示**: cache hit 时显示 ✅
5. **侧边栏**: 项目介绍 + 数据规模 + 架构 + 9 模型对比表

---

## 技术栈

| 层 | 技术 |
|----|------|
| 前端 | Streamlit |
| API | FastAPI (Docker) |
| 召回 | Two-Tower (PyTorch) |
| 排序 | LightGBM v3-mpnet |
| 缓存 | Redis |
| 编排 | Docker Compose |

---

## 完整产品架构用户浏览器
↓
Streamlit Demo (:8501)
↓ HTTP
FastAPI (:8000, Docker)
↓
[Redis 缓存] → [双塔召回 top-200] → [LightGBM 排序] → top-10---

## 面试价值

> "我的项目不只是 Jupyter notebook 里的 AUC 数字,
> 是一个**能交互的产品** — 面试官可以选用户、实时看推荐和延迟。
> 从数据处理、9 个模型对比、双塔召回、到 Docker 部署 + Streamlit demo,
> 这是一个完整的工业级 ML 系统闭环。"

---

## 下一步 (Day 12-13)

- 完善 README (架构图 + 截图 + quickstart)
- 简历精修
- v1.0 tag
