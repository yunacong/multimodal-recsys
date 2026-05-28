# Day 10 - 生产级部署 (FastAPI + Redis + Docker)

**日期**: 2026-05-28

**核心成果**: 完整容器化推荐服务, Redis 缓存 314x 加速

---

## TL;DR

把 Day 9 的召回+排序 pipeline 升级为生产级在线服务:FastAPI v2.0 提供 REST API,Redis 缓存推荐结果,Docker Compose 一键部署。缓存命中延迟从 478ms 降到 1.52ms。

---

## 架构

Client → FastAPI (recsys-api 容器) ↓ [Redis 查缓存] ──命中──→ 返回 (1.5ms) ↓ 未命中 [Two-Tower 召回 top-200] ↓ [LightGBM v3-mpnet 排序] ↓ [写 Redis 缓存 TTL=300s] → 返回 top-10

Docker Compose:





recsys-api: FastAPI 服务 (自 build)



recsys-redis: Redis 7 缓存 (官方镜像)



volume 挂载 models/ + data/processed/ (只读)

---

## API 端点

| 端点 | 方法 | 功能 |

|------|------|------|

| / | GET | 健康检查 |

| /model_info | GET | 模型信息 |

| /predict | POST | 排序 (给定 item_ids) - v1 兼容 |

| /recommend | POST | 召回+排序 (只给 user_id) - v2 新增 |

---

## 性能

| 场景 | 延迟 |

|------|------|

| 本机召回 | 14 ms |

| 本机排序 | 7 ms |

| 本机端到端 | 26 ms |

| 容器 cold start | 478 ms |

| **Redis cache hit** | **1.52 ms** ⭐ |

**Redis 加速比**: 478ms → 1.52ms = **314x** (容器), 24ms → 0.32ms = 75x (本机)

---

## 工程亮点

1. **优雅降级**: Redis 不可用时自动转无缓存模式, 保证可用性

2. **环境变量配置**: PROJECT_ROOT / REDIS_HOST 适配容器/本机

3. **Volume 挂载**: 模型 1.6GB 数据不打进镜像, 镜像精简

4. **Docker layer cache**: requirements 先 copy, 加速 rebuild

5. **.dockerignore**: 排除大文件, build context 精简

---

## 工程教训

1. **容器内路径**: __file__ 解析的 PROJECT_ROOT 在容器里是 /, 需环境变量覆盖

2. **pyarrow 依赖**: 读 parquet 需显式装, slim 镜像不含

3. **libgomp1**: LightGBM 需要, slim 镜像要 apt install

4. **Redis host**: 容器间用服务名 (redis) 而非 localhost

---

## 面试金句

> "我用 Docker Compose 部署了召回+排序推荐服务 + Redis 缓存。

> 缓存命中延迟 1.5ms (314x 加速)。设计了优雅降级 — Redis 挂了自动转无缓存。

> 关键决策: 模型用 volume 挂载而非打进镜像, 镜像从 GB 级降到 MB 级, 加速 CI/CD。

> 我没上 Kubernetes, 因为单机 Docker Compose 对这个规模足够, 避免过度工程。"

---

## 下一步 (Day 11)

- Streamlit 可交互 demo (网页输入 user_id → 可视化推荐)

- 完整模型对比 dashboard

