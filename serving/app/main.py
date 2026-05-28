"""
多模态推荐系统 - 在线服务 API v2.0 (召回 + 排序)
FastAPI + Two-Tower 召回 + LightGBM v3-mpnet 排序

启动:
    cd serving
    KMP_DUPLICATE_LIB_OK=TRUE uvicorn app.main:app --port 8000

端点:
    GET  /              健康检查
    GET  /model_info    模型信息
    POST /predict       排序 (给定 item_ids 打分) - 兼容 v1
    POST /recommend     召回+排序 (只给 user_id, 自动召回 top-200 再排序) - v2 新增
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import lightgbm as lgb
import redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# 全局对象
MODEL = None
ITEM_FEATURES = None
USER_FEATURES = None
ITEM_EMBS = None         # 召回: 所有 item 的双塔 emb (206K, 64)
ITEM_VOCAB = None        # 召回: idx -> parent_asin
USER_EMB_LOOKUP = None   # 召回: user_id -> user_emb (预计算)
REDIS = None             # 缓存层
CACHE_TTL = 300          # 推荐结果缓存 5 分钟

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).parent.parent.parent.resolve()))
DATA_PROC = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RECALL_DIR = DATA_PROC / "recall"

FEATURE_COLS = [
    "user_interaction_count", "user_avg_rating", "user_last_timestamp",
    "item_interaction_count", "item_avg_rating", "item_last_timestamp",
    "price", "price_missing", "title_length", "n_categories",
    "sub_category_id", "brand_id",
    "user_avg_price", "user_price_diff", "pop_x_activity",
    "text_cluster_id_mpnet",
]


class PredictRequest(BaseModel):
    user_id: str = Field(..., example="AGKHLEW2SOWHNMFQIJGBECAF7INQ")
    item_ids: List[str] = Field(..., min_items=1, max_items=1000)
    top_k: Optional[int] = Field(10, ge=1, le=100)


class RecommendRequest(BaseModel):
    """v2 召回+排序: 只需 user_id"""
    user_id: str = Field(..., example="AGKHLEW2SOWHNMFQIJGBECAF7INQ")
    recall_k: Optional[int] = Field(200, ge=10, le=1000, description="召回候选数")
    top_k: Optional[int] = Field(10, ge=1, le=100, description="最终返回数")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, ITEM_FEATURES, USER_FEATURES, ITEM_EMBS, ITEM_VOCAB, USER_EMB_LOOKUP

    logger.info("=" * 60)
    logger.info("启动 FastAPI v2.0 - 召回 + 排序")
    logger.info("=" * 60)

    # 1. LightGBM 排序模型
    t0 = time.time()
    MODEL = lgb.Booster(model_file=str(MODELS_DIR / "lightgbm_v3_mpnet.txt"))
    logger.info(f"✅ LightGBM 加载: {time.time()-t0:.1f}s, {MODEL.num_feature()} features")

    # 2. 商品特征
    t0 = time.time()
    item_meta = pd.read_csv(DATA_PROC / "item_meta_features.csv")
    text_clusters = pd.read_csv(DATA_PROC / "item_text_clusters_mpnet.csv")
    item_pop = pd.read_csv(DATA_PROC / "item_popularity.csv")
    items_df = item_meta.merge(text_clusters, on="parent_asin", how="left")
    items_df = items_df.merge(item_pop, on="parent_asin", how="left")
    ITEM_FEATURES = items_df.set_index("parent_asin").to_dict(orient="index")
    logger.info(f"✅ 商品特征: {len(ITEM_FEATURES):,}, {time.time()-t0:.1f}s")

    # 3. 用户特征
    t0 = time.time()
    user_act = pd.read_csv(DATA_PROC / "user_activity.csv")
    USER_FEATURES = user_act.set_index("user_id").to_dict(orient="index")
    logger.info(f"✅ 用户特征: {len(USER_FEATURES):,}, {time.time()-t0:.1f}s")

    # 4. 召回: item embeddings + vocab
    t0 = time.time()
    ITEM_EMBS = np.load(MODELS_DIR / "item_embeddings.npy").astype(np.float32)
    # item vocab (idx -> parent_asin)
    item_vocab_df = pd.read_csv(DATA_PROC / "item_text_clusters_mpnet.csv")
    # 用 recall 数据的 vocab 顺序 (sorted)
    all_items = sorted(pd.read_parquet(DATA_PROC / "train_with_all_features.parquet")["parent_asin"].unique())
    ITEM_VOCAB = {i: asin for i, asin in enumerate(all_items)}
    logger.info(f"✅ Item embeddings: {ITEM_EMBS.shape}, vocab {len(ITEM_VOCAB):,}, {time.time()-t0:.1f}s")

    # 5. 召回: 预计算的 user embeddings (val users)
    t0 = time.time()
    user_emb_path = MODELS_DIR / "val_user_embs.npz"
    if user_emb_path.exists():
        data = np.load(user_emb_path)
        user_indices = data["users"]
        user_embs = data["embs"].astype(np.float32)
        # user_idx -> emb; 但我们需要 user_id -> emb
        user_vocab_list = sorted(pd.read_parquet(DATA_PROC / "train_with_all_features.parquet")["user_id"].unique())
        USER_EMB_LOOKUP = {}
        for ui, emb in zip(user_indices, user_embs):
            USER_EMB_LOOKUP[user_vocab_list[ui]] = emb
        logger.info(f"✅ User embeddings: {len(USER_EMB_LOOKUP):,} (预计算), {time.time()-t0:.1f}s")
    else:
        USER_EMB_LOOKUP = {}
        logger.warning("⚠️ 无预计算 user embeddings, /recommend 仅支持已缓存用户")

    # 6. Redis 缓存
    global REDIS
    try:
        REDIS = redis.Redis(host=os.environ.get("REDIS_HOST", "localhost"), port=6379, decode_responses=True, socket_connect_timeout=2)
        REDIS.ping()
        logger.info("✅ Redis 缓存已连接")
    except Exception as e:
        REDIS = None
        logger.warning(f"⚠️ Redis 未连接 ({e}), 降级为无缓存模式")

    logger.info("=" * 60)
    logger.info("🚀 v2.0 就绪! http://localhost:8000/docs")
    logger.info("=" * 60)
    yield
    logger.info("应用关闭")


app = FastAPI(
    title="Multimodal Recommender System API",
    description="召回(Two-Tower) + 排序(LightGBM v3-mpnet) 两阶段推荐",
    version="2.0.0",
    lifespan=lifespan,
)


def build_features(user_id, asins):
    """为 (user_id, [asins]) 构造 LightGBM 特征矩阵"""
    user_feat = USER_FEATURES.get(user_id)
    if user_feat is None:
        return None, []
    rows, valid = [], []
    for asin in asins:
        item_feat = ITEM_FEATURES.get(asin)
        if item_feat is None:
            continue
        u_avg_p = user_feat.get("user_avg_price", 0)
        i_price = item_feat.get("price", 0)
        rows.append({
            "user_interaction_count": user_feat.get("user_interaction_count", 0),
            "user_avg_rating": user_feat.get("user_avg_rating", 4.0),
            "user_last_timestamp": user_feat.get("user_last_timestamp", 0),
            "item_interaction_count": item_feat.get("item_interaction_count", 0),
            "item_avg_rating": item_feat.get("item_avg_rating", 4.0),
            "item_last_timestamp": item_feat.get("item_last_timestamp", 0),
            "price": i_price,
            "price_missing": int(item_feat.get("price_missing", 1)),
            "title_length": item_feat.get("title_length", 0),
            "n_categories": item_feat.get("n_categories", 0),
            "sub_category_id": int(item_feat.get("sub_category_id", 0)),
            "brand_id": int(item_feat.get("brand_id", 0)),
            "user_avg_price": u_avg_p,
            "user_price_diff": i_price - u_avg_p,
            "pop_x_activity": item_feat.get("item_interaction_count", 0) * user_feat.get("user_interaction_count", 0),
            "text_cluster_id_mpnet": int(item_feat.get("text_cluster_id_mpnet", 0)),
        })
        valid.append(asin)
    if not rows:
        return None, []
    return pd.DataFrame(rows, columns=FEATURE_COLS), valid


@app.get("/")
async def root():
    return {
        "service": "Multimodal Recommender System",
        "status": "healthy" if MODEL is not None else "loading",
        "version": "2.0.0",
        "architecture": "Two-Tower recall + LightGBM ranking",
        "docs": "/docs",
    }


@app.get("/model_info")
async def model_info():
    if MODEL is None:
        raise HTTPException(503, "模型未加载")
    return {
        "ranking_model": "LightGBM v3-mpnet (AUC 0.8122)",
        "recall_model": "Two-Tower (Recall@200 0.052)",
        "n_features": MODEL.num_feature(),
        "n_items": len(ITEM_FEATURES) if ITEM_FEATURES else 0,
        "n_users": len(USER_FEATURES) if USER_FEATURES else 0,
        "n_item_embeddings": int(ITEM_EMBS.shape[0]) if ITEM_EMBS is not None else 0,
        "n_cached_user_embeddings": len(USER_EMB_LOOKUP) if USER_EMB_LOOKUP else 0,
    }


@app.post("/predict")
async def predict(request: PredictRequest):
    """v1 兼容: 给定 item_ids 排序"""
    if MODEL is None:
        raise HTTPException(503, "模型未加载")
    t0 = time.time()
    X, valid = build_features(request.user_id, request.item_ids)
    if X is None:
        raise HTTPException(404, f"用户不存在或所有商品未知")
    scores = MODEL.predict(X)
    recs = sorted([{"parent_asin": a, "score": float(s)} for a, s in zip(valid, scores)],
                  key=lambda x: x["score"], reverse=True)[:request.top_k]
    for i, r in enumerate(recs):
        r["rank"] = i + 1
    return {
        "user_id": request.user_id,
        "n_candidates": len(request.item_ids),
        "n_scored": len(valid),
        "recommendations": recs,
        "inference_time_ms": round((time.time()-t0)*1000, 2),
    }


@app.post("/recommend")
async def recommend(request: RecommendRequest):
    """v2 召回+排序: 只给 user_id, 自动召回 + 排序"""
    if MODEL is None or ITEM_EMBS is None:
        raise HTTPException(503, "模型未加载")
    t0 = time.time()

    # 0. 查缓存
    cache_key = f"rec:{request.user_id}:{request.recall_k}:{request.top_k}"
    if REDIS is not None:
        cached = REDIS.get(cache_key)
        if cached is not None:
            result = json.loads(cached)
            result["cached"] = True
            result["latency_ms"] = {"total": round((time.time()-t0)*1000, 2), "cache_hit": True}
            return result

    # 1. 召回: 查 user_emb
    user_emb = USER_EMB_LOOKUP.get(request.user_id)
    if user_emb is None:
        raise HTTPException(404, f"用户 {request.user_id} 无预计算 embedding (仅支持 val users)")

    t_recall = time.time()
    scores = user_emb @ ITEM_EMBS.T  # [206K]
    K = request.recall_k
    top_idx = np.argpartition(-scores, K)[:K]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    candidate_asins = [ITEM_VOCAB[int(i)] for i in top_idx]
    recall_ms = (time.time() - t_recall) * 1000

    # 2. 排序: LightGBM
    t_rank = time.time()
    X, valid = build_features(request.user_id, candidate_asins)
    if X is None:
        raise HTTPException(404, "候选商品特征构造失败")
    rank_scores = MODEL.predict(X)
    recs = sorted([{"parent_asin": a, "score": float(s)} for a, s in zip(valid, rank_scores)],
                  key=lambda x: x["score"], reverse=True)[:request.top_k]
    for i, r in enumerate(recs):
        r["rank"] = i + 1
    rank_ms = (time.time() - t_rank) * 1000

    result = {
        "user_id": request.user_id,
        "recall_k": request.recall_k,
        "top_k": request.top_k,
        "recommendations": recs,
        "cached": False,
        "latency_ms": {
            "recall": round(recall_ms, 2),
            "ranking": round(rank_ms, 2),
            "total": round((time.time()-t0)*1000, 2),
        },
    }
    # 写缓存
    if REDIS is not None:
        try:
            REDIS.setex(cache_key, CACHE_TTL, json.dumps(result))
        except Exception:
            pass
    return result
