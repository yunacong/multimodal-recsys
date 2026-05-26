"""
多模态推荐系统 - 在线服务 API
基于 FastAPI + LightGBM v3-mpnet 模型

启动方式:
    cd serving
    uvicorn app.main:app --reload --port 8000

API 端点:
    GET  /              - 健康检查
    GET  /model_info    - 模型信息
    POST /predict       - 单用户多商品预测
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import lightgbm as lgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ===========================================
# 全局模型 + 特征数据 (启动时加载)
# ===========================================
MODEL = None
ITEM_FEATURES = None      # parent_asin -> dict (item meta + clusters)
USER_FEATURES = None      # user_id -> dict (user 聚合)

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DATA_PROC = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# 模型用的 16 维特征 (v3-mpnet 配置)
FEATURE_COLS = [
    "user_interaction_count", "user_avg_rating", "user_last_timestamp",
    "item_interaction_count", "item_avg_rating", "item_last_timestamp",
    "price", "price_missing", "title_length", "n_categories",
    "sub_category_id", "brand_id",
    "user_avg_price", "user_price_diff", "pop_x_activity",
    "text_cluster_id_mpnet",
]
CATEGORICAL_COLS = ["sub_category_id", "brand_id", "price_missing", "text_cluster_id_mpnet"]


# ===========================================
# Pydantic Schemas (请求/响应数据格式)
# ===========================================
class PredictRequest(BaseModel):
    """预测请求"""
    user_id: str = Field(..., description="用户 ID", example="AGKHLEW2SOWHNMFQIJGBECAF7INQ")
    item_ids: List[str] = Field(..., description="待评分商品列表", min_items=1, max_items=1000)
    top_k: Optional[int] = Field(10, description="返回 top-K 商品", ge=1, le=100)


class PredictResponse(BaseModel):
    """预测响应"""
    user_id: str
    n_candidates: int
    n_scored: int
    top_k: int
    recommendations: List[dict]  # [{parent_asin, score, rank}]
    inference_time_ms: float


# ===========================================
# 模型加载 (startup 时执行)
# ===========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时加载模型,关闭时清理"""
    global MODEL, ITEM_FEATURES, USER_FEATURES
    
    logger.info("=" * 60)
    logger.info("启动 FastAPI 应用 - 加载模型 + 特征")
    logger.info("=" * 60)
    
    # 1. 加载 LightGBM v3-mpnet 模型
    model_path = MODELS_DIR / "lightgbm_v3_mpnet.txt"
    if not model_path.exists():
        logger.error(f"❌ 模型文件不存在: {model_path}")
        raise FileNotFoundError(model_path)
    
    t0 = time.time()
    MODEL = lgb.Booster(model_file=str(model_path))
    logger.info(f"✅ 模型加载完成: {time.time()-t0:.1f}s")
    logger.info(f"   特征数: {MODEL.num_feature()}")
    
    # 2. 加载商品特征(item_meta + text_cluster)
    logger.info("加载商品特征...")
    t0 = time.time()
    item_meta = pd.read_csv(DATA_PROC / "item_meta_features.csv")
    text_clusters = pd.read_csv(DATA_PROC / "item_text_clusters_mpnet.csv")
    item_pop = pd.read_csv(DATA_PROC / "item_popularity.csv")
    
    items_df = item_meta.merge(text_clusters, on="parent_asin", how="left")
    items_df = items_df.merge(item_pop, on="parent_asin", how="left")
    
    # 转 dict 加速查询: parent_asin -> {feature: value}
    ITEM_FEATURES = items_df.set_index("parent_asin").to_dict(orient="index")
    logger.info(f"✅ 商品特征加载完成: {len(ITEM_FEATURES):,} 商品, {time.time()-t0:.1f}s")
    
    # 3. 加载用户特征
    logger.info("加载用户特征...")
    t0 = time.time()
    user_act = pd.read_csv(DATA_PROC / "user_activity.csv")
    USER_FEATURES = user_act.set_index("user_id").to_dict(orient="index")
    logger.info(f"✅ 用户特征加载完成: {len(USER_FEATURES):,} 用户, {time.time()-t0:.1f}s")
    
    logger.info("=" * 60)
    logger.info("🚀 应用就绪!访问 http://localhost:8000/docs 查看 API 文档")
    logger.info("=" * 60)
    
    yield  # 应用运行中
    
    # Cleanup
    logger.info("应用关闭")


# ===========================================
# FastAPI 应用
# ===========================================
app = FastAPI(
    title="Multimodal Recommender System API",
    description="多模态推荐系统在线服务 - LightGBM v3-mpnet",
    version="1.0.0",
    lifespan=lifespan,
)


# ===========================================
# API 端点
# ===========================================
@app.get("/")
async def root():
    """健康检查"""
    return {
        "service": "Multimodal Recommender System",
        "status": "healthy" if MODEL is not None else "loading",
        "version": "1.0.0",
        "model": "LightGBM v3-mpnet",
        "docs": "/docs",
    }


@app.get("/model_info")
async def model_info():
    """返回模型详细信息"""
    if MODEL is None:
        raise HTTPException(503, "模型未加载")
    
    return {
        "model_name": "LightGBM v3-mpnet",
        "model_file": "lightgbm_v3_mpnet.txt",
        "n_features": MODEL.num_feature(),
        "n_items_in_cache": len(ITEM_FEATURES) if ITEM_FEATURES else 0,
        "n_users_in_cache": len(USER_FEATURES) if USER_FEATURES else 0,
        "feature_list": FEATURE_COLS,
        "categorical_features": CATEGORICAL_COLS,
        "training_data": "Amazon Reviews 2023 BPC, 24.6M samples",
        "val_auc": 0.8122,
        "val_ndcg_10": 0.8006,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    单用户多商品评分预测
    
    输入: user_id + item_ids 列表
    输出: 排序后的 top-K 商品 + score
    """
    if MODEL is None:
        raise HTTPException(503, "模型未加载")
    
    t_start = time.time()
    
    # 1. 查询用户特征
    user_feat = USER_FEATURES.get(request.user_id)
    if user_feat is None:
        raise HTTPException(404, f"用户 {request.user_id} 不在训练集中")
    
    # 2. 构造特征矩阵 (每个 item 一行)
    rows = []
    valid_items = []
    
    for asin in request.item_ids:
        item_feat = ITEM_FEATURES.get(asin)
        if item_feat is None:
            continue  # 跳过未知商品
        
        # 拼装一行特征 (按 FEATURE_COLS 顺序)
        row = {
            "user_interaction_count": user_feat.get("user_interaction_count", 0),
            "user_avg_rating": user_feat.get("user_avg_rating", 4.0),
            "user_last_timestamp": user_feat.get("user_last_timestamp", 0),
            "item_interaction_count": item_feat.get("item_interaction_count", 0),
            "item_avg_rating": item_feat.get("item_avg_rating", 4.0),
            "item_last_timestamp": item_feat.get("item_last_timestamp", 0),
            "price": item_feat.get("price", 0),
            "price_missing": int(item_feat.get("price_missing", 1)),
            "title_length": item_feat.get("title_length", 0),
            "n_categories": item_feat.get("n_categories", 0),
            "sub_category_id": int(item_feat.get("sub_category_id", 0)),
            "brand_id": int(item_feat.get("brand_id", 0)),
            "user_avg_price": user_feat.get("user_avg_price", 0),
            "user_price_diff": item_feat.get("price", 0) - user_feat.get("user_avg_price", 0),
            "pop_x_activity": item_feat.get("item_interaction_count", 0) * user_feat.get("user_interaction_count", 0),
            "text_cluster_id_mpnet": int(item_feat.get("text_cluster_id_mpnet", 0)),
        }
        rows.append(row)
        valid_items.append(asin)
    
    if not rows:
        raise HTTPException(400, "所有候选商品都不在训练集中")
    
    # 3. LightGBM 预测
    X = pd.DataFrame(rows, columns=FEATURE_COLS)
    scores = MODEL.predict(X)
    
    # 4. 排序并取 top-K
    recommendations = sorted(
        [{"parent_asin": asin, "score": float(s)} for asin, s in zip(valid_items, scores)],
        key=lambda x: x["score"],
        reverse=True,
    )[:request.top_k]
    
    # 加上 rank
    for i, rec in enumerate(recommendations):
        rec["rank"] = i + 1
    
    inference_ms = (time.time() - t_start) * 1000
    
    return PredictResponse(
        user_id=request.user_id,
        n_candidates=len(request.item_ids),
        n_scored=len(valid_items),
        top_k=len(recommendations),
        recommendations=recommendations,
        inference_time_ms=round(inference_ms, 2),
    )
