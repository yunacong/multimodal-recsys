"""
B端商家选品推荐逻辑层。

将 C端 user_id 推荐改造为面向商家经营目标的选品推荐：
  1. 按品类 / 价格过滤候选商品
  2. 根据经营目标调整特征权重
  3. LightGBM 排序
  4. 评论洞察驱动的推荐理由生成
"""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb

from .schemas import MerchantRecommendRequest, MerchantRecommendResponse, RecommendedSKU

# 评论洞察模块（可选，不影响主流程）
try:
    import sys
    sys.path.insert(0, str(__file__).split("serving")[0])
    from product_intelligence.review_analyzer import analyze_reviews, ReviewInsight
    from product_intelligence.mock_reviews import get_reviews_for_sku
    _REVIEW_INSIGHT_AVAILABLE = True
except Exception:
    _REVIEW_INSIGHT_AVAILABLE = False


# 经营目标 → 排序时各特征的权重调整系数（乘以 LightGBM 基础分后加权合并）
GOAL_WEIGHTS: dict[str, dict[str, float]] = {
    "提升GMV": {"item_interaction_count": 1.5, "item_avg_rating": 1.2},
    "提升CTR": {"title_length": 1.3, "text_cluster_id_mpnet": 1.0},
    "提升CVR": {"item_avg_rating": 1.5, "price": 0.8},
    "清库存": {"item_interaction_count": 0.5, "price": 1.8},
    "新品冷启动": {"item_interaction_count": 0.3, "item_avg_rating": 1.0},
}

PROBLEM_TO_CONTENT_ANGLE: dict[str, str] = {
    "CVR下降": "强化商品信任背书（评分、用量场景、成分安全）",
    "CTR下降": "优化钩子标题（痛点开场、利益点前置）",
    "曝光下降": "扩大人群定向（关键词覆盖、场景延伸）",
    "客单价下降": "突出套装价值 / 高端定位内容",
    "综合指标下滑": "全链路内容优化（标题+封面+脚本）",
}

PROBLEM_TO_RISK: dict[str, str] = {
    "CVR下降": "若详情页承接力弱，内容优化效果会打折，需同步优化落地页",
    "CTR下降": "若点击后 CVR 也低，需排查商品详情页问题",
    "曝光下降": "平台流量分配机制变化时，需配合投放预算调整",
    "客单价下降": "价格带调整可能影响转化量，需监控 GMV 整体变化",
    "综合指标下滑": "需同时排查供给、内容、投放三端，避免单点优化",
}


def _build_merchant_features(
    item_features: dict,
    goal: str,
    feature_cols: list[str],
    user_feat_defaults: dict,
) -> tuple[pd.DataFrame, list[str]]:
    """
    为商家推荐构造特征矩阵。

    使用商家平均行为（均值代替具体用户特征），
    并根据经营目标调整特征值权重。
    """
    weight_map = GOAL_WEIGHTS.get(goal, {})
    rows, valid_asins = [], []

    for asin, feat in item_features.items():
        row = {**user_feat_defaults}
        for col in feature_cols:
            val = feat.get(col, 0)
            # 应用目标权重调整
            if col in weight_map:
                val = val * weight_map[col]
            row[col] = val
        rows.append(row)
        valid_asins.append(asin)

    if not rows:
        return pd.DataFrame(), []

    df = pd.DataFrame(rows, columns=feature_cols)
    return df, valid_asins


# 中文品类关键词 → Amazon 商品文本中常见英文词的映射
# Amazon BPC 数据为英文，用关键词匹配 title / main_category 字段
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "护肤": ["skin", "skincare", "face", "cream", "serum", "toner", "lotion", "moisturizer"],
    "面膜": ["mask", "facial mask", "sheet mask"],
    "美妆": ["beauty", "makeup", "cosmetic", "foundation", "lipstick", "mascara"],
    "彩妆": ["makeup", "cosmetic", "concealer", "blush", "eyeshadow"],
    "洗护": ["shampoo", "hair", "conditioner", "scalp", "wash"],
    "香水": ["perfume", "fragrance", "cologne", "scent"],
    "保健": ["supplement", "vitamin", "health", "wellness", "capsule"],
    "口红": ["lipstick", "lip", "lipgloss", "lip color"],
    "防晒": ["sunscreen", "spf", "sun protection", "uv"],
    "眼部": ["eye", "eyebrow", "eyelash", "eyeliner"],
}


def _match_category(feat: dict, target_category: str) -> bool:
    """判断商品是否匹配目标品类（中文品类→英文关键词模糊匹配）。"""
    if not target_category:
        return True

    # 拼接商品可用文本字段
    text = " ".join([
        str(feat.get("title", "")),
        str(feat.get("main_category", "")),
        str(feat.get("categories", "")),
        str(feat.get("sub_category", "")),
    ]).lower()

    # 收集匹配关键词
    keywords: list[str] = []
    for zh_key, kws in _CATEGORY_KEYWORDS.items():
        if zh_key in target_category:
            keywords.extend(kws)

    # 无法映射时不过滤（保证结果不为空）
    if not keywords:
        return True

    return any(k in text for k in keywords)


def _filter_candidates(
    item_features: dict,
    target_category: str,
    price_range: list[float],
    min_candidates: int = 20,
) -> tuple[dict, str, str]:
    """
    分级过滤候选商品池，返回 (candidates, filter_mode, filter_note)。

    过滤优先级（严格→宽松，逐级降级）：
      strict:        品类 + 价格 同时满足
      category_only: 仅品类匹配（放开价格约束）
      price_only:    仅价格匹配（放开品类约束）
      all_fallback:  全量（两个约束均无法满足时）
    """
    min_p, max_p = price_range[0], price_range[1]
    has_category = bool(target_category)
    has_price = not (min_p == 0 and max_p >= 9999)

    def _pass_price(feat: dict) -> bool:
        price = feat.get("price", 0) or 0
        return price == 0 or (min_p <= price <= max_p)

    def _pass_category(feat: dict) -> bool:
        return _match_category(feat, target_category)

    # Level 1: 严格（品类 + 价格）
    strict = {a: f for a, f in item_features.items()
              if _pass_price(f) and _pass_category(f)}
    if len(strict) >= min_candidates:
        return strict, "strict", ""

    # Level 2: 仅品类
    if has_category:
        cat_only = {a: f for a, f in item_features.items() if _pass_category(f)}
        if len(cat_only) >= min_candidates:
            note = f"候选品类匹配 {len(strict)} 个不足{min_candidates}个，已放开价格约束"
            return cat_only, "category_only", note

    # Level 3: 仅价格
    if has_price:
        price_only = {a: f for a, f in item_features.items() if _pass_price(f)}
        if len(price_only) >= min_candidates:
            note = f"品类+价格候选不足，已放开品类约束，仅保留价格区间 [{min_p}, {max_p}]"
            return price_only, "price_only", note

    # Level 4: 全量兜底
    note = "品类和价格约束均无法满足最小候选数，已退化为全量推荐"
    return item_features, "all_fallback", note


def _get_review_insight(sku_id: str, llm_api_key: str = "") -> Optional["ReviewInsight"]:
    """获取评论洞察，失败时静默返回 None。"""
    if not _REVIEW_INSIGHT_AVAILABLE:
        return None
    try:
        reviews = get_reviews_for_sku(sku_id)
        insight = analyze_reviews(
            sku_id=sku_id,
            reviews=reviews,
            use_embedding=False,   # 服务层关闭 SBERT，避免推理延迟
            use_llm=bool(llm_api_key),
            llm_api_key=llm_api_key,
        )
        return insight
    except Exception:
        return None


def _generate_reason(
    rank: int,
    feat: dict,
    goal: str,
    problem_type: str,
    score: float,
    insight: Optional["ReviewInsight"] = None,
) -> tuple[str, list[str]]:
    """生成推荐理由和证据，优先使用评论洞察，降级到规则兜底。"""
    parts = []
    evidence = []

    rating = feat.get("item_avg_rating", 0)
    interactions = feat.get("item_interaction_count", 0)
    price = feat.get("price", 0)

    # ---- 评论洞察驱动（有则优先使用）----
    if insight and insight.positive_aspects:
        top_pros = "、".join(insight.positive_aspects[:2])
        parts.append(f"用户评论卖点：{top_pros}")
        evidence.append(f"基于 {insight.review_count} 条评论，核心卖点：{top_pros}")

    if insight and insight.negative_aspects:
        risk_hint = "、".join(insight.negative_aspects[:1])
        evidence.append(f"注意用户痛点：{risk_hint}，内容应规避或正向引导")

    if insight and insight.target_users:
        users = "、".join(insight.target_users[:2])
        parts.append(f"适合人群：{users}")

    # ---- 结构化特征规则 ----
    if rating >= 4.0:
        parts.append(f"评分 {rating:.1f} 分口碑稳定")
        evidence.append(f"平均评分 {rating:.1f}，用户满意度高")

    if interactions > 100:
        evidence.append(f"历史交互 {int(interactions)} 次，有成熟用户基础")
    elif interactions < 10:
        parts.append("新品潜力未释放")
        evidence.append("历史互动较少，适合冷启动或放量测试")

    if goal == "提升CVR" and rating >= 4.2:
        parts.append("高评分有助于提升转化信任")
    elif goal == "清库存" and price > 0:
        parts.append(f"价格带 ¥{price:.0f} 适合促销清仓")
        evidence.append(f"当前价格 ¥{price:.0f}，建议配合限时优惠")
    elif goal == "提升CTR":
        parts.append("商品特征适合内容钩子包装")

    reason = "；".join(parts) if parts else "综合评分匹配当前经营目标"
    return reason, evidence


def run_merchant_recommend(
    request: MerchantRecommendRequest,
    model: lgb.Booster,
    item_features: dict,
    feature_cols: list[str],
    user_feat_defaults: dict,
) -> MerchantRecommendResponse:
    """
    完整的 B 端选品推荐流程：过滤 → 构造特征 → LightGBM排序 → 生成理由
    """
    t0 = time.time()

    # Step 1: 分级候选过滤（严格→品类→价格→全量，每级自动记录 filter_mode）
    candidates, filter_mode, filter_note = _filter_candidates(
        item_features,
        request.target_category,
        request.price_range,
        min_candidates=max(request.top_k * 2, 20),
    )

    # Step 2: 构造特征矩阵
    X, valid_asins = _build_merchant_features(
        candidates, request.business_goal, feature_cols, user_feat_defaults
    )

    if X.empty:
        return MerchantRecommendResponse(
            merchant_id=request.merchant_id,
            business_goal=request.business_goal,
            problem_type=request.problem_type,
            recommended_skus=[],
            total_candidates=0,
            inference_time_ms=round((time.time() - t0) * 1000, 2),
            filter_mode="all_fallback",
            filter_note="特征矩阵为空，无法推荐",
        )

    # Step 3: LightGBM 打分 + 排序
    scores = model.predict(X)
    ranked = sorted(
        zip(valid_asins, scores), key=lambda x: x[1], reverse=True
    )[: request.top_k]

    # Step 4: 组装推荐结果（含评论洞察）
    llm_key = os.environ.get("DEEPSEEK_API_KEY", "")
    default_content_angle = PROBLEM_TO_CONTENT_ANGLE.get(request.problem_type, "全链路内容优化")
    default_risk = PROBLEM_TO_RISK.get(request.problem_type, "需持续监控关键指标变化")

    skus = []
    for rank_i, (asin, score) in enumerate(ranked, start=1):
        feat = candidates[asin]
        insight = _get_review_insight(asin, llm_key)
        reason, evidence = _generate_reason(
            rank_i, feat, request.business_goal, request.problem_type, float(score), insight
        )
        # 每个 SKU 独立决定 content_angle 和 risk，不互相污染
        sku_content_angle = (
            "、".join(insight.content_angles[:2])
            if insight and insight.content_angles
            else default_content_angle
        )
        sku_risk = (
            "、".join(insight.conversion_risks[:1])
            if insight and insight.conversion_risks
            else default_risk
        )
        skus.append(
            RecommendedSKU(
                sku_id=asin,
                product_name=feat.get("title", asin)[:60],
                category=str(feat.get("sub_category_id", "")),
                score=round(float(score), 4),
                reason=reason,
                content_angle=sku_content_angle,
                risk=sku_risk,
                evidence=evidence,
            )
        )

    return MerchantRecommendResponse(
        merchant_id=request.merchant_id,
        business_goal=request.business_goal,
        problem_type=request.problem_type,
        recommended_skus=skus,
        total_candidates=len(candidates),
        inference_time_ms=round((time.time() - t0) * 1000, 2),
        filter_mode=filter_mode,
        filter_note=filter_note,
    )
