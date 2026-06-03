"""
商品评论洞察与卖点挖掘模块（Baseline 版本）

Pipeline：
  评论文本
    → SBERT / BGE-M3 向量化
    → KMeans 聚类（按主题分组）
    → LLM 总结每组正向卖点 / 负向痛点
    → 输出结构化洞察

输出示例：
{
  "sku_id": "SKU_023",
  "avg_rating": 4.1,
  "review_count": 10,
  "positive_aspects": ["补水效果好", "敏感肌友好", "成分安全"],
  "negative_aspects": ["精华液偏少", "包装一般"],
  "target_users": ["熬夜党", "敏感肌", "孕期人群"],
  "content_angles": ["熬夜急救补水", "敏感肌温和修护"],
  "conversion_risks": ["包装质感偏弱，不适合高端定位"],
  "recommendation": "适合中低客单价放量，内容突出温和补水，不建议主打高端礼盒感。"
}
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
from loguru import logger


@dataclass
class ReviewInsight:
    sku_id: str
    avg_rating: float
    review_count: int
    positive_aspects: list[str] = field(default_factory=list)
    negative_aspects: list[str] = field(default_factory=list)
    target_users: list[str] = field(default_factory=list)
    content_angles: list[str] = field(default_factory=list)
    conversion_risks: list[str] = field(default_factory=list)
    recommendation: str = ""
    sentiment_score: float = 0.0   # 0~1，越高越正向


def _simple_sentiment(rating: float) -> str:
    if rating >= 4:
        return "positive"
    elif rating <= 2:
        return "negative"
    return "neutral"


def _cluster_reviews(
    texts: list[str],
    embeddings: np.ndarray,
    n_clusters: int,
) -> dict[int, list[str]]:
    """KMeans 聚类，返回 {cluster_id: [text, ...]}"""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize

    embs = normalize(embeddings)
    k = min(n_clusters, len(texts))
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(embs)

    clusters: dict[int, list[str]] = {}
    for label, text in zip(labels, texts):
        clusters.setdefault(int(label), []).append(text)
    return clusters


def _embed_texts(texts: list[str], model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> np.ndarray:
    """
    使用 sentence-transformers 编码评论文本。

    优先使用 paraphrase-multilingual-MiniLM-L12-v2（小模型，中英文都行，无需 GPU）。
    若环境中已有 BGE-M3，可换成 "BAAI/bge-m3"。
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True)


def _llm_summarize(
    reviews: list[dict],
    clusters: dict[int, list[str]],
    sku_id: str,
    api_key: str,
    base_url: str,
    model: str,
) -> dict:
    """调用 LLM 从聚类评论中提取结构化洞察。"""
    from openai import OpenAI

    # 构造评论摘要（每个 cluster 取前3条）
    cluster_samples = []
    for cid, texts in clusters.items():
        sample = texts[:3]
        cluster_samples.append(f"主题{cid+1}：" + "；".join(sample))

    pos_reviews = [r["text"] for r in reviews if r["rating"] >= 4][:5]
    neg_reviews = [r["text"] for r in reviews if r["rating"] <= 2][:5]
    avg_rating = np.mean([r["rating"] for r in reviews])

    prompt = f"""你是电商选品运营专家，请分析以下商品评论，提取结构化洞察。

商品ID：{sku_id}
平均评分：{avg_rating:.1f}（满分5分）

评论主题分组：
{chr(10).join(cluster_samples)}

好评摘录（评分4-5）：
{chr(10).join(f"- {t}" for t in pos_reviews) if pos_reviews else "无"}

差评摘录（评分1-2）：
{chr(10).join(f"- {t}" for t in neg_reviews) if neg_reviews else "无"}

请严格按以下 JSON 格式输出，不要有额外文字：
{{
  "positive_aspects": ["卖点1", "卖点2", "卖点3"],
  "negative_aspects": ["痛点1", "痛点2"],
  "target_users": ["人群1", "人群2"],
  "content_angles": ["内容角度1", "内容角度2"],
  "conversion_risks": ["风险1"],
  "recommendation": "一句话选品建议（30字内）"
}}

要求：
- positive_aspects：用户真实认可的核心卖点，3-5条
- negative_aspects：用户明确抱怨的问题，1-3条
- target_users：最适合的用户人群，2-3个
- content_angles：内容创作的切入角度，2-3个
- conversion_risks：可能影响转化的风险因素，1-2条
- recommendation：针对选品决策的一句话建议"""

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
        )
        content = resp.choices[0].message.content.strip()
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(content[start:end])
    except Exception as e:
        logger.warning(f"LLM 评论洞察失败，使用规则兜底: {e}")

    return {}


def _rule_based_fallback(reviews: list[dict]) -> dict:
    """LLM 不可用时的规则兜底。"""
    pos = [r["text"] for r in reviews if r["rating"] >= 4]
    neg = [r["text"] for r in reviews if r["rating"] <= 2]
    avg = np.mean([r["rating"] for r in reviews]) if reviews else 3.0

    # 简单关键词抽取
    pos_keywords = _extract_keywords(pos)
    neg_keywords = _extract_keywords(neg)

    return {
        "positive_aspects": pos_keywords[:3] or ["整体评价较好"],
        "negative_aspects": neg_keywords[:2] or [],
        "target_users": ["通用用户"],
        "content_angles": ["产品优势展示"],
        "conversion_risks": ["需结合评分判断放量节奏"] if avg < 4.0 else [],
        "recommendation": f"平均评分{avg:.1f}，{'口碑良好，适合放量' if avg >= 4.0 else '评分偏低，需先优化口碑'}",
    }


_POSITIVE_KEYWORDS = ["好", "棒", "赞", "推荐", "不错", "有效", "满意", "回购", "喜欢", "舒适", "温和", "滋润", "补水"]
_NEGATIVE_KEYWORDS = ["差", "失望", "退货", "过敏", "没效果", "贵", "少", "黏", "干", "普通", "一般"]


def _extract_keywords(texts: list[str]) -> list[str]:
    freq: dict[str, int] = {}
    for text in texts:
        for kw in _POSITIVE_KEYWORDS + _NEGATIVE_KEYWORDS:
            if kw in text:
                freq[kw] = freq.get(kw, 0) + 1
    return [k for k, _ in sorted(freq.items(), key=lambda x: -x[1])][:5]


def analyze_reviews(
    sku_id: str,
    reviews: list[dict],
    n_clusters: int = 3,
    use_embedding: bool = True,
    use_llm: bool = True,
    llm_api_key: str = "",
    llm_base_url: str = "https://api.deepseek.com",
    llm_model: str = "deepseek-chat",
) -> ReviewInsight:
    """
    主入口：分析一个 SKU 的评论，返回结构化洞察。

    Args:
        sku_id: 商品 ID
        reviews: list of {"rating": int, "text": str}
        n_clusters: KMeans 聚类数（建议 3-5）
        use_embedding: 是否使用 SBERT 向量化（False 时跳过聚类直接 LLM）
        use_llm: 是否调用 LLM 总结（False 时使用规则兜底）
        llm_api_key / llm_base_url / llm_model: LLM 配置
    """
    if not reviews:
        return ReviewInsight(sku_id=sku_id, avg_rating=0.0, review_count=0,
                             recommendation="无评论数据，建议先积累口碑再放量")

    texts = [r["text"] for r in reviews]
    avg_rating = float(np.mean([r["rating"] for r in reviews]))
    sentiment_score = (avg_rating - 1) / 4  # 映射到 0~1

    # Step 1: 向量化 + 聚类
    clusters: dict[int, list[str]] = {}
    if use_embedding and len(texts) >= 3:
        try:
            embeddings = _embed_texts(texts)
            clusters = _cluster_reviews(texts, embeddings, n_clusters)
            logger.info(f"[{sku_id}] SBERT 聚类完成，{len(clusters)} 个主题")
        except Exception as e:
            logger.warning(f"SBERT 向量化失败（跳过聚类）: {e}")
            clusters = {0: texts}
    else:
        clusters = {0: texts}

    # Step 2: LLM 总结 / 规则兜底
    llm_result: dict = {}
    if use_llm and llm_api_key:
        llm_result = _llm_summarize(reviews, clusters, sku_id,
                                    llm_api_key, llm_base_url, llm_model)

    if not llm_result:
        llm_result = _rule_based_fallback(reviews)

    return ReviewInsight(
        sku_id=sku_id,
        avg_rating=round(avg_rating, 2),
        review_count=len(reviews),
        positive_aspects=llm_result.get("positive_aspects", []),
        negative_aspects=llm_result.get("negative_aspects", []),
        target_users=llm_result.get("target_users", []),
        content_angles=llm_result.get("content_angles", []),
        conversion_risks=llm_result.get("conversion_risks", []),
        recommendation=llm_result.get("recommendation", ""),
        sentiment_score=round(sentiment_score, 3),
    )


def insight_to_dict(insight: ReviewInsight) -> dict:
    return asdict(insight)


def format_insight_text(insight: ReviewInsight) -> str:
    """格式化为可读文本，供 Agent 引用。"""
    lines = [
        f"**{insight.sku_id} 评论洞察**（{insight.review_count} 条，平均 {insight.avg_rating:.1f} 分）\n",
        f"✅ 核心卖点：{'、'.join(insight.positive_aspects) or '无'}",
        f"⚠️  用户痛点：{'、'.join(insight.negative_aspects) or '无'}",
        f"👥 目标人群：{'、'.join(insight.target_users) or '通用'}",
        f"📢 内容角度：{'、'.join(insight.content_angles) or '产品优势展示'}",
        f"🔴 转化风险：{'、'.join(insight.conversion_risks) or '无明显风险'}",
        f"📌 选品建议：{insight.recommendation}",
    ]
    return "\n".join(lines)
