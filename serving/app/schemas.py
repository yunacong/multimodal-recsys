"""B端商家推荐接口的请求/响应 Schema"""
from pydantic import BaseModel, Field
from typing import Optional


class MerchantRecommendRequest(BaseModel):
    merchant_id: str = Field(..., example="m_001")
    business_goal: str = Field(..., example="提升GMV",
                               description="提升GMV / 提升CTR / 提升CVR / 清库存 / 新品冷启动")
    problem_type: str = Field(..., example="CVR下降",
                              description="CVR下降 / CTR下降 / 曝光下降 / 客单价下降 / 综合指标下滑")
    target_category: Optional[str] = Field("", example="护肤/面膜")
    price_range: Optional[list[float]] = Field([0, 9999], example=[50, 150])
    top_k: Optional[int] = Field(10, ge=1, le=50)
    recall_k: Optional[int] = Field(200, ge=20, le=1000)


class RecommendedSKU(BaseModel):
    sku_id: str
    product_name: str
    category: str
    score: float
    reason: str
    content_angle: str
    risk: str
    evidence: list[str]


class MerchantRecommendResponse(BaseModel):
    merchant_id: str
    business_goal: str
    problem_type: str
    recommended_skus: list[RecommendedSKU]
    total_candidates: int
    inference_time_ms: float
    filter_mode: str = "strict"          # strict | category_only | price_only | all_fallback
    filter_note: str = ""                # 说明为何触发兜底
