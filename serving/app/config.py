"""推荐系统全局配置 (serving 自包含副本)

内容与 src/config.py 保持同步。
LightGBM v3-mpnet 特征列表、召回参数、交叉特征计算函数。
所有交叉特征公式以 notebooks/04_meta_features.ipynb 为准。
"""

import math

# ============================================================
# LightGBM 特征列 (16 维, 顺序不可改动 — 与模型训练时一致)
# ============================================================
FEATURE_COLS = [
    "user_interaction_count", "user_avg_rating", "user_last_timestamp",
    "item_interaction_count", "item_avg_rating", "item_last_timestamp",
    "price", "price_missing", "title_length", "n_categories",
    "sub_category_id", "brand_id", "user_avg_price",
    "user_price_diff", "pop_x_activity",
    "text_cluster_id_mpnet",
]

# ============================================================
# 召回参数
# ============================================================
RECALL_K = 200


# ============================================================
# 交叉特征计算函数
# ============================================================

def compute_user_price_diff(item_price, user_avg_price) -> float:
    """价格偏离度 (归一化绝对差)

    公式: |item_price - user_avg_price| / (user_avg_price + 1e-6)

    训练来源 (notebooks/04_meta_features.ipynb):
        df_main['user_price_diff'] = (
            (df_main['price'] - df_main['user_avg_price']).abs()
            / (df_main['user_avg_price'] + 1e-6)
        ).astype('float32')

    Args:
        item_price:     商品价格
        user_avg_price: 用户历史平均购买价格

    Returns:
        float, 归一化后的价格偏差; 输入为 NaN/None 时返回 0.0
    """
    try:
        p = float(item_price)
        avg = float(user_avg_price)
        if math.isnan(p) or math.isnan(avg):
            return 0.0
        return abs(p - avg) / (avg + 1e-6)
    except (TypeError, ValueError):
        return 0.0


def compute_pop_x_activity(user_count, item_count) -> float:
    """用户活跃度 × 商品热度 (对数乘积)

    公式: log1p(user_count) * log1p(item_count)

    训练来源 (notebooks/04_meta_features.ipynb):
        df_main['pop_x_activity'] = (
            np.log1p(df_main['user_interaction_count'])
            * np.log1p(df_main['item_interaction_count'])
        ).astype('float32')

    Args:
        user_count: 用户交互次数
        item_count: 商品交互次数

    Returns:
        float, 对数乘积值; 输入为 None 时视为 0
    """
    try:
        return math.log1p(float(user_count or 0)) * math.log1p(float(item_count or 0))
    except (TypeError, ValueError):
        return 0.0
