"""
Demo 用 mock 评论数据。

真实场景：从 Amazon Reviews CSV 中按 parent_asin 过滤。
Demo 场景：按 sku_id 返回预设评论，覆盖多个品类和正负评特征。

使用方式：
    from product_intelligence.mock_reviews import get_reviews_for_sku
    reviews = get_reviews_for_sku("SKU_023")
"""
from __future__ import annotations
import random

_REVIEW_POOL: dict[str, list[dict]] = {
    # ---- 护肤/面膜 高评分商品 ----
    "SKU_023": [
        {"rating": 5, "text": "补水效果超级好，用完第二天皮肤水润有光泽，敏感肌也没有刺激，强烈推荐"},
        {"rating": 5, "text": "熬夜党福音！敷完真的能感觉皮肤喝饱水了，精华液很充足不会干"},
        {"rating": 4, "text": "补水保湿效果不错，成分温和，敏感肌可以放心用，就是价格稍微贵一点"},
        {"rating": 5, "text": "买了好多次了，每次熬夜后急救用，效果很稳定，皮肤不紧绷了"},
        {"rating": 3, "text": "补水效果有但不算惊艳，包装一般，精华液有点少，感觉性价比一般"},
        {"rating": 4, "text": "适合换季干燥时候用，敷20分钟皮肤明显水润，不会油腻"},
        {"rating": 2, "text": "感觉没什么效果，精华液一上脸就干了，可能不适合我的肤质"},
        {"rating": 5, "text": "成分简单安全，孕期也能用，补水效果很好，闺蜜也在用"},
        {"rating": 4, "text": "质地轻薄，敷完不黏腻，补水效果持续时间比较长，会回购"},
        {"rating": 3, "text": "包装比较简陋，但效果还行，补水保湿够用，就是不够高端"},
    ],
    # ---- 护肤/面膜 中评分商品（CVR低风险）----
    "SKU_041": [
        {"rating": 3, "text": "没想象中好用，补水效果一般，上脸有点黏"},
        {"rating": 2, "text": "过敏了！脸红了一整天，成分表里有我过敏的成分，退货"},
        {"rating": 4, "text": "还可以吧，补水有效果，但比同价位其他品牌差一点"},
        {"rating": 3, "text": "精华液太少了，面膜纸不够贴合，效果打折"},
        {"rating": 2, "text": "买了五片，两片有异味，怀疑是过期产品"},
        {"rating": 4, "text": "普通的补水面膜，效果中规中矩，价格合理"},
        {"rating": 3, "text": "感觉和超市十块钱的差不多，溢价不值"},
        {"rating": 1, "text": "完全没效果，敷完还是干，浪费钱"},
        {"rating": 4, "text": "还行，补水够，敏感肌没有不适，可以接受"},
        {"rating": 3, "text": "成分还行，但味道有点奇怪，包装便宜"},
    ],
    # ---- 彩妆/口红 ----
    "SKU_089": [
        {"rating": 5, "text": "颜色太美了，滋润度超好，涂完嘴唇水润润的，显气色"},
        {"rating": 5, "text": "这个色号太绝了！日常上班涂很高级，持妆4-5小时没问题"},
        {"rating": 4, "text": "滋润型口红，显色度很好，就是持久度一般，吃完饭要补涂"},
        {"rating": 5, "text": "已经回购三支了，颜色正，滋润不拔干，嘴唇干的人必备"},
        {"rating": 4, "text": "质地很滋润，上妆后嘴巴有光泽，但持妆力弱是缺点"},
        {"rating": 3, "text": "颜色比较暗沉，实物和图片差别比较大，有点失望"},
        {"rating": 5, "text": "平价里这个算很好的，滋润持久，气质好看"},
        {"rating": 2, "text": "拔干！涂完嘴巴更干了，完全没有滋润效果"},
        {"rating": 4, "text": "日常色，百搭，推给不会选色的姐妹"},
        {"rating": 5, "text": "涂上去很嫩很水润，男友说嘴巴好看哈哈，回购"},
    ],
    # ---- 健康/保健品 ----
    "SKU_157": [
        {"rating": 5, "text": "吃了两个月，睡眠明显改善，不再失眠了，效果很好"},
        {"rating": 4, "text": "睡眠有改善，但是要配合规律作息，不能指望全靠它"},
        {"rating": 5, "text": "熬夜党救星，吃完晚上真的困了，而且不会有睡不醒的感觉"},
        {"rating": 3, "text": "效果一般，可能我体质问题，吃了一个月没什么明显变化"},
        {"rating": 4, "text": "改善睡眠有效，不过价格有点高，希望出促销"},
        {"rating": 2, "text": "完全没效果，浪费钱，还不如喝杯热牛奶"},
        {"rating": 5, "text": "出差必备，调整时差很有用，没有依赖性"},
        {"rating": 4, "text": "助眠效果真实，吃完半小时有困意，睡眠质量变好了"},
        {"rating": 5, "text": "家里老人也在用，睡得好多了，安全无依赖"},
        {"rating": 3, "text": "短期有效，但长期用感觉效果下降，可能产生了耐受"},
    ],
}

# 默认通用评论（当 sku_id 不在池中时使用）
_DEFAULT_REVIEWS = [
    {"rating": 4, "text": "质量不错，物有所值，会推荐给朋友"},
    {"rating": 5, "text": "商品和描述一致，快递很快，非常满意"},
    {"rating": 3, "text": "一般般，没有想象中好，但也没有差到退货"},
    {"rating": 4, "text": "性价比可以，功能正常，包装完好"},
    {"rating": 2, "text": "有点失望，效果不如描述，沟通客服解决了"},
    {"rating": 5, "text": "已经是第三次回购了，信赖这个牌子"},
    {"rating": 3, "text": "效果一般，价格偏高，不一定会回购"},
    {"rating": 4, "text": "挺好的，日常使用够用，偶尔促销值得入手"},
]


def get_reviews_for_sku(sku_id: str, n: int = 10) -> list[dict]:
    """
    获取某商品的评论列表。

    真实场景：从 Amazon Reviews 数据集中按 parent_asin 查询。
    Demo 场景：从 mock 池中返回（如 sku_id 不存在则返回通用评论）。

    Returns:
        list of {"rating": int, "text": str}
    """
    pool = _REVIEW_POOL.get(sku_id, _DEFAULT_REVIEWS)
    if len(pool) <= n:
        return pool
    return random.sample(pool, n)


def get_all_mock_skus() -> list[str]:
    return list(_REVIEW_POOL.keys())
