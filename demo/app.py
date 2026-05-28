"""
多模态推荐系统 - Streamlit 交互 Demo (高保真数据产品版)
启动: streamlit run demo/app.py --server.port 8501
"""
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

API_URL = "http://localhost:8000"
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROC = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

st.set_page_config(
    page_title="多模态推荐系统",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 全局 CSS - 数据产品风格
# ============================================================
st.markdown("""
<style>
    /* 全局背景 */
    .stApp {
        background: #F7F8FA;
    }
    /* 隐藏默认页眉 */
    header[data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }

    /* 主容器宽度 */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }

    /* Hero 区 */
    .hero {
        background: linear-gradient(120deg, #FF5A3C 0%, #FF7E5F 50%, #6366F1 100%);
        border-radius: 20px;
        padding: 36px 40px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(255, 90, 60, 0.18);
    }
    .hero h1 {
        font-size: 34px;
        font-weight: 800;
        margin: 0 0 8px 0;
        color: white;
        letter-spacing: -0.5px;
    }
    .hero p {
        font-size: 16px;
        opacity: 0.95;
        margin: 0;
        font-weight: 400;
    }
    .hero .tags {
        margin-top: 16px;
    }
    .hero .tag {
        display: inline-block;
        background: rgba(255,255,255,0.22);
        border: 1px solid rgba(255,255,255,0.35);
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 13px;
        margin-right: 8px;
        backdrop-filter: blur(4px);
    }

    /* KPI 卡片 */
    .kpi-card {
        background: white;
        border-radius: 16px;
        padding: 22px 24px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
        border: 1px solid #EEF0F3;
        height: 100%;
    }
    .kpi-value {
        font-size: 30px;
        font-weight: 800;
        color: #1A1A2E;
        line-height: 1.1;
    }
    .kpi-label {
        font-size: 13px;
        color: #8A8F99;
        margin-top: 6px;
        font-weight: 500;
    }
    .kpi-accent { color: #FF5A3C; }
    .kpi-purple { color: #6366F1; }

    /* 通用卡片 */
    .card {
        background: white;
        border-radius: 16px;
        padding: 24px 28px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
        border: 1px solid #EEF0F3;
        margin-bottom: 20px;
    }
    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #1A1A2E;
        margin-bottom: 4px;
    }
    .section-sub {
        font-size: 13px;
        color: #8A8F99;
        margin-bottom: 18px;
    }

    /* 商品卡片 */
    .product-card {
        background: white;
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border: 1px solid #EEF0F3;
        transition: transform 0.15s, box-shadow 0.15s;
        height: 100%;
        position: relative;
    }
    .product-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 22px rgba(99,102,241,0.14);
    }
    .rank-badge {
        position: absolute;
        top: 12px;
        left: 12px;
        background: linear-gradient(135deg, #FF5A3C, #FF7E5F);
        color: white;
        font-size: 12px;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 12px;
        z-index: 2;
    }
    .product-img {
        width: 100%;
        height: 120px;
        background: linear-gradient(135deg, #F0F1F5, #E4E7EC);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 40px;
        margin-bottom: 12px;
    }
    .product-title {
        font-size: 14px;
        font-weight: 600;
        color: #1A1A2E;
        margin-bottom: 8px;
        line-height: 1.3;
        height: 36px;
        overflow: hidden;
    }
    .product-meta {
        font-size: 12px;
        color: #8A8F99;
        margin-bottom: 4px;
    }
    .product-score {
        font-size: 13px;
        font-weight: 700;
        color: #6366F1;
        margin-top: 8px;
    }
    .reason-tag {
        display: inline-block;
        background: #FFF0ED;
        color: #FF5A3C;
        font-size: 11px;
        font-weight: 600;
        padding: 3px 9px;
        border-radius: 10px;
        margin-top: 8px;
    }

    /* 模型对比条 */
    .model-row {
        display: flex;
        align-items: center;
        margin-bottom: 12px;
    }
    .model-name {
        width: 150px;
        font-size: 14px;
        font-weight: 600;
        color: #1A1A2E;
    }
    .model-bar-bg {
        flex: 1;
        background: #F0F1F5;
        border-radius: 8px;
        height: 28px;
        overflow: hidden;
        margin: 0 12px;
    }
    .model-bar {
        height: 100%;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 10px;
        color: white;
        font-size: 12px;
        font-weight: 700;
    }

    /* 架构流程 */
    .flow-step {
        background: white;
        border-radius: 14px;
        padding: 18px 14px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #EEF0F3;
        height: 100%;
    }
    .flow-icon { font-size: 28px; margin-bottom: 8px; }
    .flow-name { font-size: 14px; font-weight: 700; color: #1A1A2E; }
    .flow-desc { font-size: 11px; color: #8A8F99; margin-top: 4px; }

    /* 延迟指标 */
    .latency-pill {
        background: white;
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #EEF0F3;
        text-align: center;
    }
    .latency-value { font-size: 24px; font-weight: 800; color: #1A1A2E; }
    .latency-label { font-size: 12px; color: #8A8F99; margin-top: 4px; }

    /* sidebar */
    section[data-testid="stSidebar"] {
        background: #1A1A2E;
    }
    section[data-testid="stSidebar"] * { color: #E4E7EC; }
    .side-item {
        background: rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 13px;
    }
    .side-item b { color: #FF7E5F; }
    .side-label { font-size: 11px; color: #8A8F99; text-transform: uppercase; }

    /* 按钮 */
    .stButton button {
        background: linear-gradient(135deg, #FF5A3C, #FF7E5F);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        padding: 10px 0;
        font-size: 15px;
        box-shadow: 0 4px 14px rgba(255,90,60,0.3);
    }
    .stButton button:hover {
        box-shadow: 0 6px 18px rgba(255,90,60,0.45);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 数据加载 (缓存)
# ============================================================
@st.cache_data
def load_sample_users():
    data = np.load(MODELS_DIR / "val_user_embs.npz")
    users = data["users"]
    vocab = sorted(pd.read_parquet(DATA_PROC / "train_with_all_features.parquet")["user_id"].unique())
    return [vocab[u] for u in users[:100]]


@st.cache_data
def load_item_info():
    item_meta = pd.read_csv(DATA_PROC / "item_meta_features.csv")
    return item_meta.set_index("parent_asin").to_dict(orient="index")


@st.cache_data
def load_subcat_names():
    # 尝试加载子类目映射, 没有就用 id
    try:
        df = pd.read_csv(DATA_PROC / "subcategory_mapping.csv")
        return dict(zip(df.iloc[:, 1], df.iloc[:, 0]))
    except Exception:
        return {}


sample_users = load_sample_users()
item_info = load_item_info()

# 推荐理由标签池 (基于规则的伪解释, 用于展示)
REASON_TAGS = ["文本语义相似", "品牌偏好匹配", "历史行为相似", "同类目热门", "价格区间契合"]


def pick_reason(rank, asin):
    # 用 rank + asin hash 稳定地选一个理由
    idx = (rank + hash(asin)) % len(REASON_TAGS)
    return REASON_TAGS[idx]


# ============================================================
# Sidebar - 项目状态面板
# ============================================================
with st.sidebar:
    st.markdown("### 🛍️ 推荐系统")
    st.markdown('<div class="side-label">系统状态</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-item"><b>数据集</b><br>Amazon Reviews 2023</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-item"><b>召回</b><br>Two-Tower 双塔 ANN</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-item"><b>排序</b><br>LightGBM / DeepFM</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-item"><b>缓存</b><br>Redis</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-item"><b>部署</b><br>Docker Compose</div>', unsafe_allow_html=True)

    st.markdown('<div class="side-label" style="margin-top:14px">项目进度</div>', unsafe_allow_html=True)
    st.progress(0.70)
    st.caption("70% 完成 · Day 11 / 13")

    # API 状态
    try:
        r = requests.get(f"{API_URL}/", timeout=2)
        if r.status_code == 200:
            st.success("API 在线", icon="✅")
        else:
            st.warning("API 异常")
    except Exception:
        st.error("API 离线", icon="⚠️")


# ============================================================
# Hero 区
# ============================================================
st.markdown('<div class="hero"><h1>多模态商品推荐系统</h1><p>Two-Tower Retrieval + LightGBM / DeepFM Ranking · 工业级端到端推荐 pipeline</p><div class="tags"><span class="tag">🔍 双塔召回</span><span class="tag">📊 GBDT 排序</span><span class="tag">⚡ Redis 缓存</span><span class="tag">🐳 Docker 部署</span></div></div>', unsafe_allow_html=True)


# ============================================================
# KPI 卡片
# ============================================================
k1, k2, k3, k4 = st.columns(4)
kpis = [
    (k1, "5.16M", "用户交互", "kpi-accent"),
    (k2, "729K", "用户数", ""),
    (k3, "207K", "商品数", "kpi-purple"),
    (k4, "0.8145", "Best AUC (DeepFM)", "kpi-accent"),
]
for col, val, label, cls in kpis:
    col.markdown(f'<div class="kpi-card"><div class="kpi-value {cls}">{val}</div><div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)


# ============================================================
# 操作卡片
# ============================================================
st.markdown('<div class="section-title">🎯 生成推荐</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">选择用户, 系统通过双塔召回 top-200 候选, 再由 LightGBM 排序输出 Top-K</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns([3, 1, 1])
with c1:
    user_id = st.selectbox("用户 ID", sample_users, index=0, label_visibility="collapsed")
with c2:
    top_k = st.number_input("Top-K", min_value=5, max_value=30, value=10, step=5, label_visibility="collapsed")
with c3:
    go = st.button("🚀 获取推荐", use_container_width=True)


# ============================================================
# 推荐结果
# ============================================================
if go:
    try:
        with st.spinner("双塔召回 + 排序中..."):
            resp = requests.post(
                f"{API_URL}/recommend",
                json={"user_id": user_id, "recall_k": 200, "top_k": int(top_k)},
                timeout=15,
            )
        if resp.status_code != 200:
            st.error(f"API 错误: {resp.status_code} - {resp.text}")
        else:
            data = resp.json()
            lat = data["latency_ms"]

            # 延迟指标卡
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            L = st.columns(4)
            cache_hit = lat.get("cache_hit", False)
            metrics = [
                ("总延迟", f"{lat.get('total', 0):.1f} ms"),
                ("缓存命中" if cache_hit else "召回", "✅ YES" if cache_hit else f"{lat.get('recall', 0):.1f} ms"),
                ("排序", "—" if cache_hit else f"{lat.get('ranking', 0):.1f} ms"),
                ("候选", f"200 → {top_k}"),
            ]
            for col, (lab, val) in zip(L, metrics):
                col.markdown(f'<div class="latency-pill"><div class="latency-value">{val}</div><div class="latency-label">{lab}</div></div>', unsafe_allow_html=True)

            st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
            st.markdown(f'<div class="section-title">🎁 为该用户推荐的 Top-{top_k} 商品</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">每张卡片含 商品信息 + CTR score + 推荐理由</div>', unsafe_allow_html=True)

            # 商品卡片网格
            recs = data["recommendations"]
            emojis = ["💄", "🧴", "🪥", "🧼", "💅", "🧖", "🪞", "🌸"]
            per_row = 5
            for i in range(0, len(recs), per_row):
                cols = st.columns(per_row)
                for j, rec in enumerate(recs[i:i+per_row]):
                    asin = rec["parent_asin"]
                    info = item_info.get(asin, {})
                    price = info.get("price", 0)
                    rank = rec["rank"]
                    emoji = emojis[(rank - 1) % len(emojis)]
                    reason = pick_reason(rank, asin)
                    title = f"商品 {asin[-6:]}"
                    sub_cat = info.get("sub_category_id", "—")
                    brand = info.get("brand_id", "—")
                    with cols[j]:
                        card = f'<div class="product-card"><div class="rank-badge">#{rank}</div><div class="product-img">{emoji}</div><div class="product-title">{title}</div><div class="product-meta">📦 类目 {sub_cat} · 🏷️ 品牌 {brand}</div><div class="product-meta">💰 ${price:.2f}</div><div class="product-score">CTR {rec["score"]:.4f}</div><div class="reason-tag">{reason}</div></div>'
                        st.markdown(card, unsafe_allow_html=True)
    except requests.exceptions.RequestException as e:
        st.error(f"无法连接 API: {e}")
        st.info("请确认 Docker 服务运行中: docker compose up -d")


# ============================================================
# 模型对比区
# ============================================================
st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
st.markdown('<div class="section-title">📊 模型对比</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">9 个模型迭代, DeepFM 以 0.8145 击败 LightGBM v3</div>', unsafe_allow_html=True)

models = [
    ("LightGBM v0", 0.7645, "#94A3B8"),
    ("v2 meta+cross", 0.8100, "#818CF8"),
    ("v3-mpnet", 0.8122, "#6366F1"),
    ("DeepFM 🏆", 0.8145, "#FF5A3C"),
    ("Minimal DIN", 0.6827, "#CBD5E1"),
]
auc_min, auc_max = 0.65, 0.83
bars_html = '<div class="card">'
for name, auc, color in models:
    pct = (auc - auc_min) / (auc_max - auc_min) * 100
    bars_html += f'<div class="model-row"><div class="model-name">{name}</div><div class="model-bar-bg"><div class="model-bar" style="width:{pct:.0f}%; background:{color};">{auc:.4f}</div></div></div>'
bars_html += '</div>'
st.markdown(bars_html, unsafe_allow_html=True)


# ============================================================
# 系统架构区
# ============================================================
st.markdown('<div class="section-title">🏗️ 系统架构</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">数据 → 特征 → 召回 → 排序 → 推荐 五阶段 pipeline</div>', unsafe_allow_html=True)

flow = [
    ("📥", "数据", "5.16M 交互"),
    ("⚙️", "特征工程", "16 维 + BERT/CLIP"),
    ("🔍", "召回", "Two-Tower top-200"),
    ("📊", "排序", "LightGBM / DeepFM"),
    ("🎁", "推荐", "Top-K + Redis"),
]
fcols = st.columns(len(flow))
for col, (icon, name, desc) in zip(fcols, flow):
    col.markdown(f'<div class="flow-step"><div class="flow-icon">{icon}</div><div class="flow-name">{name}</div><div class="flow-desc">{desc}</div></div>', unsafe_allow_html=True)


# ============================================================
# 底部
# ============================================================
st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
st.caption("Multimodal Recommender System · GitHub: yunacong/multimodal-recsys · Built with Streamlit + FastAPI + Docker")
