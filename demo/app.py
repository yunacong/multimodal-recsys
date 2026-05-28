"""
多模态推荐系统 - Streamlit 交互 Demo
启动: streamlit run demo/app.py
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

st.set_page_config(page_title="多模态推荐系统", page_icon="🛍️", layout="wide")


# ============ 加载示例 user_ids + 商品信息 (缓存) ============
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


sample_users = load_sample_users()
item_info = load_item_info()


# ============ 侧边栏: 项目介绍 ============
with st.sidebar:
    st.title("🛍️ 多模态推荐系统")
    st.markdown("""
    **端到端推荐系统** | Amazon Reviews 2023 BPC

    数据规模:
    - 5.16M 交互, 729K 用户, 207K 商品

    架构:
    - **召回**: Two-Tower 双塔 (FAISS-style ANN)
    - **排序**: LightGBM v3-mpnet (AUC 0.8122)
    - **缓存**: Redis (314x 加速)
    - **部署**: Docker Compose
    """)

    st.divider()
    st.subheader("📊 模型对比")
    model_df = pd.DataFrame({
        "模型": ["LightGBM v0", "v2 meta+cross", "v3-mpnet", "DeepFM", "Minimal DIN"],
        "Val AUC": [0.7645, 0.8100, 0.8122, 0.8145, 0.6827],
    })
    st.dataframe(model_df, hide_index=True, use_container_width=True)
    st.caption("🏆 DeepFM 0.8145 击败 LightGBM")


# ============ 主页面 ============
st.title("商品推荐 Demo")
st.markdown("选择一个用户, 系统将通过 **双塔召回 + LightGBM 排序** 推荐 top-10 商品")

col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    user_id = st.selectbox("选择用户 ID", sample_users, index=0)
with col2:
    top_k = st.number_input("Top-K", min_value=5, max_value=50, value=10, step=5)
with col3:
    st.write("")
    st.write("")
    go = st.button("🚀 获取推荐", type="primary", use_container_width=True)


if go:
    try:
        with st.spinner("召回 + 排序中..."):
            resp = requests.post(
                f"{API_URL}/recommend",
                json={"user_id": user_id, "recall_k": 200, "top_k": int(top_k)},
                timeout=10,
            )
        if resp.status_code != 200:
            st.error(f"API 错误: {resp.status_code} - {resp.text}")
        else:
            data = resp.json()

            # 延迟指标
            lat = data["latency_ms"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("总延迟", f"{lat.get('total', 0):.2f} ms")
            if lat.get("cache_hit"):
                m2.metric("缓存", "✅ 命中")
            else:
                m2.metric("召回", f"{lat.get('recall', 0):.1f} ms")
                m3.metric("排序", f"{lat.get('ranking', 0):.1f} ms")
            m4.metric("候选数", "200 → " + str(top_k))

            st.divider()
            st.subheader(f"为用户推荐的 Top-{top_k} 商品")

            # 商品卡片
            recs = data["recommendations"]
            cols_per_row = 5
            for i in range(0, len(recs), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, rec in enumerate(recs[i:i+cols_per_row]):
                    asin = rec["parent_asin"]
                    info = item_info.get(asin, {})
                    with cols[j]:
                        st.markdown(f"**#{rec['rank']}**")
                        st.markdown(f"`{asin}`")
                        price = info.get("price", 0)
                        st.caption(f"💰 ${price:.2f}" if price else "💰 N/A")
                        st.caption(f"⭐ score: {rec['score']:.4f}")
    except requests.exceptions.RequestException as e:
        st.error(f"无法连接 API: {e}")
        st.info("请确认 Docker 服务运行中: docker compose up -d")


# ============ 底部 ============
st.divider()
st.caption("Multimodal Recommender System | GitHub: yunacong/multimodal-recsys")
