# Multimodal Recommender System(多模态推荐系统)

基于 Amazon Reviews 2023(Beauty 子类目)的端到端多模态推荐系统项目。

## 项目目标

面向 2026 年算法岗(推荐方向)校招/社招的实战项目,从数据清洗到多模态深度模型,完整覆盖工业级推荐系统的核心流程。

## 路线图

- **Week 1**: 数据清洗 + 样本构造 + LightGBM Hello World
- **Week 2+**: 多模态特征工程(CLIP 图像 embedding、Sentence-BERT 文本 embedding)
- **后续**: 双塔模型、序列推荐、在线服务化

## 数据集

- **来源**: McAuley-Lab/Amazon-Reviews-2023(HuggingFace)
- **子类目**: All_Beauty
- **规模**: 701,528 条交互 × 112,590 个商品
- **文件**:
  - All_Beauty.jsonl(311 MB,用户-商品交互数据)
  - meta_All_Beauty.jsonl(203 MB,商品元数据,含标题/描述/图片 URL/价格)

## 环境配置

创建虚拟环境并安装依赖:

    conda create -n recsys python=3.11 -y
    conda activate recsys
    pip install -r requirements.txt

注意: Apple Silicon Mac 用户需要先运行 `brew install libomp` 解决 LightGBM 的 OpenMP 依赖。

## 目录结构

- data/        原始与处理后的数据(已通过 .gitignore 忽略,不上传仓库)
- notebooks/   数据探索 Jupyter notebook
- src/         正式 Python 脚本(数据清洗、负采样、训练等)
- reports/     项目日志(daily / weekly / 八股笔记)
- models/      训练后的模型文件
- configs/     超参数和配置文件

## 项目进度

详见 reports/daily/ 下每日日志。

## 作者

yunacong — 2026 年算法岗(推荐方向)求职项目。
