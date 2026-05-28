# 多模态推荐系统 - FastAPI 服务镜像
FROM python:3.11-slim

WORKDIR /app

# 系统依赖 (LightGBM 需要 libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖 (先 copy requirements, 利用 Docker layer cache)
COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy 应用代码
COPY serving/app ./app

# 模型 + 数据通过 volume 挂载 (不打进镜像)
# 暴露端口
EXPOSE 8000

ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV PYTHONUNBUFFERED=1

# 启动
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
