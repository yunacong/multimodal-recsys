# 八股 #1: 逻辑回归 (Logistic Regression)

## 核心一句话

逻辑回归 = 线性回归 + sigmoid 挤压 + Binary Cross Entropy 损失。
本质是把"无界分数"翻译成"0-1 概率",再用 log 损失训练。

## 三大组件

### 1. 线性部分(计算综合分数)
score = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
其中:

x = 输入特征(年龄、收入、商品热度...)
w = 权重(模型要学的参数)
b = 偏置(模型要学的参数)
score 范围: (-∞, +∞)
### 2. Sigmoid (把分数挤压成概率)
p = σ(score) = 1 / (1 + e^(-score))
输入: (-∞, +∞)
输出: (0, 1) ← 这就是"是正样本的概率"
关键点:

score = 0 → p = 0.5 (中性)
score → +∞ → p → 1 (强正样本)
score → -∞ → p → 0 (强负样本)
### 3. Binary Cross Entropy 损失 (衡量预测和真实的差距)
Loss = -[y·log(p) + (1-y)·log(1-p)]
其中:

y = 真实标签 (0 或 1)
p = 模型预测的概率

简化看:

当 y = 1: Loss = -log(p)        预测越接近 1 损失越小
当 y = 0: Loss = -log(1-p)      预测越接近 0 损失越小

为什么用 log 而不是 (y-p)²?

log 在错得离谱时损失爆炸(p=0.001 vs y=1 损失 6.9)
让模型"严厉惩罚严重错误",学得更快
## 关键问答

### Q1: 为什么不能用线性回归做分类?
A: 线性回归输出范围 (-∞, +∞),没法解释成 0/1 概率。
   sigmoid 加进来挤压成 (0, 1) 才能跟真实标签做对比。

### Q2: sigmoid 输出范围?
A: (0, 1),严格不到 0 也不到 1。
   输入越大输出越接近 1,输入越小越接近 0,输入 0 输出 0.5。

### Q3: 推理时的 0.5 阈值固定吗?
A: 不固定!业务驱动调整:
   - 反欺诈用 0.7-0.9(谨慎,宁可漏过欺诈也不冤枉好人)
   - 推荐系统用 0.2-0.5(宽松,多推一些)
   - 医疗诊断用 0.05-0.2(漏诊代价大于误诊)

### Q4: 为什么二分类用 BCE 而不是 MSE?
A: BCE 用 log,对"严重错误"惩罚极大(log 在 0 附近是 -∞)。
   MSE 损失差异不够大,模型学得慢。
   而且 BCE 是极大似然估计的自然推导结果(从概率角度严谨)。

### Q5: 逻辑回归 vs LightGBM 的关系?
A: LightGBM 二分类时,最终输出经过 sigmoid,损失用 binary_logloss
   (就是 BCE)。可以理解为"用很多决策树代替线性部分"。

### Q6: 极大似然推导(进阶)
A: 假设 P(y=1|x) = p,P(y=0|x) = 1-p
   样本似然 L = Π p^y · (1-p)^(1-y)
   取 log: log L = Σ [y·log(p) + (1-y)·log(1-p)]
   加负号(最大化变最小化): Loss = -log L = BCE
   
   → BCE 不是凭空设计的,是从概率论严格推导出来的!

## 在你的项目里怎么用

Week 1 LightGBM 二分类:
- 训练数据: train_with_neg.csv (24.6M 样本, 1:5 正负比)
- 特征: user_activity, item_popularity 等
- 标签: label (1=正样本, 0=负样本)
- 损失: binary_logloss = BCE
- 训练: 让模型预测的 p 越接近真实 label 越好

具体代码:
```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    objective='binary',         # 二分类
    metric='binary_logloss',    # 用 BCE 评估
    n_estimators=100
)
model.fit(X_train, y_train)

# 预测
p = model.predict_proba(X_test)[:, 1]  # 第二列是 p(y=1)
predictions = (p > 0.5).astype(int)    # 阈值 0.5
```

## 面试金句

"逻辑回归是所有二分类模型的根基。它通过 sigmoid 把线性回归
的无界输出挤压到 (0, 1),变成概率;用 BCE 损失训练,这种损失
对错误预测的惩罚是非对称的(log 特性),让模型快速避开严重错误。
LightGBM 二分类的 binary_logloss 就是 BCE,深度学习的二分类
输出层也是 sigmoid + BCE。理解逻辑回归就理解了二分类的本质。"
