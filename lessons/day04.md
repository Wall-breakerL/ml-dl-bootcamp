# Day 4：补齐传统机器学习和评估方法（270 分钟）

核心问题：一个小型数据集为什么未必需要深度网络？怎么公平比较方法？

| 分钟 | 动作 |
|---:|---|
| 0–30 | 读 sklearn [Getting Started](https://scikit-learn.org/stable/getting_started.html)、[Common Pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)；区分训练、验证、测试和 CV |
| 30–80 | 用下表理解模型假设；只查不理解的官方模型文档 |
| 80–95 | 休息 |
| 95–175 | 跑模型对照，检查 Pipeline 与 3 折 CV；用 macro-F1 选择模型 |
| 175–220 | PCA、K-means；读官方 [降维](https://scikit-learn.org/stable/modules/decomposition.html#pca) 与 [聚类](https://scikit-learn.org/stable/modules/clustering.html) 的相关小节 |
| 220–270 | 整理比较表，冻结方案后才可加 --test；口头解释失败条件 |

## 先看假设，再看分数

| 模型 | 需要抓住的机制 | 当天深度 |
|---|---|---|
| LogisticRegression | 线性决策得分 + 分类似然 + 正则化 | 接上 Day 2 Softmax |
| kNN | 距离近的样本投票 | 解释缩放和 k 的影响 |
| GaussianNB | 类条件特征独立、高斯分布假设 | 能写贝叶斯公式，知道假设可能不成立 |
| SVM | 间隔与惩罚、核隐含的非线性 | 比较缩放和 C，不推全部对偶 |
| 决策树 | 局部特征划分、叶节点预测 | 看 max_depth 与过拟合 |
| 随机森林 | 多棵随机化的树汇总 | 理解与单棵树的区别 |
| 梯度提升 | 后续模型逐步修正当前目标的残差/梯度 | 概念与调用，不实现整套算法 |

这些解释是入口。结果只针对本次 digits 数据和默认/指定参数，不能推出“某算法普遍最好”。

## 实验

```bash
python -m labs.classical
# 所有开发集比较结束、选模规则冻结之后才运行
python -m labs.classical --test
```

数据为 sklearn 自带 8×8 digits，与前两天 Fashion-MNIST 不同，**不能直接用准确率跨数据集排名**。固定 80% 开发集、20% 留出测试；开发集内部做分层 3 折。StandardScaler 放在 Pipeline 内，使每折都只拟合该折训练数据。

1. 读训练入口，找出为什么没有先对全体 `X` 做标准化。
2. 把运行结果填成“模型 / CV macro-F1 均值 / 折间标准差 / 假设 / 易失败情况”表。
3. 选一个因素：SVM 的 C、树的 max_depth 或 kNN 的 k，只在开发集/CV 比较。修改代码后记录具体差异。
4. 查看 PCA 保留 95% 方差所需的维数；改成二维画图。PCA 是方差目标，不保证分类最优。
5. K-means 拟合不使用标签；ARI 只用于事后比较。聚类编号可以任意置换，不能直接当分类标签算 accuracy。
6. 讨论现实中的按人/动物/设备分组与按时间划分。若同一动物相邻片段随机散在训练和测试中，性能可能高估。

本仓库为节省时间只对分类模型做 CV，PCA/K-means 的图和 ARI 是开发集探索结果，不是独立泛化测试。

## 验收

- 为什么还需要 dummy baseline？
- 为什么树常常不需要特征标准化，而距离/正则化相关模型通常需要？
- 如果类别严重不平衡，accuracy 有什么问题？macro-F1 在平均什么？
- 用测试集选超参数和用验证集选超参数差在哪？
- CV 的标准差是否就是置信区间？（不是，折之间也非完全独立。）

交付：一张比较表、一项单变量对照、三条关于数据泄漏/评估边界的说明。
