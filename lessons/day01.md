# Day 1：模型到底是怎样学出来的（270 分钟）

核心问题：从一组 `(X,y)` 到可以预测新样本的参数，中间究竟发生了什么？

## 时间表

| 分钟 | 动作 | 到哪里就停 |
|---:|---|---|
| 0–15 | 完成课表中的五项自测，运行环境检查 | 写下薄弱项，不临时补整套数学 |
| 15–60 | [D2L 张量](https://zh.d2l.ai/chapter_preliminaries/ndarray.html)、[线性代数](https://zh.d2l.ai/chapter_preliminaries/linear-algebra.html)、[微积分](https://zh.d2l.ai/chapter_preliminaries/calculus.html)、[概率](https://zh.d2l.ai/chapter_preliminaries/probability.html) 快速选读 | 掌握 shape、矩阵乘法、链式法则、期望/方差与条件概率；会的部分略过 |
| 60–105 | [自动微分](https://zh.d2l.ai/chapter_preliminaries/autograd.html)、[线性回归](https://zh.d2l.ai/chapter_linear-networks/linear-regression.html)、[从零实现](https://zh.d2l.ai/chapter_linear-networks/linear-regression-scratch.html)；科学空间 [4277](https://www.spaces.ac.cn/archives/4277) 的梯度下降段 | 把训练拆成预测、loss、求梯度、更新、清梯度 |
| 105–120 | 休息 | |
| 120–210 | 完成下面的手算、代码与学习率对照 | 保存至少两条曲线 |
| 210–235 | 跑 dummy + LogisticRegression，对比回归与分类 | 先知道传统 ML 也在做拟合和泛化评估 |
| 235–270 | 关掉代码复述、填日志、做验收 | 不理解的问题保留到次日开头 |

## 公式对应什么代码

输入 `X:[N,d]`、`w:[d,1]`、`b:[1]`、`y:[N,1]`。预测是 `X @ w + b`，均方误差是 `mean((prediction-y)**2)`。其梯度为

\[
\nabla_w L=\frac{2}{N}X^T(Xw+b-y),\qquad
\frac{\partial L}{\partial b}=\frac{2}{N}\sum_i(\hat y_i-y_i).
\]

沿负梯度更新是局部一阶近似下的选择，步长过大可能发散。D2L 中平方损失常带 `1/2`；本实验使用普通 MSE，所以梯度多出系数 2。对齐 loss 定义后再比较学习率。

## 复现任务

```bash
python -m labs.linear
python -m labs.linear --lr 0.01
python -m labs.classical --quick
```

1. 在工作本手算 `x=2,y=5,w=1,b=0` 的 loss、两个梯度及一步更新，随后用 autograd 核对。
2. 读 `labs/linear.py`，标出输入、参数、计算图、更新语句、验证和解析解。此处为便于比较使用全批量梯度下降；再自己改成 minibatch，理解 SGD 的噪声来源。
3. 不复制参考代码，在空白单元重写预测、loss、更新三步；可以调用 `backward()`。接着把权重梯度换成手写公式。
4. 先预测 `lr=0.01` 相比 `0.1` 的变化，再比较验证曲线。扩展：`lr=2` 观察不稳定，不把发散当程序安装错误。
5. 解释为什么先划分再拟合；对任何预处理只在训练部分拟合。经典基线使用 3 折 CV，此时无需深入所有模型。

`runs/.../metrics.json` 中的 `w,b,least_squares,manual_autograd_max_error` 是核对入口。默认例子的真实参数 `[2,-3.4], b=4.2` 已知，因此能检查参数恢复；真实任务通常没有这个条件。

## 验收与复述

- 不看代码讲一遍：输入 → 前向 → 损失 → 梯度 → 更新 → 验证。
- 为什么 `y:[N]` 和预测 `[N,1]` 相减可能得到错误的 `[N,N]`？
- 为什么 `backward()` 不会自己更新参数？为什么每步要清梯度？
- 训练误差低意味着什么？为什么不直接用测试集调学习率？

交付：自己的核心实现、一张学习率对照图或两份运行路径、200 字机制解释。不会这些内容就把 Day 2 的扩展阅读让给复习。
