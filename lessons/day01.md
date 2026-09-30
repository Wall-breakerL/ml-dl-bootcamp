# Day 1：模型怎样学出来（270 分钟）

核心问题：从一组 `(X,y)` 到能预测新样本的参数，中间发生了什么？本日属于 [M0](../modules/00-foundations.md)。

## 时间表

| 分钟 | 动作 | 到哪里就停 |
|---:|---|---|
| 0–15 | 完成课表自测，确认使用的 Python 环境 | 记录薄弱项 |
| 15–55 | D2L [张量](https://zh.d2l.ai/chapter_preliminaries/ndarray.html)、[线性代数](https://zh.d2l.ai/chapter_preliminaries/linear-algebra.html) 的 shape、广播、矩阵乘法 | 写出 `[N,d] @ [d,1]`，不补整章证明 |
| 55–95 | [微积分](https://zh.d2l.ai/chapter_preliminaries/calculus.html)、[自动微分](https://zh.d2l.ai/chapter_preliminaries/autograd.html) 的导数与链式法则 | 手算一个梯度并核对 |
| 95–110 | 休息 | |
| 110–180 | [线性回归](https://zh.d2l.ai/chapter_linear-networks/linear-regression.html)、[从零实现](https://zh.d2l.ai/chapter_linear-networks/linear-regression-scratch.html)；读并运行本仓库回归 | 预测→loss→梯度→更新→验证；科学空间梯度段按需替换部分阅读 |
| 180–215 | 预测并运行学习率对照，理解训练/验证/测试用途 | 两份结果，不提前学 CV |
| 215–240 | 不复制参考文件，重写核心更新步骤 | 能解释每个张量 |
| 240–270 | 关闭资料复述、写日志 | 留下一个还没想明白的问题 |

## 公式与代码

输入 `X:[N,d]`、`w:[d,1]`、`b:[1]`、`y:[N,1]`，预测是 `X @ w + b`，损失是 `mean((prediction-y)**2)`：

\[
\nabla_w L=\frac{2}{N}X^T(Xw+b-y),\qquad
\frac{\partial L}{\partial b}=\frac{2}{N}\sum_i(\hat y_i-y_i).
\]

负梯度来自局部一阶下降方向，步长过大可能发散。D2L 的平方损失常带 `1/2`，本实验普通 MSE 的梯度多一个系数 2。先对齐定义，再比较学习率。

## 复现任务

```bash
python -m labs.linear
python -m labs.linear --lr 0.01
```

1. 手算 `x=2,y=5,w=1,b=0` 的 loss、两个梯度和 `lr=0.1` 的一步更新，用 autograd 核对。
2. 读 `labs/linear.py`，标出输入、参数、计算图、更新和验证。它每步使用全部训练数据，是全批量梯度下降；minibatch 留到 Day 2。
3. 自己写预测、loss、反传、更新和清梯度，再用手写权重梯度核对 autograd。
4. 先预测 `lr=0.01` 与默认 `0.1` 的区别，再看曲线；其他设置保持一致。
5. 解释训练与验证误差分别回答什么问题，为什么测试集不能用来选学习率。

`runs/.../metrics.json` 中的 `w,b,least_squares,manual_autograd_max_error` 是核对入口。合成数据参数已知，能检查参数恢复；真实任务通常没有这个条件。科学空间 [4277](https://www.spaces.ac.cn/archives/4277) 的梯度段为按需解释，阅读边界见 [资料页](../docs/RESOURCES.md)。

## 验收

不看代码复述一轮训练；解释 `[N]` 与 `[N,1]` 相减的广播风险、`backward()` 为什么不更新参数、为什么需要清梯度。交付自己的核心实现、两份学习率运行路径和约 200 字机制说明。传统 ML 基线、交叉验证和 SGD 噪声分析移到后续模块。
