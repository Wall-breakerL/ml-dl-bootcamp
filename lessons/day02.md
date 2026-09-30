# Day 2：从线性分类到多层感知机（270 分钟）

核心问题：分类为什么需要概率和交叉熵？增加隐藏层到底改变了什么？

| 分钟 | 动作 |
|---:|---|
| 0–15 | 不看笔记复述一轮训练；修正 Day 1 的一个问题 |
| 15–60 | [D2L Softmax](https://zh.d2l.ai/chapter_linear-networks/softmax-regression.html)、[从零实现](https://zh.d2l.ai/chapter_linear-networks/softmax-regression-scratch.html)；科学空间 [4277](https://www.spaces.ac.cn/archives/4277) 最大似然段 |
| 60–100 | [MLP](https://zh.d2l.ai/chapter_multilayer-perceptrons/mlp.html)、[反向传播](https://zh.d2l.ai/chapter_multilayer-perceptrons/backprop.html)；用两个神经元解释链式法则 |
| 100–115 | 休息 |
| 115–205 | 画数据样本、训练 Softmax 和 MLP、观察曲线、重写分类训练循环 |
| 205–245 | 读 [过拟合](https://zh.d2l.ai/chapter_multilayer-perceptrons/underfit-overfit.html)、[权重衰减](https://zh.d2l.ai/chapter_multilayer-perceptrons/weight-decay.html)、[Dropout](https://zh.d2l.ai/chapter_multilayer-perceptrons/dropout.html) 的核心式，并完成一次消融 |
| 245–270 | 复述与日志 |

## 机制与形状

Fashion-MNIST 每张图是 `[1,28,28]`，展平为 784 个输入。线性分类器给出 10 个 logits。Softmax 将 logits 变成概率，交叉熵对整数标签取真实类别的负对数概率。

\[
p_k=\frac{e^{z_k}}{\sum_j e^{z_j}},\quad L=-\log p_y,
\quad \frac{\partial L}{\partial z_k}=p_k-\mathbf1[k=y].
\]

数值实现用减去最大值或 `logsumexp`，避免直接指数溢出。PyTorch `cross_entropy(logits, labels)` 接收 logits，不能在其前面再加一次 Softmax。MLP 用非线性打破“多次线性映射仍然是线性”的限制。

## 实验

```bash
python -m labs.vision --model softmax --epochs 10
python -m labs.vision --model mlp --epochs 10
python -m labs.vision --model mlp --train-size 2000 --epochs 20
python -m labs.vision --model mlp --train-size 2000 --epochs 20 --dropout 0.3
```

前两个是基线比较；后两个只改变 Dropout，验证集划分不变。时间不足时后两条只做一组、减少到 10 epochs，记录改动。读 loss 与 accuracy 时注意单位不同，参考图把它们画在一张图只用于观察趋势，严谨比较时自己拆成两张图。

1. 看 16 张训练图、标签分布，确认标签含义。
2. 在工作本手写稳定 Softmax 与负 log 概率，并与 `cross_entropy` 对齐。
3. 重写一次 minibatch 训练循环，给每个张量写 shape。
4. 解释 Dropout 的训练/推理行为；观察正则化可能降低训练准确率，但是否改善验证结果必须看本次实验。
5. 扩展题：只改变 `weight_decay` 做 L2 对照。不要同时改隐藏宽度、学习率和优化器。

## 验收

- 对 logits `[2,0,-1]` 和标签 0，写出概率与 loss 的计算路径。
- 为何分类不能只用 argmax 后的 0/1 错误作为普通反传目标？
- 两层全连接若去掉 ReLU，会怎样？
- 为什么验证时需要 `model.eval()`，以及为什么它不等于 `no_grad()`？
- loss 降低但 accuracy 不变是否可能？给一个具体预测概率例子。

交付：Softmax/MLP 对照、一次消融的假设和结果、自己的 Softmax 与训练循环。测试集保持不用于调参。
