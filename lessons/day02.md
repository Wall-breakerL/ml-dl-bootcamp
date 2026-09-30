# Day 2：从线性分类到 MLP（270 分钟）

核心问题：分类为何使用交叉熵，隐藏层改变了什么？本日属于 [M1](../modules/01-deep-learning.md)。

| 分钟 | 动作 |
|---:|---|
| 0–20 | 关闭笔记复述训练步骤，修正 Day 1 的一个问题 |
| 20–60 | 看 Fashion-MNIST 样本/标签，读 D2L [Softmax 回归](https://zh.d2l.ai/chapter_linear-networks/softmax-regression.html) 的输入、logits 与概率 |
| 60–100 | [Softmax 从零实现](https://zh.d2l.ai/chapter_linear-networks/softmax-regression-scratch.html) 的交叉熵和数值稳定性；手写一个例子，最大似然卡点可选读科学空间 4277 相应段 |
| 100–115 | 休息 |
| 115–150 | [MLP](https://zh.d2l.ai/chapter_multilayer-perceptrons/mlp.html) 与 [反向传播](https://zh.d2l.ai/chapter_multilayer-perceptrons/backprop.html) 的关键链式法则 |
| 150–225 | 运行两个相同数据的基线，读曲线与训练循环 |
| 225–255 | 独立改写 minibatch 更新和验证步骤，标注 shape |
| 255–270 | 复述和记录，判断是否能进入 Day 3 |

## 机制与形状

图像 `[B,1,28,28]` 展平为 `[B,784]`，分类器输出 `[B,10]` logits；标签是 `[B]` 的整数类别。

\[
p_k=\frac{e^{z_k}}{\sum_j e^{z_j}},\quad L=-\log p_y,
\quad \frac{\partial L}{\partial z_k}=p_k-\mathbf1[k=y].
\]

用减去最大值或 `logsumexp` 实现稳定计算。`cross_entropy(logits, labels)` 接收 logits，不先重复 Softmax。多个线性层若没有非线性，仍可合成一个线性映射；MLP 的 ReLU 改变了表达能力。

## 两个必做实验

```bash
python -m labs.vision --model softmax --train-size 6000 --epochs 10 --seed 42
python -m labs.vision --model mlp --train-size 6000 --epochs 10 --seed 42
```

两者使用同一个固定 54k/6k 划分，再从训练部分取同一批 6,000 样本；验证仍为 6,000。学习率、epoch、优化器和 seed 保持相同，默认优化器为 Adam。GPU 仅用于集中实验；手算与写代码可以本地做。

1. 看 16 张训练图片、类别和标签分布。
2. 对 logits `[2,0,-1]`、标签 0 手写稳定 Softmax 和负 log 概率，与 `cross_entropy` 对齐。
3. 重写一次 minibatch 训练循环，解释 `zero_grad/backward/step`；验证时使用 `eval()` 和 `no_grad()`。
4. 比较训练/验证曲线。loss 与 accuracy 单位不同，严谨比较时分开画；不要求 MLP 在这一次小实验中必然更好。

## 验收与降载

能解释为什么 argmax 后的 0/1 错误不适合作为普通反传目标，为什么去掉 ReLU 会退化，为什么 loss 降低但 accuracy 可以不变。能说明 `eval()` 与 `no_grad()` 分别改变什么，且能不看参考代码写出核心循环。

交付两个基线的结果和自己的实现。若还做不到，Day 3 用于复写与排错，卷积顺延。Dropout/L2、2,000 样本过拟合和更多消融均放在 M1 后续，工作本中的相关命令默认注释。测试集保持不用于调参。
