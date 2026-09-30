# 序列与注意力专题（保留 day05 入口）

归属 [M1](../modules/01-deep-learning.md)。原单日时间表保留为参考，可拆成多个学习时段；须在前置能力具备后使用，不承诺在第 5 个学习日完成本专题。

核心问题：过去的信息如何传到当前预测？Attention 究竟在混合什么？

| 分钟 | 动作 |
|---:|---|
| 0–15 | 回忆矩阵乘法、交叉熵和 residual |
| 15–55 | D2L [序列模型](https://zh.d2l.ai/chapter_recurrent-neural-networks/sequence.html)、[RNN](https://zh.d2l.ai/chapter_recurrent-neural-networks/rnn.html)、[BPTT](https://zh.d2l.ai/chapter_recurrent-neural-networks/bptt.html)、[GRU](https://zh.d2l.ai/chapter_recurrent-modern/gru.html) 的机制；LSTM只看门的作用 |
| 55–95 | 时序预测实验与 last-value 基线 |
| 95–110 | 休息 |
| 110–155 | D2L [注意力评分](https://zh.d2l.ai/chapter_attention-mechanisms/attention-scoring-functions.html)、[多头注意力](https://zh.d2l.ai/chapter_attention-mechanisms/multihead-attention.html)、[位置编码](https://zh.d2l.ai/chapter_attention-mechanisms/self-attention-and-positional-encoding.html)；科学空间 [4765](https://spaces.ac.cn/archives/4765/comment-page-1?replyTo=8871) 按可访问情况对照 |
| 155–205 | 手写缩放点积 Attention，核对 shape、Softmax 维度、causal mask |
| 205–255 | 读 D2L [Transformer](https://zh.d2l.ai/chapter_attention-mechanisms/transformer.html) 的结构部分；运行微型 block，检查训练/验证 |
| 255–300 | 完成结课题与 10 分钟复述，留下后续问题 |

## 序列实验

```bash
python -m labs.sequence --model rnn --epochs 15
python -m labs.sequence --model gru --epochs 15
```

输入 `[batch,24,1]`，输出下一个时刻的一个标量。合成序列由两个正弦和噪声组成；原始时间轴先拆成 3600/1200/1200 三段，再在每段内部切窗口，防止重叠窗口跨划分共享观测。

比较神经网络与“下一步等于最近值”的基线。没胜过基线也要如实记录，检查信号频率、噪声和训练充分性。这个简单任务不能证明 GRU 普遍优于 RNN，更不能代表真实语言建模能力。

画 `h_t=f(x_t,h_{t-1})`，解释时间展开后的链式法则。GRU 的门控制保留与更新信息；LSTM 另有 cell state。今天不要求从零推导所有门的梯度。

## Attention 实验

\[
A=\operatorname{softmax}(QK^T/\sqrt{d_k}+M),\qquad O=AV.
\]

把每个维度写清：Q、K、V 为 `[B,H,T,D]`，分数为 `[B,H,T,T]`；最后一维对应每个 query 可选择的 key。causal mask 令未来位置分数为负无穷，Softmax 后对应权重为 0。

```bash
python -m labs.attention --steps 400
```

参考脚本包含三项：手写结果与 PyTorch SDPA 数值对照；改变未来 K/V 不影响过去位置的检查；一层微型 Transformer 在首词元检索任务上的训练。

任务输入为 8 个随机数字词元，加一个末尾查询词元；要求末尾输出第一个数字。这个合成任务使跨位置取信息容易观察，**不是 D2L 的完整 encoder-decoder 翻译，也不是 GPT 训练**。block 包含 QKV 投影、多头拼接、残差、LayerNorm、FFN 和可学习位置参数。

1. 在工作本独立写 `scores → mask → softmax → weights @ V`，与参考函数比较。
2. 手算 `Q:[1,1,2,2]` 的一个两词元例子；说明第二行为什么能看第一行，第一行为什么不能看第二行。
3. 去掉 mask 后重新做“扰动未来”检查，预测哪条断言会失败。这个小实验直接检验因果约束。
4. 画一个 block 的输入输出，不把 Q/K/V 当成预先存在于文本里的三个不同东西。
5. 可选消融：去掉位置参数。因果结构也能携带某些顺序信息，因此结果不能当作“所有模型必须显式位置编码”的普遍证明。

## 结课

- RNN 的递归状态与 self-attention 的两两交互有何区别？
- 缩放 `sqrt(d_k)` 想控制什么？Softmax 前分数过大会怎样？
- 注意力矩阵为什么随序列长度平方增长？
- residual、LayerNorm、FFN 分别做什么？Attention 不等于整个 Transformer。
- 此处分类目标与语言模型的 next-token prediction 有什么不同？

交付：两种时序模型与简单基线对照、自己的 Attention 实现、mask 检查结果、微型任务的验证曲线、`notes/FINAL_CHECK.md` 的个人答案。
