# Day 3：卷积入门与可信的训练结果（270 分钟）

核心问题：怎样利用图像结构，怎样解释一个训练结果？前置是能独立解释 Day 2 训练循环；否则本日先巩固 MLP，卷积顺延。本日属于 [M1](../modules/01-deep-learning.md)。

| 分钟 | 动作 |
|---:|---|
| 0–15 | 回忆输入、logits、loss 和参数更新 |
| 15–55 | 用 Day 2 曲线讲训练/验证差别，区分数据、优化与泛化问题；读 [Karpathy](https://karpathy.github.io/2019/04/25/recipe/) 的数据与简单基线片段 |
| 55–95 | D2L [卷积](https://zh.d2l.ai/chapter_convolutional-neural-networks/conv-layer.html)、[填充/步幅](https://zh.d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html)、[通道](https://zh.d2l.ai/chapter_convolutional-neural-networks/channels.html)、[池化](https://zh.d2l.ai/chapter_convolutional-neural-networks/pooling.html) 的必要概念；手算小矩阵 |
| 95–110 | 休息 |
| 110–150 | [LeNet](https://zh.d2l.ai/chapter_convolutional-neural-networks/lenet.html)：画网络与逐层 shape |
| 150–205 | 运行一个 LeNet 实验，检查权重加载后的预测；按需查 [保存/加载](https://zh.d2l.ai/chapter_deep-learning-computation/read-write.html) 与 [GPU](https://zh.d2l.ai/chapter_deep-learning-computation/use-gpu.html) |
| 205–240 | 画验证集错例，比较基线，解释至少两个失败样本 |
| 240–270 | 三日检查与复述 |

## 手算与实验

无 dilation 时，输出空间大小是 `floor((n+2p-k)/s)+1`。带 bias 的卷积层参数量是 `Cout*(Cin*k*k+1)`。卷积共享权重；池化没有相同形式的可学习核。

```bash
python -m labs.vision --model lenet --train-size 6000 --epochs 10 --lr 0.003 --seed 42
```

复用 Day 2 的训练/验证样本。LeNet 使用 sigmoid/平均池化，其学习率设为 0.003；因此与 MLP 的比较是不同配置的基线观察，不能把差异全部归因为卷积。记录超参数，不能要求它必然胜过 MLP。

1. 逐层列出 shape：空间尺寸 `28→28→14→10→5`，展平后 `400→120→84→10`；同时写出通道数和 batch 维度，用前向输出核对。
2. 用 4×4 输入和 2×2 核手算二维互相关，对照 `conv2d`。
3. 保存/加载 `state_dict`，对同一批输入检查预测一致性；这只验证保存路径，不证明模型学得好。
4. 找 16 个验证集错例，分析两种容易混淆的类别及可能原因，区分观察与猜测。

如果 loss 不动，先检查输入/标签、loss 和梯度；时间不足就保留排错记录。小样本长训练、残差网络、更多超参数搜索放到 M1 后续，不为凑实验数量占满本日。

## 三日验收

面对 `[B,1,28,28]` 输入，能写出模型和核心训练/验证步骤，解释 shape、数据划分、指标与模型差异。交付自己的核心代码、至少一次单变量对照（Day 1 学习率）、三类模型的验证记录和错例分析。

完成 [阶段检查](../notes/FINAL_CHECK.md) 的第 1–4、6 题，口述 10 分钟。前三天只验收基础监督训练；传统 ML 全景、序列/Transformer、LLM、RL 和具身均按长期路线后续学习。此阶段不看测试集。
