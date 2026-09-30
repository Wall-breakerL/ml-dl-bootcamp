# Day 3：卷积、残差与可信的训练结果（270 分钟）

核心问题：图像结构怎样改变网络设计？如何知道训练结果值得相信？

| 分钟 | 动作 |
|---:|---|
| 0–15 | 回忆 logits、loss、梯度与正则化 |
| 15–65 | D2L [图像卷积](https://zh.d2l.ai/chapter_convolutional-neural-networks/conv-layer.html)、[填充和步幅](https://zh.d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html)、[多通道](https://zh.d2l.ai/chapter_convolutional-neural-networks/channels.html)、[池化](https://zh.d2l.ai/chapter_convolutional-neural-networks/pooling.html)；用小矩阵手算 |
| 65–100 | [LeNet](https://zh.d2l.ai/chapter_convolutional-neural-networks/lenet.html)、[残差网络](https://zh.d2l.ai/chapter_convolutional-modern/resnet.html)；画结构与 shape |
| 100–115 | 休息 |
| 115–205 | 训练 LeNet、对比 MLP；小样本过拟合和保存/加载检查 |
| 205–245 | [Karpathy 训练经验](https://karpathy.github.io/2019/04/25/recipe/) 中数据、简单基线、overfit 小数据段；检查错例；按需查 D2L [GPU](https://zh.d2l.ai/chapter_deep-learning-computation/use-gpu.html) 与 [读写文件](https://zh.d2l.ai/chapter_deep-learning-computation/read-write.html) |
| 245–270 | 三日验收与日志 |

## 手算与复现

对核大小 `k`、padding `p`、stride `s`，无 dilation 时输出大小为 `floor((n+2p-k)/s)+1`。单个卷积层参数量为 `Cout*(Cin*k*k+1)`（带 bias）。卷积共享权重；池化不学习同样的参数。

```bash
python -m labs.vision --model lenet --epochs 10 --lr 0.003
python -m labs.vision --model mlp --train-size 64 --epochs 100
# 有时间再运行缩小残差示例；不把它称为 ResNet-18
python -m labs.vision --model residual --epochs 10
```

LeNet 使用 sigmoid/平均池化，较深的 sigmoid 结构可能对初始化、学习率敏感；不要求它必然优于 Day 2 的 MLP。若曲线不动，先检查 logits、梯度和小样本能否拟合，再逐一改学习率/激活函数。

1. 逐层列出 LeNet 的 shape：`28→28→14→10→5→400→120→84→10`。用实际 Tensor 前向核对。
2. 用 4×4 输入和 2×2 核手算一次二维互相关，和 `Conv2d` 对照。
3. 对 64 个样本的 MLP 关闭正则化，尝试接近拟合训练集。小样本拟合是排错工具，不是泛化成绩。
4. 保存/加载 `state_dict` 后，检查同一批输入预测是否一致。工作本有入口。
5. 观察验证集错例，写出两种容易混淆的类别和可能原因；不能只归因于“模型太小”。
6. 有余力再读 `x+F(x)`：当 `F` 初始很小时网络能保留什么？维度不同时为什么需要投影？

## 三日验收

给你新的一批 `[B,1,28,28]` 图片，能否从头写出模型、DataLoader、训练/验证循环，并解释每行代码？给出训练/验证曲线、参数量、至少一组对照和错误预测分析。

完成前三天后，可以称为“打通基础监督学习与神经网络训练”，传统模型全景和时序/Transformer 留到后两天。最终模型与超参数冻结后才运行一次 `--test`；测试表现不理想也如实记录。
