# 机器学习与深度学习：5 天基础速学

以《动手学深度学习》中文版 PyTorch 路线为主线，配合苏剑林科学空间的机制解释和 scikit-learn 的传统机器学习实验。每天 4–5 小时，推荐 **5 天约 23 小时**；3 天完成前半段核心流程。

**这次的“完成”指：能够解释基础机制，独立改写关键步骤，在小数据集上复现代表性实验，并给出可靠评估。** 15–25 小时不能等同于学完整个机器学习领域、逐章复现 D2L 或具备研究级熟练度。

入口：[完整课表](docs/CURRICULUM.md) · [资料筛选与阅读顺序](docs/RESOURCES.md) · [服务器使用](docs/SERVER.md) · [实测检查记录](docs/VERIFICATION.md) · [学习进度](notes/PROGRESS.md)

## 每天做什么

| 天 | 主题 | 时间（含休息） | 当天交付 |
|---|---|---:|---|
| [Day 1](lessons/day01.md) | 张量、梯度、线性回归、数据划分 | 4.5h | 手算/自动微分对照、SGD 与解析解、学习率实验 |
| [Day 2](lessons/day02.md) | Softmax、交叉熵、MLP、正则化 | 4.5h | Fashion-MNIST 分类、训练/验证曲线、一次消融 |
| [Day 3](lessons/day03.md) | 卷积、LeNet、残差、训练调试 | 4.5h | CNN 基线、逐层 shape、错误预测分析 |
| [Day 4](lessons/day04.md) | 传统 ML、交叉验证、PCA、K-means | 4.5h | 模型比较表、无泄漏 Pipeline、泛化说明 |
| [Day 5](lessons/day05.md) | RNN/GRU、注意力、Transformer | 5h | 时序基线、手写 Attention、微型 Transformer 实验 |

默认前提：会写 Python 函数/类，能读矩阵乘法、求导和基础概率。先做 [15 分钟自测](docs/CURRICULUM.md#开始前的自测)，据此降载。这里按天编号，不绑定日历日期。

## 开始学习

服务器上的课程目录已经准备在 `/root/autodl-tmp/ml-dl-bootcamp`。进入后：

```bash
cd /root/autodl-tmp/ml-dl-bootcamp
source .venv/bin/activate
python -m labs.environment
python -m labs.linear
```

再打开 [Day 1 工作本](notebooks/day01.ipynb)，按照对应 lesson 完成推导、改写和复述。Jupyter/VS Code 的连接方式见 [服务器说明](docs/SERVER.md)。运行参考代码只是第一步；所有工作本的个人作答区仍待你完成。

## 仓库怎么用

- `lessons/`：每日精确到分钟的安排、阅读章节、练习和验收。
- `notebooks/`：每日工作本，含实验入口和独立作答区。
- `labs/`：短小、可直接阅读的参考实现；训练循环不隐藏在 `d2l` helper 中。
- `notes/`：学习日志、结课检查和进度；学习完成状态由你实际作答决定。
- `references/sources.lock.json`：D2L 固定版本、资料核验日期。
- `runs/`：运行自动生成的参数、指标、曲线；默认不进 Git。
- `data/`：缓存 Fashion-MNIST；不进 Git。

每次实验生成独立目录，不覆盖已有结果。课程默认只观察验证集；确定最终配置后才加 `--test`。不要为了超过一个目标数字反复查看测试集。

## 与原教材的关系

D2L 中文源码固定为 [`e6b18ccea714`](https://github.com/d2l-ai/d2l-zh/tree/e6b18ccea71451a55fcd861d7b96fddf2587b09a)。课程保留基础模型与训练概念，使用服务器现有 PyTorch，并补入传统 ML 和评估训练。

`linear` / `vision` 是基础算法的教学改编；残差 CNN 是缩小示例，**不是完整 ResNet**。`sequence` 使用合成时序；`attention` 使用首词元检索任务，**不是语言模型或机器翻译复现**。原书各章仍是阅读与进一步复现的依据，参见 [覆盖层级](docs/CURRICULUM.md#覆盖层级与延期内容)。

资料版权归各作者。仓库不转载科学空间或经验贴全文；来源与改编说明见 [ATTRIBUTION](ATTRIBUTION.md)。
