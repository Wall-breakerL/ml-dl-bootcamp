# AI 基础学习空间

以五套开源材料为来源，按共同基础 → 深度学习 → LLM / 强化学习 → 具身智能组织长期学习。保留 `ml-dl-bootcamp` 仓库名，前三天是起步课，后续逐步积累推导、代码、实验和复述记录。

**目标是能解释机制、独立改写关键步骤，并在小任务上验证。** 按已有 Python 和大学基础数学估计，去掉重复内容后约需 **200–300 小时**；每天 4–5 小时，约 40–75 个学习日。这不包含逐题完成五套材料、完整训练大模型或掌握具身指南的全部外链课程。

入口：[长期路线](docs/ROADMAP.md) · [五套来源与阅读顺序](docs/RESOURCES.md) · [前三天课表](docs/CURRICULUM.md) · [算力与预算](docs/COMPUTE_BUDGET.md) · [学习进度](notes/PROGRESS.md)

**当前接续（2026-09-30）：** 正在 [Day 0：1–2 小时预习](lessons/day00.md)，暂时停止出题，以阅读和解释为主。新会话先读 [接续说明](notes/SESSION_HANDOFF.md) 和学习进度。[补充材料调研](docs/SUPPLEMENTARY_RESOURCES.md)保留为候选，尚未全部纳入必修。

**框架约定：各模块的神经网络实现以 PyTorch 为主。** 入门阶段优先原生 PyTorch，先理解完整训练循环，再按需使用其上层工具。其他框架的资料用于参考原理，复现时优先选择或改写为 PyTorch；数学与数据处理可用 NumPy，传统机器学习保留 scikit-learn，仿真器按任务选型。

## 教学模块

| 模块 | 内容与来源 | 预计学习时间 | 仓库当前准备情况 |
|---|---|---:|---|
| [M0 共同基础](modules/00-foundations.md) | 数学、PyTorch、传统 ML、数据划分；D2L + sklearn | 20–30h | 回归和传统 ML 参考实验可用 |
| [M1 深度学习](modules/01-deep-learning.md) | D2L 主线，吴恩达补充优化、诊断和项目方法 | 50–70h | CNN、序列、注意力的小型参考实验可用 |
| [M2 大语言模型](modules/02-llm.md) | Happy-LLM；复用 M1 的 Transformer 基础 | 35–50h | 已设计章节映射与任务，实验待建设 |
| [M3 强化学习](modules/03-reinforcement-learning.md) | Easy-RL；MDP、价值、策略、DQN/PPO | 35–50h | 已设计章节映射与任务，实验待建设 |
| [M4 具身智能](modules/04-embodied-ai.md) | Embodied-AI-Guide；仿真、模仿学习、一个操作任务 | 35–55h | 已限定范围与验收，环境和实验待建设 |
| [M5 综合复盘](modules/05-integration.md) | 串联模型、目标、数据、决策与评估 | 15–25h | 复盘问题和交付要求已列出 |

逐模块估算合计 190–280 小时，整体预算取整为 200–300 小时。前三天的 13.5 小时已经包含在 M0/M1 中，不重复计算。学习时长均为规划估计，不是实测完成时长。

## 先完成前三天

| 天 | 主题 | 时间（含休息） | 当天交付 |
|---|---|---:|---|
| [Day 1](lessons/day01.md) | 张量、梯度、线性回归、数据划分 | 4.5h | 手算/自动微分对照、全批量梯度下降与解析解、学习率对照 |
| [Day 2](lessons/day02.md) | Softmax、交叉熵、MLP、训练循环 | 4.5h | 相同数据上的两个分类基线、自己写的核心循环 |
| [Day 3](lessons/day03.md) | 评估、卷积、LeNet、结果分析 | 4.5h | 逐层 shape、一个 CNN 实验、验证集错例分析 |

进入 Day 1 且恢复练习后，可做 [15 分钟自测](docs/CURRICULUM.md#开始前的自测)。若 Day 2 结束仍无法独立解释训练循环，Day 3 用于巩固 MLP，卷积顺延。原 `day04` / `day05` 保留为传统 ML 与序列/注意力的后续专题，完成时间由前置能力决定。

## 开始学习

已准备的服务器课程目录是 `/root/autodl-tmp/ml-dl-bootcamp`：

```bash
cd /root/autodl-tmp/ml-dl-bootcamp
source .venv/bin/activate
python -m labs.environment
python -m labs.linear
```

打开 [Day 1 工作本](notebooks/day01.ipynb)，按 lesson 推导、改写和复述。[服务器说明](docs/SERVER.md)包含连接方式；[实测记录](docs/VERIFICATION.md)区分准备检查和正式学习实验。前两天的数学和小型 CPU 练习可在本地完成，但本地 Python 环境尚未核验。所有个人作答区保持待填写。

## 仓库怎么用

- `modules/`：长期模块的前置能力、阅读映射、实验任务与验收。
- `lessons/`、`notebooks/`：起步课和已有专题工作本。
- `labs/`：短小、可读的参考实现；训练循环不隐藏在 `d2l` helper 中。
- `notes/`：学习日志、阶段检查与进度，以实际作答为准。
- `references/sources.lock.json`：五个上游仓库的固定版本及核验范围；从资料页访问原文。
- `runs/`、`data/`：实验输出与数据缓存，默认不进 Git。

五套材料以来源索引、固定版本和教学映射纳入；原文、原始代码与许可在各自上游保留，不整库复制。D2L 是 DL 主线，吴恩达用于补充，重复的反传/CNN/Attention 不重学一遍。科学空间用于当前机制问题的定向阅读。

每次实验生成独立目录。模型选择只看训练/验证，最终配置冻结后才使用 `--test`。已有残差 CNN 是缩小示例；序列与注意力使用合成任务，不等同于完整 ResNet、语言模型或机器翻译复现。版权与改编范围见 [ATTRIBUTION](ATTRIBUTION.md)。
