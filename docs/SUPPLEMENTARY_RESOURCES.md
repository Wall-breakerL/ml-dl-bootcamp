# 补充材料调研：按缺口选读

整理日期：2026-09-30。以下保存本次讨论中的候选建议，**尚未全部纳入必修**。英文材料可用，讲解使用中文；神经网络实践以 PyTorch 为主。现有五套来源及科学空间入口仍见 [资料索引](RESOURCES.md)。

目的不是再完整读一批书，而是补足统计判断、机制解释、可读实现与机器人基础。能解释同一概念的材料优先互相替换，避免叠加课时。长期 200–300 小时的估计不包含完整学习下列所有课程。

## 六个优先讨论的补充来源

| 对应模块 | 来源 | 补什么、先读哪里 | 与当前路线的关系 |
|---|---|---|---|
| M0 共同基础 | [An Introduction to Statistical Learning with Applications in Python（ISLP）](https://www.statlearning.com/) | 统计学习、泛化、模型评估与选择；先第 2、5、6 章的关键内容，再按模型需要选回归、分类、树方法 | 补足仅运行 sklearn 示例不容易建立的统计判断；传统 ML 用 Python/sklearn，不要求转成 PyTorch |
| M1 深度学习 | [Understanding Deep Learning — Simon J. D. Prince](https://udlbook.github.io/udlbook/) | 先第 5 章损失、第 6 章拟合、第 7 章梯度；以后按需要补正则化与注意力 | 作为 D2L 的第二解释和练习来源，不额外顺序学完整本书；实践挑 PyTorch 或用于数值理解的 NumPy 示例 |
| M2 大语言模型 | [Build a Large Language Model (From Scratch) — Sebastian Raschka](https://github.com/rasbt/LLMs-from-scratch) | 第 2 章文本、第 3 章注意力、第 4 章 GPT、第 5 章预训练；之后选指令微调与 LoRA | Happy-LLM 提供路线，这套 PyTorch 代码用于逐步读懂实现；公开代码与需购买的正式书正文要区分 |
| M3 强化学习 | [Reinforcement Learning: An Introduction, 2nd ed. — Sutton & Barto](https://mitpress.mit.edu/9780262039246/reinforcement-learning/) | 第 3 章 MDP、第 4–6 章 DP/MC/TD，再读第 13 章策略梯度 | 给 Easy-RL 补清楚回报、价值、采样与自举的概念关系；先理论和小型表格实验，再进入深度 RL |
| M4 具身智能 | [Modern Robotics — Lynch & Park](https://modernrobotics.org/) | 第 2 章构型、第 3 章刚体运动、第 4 章正运动学、第 5 章雅可比、第 6 章逆运动学 | 补足坐标系、位姿和动作表示，再接具身指南中的仿真与策略；运动学计算可用 NumPy/Python |
| M5 综合复盘 | [Full Stack Deep Learning 2022](https://fullstackdeeplearning.com/course/2022/) | 实验管理 Lab 4、Lecture 3 Troubleshooting & Testing、Lecture 4 Data Management | 把检查数据、小样本记忆、跟踪实验和错例分析变成习惯；按主题使用，不整体重建旧课程环境 |

其中 **ISLP 与 Modern Robotics 是优先建议补入的两个缺口**；这仍是讨论建议。其余优先用于当前章节的替换或补讲。前三天的结构不变，不在 Day 0 同时打开六套材料。

## 遇到具体问题再查

| 来源 | 使用时机 | 范围与框架 |
|---|---|---|
| [Mathematics for Machine Learning](https://mml-book.github.io/) | 数学概念卡住时 | 第 2 章线性代数、第 4 章矩阵分解、第 5 章向量微积分、第 6 章概率、第 7 章优化；按需查阅，不从头补完才能开始 |
| [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1) | 已理解基础模型，开始使用训练工具时 | 选 Transformers、Datasets、Tokenizers 和微调；神经网络示例选 PyTorch，与 Happy-LLM 重复部分合并 |
| [CleanRL](https://docs.cleanrl.dev/) | 理解 DQN/PPO 后，想把公式映射到短代码时 | 先经典控制任务的 PyTorch `dqn.py`、`ppo.py`；跳过 JAX 变体，不以 Atari 大规模基准作为入门要求 |
| [MIT Robotic Manipulation — Russ Tedrake](https://manipulation.csail.mit.edu/) | 需要把感知、位姿、规划与控制串起来时 | 选 Basic Pick and Place 与 Geometric Pose Estimation；Drake 相关内容先作阅读参考，不自动替换既有仿真任务 |
| [The RLHF Book — Nathan Lambert](https://rlhfbook.com/) | M2、M3 都有基础，准备理解模型对齐时 | 串联 SFT、奖励模型、策略优化和直接偏好方法；阅读不意味着立刻安排完整 RLHF 训练 |

[Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) 放到第二轮深化。它要求较强的 Python、深度学习和系统基础，完整作业会明显扩展范围，不属于当前 200–300 小时的默认必修任务。

## 本次调研覆盖与限制

- 核对了官方课程/作者/出版社入口及对应目录，部分代表章节已阅读；未声称通读全部书，也未运行这些课程的实验。新增来源还没有纳入五套仓库的固定版本文件。
- ISLP 代表性查阅：[第 5 章交叉验证实验](https://intro-stat-learning.github.io/ISLP/labs/Ch05-resample-lab.html)，用于理解划分变化与估计结果的关系；官网提供书籍与 Python 实验入口。
- Full Stack Deep Learning 代表性查阅：[Lecture 3](https://fullstackdeeplearning.com/course/2022/lecture-3-troubleshooting-and-testing/)，重点为数据检查、小样本记忆和失败分析。2022 年工具版本不能直接当作当前环境配置。
- CleanRL 代表性查阅：[DQN 文档](https://docs.cleanrl.dev/rl-algorithms/dqn/)，用于定位实现、目标网络与日志；尚未完成本机或服务器兼容性验证。
- Sutton & Barto 的出版社目录已核对，作者站直连曾失败；未成功下载并阅读全文。Understanding Deep Learning 的官方页面正文抓取也有限，不能据此声称所有配套代码均已检查。
- 进入具体模块时再确认对应版本、可访问的章节、许可、依赖与实验规模。资料索引不包含收费正文的镜像，也不提前估计未实测的 GPU 耗时。
