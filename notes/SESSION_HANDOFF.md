# 新会话接续说明

更新日期：2026-09-30。此文件负责接续规则；具体学习证据以 [PROGRESS.md](PROGRESS.md) 为准。

## 先读什么

1. [AGENTS.md](../AGENTS.md)：协作和框架约定。
2. [README.md](../README.md)：长期模块与仓库准备情况。
3. [PROGRESS.md](PROGRESS.md)：当前学到哪里、哪些内容尚未确认。
4. [Day 0](../lessons/day00.md)：今天 1–2 小时的阅读安排。准备进入正式三天课时再读 [课表](../docs/CURRICULUM.md) 和对应 lesson。

## 当前学习方式

- 用户在 Day 0，已讨论线性预测、参数和简单 MSE 比较；尚未确认整节阅读或独立实验完成。
- 用户明确说“题可以先不做了”。**不要自动出新题、重做诊断、要求复述或填工作本。** 先按用户选定的小节阅读、讲解；明确恢复练习后再继续检查理解。
- 下一个建议入口是 D2L 2.1.1–2.1.4 的 PyTorch 张量内容。若用户直接指定其他阅读位置，以新指示为准。
- 英文材料可以；讨论和讲解用中文。神经网络统一以 PyTorch 为主，入门优先显式的原生训练循环。
- 先讲清输入、具体步骤、输出和原因，再按需要补公式。课程用于覆盖知识，科学空间围绕当前机制问题选读。

## 路线与资料的决定

五套来源及固定版本见 [资料索引](../docs/RESOURCES.md)；模块主线已经整理到 `modules/`。前三天每天约 4.5 小时，只是 M0/M1 的起步课；长期 200–300 小时是规划估计，不是把五套材料逐页学完的承诺。

[新增材料调研](../docs/SUPPLEMENTARY_RESOURCES.md)仍是候选。ISLP、Understanding Deep Learning、LLMs from Scratch、Sutton & Barto、Modern Robotics、Full Stack Deep Learning 等不能自动全部变成新必修课。通过选读或替换重复内容控制总量，后续和用户讨论具体取舍。

## 本地与服务器边界

- 本机仓库：`/Users/wa11/ml-dl-bootcamp`；GitHub：[Wall-breakerL/ml-dl-bootcamp](https://github.com/Wall-breakerL/ml-dl-bootcamp)。
- 已有参考实现和学习工作本；M2–M4 的课程设计不代表实验环境已经搭好。工作本作答区由学习者填写。
- 本地 Python/PyTorch 环境尚未核验；本地已有源码与资料不等于所有实验已能运行。
- 服务器保留初始课程副本；后续更新尚未同步。上次连接被拒绝，本次资料整理没有重新连接，当前开关机与计费状态未知。
- Day 0 不需要 GPU；不要因开启新会话自动连接服务器、启动训练或下载大模型。以后需要实验时先核对 [服务器说明](../docs/SERVER.md) 与 [预算](../docs/COMPUTE_BUDGET.md)。

本次只更新课程、资料索引与接续记录，未替学习者完成作答，也没有新增实验成绩。
