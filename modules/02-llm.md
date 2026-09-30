# M2：大语言模型基础

预计 35–50 小时。前置：[M1](01-deep-learning.md) 的 Transformer、交叉熵、mask 和训练诊断。来源：[Happy-LLM](https://github.com/datawhalechina/happy-llm)，固定版本见 [来源锁定](../references/sources.lock.json)。**本模块当前是教学设计，尚未新增或验证 LLM 实验。**

## 章节映射

| 内容 | Happy-LLM | 预算 |
|---|---|---:|
| NLP、tokenizer、语言建模、PLM/LLM 图景 | 第 1–4 章；第 2 章 Transformer 以复习和查漏为主 | 6–9h |
| decoder、next-token loss、数据流、小模型训练 | 第 5 章 | 10–14h |
| Transformers、SFT、PEFT、LoRA/QLoRA | 第 6 章 | 10–14h |
| 评估、RAG/Agent 最小例、误差分析 | 第 7 章 | 9–13h |

第 8 章 GRPO、OPD 等在 [M3](03-reinforcement-learning.md) 后选读，不计入本轮必须完整复现的范围。

## 实验设计

1. 选一个固定的小型文本集，明确文档划分、tokenizer、特殊 token 和 padding；跟踪一个 batch 从文字到输入/标签的位移。
2. 从小型 decoder 训练开始，检查 loss 和生成结果，能解释 teacher forcing 与推理的区别。规模、步数和停止条件在开跑前按服务器实测确定；不预设达到上游 215M 模型的效果。
3. 选一个单卡可承受的现成小模型做一次 LoRA/SFT，冻结训练集、提示模板和评估问题集，比较微调前后；记录模板泄漏与遗忘风险。
4. 用同一组问题比较基础回答和一个最小 RAG 流程；Agent 做工具调用机制演示，不扩成生产系统。

## 环境与预算边界

上游建议按章节分环境：第 5 章单卡，第 6 章有多卡建议和可缩小调试范围；第 8 章依赖远程训练服务及账户/额度，不能把其费用当作本地 GPU 租金。现有 D2L 的 Python 3.12 环境不直接作为全部章节兼容承诺。开始时按固定版本的环境说明另建环境，先做最小检查，再定规模。

## 完成门槛

解释 token、embedding、causal mask、next-token loss 的关系；区分预训练、SFT、偏好/RL 后训练与 RAG；能指出哪些参数被 LoRA 更新；给出固定问题集上可追溯的前后比较，说明小型实验不能证明什么。输出自己的数据流图、一个训练记录和错误分析。
