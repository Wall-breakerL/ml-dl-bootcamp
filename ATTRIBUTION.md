# 来源与改编说明

- 课程主线来源：[Dive into Deep Learning 中文版](https://github.com/d2l-ai/d2l-zh)，作者 Aston Zhang、Zachary C. Lipton、Mu Li、Alexander J. Smola；固定 commit 见 `references/sources.lock.json`。
- 本仓库的线性回归、Softmax/MLP、LeNet 参考了教材中的基础模型和教学路径，训练/评估/记录代码重新组织。`lenet` 使用教材中的网络层次与 sigmoid/平均池化形式。
- D2L 根目录 [LICENSE](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/LICENSE) 为 Apache-2.0；上游 `setup.py` 的包许可字段另写 MIT-0。保留上游根许可文本于 `references/D2L-LICENSE`，不对上游所有材料重新声明统一许可。
- 科学空间与经验博客仅链接、少量概述和提出阅读问题，不复制正文或图像。
- Fashion-MNIST 从作者 [zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist) 下载，数据不随仓库发布。
- 时序与注意力任务是本课程的合成教学示例，不是教材 NLP 章节原实验。残差网络也是缩小教学示例。
- 本仓库目前为个人学习用途，未为所有新作材料选择统一开源许可；不能把外部内容视为自由转载。

## 长期路线的新增来源

- [fengdu78/deeplearning_ai_books](https://github.com/fengdu78/deeplearning_ai_books)：吴恩达深度学习课程的中文笔记与相关材料；原课程和整理内容归相应权利人。
- [datawhalechina/happy-llm](https://github.com/datawhalechina/happy-llm)：LLM 章节、学习顺序与环境要求的参考。
- [datawhalechina/easy-rl](https://github.com/datawhalechina/easy-rl)：强化学习章节、问题与算法教学的参考。
- [TianxingChen/Embodied-AI-Guide](https://github.com/TianxingChen/Embodied-AI-Guide)：具身领域索引及 RoboTwin/ACT 实战路线的参考。

这里只加入链接、版本索引、课程映射及原创教学安排，未复制上述四个仓库的全文、代码或图片。后续如引入实际代码/数据，需按所用文件的许可与署名要求处理。各自许可以上游固定版本为准；不能把 D2L 的许可推广到其他材料。
