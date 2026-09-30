# 资料筛选：读什么、为什么读、读到哪里

核验日期：2026-09-30。以下是面向 3–5 天预算的精选，不是全网经验调查。技术结论优先回到教材、作者原文和官方文档；个人笔记只作为学习组织和踩坑线索。

## 主线教程

| 资料 | 用法 | 本次安排 |
|---|---|---|
| [D2L 中文第二版](https://zh.d2l.ai/) / [指定 GitHub 仓库](https://github.com/d2l-ai/d2l-zh) | 章节顺序、公式、模型和代码相互对应 | Day 1–3 主线，Day 5 定向选读 |
| [李沐课程页](https://courses.d2l.ai/zh-v2/) | 官方视频与章节入口 | 只看当前卡住的内容，替代部分阅读时间 |
| [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) | Tensor、Dataset、autograd、优化循环、保存/加载 | API不熟时作为第二解释来源 |
| [sklearn Getting Started](https://scikit-learn.org/stable/getting_started.html) | fit/predict、Pipeline、交叉验证 | Day 1 快速基线与 Day 4 |
| [sklearn Common Pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) | 预处理一致性、泄漏、随机性 | Day 4 必看；拆分后再拟合预处理 |

D2L 中文仓库在本次核验时的 `master` 是 `e6b18ccea71451a55fcd861d7b96fddf2587b09a`，提交日期为 2023-08-18。固定版本的 [安装说明](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_installation/index.md) 使用 Python 3.9、torch 1.12.0、torchvision 0.13.0、d2l 0.17.6；[setup.py](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/setup.py) 也固定了旧 NumPy/Pandas。因此课程采用现代环境的独立教学实现，避免把旧依赖覆盖到当前 GPU 环境。网页更新状态与这个固定源码版本不应混为一谈。

## 科学空间：带着问题读两篇主文

1. [《梯度下降和 EM 算法：系出同源，一脉相承》](https://www.spaces.ac.cn/archives/4277)（2017-03-23）。Day 1 只读牛顿迭代与梯度下降段，回答“为什么更新式里有负梯度与步长”；Day 2 读最大似然段，连接概率乘积、log 和交叉熵；Day 4 有余力再看 K-means 段。每次约 15–20 分钟。**边界：** 文中把若干迭代思想统称为 EM 的讲法较宽泛，不等于标准统计定义下梯度下降就是 EM；正步长也不自动保证收敛，要考虑光滑性、步长和目标函数。
2. [《Attention is All You Need》浅读（简介+代码）](https://spaces.ac.cn/archives/4765/comment-page-1?replyTo=8871)（原站可访问的分页入口）。Day 5 阅读注意力公式和模型组成，用自己的 `[B,H,T,D]` 张量逐项对照。文章中的历史框架代码用来理解结构，课程代码使用 PyTorch；不把旧实现命令直接搬入环境。

扩展线索：[《梯度流：探索通向最小值之路》](https://www.spaces.ac.cn/archives/9660)、[《缓解交叉熵过度自信的一个简明方案》](https://www.spaces.ac.cn/archives/9526)。适合结课后分别追问优化轨迹和概率校准；本次只获取到搜索索引的部分正文，未完成全文核验，故不列必读，也不据此安排复现。

访问限制：本次 `kexue.fm` 多个直连返回 403。4277 通过原站 `spaces.ac.cn` 获取并阅读了指定正文；4765 最初返回文章索引/页面信息，但正文复查又返回 403，故作为定向对照阅读，公式与代码以 D2L 和本课程的数值检查为准，不声称已逐行核验该博客全文。未访问到的扩展文章也没有伪装成已精读。

## 训练经验与个人学习笔记

| 来源 | 吸收什么 | 如何转成课程动作 |
|---|---|---|
| [Karpathy: A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)（2019） | 先理解数据，建立简单基线，再逐步增加复杂度；训练可能静默出错 | 看样本与标签分布；小样本过拟合；一次改一个变量；检查曲线和错例 |
| [FanZheng：《动手学深度学习》学习记录](https://fanzheng.org/post/d2l-notes.html)（2026-07-11） | 学习者按章节记录疑问与补充解释，并报告旧依赖/辅助函数问题 | 保留“问题—推导—实验—边界”日志；其安装建议与技术断言需另行核对，不直接当标准答案 |
| [Karpathy: Neural Networks Zero to Hero](https://github.com/karpathy/nn-zero-to-hero) | 用小例子逐步搭建神经网络的实践路线 | 反传卡住时选 micrograd 片段；完整系列放在五天之后 |

本课表的时间压缩方案是针对你预算的设计，不是这些作者声称的“三天学完”。Karpathy 2019 文中的部分时代性判断不作为今天的方法推荐；个人学习笔记也不能证明代码或数学解释都正确。

## 传统 ML 按需查阅

- [监督学习总览](https://scikit-learn.org/stable/supervised_learning.html)：从对应模型名称跳转，不顺序读整章。
- [交叉验证](https://scikit-learn.org/stable/modules/cross_validation.html)：训练/验证/测试的用途与分层、分组、时序划分。
- [PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)：降维保留方差，不能保证保留最有用的分类信号。
- [聚类](https://scikit-learn.org/stable/modules/clustering.html)：K-means 假设和聚类指标。
- [Fashion-MNIST 作者仓库](https://github.com/zalandoresearch/fashion-mnist)：数据源；下载脚本使用作者发布的四个文件，并核对 torchvision 给定的 MD5。

以上 sklearn 细分页是官方补充阅读入口；主线已核验 Getting Started 与 Common Pitfalls。后续深入某个模型时应再对照当前 API。
