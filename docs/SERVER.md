# 服务器与实验使用

课程目录：`/root/autodl-tmp/ml-dl-bootcamp`。机器登录地址与密码不写入仓库。

2026-09-30 长期课程调整时，SSH 连接被拒绝，新课表尚未同步到服务器副本。下次连接成功后先同步 GitHub 上的新源码；初始环境与实验记录见下文。

## 已核验配置

2026-09-30：Ubuntu 22.04.5，NVIDIA RTX 4090 24564 MiB，驱动 580.105.08，CUDA runtime 13.0，Python 3.12.3。容器 `cpu.max` 对应 16 核额度，`memory.max` 对应 120 GiB；宿主 `free` 展示的约 1 TiB 不是本容器可用配额。数据盘检查时约 50GB 可用。

预装 PyTorch 2.12.1+cu130、torchvision 0.27.1+cu130，CUDA 前向/反向已实测。新建 `.venv` 使用 `--system-site-packages` 复用这套可用 GPU 包，在 venv 中补 pandas/sklearn 等课程依赖，不升级系统 torch。它在依赖层面仍继承基础环境，基础环境改变后需要重新验证。

具体版本快照见 `requirements-server.lock.txt`。这是已验证服务器的版本记录，不是跨平台万能 lockfile。未验证用它在全新机器一键重建；换机器时先按 [PyTorch 官方安装入口](https://pytorch.org/get-started/locally/) 选择匹配 CUDA 的 torch/torchvision，再安装 `requirements.txt`。

## 终端方式

使用租赁平台给出的 SSH 命令登录，在服务器内运行：

```bash
cd /root/autodl-tmp/ml-dl-bootcamp
source .venv/bin/activate
python -m labs.environment
python -m labs.linear
```

不要依赖非交互 SSH 的默认 PATH，直接使用 `.venv/bin/python` 更可靠。`nvidia-smi` 的 CUDA Version 是驱动支持信息，实验中还需要检查 `torch.version.cuda` 与 `torch.cuda.is_available()`。

## Jupyter / VS Code

最直接的方式是 VS Code Remote-SSH 连接服务器，打开课程目录，选择解释器 `.venv/bin/python` 和 `ML-DL Bootcamp` 内核，打开 `notebooks/day01.ipynb`。

使用浏览器时，在服务器终端运行：

```bash
cd /root/autodl-tmp/ml-dl-bootcamp
bash scripts/start_jupyter.sh
```

在自己的电脑另开终端，保留原 SSH 命令的用户名、主机和端口，添加本地转发：

```bash
ssh -N -L 8890:127.0.0.1:8890 -p YOUR_PORT YOUR_USER@YOUR_HOST
```

打开 `http://127.0.0.1:8890`，使用启动终端显示的 token。服务只绑定回环地址并保留 Jupyter 认证。结束后分别 Ctrl-C 停掉 Jupyter 与转发；课程没有设置常驻训练或自动开机任务。

## 数据与运行时间

Fashion-MNIST 缓存在 `data/`，60,000 个训练样本、10,000 个测试样本。参考实验从原训练集固定抽出 6,000 验证样本。下载与校验：

```bash
python scripts/download_data.py
```

传统 ML 用 sklearn 自带的 digits，不联网下载。时序与注意力任务合成数据，便于短时间检查机制。`--smoke` 只检查少量步骤；完整实验的 epoch 数以每日日程为准。

训练日志保存到新的 `runs/<时间>-<实验>/`，JSON 包含参数和版本；部分实验有曲线和模型文件。运行前用 `nvidia-smi` 看是否有其他任务；当前起步课不需要多卡或大模型下载；后续模块按各自规模另行准备。

## 保存学习成果

本地仓库是 GitHub 的推送端。服务器有同一 Git 源码快照，但未配置 GitHub 账号令牌。可以在本地运行：

```bash
# 替换登录参数，将笔记与小型指标同步回来再检查提交。
scp -P YOUR_PORT -r YOUR_USER@YOUR_HOST:/root/autodl-tmp/ml-dl-bootcamp/notes ./server-notes
```

正式提交自己的笔记、改写代码和少量整理后的指标。数据、权重、登录信息和完整运行目录不纳入 Git。删除租赁实例前先保存需要的内容；系统/数据盘的关机保留政策以平台页面为准。

## 长期模块与按需开机

[算力预算](COMPUTE_BUDGET.md)区分学习时间、训练时间和实例计费时间。前三天统一缩小图片训练样本数，正式配置的耗时尚未测量。阅读、推导和小型 CPU 练习可以本地完成，但本地 Python 环境还未验证。

当前 `.venv` 只对已有参考实验做过检查。Happy-LLM、RL 和机器人仿真以后使用独立环境，在各模块开始时核验版本、显存、磁盘与运行规模。
