# 🎯 DQN 作业说明

## 任务概述

请在 `alg/agent.py` 中实现 DQN 算法的关键部分，**主要是 `update()` 方法**。完成后，你可以运行训练脚本 `train.py` 来训练模型，并使用 `evaluate.py` 评估训练结果。

## 获取作业仓库

1. **点击链接 → 自动创建私有仓库，如 studentA-DQN-Assignment。**

2. **在本地执行：**
```bash
git clone https://github.com/sysu-qhxx/DQN-Assignment.git
git checkout -b dev
```

3. **修改 agent.py 的 TODO 部分。**

4. **提交并推送：**
```bash
git add .
git commit -m "Implement DQN update"
git push origin dev
```

5. **提交 pull request 到 `main` 分支。**
---
## 环境准备

以下步骤以 Windows/macOS/Linux 通用命令为准。请在终端（Anaconda Prompt 或 系统终端）中逐行执行。

1. **创建 conda 环境（推荐 Python 3.11）**

```bash
conda create -n dqn python=3.11 -y
```

2. **激活新环境**

```bash
conda activate dqn
```


3. **进入项目目录（假设你已把仓库 clone 到本地）**

```bash
cd DQN-Assignment
```

4. **安装依赖**

使用 `requirements.txt`：

```bash
pip install -r requirements.txt
```
5. **（可选）如果你想使用 GPU，请根据你的 CUDA 版本安装对应的 PyTorch。**
   建议访问 [https://pytorch.org](https://pytorch.org) 获取官方安装命令。若不确定或使用 CPU，下面的 pip 安装命令也可以。
如果你更愿意用 conda 安装 PyTorch（推荐 GPU 用户），可以先用 conda 安装 pytorch，再 `pip install` 其余依赖：

```bash
# 示例（请按官方页面选择具体命令）
conda install pytorch torchvision torchaudio -c pytorch
pip install -r requirements.txt
```

---

## 文件与目录说明

```
DQN-Assignment/
│
├── alg/
│   ├── network.py        # Q 网络结构
│   ├── replay_buffer.py  # 经验回放缓冲区
│   └── agent.py          # TODO: 完成 update() 方法
│
├── train.py              # 训练入口脚本
├── evaluate.py           # 评估脚本
├── tests/
│   └── test_update.py    # 单元测试（用于自动批改）
├── requirements.txt
└── README.md
```

---

## 如何运行（按步骤）

1. **确认当前激活 conda 环境为 `dqn`**：

```bash
conda info --envs
# 或者直接看命令行提示符，确保显示 (dqn)
```

2. **安装完成后，先运行单元测试（本地自测）**：

* 使用 `unittest`：

```bash
python -m unittest -v tests.test_update.test_update_logic_exists
```

* 或者使用 `pytest`（如果你安装了 pytest）：

```bash
pytest -q
```

3. **训练模型（用于观察训练过程）**：

```bash
python train.py
```

训练结束会保存模型为 `dqn_model.pt`（脚本中有保存语句）。

4. **评估训练结果**：

```bash
python evaluate.py
```

---

## 单元测试说明（自动化批改参考）

* `tests/test_update.py` 会检测 `DQNAgent.update()` 中是否仍含 `pass`，以及在经验池样本充足时是否能正常运行。
* 教师可在 GitHub Actions 中配置自动运行 `pytest` 或 `unittest` 来实现自动批改。详见课程说明。

---

## 常见问题与排查

* **没有找到 conda 命令**：请先安装 Miniconda 或 Anaconda，并重启终端。
* **PyTorch 安装出现问题**：请参考 PyTorch 官方安装页面，根据你的 CUDA 版本选择正确命令；若使用 CPU，可安装 CPU 版本。（cpu版本安装快速且简单）
* **依赖冲突或安装失败**：建议创建全新 conda 环境（如上所示）并重试。
* **单元测试通过但训练不收敛**：DQN 收敛受超参数和随机种子影响较大。建议先确认 `update()` 的实现是否正确，再调小学习率或增加训练轮数测试。

---

## 作业提交要求

1. 在自己的仓库（或分支）中完成 `dqn/agent.py` 的实现。
2. 确保 `pytest` 或 `unittest` 中的测试通过。
3. 提交包含以下内容：

   * `dqn/agent.py` 完整实现
   * 可选：`train.log` 或训练曲线截图
   * 提交信息中写明你实现了哪些部分


