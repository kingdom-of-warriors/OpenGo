# OpenGo: 一个基于深度学习与强化学习的围棋AI实现

本项目旨在复现 AlphaGo 论文中描述的核心算法，通过监督学习（Supervised Learning）和强化学习（Reinforcement Learning）来训练一个具有竞争力的围棋AI。

## 项目路线图 (Roadmap)
- [X] 实现并行化的自我对弈（self-play）流程
- [X] 在自我对弈中引入**温度（temperature）**参数以增加探索性
- [X] 实现一个更直观的棋盘命令行打印函数 `print_board`
- [ ] 完整实现价值网络（Value Network）的训练
- [ ] 引入蒙特卡洛树搜索（MCTS）以增强决策能力

## 环境安装
推荐使用 Conda 来管理项目环境，以确保依赖库的一致性。

```bash
# 1. 创建并激活Conda环境
conda create -n opengo python=3.10
conda activate opengo

# 2. 安装所有必需的依赖库
pip install -r requirements.txt
```

-----

## 训练流程

整个训练流程分为两个主要阶段：首先通过**监督学习**让模型掌握围棋的基本知识，然后通过**强化学习**让模型在自我对弈中不断进化。

### 第一阶段：监督学习 (Supervised Learning) - 学习专家知识

在这一阶段，我们的目标是让策略网络（Policy Network）通过模仿高质量的人类和AI棋谱，学习围棋的基本走法和布局。

#### 1\. 数据集准备

本项目选用了 [computer-go-dataset](https://github.com/yenw/computer-go-dataset)，这是一个包含超过200万局高质量围棋对局的公开数据集。

  - **数据来源**: 包含职业棋手对局、AlphaGo系列、绝艺、DeepZenGo等顶级AI的对局及自对弈数据。
  - **数据格式**: 所有棋谱均已处理为标准的SGF格式。
  - **下载地址**: `https://huggingface.co/datasets/jiarui1/GoDataset/tree/main`

**操作步骤**:

```bash
# 进入OpenGo项目根目录
cd OpenGo

# 创建数据集文件夹，并将下载好的AI和Human两个文件夹解压至此
mkdir GoDataset
cd GoDataset
unzip AI.zip
unzip Human.zip
```

#### 2\. 数据预处理

原始的SGF棋谱文件需要被转换为模型可以直接读取的PyTorch张量（`.pt`）格式。

```bash
# 处理AI棋谱
python data_utils/read.py --sgf_dir GoDataset/AI/ --output_dir GoDataset/AI_pt

# 处理人类棋谱
python data_utils/read.py --sgf_dir GoDataset/Human/ --output_dir GoDataset/Human_pt
```

#### 3\. 执行监督学习训练

我们使用PyTorch的分布式数据并行（Distributed Data Parallel, DDP）来实现高效的多GPU训练。

  - **启动命令** (`$NUM_GPUS` 请替换为您机器上可用的GPU数量):
    ```bash
    # 使用全部AI和人类数据集进行训练 (内存需求高，约125G/进程)
    torchrun --standalone --nproc_per_node=$NUM_GPUS sl_train_dl/sl_train.py

    # 仅使用AI数据集进行训练 (内存需求中，约32G/进程)
    torchrun --standalone --nproc_per_node=$NUM_GPUS sl_train_dl/sl_train.py --data_dirs "GoDataset/AI_pt/"

    # 仅使用人类数据集进行训练 (内存需求高，约93G/进程)
    torchrun --standalone --nproc_per_node=$NUM_GPUS sl_train_dl/sl_train.py --data_dirs "GoDataset/Human_pt/"
    ```
  - **预期结果**: 训练完成后，模型在测试集上预测下一步的准确率可以达到 **51.6%** 左右，这与AlphaGo论文中仅使用历史信息作为特征的策略网络（55.7%）的结果相近。

#### 4\. 与SL模型对弈

训练完成后，您可以与您亲手训练的AI进行一盘有趣的对局！

```bash
python sl_train_dl/eval.py --ckpt_path ckpt/AI_12_192.pth
```

-----

### 第二阶段：强化学习 (Reinforcement Learning) - 从自我对弈中进化

在监督学习的基础上，我们使用强化学习通过自我对弈的方式，让策略网络不断超越过去的自己，从而进一步提升棋力。

#### 1\. 准备工作

在开始训练之前，我们需要一个初始的“对手”模型，这个对手就是在监督学习阶段训练好的模型本身。

1.  **创建对手文件夹**：

    ```bash
    mkdir -p ckpt/enemies
    ```

2.  **复制初始模型**：
    将您通过监督学习训练得到的模型权重（例如 `ckpt/AI_12_192.pth`）复制一份到 `ckpt/enemies/` 目录下。这将作为智能体在训练初期遇到的第一个对手。

    ```bash
    cp ckpt/AI_12_192.pth ckpt/enemies/opponent_initial.pth
    ```

#### 2\. 参数详解

您可以通过命令行参数来控制强化学习的训练过程：

  * `--ckpt_path`: **[必需]** 指定从哪个监督学习模型开始进行强化学习。
  * `--num_iterations`: 训练的总迭代次数。每完成一个 `minibatch` 的对弈和学习，算作一次迭代。
  * `--minibatch`: 组成一个训练批次所需的对弈棋局数量。例如，设置为 $32$ 意味着程序会完整地进行 $32$ 盘棋的自我对弈，然后用这些数据对模型进行一次更新（alphago原论文是 $128$ ）。
  * `--num_parallel`: **[核心参数]** 设置在**所有可用GPU上**并行运行的自我对弈**棋局总数**。脚本会自动将这些并行任务平均分配给检测到的所有GPU。
  * `--max_step`: 限制每盘棋的最大步数，防止对局无限进行。

#### 3\. 硬件与显存说明

  * **多GPU支持**：脚本会自动通过 `torch.cuda.device_count()` 检测可用的GPU数量，并充分利用它们进行并行计算。
  * **显存占用**：在自我对弈时，为了计算梯度，当前模型的每一步都需要保存计算图，这会占用大量显存。
      * **经验参考**：根据测试，当 `max_step` 设置为360时，单盘棋局大约需要 **14GB** 的显存。
      * **如何估算**：您可以使用这个参考值来估算单张GPU可以承载的并行棋局数。例如，对于一张80G显存的A800，理论上可以承载 `80 / 14 ≈ 5` 盘棋。因此，**单张GPU**的并行数 (`--num_parallel` / `显卡数量`) 建议设置在`1-5`之间。

#### 4\. 启动训练

下面是一个在 **4张A800 (80G)** 服务器上运行的可行配置示例：

```bash
# --ckpt_path: 指定初始模型路径
# --num_parallel 16: 在4张卡上总共并行16盘棋，即每张卡并行4盘 (4 * 14G ≈ 56G显存，在安全范围内)
# --minibatch 32: 设置批次大小为32，意味着需要进行两轮并行生成 (2 * 16 = 32)，才能构成一个minibatch用于训练
python sl_train_rl/run_rl.py \
    --ckpt_path ckpt/AI_12_192.pth \
    --num_parallel 16 \
    --minibatch 32
```


### 第三阶段：价值网络 (Value Network) - 学习局势评估

### 第四阶段：蒙特卡洛树搜索 (MCTS) - 探索与决策
