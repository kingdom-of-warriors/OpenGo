## TODO list
- [X] 实现自我对弈的并行过程
- [ ] 自我对弈时增加 **温度** 这个参数
- [X] 实现一个直观的`print_board`函数
- [ ] 实现价值网络的训练
- [ ] 实现蒙特卡洛搜索

## 安装环境
```bash
conda create -n opengo -n python=3.10
conda activate opengo
pip install -r requirements.txt
```

## 深度学习训练policy network
### 数据集
选择了[computer-go-dataset](https://github.com/yenw/computer-go-dataset)，包含超过200万局围棋对局记录
- 人类对局：Professional2000+
- AI 对局：AlphaGo系列、绝艺、DeepZenGo、Leela等AI对局及自对弈数据
- 数据格式：已处理为标准SGF格式
- 许可证：GPL-3.0
- 仓库地址: `https://huggingface.co/datasets/jiarui1/GoDataset/tree/main`

#### 使用
1. 下载并解压到本地
```bash
cd OpenGo
mkdir GoDataset # 把AI和Human两个文件夹放在GoDataset路径下并解压
```
2. 通过`data_utils/read.py`解析为带有历史信息的pt文件
```bash
python data_utils/read.py --sgf_dir GoDataset/AI/ --output_dir GoDataset/AI_pt # 处理AI棋谱
python data_utils/read.py --sgf_dir GoDataset/Human/ --output_dir GoDataset/Human_pt # 处理人类棋谱
```

### 深度学习训练策略网络
1. 策略网络支持多卡ddp训练，启动命令如下，$NUM_GPUS是你所有的显卡数量：
    ```bash
    torchrun --standalone --nproc_per_node=$NUM_GPUS sl_train_dl/sl_train.py # AI 人类数据集全部训练，一个进程大约需要125G内存
    torchrun --standalone --nproc_per_node=$NUM_GPUS sl_train_dl/sl_train.py --data_dirs "GoDataset/AI_pt/" # 只训练AI数据集，一个进程32G内存
    torchrun --standalone --nproc_per_node=$NUM_GPUS sl_train_dl/sl_train.py --data_dirs "GoDataset/Human_pt/" # 只训练人类数据集，一个进程93G内存
    ```
    最终模型在测试集上预测下一步的准确率会达到 51.6% 左右，比`alphago`论文中的只用历史信息的网络稍低一些(55.7%)。

2. 在训练完策略网络之后，你可以与它激情对局一盘！
    ```bash
    python sl_train_dl/eval.py --ckpt_path ckpt/AI_12_192.pth
    ```

### 强化学习训练策略网络


