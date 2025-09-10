## TODO list

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
2. 通过`data_utils/read.py`解析为pt文件
```bash
python data_utils/read.py --sgf_dir GoDataset/AI/ --output_dir GoDataset/AI_pt # 处理AI棋谱
python data_utils/read.py --sgf_dir GoDataset/Human/ --output_dir GoDataset/Human_pt # 处理人类棋谱
```

### 训练代码

