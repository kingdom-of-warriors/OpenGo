import argparse

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='围棋策略网络训练')
    
    # 必需参数
    parser.add_argument('--model', type=str, choices=['resnet', 'winner'], 
                       default='resnet', help='模型类型 (resnet)')
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='学习率 (默认: 0.001)')
    parser.add_argument('--data_num', type=int, default=10, 
                       help='数据集数量')
    parser.add_argument('--input_channels', type=int, default=27, help='输入通道数 (默认: 19)')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=128, 
                       help='批处理大小 (默认: 128)')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='训练轮数 (默认: 100)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                       help='权重衰减 (默认: 1e-4)')
    parser.add_argument('--data_dirs', type=str, nargs='+', default=['GoDataset/AI_pt_2', 'GoDataset/Human_pt'], 
                       help='数据集目录（一个或多个）')
    parser.add_argument('--scheduler', type=str, default='steplr')
    parser.add_argument('--ckpt_dir', type=str, default='ckpt/',
                        help='检查点保存目录 (默认: ckpt/)')
    
    # Model 参数
    parser.add_argument('--resnet_blocks', type=int, default=8, 
                       help='ResNet 残差块数量 (默认: 8)')
    parser.add_argument('--resnet_filters', type=int, default=192, 
                       help='ResNet 滤波器数量 (默认: 192)')
    
    # eval 参数
    parser.add_argument('--ckpt_path', type=str, default=None,
                    help='检查点文件路径')
    
    # self play 参数
    parser.add_argument('--enemies_ckpt_dir', type=str, default='ckpt/enemies/',
                        help='对手检查点保存目录')
    parser.add_argument('--minibatch', type=int, default=16, help='每个minibatch的自对弈数量 (默认: 16)')
    parser.add_argument('--max_step', type=int, default=400, help='最大自对弈手数')
    parser.add_argument('--save_enemy', type=int, default=20, help='每N个minibatch保存一个对手模型 (默认: 20)')
    parser.add_argument('--save_model', type=int, default=20, help='每N个minibatch保存一个自己的模型 (默认: 20)')
    parser.add_argument('--rl_lr', type=float, default=5e-5, help='强化学习学习率 (默认: 5e-5)')
    return parser.parse_args()