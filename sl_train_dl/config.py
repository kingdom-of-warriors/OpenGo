import argparse

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='围棋策略网络训练')
    
    # 必需参数
    parser.add_argument('--model', type=str, choices=['resnet'], 
                       default='resnet', help='模型类型 (resnet)')
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='学习率 (默认: 0.001)')
    parser.add_argument('--data_num', type=int, default=10, 
                       help='数据集数量')
    parser.add_argument('--input_channels', type=int, default=27, help='输入通道数 (默认: 19)')
    
    # 可选参数
    parser.add_argument('--batch_size', type=int, default=128, 
                       help='批处理大小 (默认: 128)')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='训练轮数 (默认: 100)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                       help='权重衰减 (默认: 1e-4)')
    parser.add_argument('--data_dir', type=str, default='GoDataset/AI_pt', 
                       help='数据集目录')
    parser.add_argument('--scheduler', type=str, default='steplr')
    parser.add_argument('--ckpt_dir', type=str, default='ckpt/')
    
    # ResNet 特定参数
    parser.add_argument('--resnet_blocks', type=int, default=8, 
                       help='ResNet 残差块数量 (默认: 8)')
    parser.add_argument('--resnet_filters', type=int, default=128, 
                       help='ResNet 滤波器数量 (默认: 128)')

    
    return parser.parse_args()