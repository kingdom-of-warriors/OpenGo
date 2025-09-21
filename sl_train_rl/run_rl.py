
import torch
import os
import sys

# 确保项目根目录在sys.path中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.insert(0, project_root)

from models.policy_networks import create_model
from config import parse_args
from sl_train_rl.trainer import train

def main():
    """主函数：初始化并启动训练。"""
    args = parse_args()
    args.rl_lr = 5e-5 
    args.num_iterations = 10000
    args.minibatch = 4 
    args.save_enemy = 50
    args.save_model = 100
    args.max_step = 360

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Reinforcement learning rate: {args.rl_lr}")

    # 创建和加载主模型
    model = create_model(args, device)
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {args.ckpt_path}")
    os.makedirs(args.enemies_ckpt_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # 启动训练
    train(model, args, device)

if __name__ == '__main__':
    main()