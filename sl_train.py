# train_ddp.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import time
import os

# ===> DDP变更 1: 导入必要的分布式训练库 <===
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from config import parse_args
from data_utils.go_dataset import GoDataset
from models.policy_networks import create_model
from engine import train_epoch, validate_epoch

# ===> DDP变更 2: 创建用于初始化和清理分布式环境的函数 <===
def setup_ddp(rank, world_size):
    """初始化DDP环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' # 使用一个未被占用的端口
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """清理DDP环境"""
    dist.destroy_process_group()

def main():
    args = parse_args()
    
    # ===> DDP变更 3: 获取由torchrun注入的环境变量 <===
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"]) # 总进程数，即GPU数量
    
    # 初始化DDP
    setup_ddp(local_rank, world_size)

    if local_rank == 0:
        print("--- 训练参数 ---")
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")
        print("-" * 20)

    # 2. 准备数据
    # 为了保证所有进程使用相同的数据分割，我们在主进程上确定随机种子
    torch.manual_seed(42)
    dataset = GoDataset(data_dir=args.data_dir, data_num=args.data_num, board_size=19, enable_augmentation=True)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # ===> DDP变更 4: 使用 DistributedSampler 为每个进程分配不同的数据子集 <===
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)

    # shuffle=False 因为sampler已经负责了随机化
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, sampler=test_sampler)
    
    if local_rank == 0:
        print(f"数据集大小: 训练={len(train_dataset)}, 验证={len(val_dataset)}, 测试={len(test_dataset)}")

    # 3. 创建模型和优化器
    device = torch.device("cuda", local_rank)
    model = create_model(args, device) 
    
    # ===> DDP变更 5: 使用DDP包装模型 <===
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == 'steplr':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # 4. 训练主循环
    best_val_acc = 0.0
    if local_rank == 0:
        checkpoint_dir = f'ckpt/{args.model}'
        os.makedirs(checkpoint_dir, exist_ok=True)
        print("\n--- 开始DDP训练 ---")
        
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # ===> DDP变更 6: 每个epoch前需要设置sampler的epoch <===
        train_loader.sampler.set_epoch(epoch)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        scheduler.step()
        epoch_time = time.time() - start_time
        
        # ===> DDP变更 7: 只有主进程(rank=0)进行打印和保存 <===
        if local_rank == 0:
            print(f'Epoch [{epoch+1}/{args.epochs}] ({epoch_time:.1f}s) | '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | '
                  f'LR: {scheduler.get_last_lr()[0]:.6f}')
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = f'{checkpoint_dir}/test1.pth'
                torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'args': args}, save_path)
                print(f"--> 新的最佳模型已保存至 {save_path} (Val Acc: {val_acc:.2f}%)")

    if local_rank == 0:
        print("\n--- 训练完成 ---")

    # ===> DDP变更 8: 训练结束后，清理DDP环境 <===
    cleanup_ddp()

if __name__ == "__main__":
    main()