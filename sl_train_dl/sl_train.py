import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import time
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import random
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import parse_args
from sl_train_dl.engine import train_epoch, validate_epoch
from data_utils.go_dataset import GoDataset, WinnerDataset
from models.policy_networks import create_model

def setup_ddp(rank, world_size):
    """初始化DDP环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp(): dist.destroy_process_group()

def seed_worker(worker_id: int):
    """
    为 DataLoader 的工作进程设置独立的随机种子。
    """
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(worker_seed)

def main():
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    setup_ddp(local_rank, world_size)

    if local_rank == 0:
        print("--- 训练参数 ---")
        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")
        print("-" * 20)

    torch.manual_seed(42)
    if args.model == "resnet":
        dataset = GoDataset(data_dirs=args.data_dirs, data_num=args.data_num, board_size=19, enable_augmentation=True)
    # elif args.model == "winner":
    #     dataset = WinnerDataset(data_dirs=args.data_dirs, data_num=args.data_num, enable_augmentation=True)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset= random_split(dataset, [train_size, val_size])
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler, worker_init_fn=seed_worker)
    
    if local_rank == 0: print(f"数据集大小: 训练={len(train_dataset)}, 验证={len(val_dataset)}")

    device = torch.device("cuda", local_rank)
    model = create_model(args, device) 
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == 'steplr':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        start_time = time.time()
        train_loader.sampler.set_epoch(epoch)
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()
        epoch_time = time.time() - start_time
        
        if local_rank == 0:
            print(f'Epoch [{epoch+1}/{args.epochs}] ({epoch_time:.1f}s) | '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | '
                  f'LR: {scheduler.get_last_lr()[0]:.6f}')
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = f'{args.ckpt_dir}/AI_12_192.pth'
                torch.save({'model_state_dict': model.module.state_dict(), 'args': args}, save_path)
                print(f"--> 新的最佳模型已保存至 {save_path} (Val Acc: {val_acc:.2f}%)")

    if local_rank == 0: print("\n--- 训练完成 ---")
    cleanup_ddp()

if __name__ == "__main__":
    main()