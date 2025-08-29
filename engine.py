import torch
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(train_loader):
        boards = batch['board'].float().to(device)
        move_idx = batch['move'].to(device)
        
        optimizer.zero_grad()
        policy_logits = model(boards)   
        loss = criterion(policy_logits, move_idx)
    
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 累计每个进程自己的统计数据
        total_loss += loss.item() * boards.size(0) 
        _, predicted = torch.max(policy_logits.data, 1)
        total += move_idx.size(0)
        correct += (predicted == move_idx).sum().item()
    
    stats = torch.tensor([total_loss, correct, total], dtype=torch.float32, device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    
    total_loss_global = stats[0].item()
    correct_global = stats[1].item()
    total_global = stats[2].item()
    avg_loss = total_loss_global / total_global if total_global > 0 else 0
    accuracy = 100. * correct_global / total_global if total_global > 0 else 0
    
    return avg_loss, accuracy

def validate_epoch(model, val_loader, criterion, device):
    """验证一个epoch（DDP兼容版）"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            boards = batch['board'].float().to(device)
            move_idx = batch['move'].to(device)
            
            policy_logits = model(boards)
            loss = criterion(policy_logits, move_idx)
            
            total_loss += loss.item() * boards.size(0)
            _, predicted = torch.max(policy_logits.data, 1)
            total += move_idx.size(0)
            correct += (predicted == move_idx).sum().item()
            
    # ===> DDP变更: 同步验证集的统计数据 <===
    stats = torch.tensor([total_loss, correct, total], dtype=torch.float32, device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    
    total_loss_global = stats[0].item()
    correct_global = stats[1].item()
    total_global = stats[2].item()

    avg_loss = total_loss_global / total_global if total_global > 0 else 0
    accuracy = 100. * correct_global / total_global if total_global > 0 else 0
    
    # 只有主进程需要返回准确的指标用于打印和保存
    return avg_loss, accuracy