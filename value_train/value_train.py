import torch
import numpy as np
import os
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import parse_args
from data_utils.go_dataset import ValueDataset
from models.policy_networks import create_model

args = parse_args()
args.model = 'value'
device = 'cuda'
value_model = create_model(args, device)

value_dataset = ValueDataset(data_dir='GoDataset/Value', data_num=96, enable_augmentation=True)
total_size = len(value_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size

train_dataset, val_dataset= random_split(value_dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
criterion = nn.MSELoss()
optimizer = optim.Adam(value_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

for epoch in range(100):
    value_model.train()
    train_losses = []
    val_losses = []
    train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{100} [Train]")
    for batch in train_pbar:
        states = batch['states'].float().to(device)
        outcomes = batch['outcomes'].float().to(device)

        optimizer.zero_grad()
        preds = value_model(states)
        loss = criterion(preds, outcomes)
        train_losses.append(loss.item())
        loss.backward()
        clip_grad_norm_(value_model.parameters(), max_norm=1.0)
        optimizer.step()
        train_pbar.set_postfix({"loss": np.mean(train_losses)})                                                                                   
    train_losses = np.array(train_losses)

    value_model.eval()
    for batch in val_dataloader:
        states = batch['states'].float().to(device)
        outcomes = batch['outcomes'].float().to(device)
        with torch.no_grad():
            preds = value_model(states)
            loss = criterion(preds, outcomes)
            val_losses.append(loss.item())
    val_losses = np.array(val_losses)

    print(f"Epoch {epoch+1}/{100} -> Train Loss: {np.mean(train_losses):.4f}, Val Loss: {np.mean(val_losses):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    scheduler.step()
