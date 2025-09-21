import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from tqdm import tqdm
from collections import deque
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.policy_networks import create_model
from config import parse_args
from rl_utils import game_state_to_tensor
from dlgo.goboard_slow import GameState, Move
from dlgo.gotypes import Player, Point

# --- 2. 数据生成 ---
def generate_one_game_sample(sl_model, rl_model, board_size=19, history_length=8):
    """
    完整复现 AlphaGo 论文中为价值网络生成一个训练样本的过程。
    返回: (s_U+1 的特征张量, 最终胜负 z)
    """
    game_state = GameState.new_game(board_size)
    history = deque(maxlen=history_length)
    
    # 随机选择一个时间步 U
    U = random.randint(1, 350)
    s_U_plus_1 = None

    # 阶段一: 使用 SL 模型下棋 (1 到 U-1 步)
    for t in range(1, U):
        if game_state.is_over():
            break
        
        history.append(game_state)
        input_tensor = game_state_to_tensor(game_state, history, history_length).unsqueeze(0).to(device)
        
        # 使用 SL 模型采样
        with torch.no_grad():
            move_probs = sl_model.sample(input_tensor).squeeze(0)
        
        # 从合法走法中采样
        legal_moves = game_state.legal_moves()
        if not legal_moves: break
        
        legal_indices = [move.point.row * board_size + move.point.col for move in legal_moves]
        masked_probs = torch.zeros_like(move_probs)
        masked_probs[legal_indices] = move_probs[legal_indices]
        if masked_probs.sum() > 0:
            masked_probs /= masked_probs.sum()
        else: # 如果所有合法位置概率都为0，则均匀选择
            masked_probs[legal_indices] = 1.0 / len(legal_indices)

        move_idx = torch.multinomial(masked_probs, 1).item()
        point = Point(row=move_idx // board_size, col=move_idx % board_size)
        game_state = game_state.apply_move(Move.play(point))

    # 阶段二: 随机下一步 (第 U 步)
    if not game_state.is_over():
        history.append(game_state)
        legal_moves = game_state.legal_moves()
        if legal_moves:
            random_move = random.choice(legal_moves)
            game_state = game_state.apply_move(random_move)
            # 保存这个关键的局面 s_U+1
            s_U_plus_1 = game_state
            
    # 如果棋局在随机步骤前就结束了，或者随机步骤后无法再走，则此样本无效
    if s_U_plus_1 is None or game_state.is_over():
        return None, None

    # 阶段三: 使用 RL 模型下棋 (U+1 到 T 步)
    while not game_state.is_over():
        history.append(game_state)
        input_tensor = game_state_to_tensor(game_state, history, history_length).unsqueeze(0).to(device)
        
        # 使用 RL 模型采样
        with torch.no_grad():
            move_probs = rl_model.sample(input_tensor).squeeze(0)

        legal_moves = game_state.legal_moves()
        if not legal_moves: break

        legal_indices = [move.point.row * board_size + move.point.col for move in legal_moves]
        masked_probs = torch.zeros_like(move_probs)
        masked_probs[legal_indices] = move_probs[legal_indices]
        if masked_probs.sum() > 0:
            masked_probs /= masked_probs.sum()
        else:
            masked_probs[legal_indices] = 1.0 / len(legal_indices)

        move_idx = torch.multinomial(masked_probs, 1).item()
        point = Point(row=move_idx // board_size, col=move_idx % board_size)
        game_state = game_state.apply_move(Move.play(point))
        
    # 确定最终胜负
    winner = game_state.winner()
    # z 是从 s_U+1 的视角来看的胜负
    if winner == s_U_plus_1.next_player:
        z = 1.0
    else:
        z = -1.0
        
    # 提取 s_U+1 的特征
    # 注意：需要为 s_U+1 创建正确的历史记录
    s_U_plus_1_history = deque(list(history)[:U], maxlen=history_length)
    feature_tensor = game_state_to_tensor(s_U_plus_1, s_U_plus_1_history, history_length)

    return feature_tensor, z

# --- 3. 训练主循环 ---
def train_value_network(args, device, sl_model, rl_model, value_model):
    """价值网络的主训练循环"""
    optimizer = optim.Adam(value_model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # 根据论文，训练了5000万个批次，这里我们只示意性地训练一部分
    num_steps = args.num_steps
    batch_size = args.batch_size
    save_interval = args.save_interval
    
    value_model.train()

    pbar = tqdm(range(num_steps))
    for step in pbar:
        batch_states = []
        batch_outcomes = []
        
        # 动态生成一个批次的数据
        while len(batch_states) < batch_size:
            state, outcome = generate_one_game_sample(sl_model, rl_model, args.board_size, args.history_length)
            if state is not None:
                batch_states.append(state)
                batch_outcomes.append(outcome)

        states_tensor = torch.stack(batch_states).to(device)
        outcomes_tensor = torch.tensor(batch_outcomes, dtype=torch.float32).unsqueeze(1).to(device)

        # 训练步骤
        optimizer.zero_grad()
        predicted_values = value_model(states_tensor)
        loss = loss_fn(predicted_values, outcomes_tensor)
        loss.backward()
        optimizer.step()

        pbar.set_description(f"Step {step+1}/{num_steps} | Loss: {loss.item():.4f}")

        # 定期保存模型
        if (step + 1) % save_interval == 0:
            os.makedirs("ckpt", exist_ok=True)
            checkpoint_path = f"ckpt/value_net_step_{step+1}.pth"
            torch.save({
                'step': step + 1,
                'model_state_dict': value_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            print(f"\n模型已保存至: {checkpoint_path}")


if __name__ == '__main__':
    # --- 初始化和加载模型 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    args = parse_args()
    
    # 修改 input_channels 以匹配我们的特征提取器
    args.input_channels = args.history_length * 2 + 1
    
    # 加载sl_model
    sl_model = create_model(args, device)
    sl_model_path = "ckpt/sl_policy_net.pth" # 假设这是你的SL模型路径
    print(f"加载SL模型: {sl_model_path}")
    sl_ckpt = torch.load(sl_model_path, map_location=device)
    sl_model.load_state_dict(sl_ckpt['model_state_dict'])
    sl_model.eval()

    # 加载rl_model
    rl_model = create_model(args, device)
    rl_model_path = "ckpt/rl_policy_net.pth" # 假设这是你的RL模型路径
    print(f"加载RL模型: {rl_model_path}")
    rl_ckpt = torch.load(rl_model_path, map_location=device)
    rl_model.load_state_dict(rl_ckpt['model_state_dict'])
    rl_model.eval()

    # 创建value_model
    args.model = 'value'
    value_model = create_model(args, device)
    print("价值网络已创建。")
    
    # --- 开始训练 ---
    train_value_network(args, device, sl_model, rl_model, value_model)
