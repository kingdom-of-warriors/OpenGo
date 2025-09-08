import torch
import glob
import random
import os
import numpy as np
import sys
from tqdm import tqdm
from collections import deque
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dlgo.goboard_slow import GameState, Move
from dlgo.gotypes import Player, Point
from dlgo.scoring import compute_game_result
from models.policy_networks import create_model, PolicyNetwork_resnet
from sl_train_dl.config import parse_args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = parse_args()
args.lr = 1e-4
enemy_dir = args.enemies_ckpt_dir # 敌人模型文件储存路径
model_dir = args.ckpt_dir         # 我方模型文件储存路径
model_path = args.ckpt_path       # 我方模型文件路径

model = create_model(args, device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
os.makedirs(enemy_dir, exist_ok=True)

def game_state_to_tensor(next_player, game_history: deque, board_size=19):
    """
    将GameState和历史状态转换为模型输入张量
    """
    input_tensor = np.zeros((27, board_size, board_size), dtype=np.float32)
    # 第1-3维：当前状态
    if next_player == Player.black:
        input_tensor[0] = (game_history[-1] == 1).astype(np.float32)
        input_tensor[1] = (game_history[-1] == 2).astype(np.float32)
        input_tensor[2] = np.ones((board_size, board_size), dtype=np.float32)
    else:
        input_tensor[0] = (game_history[-1] == 2).astype(np.float32)
        input_tensor[1] = (game_history[-1] == 1).astype(np.float32)
        input_tensor[2] = np.zeros((board_size, board_size), dtype=np.float32)
    
    # 第4-27维：历史状态 (t-1 到 t-12)
    for i in range(min(12, len(game_history) - 1)):
        hist_idx = -i - 2 # 由于是deque，越新的层进来时间越后，所以索引反而越后
        hist_board = game_history[hist_idx]
        if next_player == Player.black:
            input_tensor[3 + i*2] = (hist_board == 1).astype(np.float32)  
            input_tensor[4 + i*2] = (hist_board == 2).astype(np.float32)
        else:
            input_tensor[3 + i*2] = (hist_board == 2).astype(np.float32)
            input_tensor[4 + i*2] = (hist_board == 1).astype(np.float32)
    
    return torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0).to(device)

def get_model_prediction(model: PolicyNetwork_resnet, game_state: GameState, game_history: deque, board_size=19, requires_grad=False):
    """输出模型的落子概率分布，屏蔽掉非法落子"""
    input_tensor = game_state_to_tensor(game_state.next_player, game_history, board_size)
    move_probs = model.sample(input_tensor, requires_grad=requires_grad).squeeze(0) 
    # 屏蔽非法落子
    legal_mask = torch.zeros_like(move_probs)
    for row in range(board_size):
        for col in range(board_size):
            point = Point(row + 1, col + 1)
            if game_state.is_valid_move(Move.play(point)):
                legal_mask[row * board_size + col] = 1.0
    
    move_probs_new = move_probs * legal_mask
    prob_sum = move_probs_new.sum()
    move_probs_new = move_probs_new / prob_sum 
    
    return move_probs_new

def load_random_opponent(args, device):
    """从enemy_path加载一个随机的对手模型"""
    opponent_files = glob.glob(os.path.join(enemy_dir, "*.pth"))
    opponent_ckpt_path = random.choice(opponent_files)
    print(f"加载对手: {opponent_ckpt_path}")
    
    opponent_model = create_model(args, device)
    checkpoint = torch.load(opponent_ckpt_path, map_location=device)
    opponent_model.load_state_dict(checkpoint['model_state_dict'])
    opponent_model.eval()
    return opponent_model

def play_game(current_model: PolicyNetwork_resnet, opponent_model: PolicyNetwork_resnet, max_step: int, board_size=19):
    """
    在游戏过程中直接计算训练模型的loss
    """
    game_state = GameState.new_game(board_size)
    states, moves = [], []
    step_count = 0
    game_history = deque([], maxlen=13) # T -- T-13 共13个棋盘
    training_losses_abs = []  # 存储loss的绝对值

    if random.random() < 0.5:
        players = {Player.white: current_model, Player.black: opponent_model}
        current_model_color = Player.white
    else:
        players = {Player.black: current_model, Player.white: opponent_model}
        current_model_color = Player.black

    pbar = tqdm(total=max_step, desc="Game Progress", leave=False, bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} steps')
    while not game_state.is_over() and step_count < max_step: # 设定max_step后停止
        current_board = np.zeros((board_size, board_size), dtype=int)
        for row in range(1, board_size + 1):
            for col in range(1, board_size + 1):
                stone = game_state.board.get(Point(row=row, col=col))
                if stone == Player.black: current_board[row - 1, col - 1] = 1
                elif stone == Player.white: current_board[row - 1, col - 1] = 2
        
        # 添加新的棋盘状态
        game_history.append(current_board) # T-12 -- T共13个棋盘
        active_model = players[game_state.next_player]
        move_probs = get_model_prediction(active_model,
                                          game_state,
                                          game_history,
                                          board_size, 
                                          requires_grad=(active_model == current_model))
        move_idx = torch.multinomial(move_probs, 1).item()  # 根据概率随机采样下一步
        if active_model == current_model:
            loss_abs = -torch.log(move_probs[move_idx] + 1e-8)
            training_losses_abs.append(loss_abs)

        point = Point(move_idx // board_size + 1, move_idx % board_size + 1)
        move = Move.play(point)
        states.append(game_state)
        moves.append(move)
        game_state = game_state.apply_move(move)
        step_count += 1
        pbar.update(1)
    
    pbar.close()
    # 计算结果和loss
    game_result = compute_game_result(game_state)
    current_model_won = (game_result.winner == current_model_color)
    training_losses = []
    z_t = +1.0 if current_model_won else -1.0
    training_losses = [loss * z_t for loss in training_losses_abs]

    return {
        'current_model_won': current_model_won,
        'training_losses': training_losses,
    }

def train(num_iterations=1000):
    """强化学习主训练循环（已使用梯度累积进行优化）"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for i in range(num_iterations):
        print(f"--- 第 {i + 1} 轮 (Minibatch) ---")
        opponent_model = load_random_opponent(args, device)
        minibatch = 8
        wins = 0
        accumulated_loss_value = 0.0
        optimizer.zero_grad()
        
        # 评估模式生成最优质数据
        model.eval()
        for j in tqdm(range(minibatch)):
            game_data = play_game(model, opponent_model, max_step=250)
            game_loss = torch.stack(game_data['training_losses']).mean()
            accumulated_loss_value += game_loss.item()
            (game_loss / minibatch).backward()
            if game_data['current_model_won']: wins += 1
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        avg_loss = accumulated_loss_value / minibatch if minibatch > 0 else 0
        print(f"Minibatch {i+1} 完成:")
        print(f"  胜率: {wins}/{minibatch} = {wins/minibatch:.2%}")
        print(f"  平均loss: {avg_loss:.4f}")

        # 每50个minibatch更新一次对手池
        if (i + 1) % 50 == 0:
            opponent_save_path = os.path.join(enemy_dir, f"opponent_iter_{i+1}.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'iteration': i + 1,
            }, opponent_save_path)
            print(f"新对手模型已保存至: {opponent_save_path}")
            
        # 每100个minibatch保存检查点
        if (i + 1) % 100 == 0:
            checkpoint_path = os.path.join(model_dir, f"AI_RL_iter_{i+1}.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'iteration': i + 1
            }, checkpoint_path)
            print(f"检查点已保存至: {checkpoint_path}")

if __name__ == '__main__':
    print(f"设备: {device}")
    print(f"学习率: {args.lr}")
    train()