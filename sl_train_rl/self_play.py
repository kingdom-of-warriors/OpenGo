import torch
import glob
import random
import os
import numpy as np
from datetime import datetime

from dlgo.goboard_slow import GameState, Move
from dlgo.gotypes import Player, Point
from dlgo.scoring import compute_game_result
from dlgo.utils import print_board, print_move
from models.policy_networks import PolicyNetwork_resnet
from models.policy_networks import create_model
from sl_train_dl.config import parse_args
from dlgo import sgf

enemy_path = "ckpt/enemies/"
model_path = "ckpt/AI_12_192.pth"
game_path = "GoDataset/self_play/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = parse_args()
model = create_model(args, device)
ckpt_path = "ckpt/AI_Human_12_192.pth"
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.train() # 设置为训练模式

# 确保目录存在
os.makedirs(enemy_path, exist_ok=True)
os.makedirs(game_path, exist_ok=True)

def game_state_to_tensor(game_state, game_history, board_size=19):
    """
    将GameState和历史状态转换为模型输入张量
    输入格式: (27, 19, 19) - 包含当前状态(3层) + 历史状态(24层)
    """
    input_tensor = np.zeros((27, board_size, board_size), dtype=np.float32)
    
    # 获取当前棋盘状态
    current_board = np.zeros((board_size, board_size), dtype=int)
    for row in range(1, board_size + 1):
        for col in range(1, board_size + 1):
            stone = game_state.board.get(Point(row=row, col=col))
            if stone == Player.black:
                current_board[row-1, col-1] = 1
            elif stone == Player.white:
                current_board[row-1, col-1] = 2
    
    # 第1-3维：当前状态
    if game_state.next_player == Player.black:
        input_tensor[0] = (current_board == 1).astype(np.float32)  # 我方（黑棋）
        input_tensor[1] = (current_board == 2).astype(np.float32)  # 敌方（白棋）
        input_tensor[2] = np.ones((board_size, board_size), dtype=np.float32)  # 黑棋回合
    else:
        input_tensor[0] = (current_board == 2).astype(np.float32)  # 我方（白棋）
        input_tensor[1] = (current_board == 1).astype(np.float32)  # 敌方（黑棋）
        input_tensor[2] = np.zeros((board_size, board_size), dtype=np.float32)  # 白棋回合
    
    # 第4-27维：历史状态 (t-1 到 t-12)
    for i in range(min(12, len(game_history))):
        hist_idx = len(game_history) - 1 - i
        hist_board = game_history[hist_idx]
        
        if game_state.next_player == Player.black:
            input_tensor[3 + i*2] = (hist_board == 1).astype(np.float32)     # 我方历史
            input_tensor[4 + i*2] = (hist_board == 2).astype(np.float32)     # 敌方历史
        else:
            input_tensor[3 + i*2] = (hist_board == 2).astype(np.float32)     # 我方历史
            input_tensor[4 + i*2] = (hist_board == 1).astype(np.float32)     # 敌方历史
    
    return torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0).to(device)

def get_model_prediction(model, game_state, game_history, board_size=19):
    """获取模型对当前局面的预测"""
    input_tensor = game_state_to_tensor(game_state, game_history, board_size)
    
    with torch.no_grad():
        move_probs = model.sample(input_tensor).squeeze(0)  # (361,)
        # 检查合法性并调整概率
        legal_moves = []
        for row in range(board_size):
            for col in range(board_size):
                point = Point(row + 1, col + 1)
                if game_state.is_valid_move(Move.play(point)):
                    legal_moves.append(row * board_size + col)
        
        # 将非法位置的概率设为0
        legal_mask = torch.zeros_like(move_probs)
        for move_idx in legal_moves:
            legal_mask[move_idx] = 1.0
        # 归一化
        move_probs = move_probs * legal_mask
        move_probs = move_probs / move_probs.sum()
    
    return move_probs

def load_random_opponent(args, device):
    """从enemy_path加载一个随机的对手模型"""
    opponent_files = glob.glob(os.path.join(enemy_path, "*.pth"))
    opponent_ckpt_path = random.choice(opponent_files)
    print(f"加载对手: {opponent_ckpt_path}")
    
    opponent_model = create_model(args, device)
    checkpoint = torch.load(opponent_ckpt_path, map_location=device)
    opponent_model.load_state_dict(checkpoint['model_state_dict'])
    opponent_model.eval()
    return opponent_model

def play_game(current_model, opponent_model, max_step, board_size=19):
    """
    模拟当前模型与对手模型的一局对弈。
    返回: (获胜方, 游戏状态列表, 动作列表)
    """
    game_state = GameState.new_game(board_size)
    states, moves = [], []
    step_count = 0
    game_history = []  # 存储历史棋盘状态

    # 随机决定当前模型执黑或执白
    players = {Player.black: current_model, Player.white: opponent_model}
    if random.random() < 0.5:
        players = {Player.white: current_model, Player.black: opponent_model}

    while not game_state.is_over() and step_count < max_step:
        # 保存当前棋盘状态到历史
        current_board = np.zeros((board_size, board_size), dtype=int)
        for row in range(1, board_size + 1):
            for col in range(1, board_size + 1):
                stone = game_state.board.get(Point(row=row, col=col))
                if stone == Player.black:
                    current_board[row-1, col-1] = 1
                elif stone == Player.white:
                    current_board[row-1, col-1] = 2
        game_history.append(current_board.copy())
        active_model = players[game_state.next_player]
        move_probs = get_model_prediction(active_model, game_state, game_history, board_size)
        
        # 从概率分布中抽样一个动作
        move_idx = torch.multinomial(move_probs, 1).item()
        row, col = move_idx // board_size, move_idx % board_size
        point = Point(row + 1, col + 1)
        move = Move.play(point)
        states.append(game_state)
        moves.append(move)
        game_state = game_state.apply_move(move)
        step_count += 1

    game_result = compute_game_result(game_state)
    winner = game_result.winner
    
    # 判断当前训练的模型是否获胜
    if (players[Player.black] == current_model and winner == Player.black) or \
       (players[Player.white] == current_model and winner == Player.white):
        current_model_won = True
    else:
        current_model_won = False

    return current_model_won, states, moves, game_result

def train(num_iterations=10000):
    """强化学习主训练循环"""
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr) # 示例优化器

    for i in range(num_iterations):
        print(f"--- 第 {i + 1} 轮 ---")
        # 1. 加载一个随机对手
        opponent_model = load_random_opponent(args, device)
        if opponent_model is None:
            opponent_model = model # 如果没有对手，就和自己对弈

        # 2. 进行对弈
        won, states, moves, result = play_game(model, opponent_model, args.board_size)
        print(f"游戏结束. 胜利者: {result.winner}, 分数: {result.w_score}-{result.b_score}")
        
        # 在这里执行策略梯度更新 (伪代码)
        # reward = 1 if won else -1
        # update_policy(model, optimizer, states, moves, reward)
        
        # 3. 保存棋谱
        sgf_content = sgf.sgf_from_game_state(
            states[0], 
            result, 
            moves, 
            black_player='current_model', 
            white_player='opponent'
        )
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        sgf_filename = os.path.join(game_path, f"game_{i+1}_{timestamp}.sgf")
        with open(sgf_filename, "w") as f:
            f.write(sgf_content)
        print(f"棋谱已保存至: {sgf_filename}")

        # 4. 每500轮更新一次对手池
        if (i + 1) % 500 == 0:
            opponent_save_path = os.path.join(enemy_path, f"opponent_iter_{i+1}.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
            }, opponent_save_path)
            print(f"新对手模型已保存至: {opponent_save_path}")

if __name__ == '__main__':
    train()