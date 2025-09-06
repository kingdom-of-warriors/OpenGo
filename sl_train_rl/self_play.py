import torch
import glob
import random
import os
import numpy as np
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from dlgo.goboard_slow import GameState, Move
from dlgo.gotypes import Player, Point
from dlgo.scoring import compute_game_result
from models.policy_networks import create_model
from sl_train_dl.config import parse_args
from models.policy_networks import create_model


enemy_path = "ckpt/enemies/"
model_path = "ckpt/AI_12_192.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = parse_args()
model = create_model(args, device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
os.makedirs(enemy_path, exist_ok=True)

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

def get_model_prediction(model, game_state, game_history, board_size=19, requires_grad=False):
    input_tensor = game_state_to_tensor(game_state, game_history, board_size)
    if requires_grad:
        move_probs = model.sample_rl(input_tensor).squeeze(0) 
    else: 
        with torch.no_grad():  move_probs = model.sample_inference(input_tensor).squeeze(0)
    
    legal_mask = torch.zeros_like(move_probs)
    for row in range(board_size):
        for col in range(board_size):
            point = Point(row + 1, col + 1)
            if game_state.is_valid_move(Move.play(point)):
                legal_mask[row * board_size + col] = 1.0
    
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
    在游戏过程中直接计算训练模型的loss
    """
    game_state = GameState.new_game(board_size)
    states, moves = [], []
    step_count = 0
    game_history = []  # 累积历史状态
    training_losses_abs = []  # 存储loss的绝对值

    if random.random() < 0.5:
        players = {Player.white: current_model, Player.black: opponent_model}
        current_model_color = Player.white
    else:
        players = {Player.black: current_model, Player.white: opponent_model}
        current_model_color = Player.black

    while not game_state.is_over() and step_count < max_step:
        # 保存当前历史状态的副本
        current_history = game_history.copy()
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
        
        if active_model == current_model:
            move_probs = get_model_prediction(active_model, 
                                              game_state, 
                                              current_history, 
                                              board_size, 
                                              requires_grad=True)
            move_idx = torch.multinomial(move_probs, 1).item()
            loss_abs = -torch.log(move_probs[move_idx] + 1e-8)
            training_losses_abs.append(loss_abs)

        else:
            move_probs = get_model_prediction(active_model, 
                                              game_state, 
                                              current_history, 
                                              board_size, 
                                              requires_grad=False)
            move_idx = torch.multinomial(move_probs, 1).item()
        
        point = Point(move_idx // board_size + 1, move_idx % board_size + 1)
        move = Move.play(point)
        states.append(game_state)
        moves.append(move)
        game_state = game_state.apply_move(move)
        step_count += 1

    # 计算结果和loss
    game_result = compute_game_result(game_state)
    winner = game_result.winner
    current_model_won = (winner == current_model_color)
    training_losses = []
    z_t = +1.0 if current_model_won else -1.0
    training_losses = [loss * z_t for loss in training_losses_abs]

    return {
        'current_model_won': current_model_won,
        'training_losses': training_losses,
        'num_training_steps': len(training_losses)
    }

def train(num_iterations=1000):
    """强化学习主训练循环"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for i in range(num_iterations):
        print(f"--- 第 {i + 1} 轮 (Minibatch) ---")
        opponent_model = load_random_opponent(args, device)
        minibatch_losses = []
        total_games = 8
        wins = 0
        total_training_steps = 0
        model.eval() # 生成数据使用评估模式

        for j in range(total_games):
            game_data = play_game(model, opponent_model, max_step=300)
            minibatch_losses.extend(game_data['training_losses'])
            total_training_steps += game_data['num_training_steps']
            if game_data['current_model_won']: wins += 1
        
        optimizer.zero_grad()
        
        # 计算平均loss
        model.train() # 更新权重使用训练模式
        total_loss = torch.stack(minibatch_losses).mean()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        print(f"Minibatch {i+1} 完成:")
        print(f"  胜率: {wins}/{total_games} = {wins/total_games:.2%}")
        print(f"  平均loss: {total_loss.item():.4f}")

        # 4. 每50个minibatch更新一次对手池
        if (i + 1) % 50 == 0:
            opponent_save_path = os.path.join(enemy_path, f"opponent_iter_{i+1}.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'iteration': i + 1,
            }, opponent_save_path)
            print(f"新对手模型已保存至: {opponent_save_path}")
            
        # 5. 每100个minibatch保存检查点
        if (i + 1) % 100 == 0:
            checkpoint_path = os.path.join("ckpt", f"AI_RL_iter_{i+1}.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'iteration': i + 1
            }, checkpoint_path)
            print(f"检查点已保存至: {checkpoint_path}")

if __name__ == '__main__':
    print(f"设备: {device}")
    print(f"学习率: {args.lr}")
    train()