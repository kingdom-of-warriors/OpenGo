import torch
import numpy as np
from tqdm import tqdm
from collections import deque
import random
import os

from dlgo.goboard_slow import GameState, Move
from dlgo.gotypes import Player, Point
from dlgo.scoring import compute_game_result
from models.policy_networks import PolicyNetwork
from .rl_utils import game_state_to_tensor, save_game_to_sgf
from models.policy_networks import create_model

def get_model_prediction(model: PolicyNetwork, game_state: GameState, 
                         game_history: deque, board_size: int, 
                         requires_grad: bool, device: torch.device,
                         temperature: float = 0.2):
    """输出模型的落子概率分布，屏蔽掉非法落子。"""
    input_tensor = game_state_to_tensor(game_state.next_player, 
                                        game_history, board_size).to(device)
    move_probs = model.sample(input_tensor, requires_grad=requires_grad).squeeze(0)
    invalid_move = 0
    legal_mask = torch.zeros_like(move_probs)
    for row in range(board_size):
        for col in range(board_size):
            point = Point(row + 1, col + 1)
            if not game_state.is_valid_move(Move.play(point)):
                legal_mask[row * board_size + col] = -float('inf')
                invalid_move += 1 # 计算不合法的步数

    if invalid_move == 19 * 19: # 没有合法的位置
        return None 
    move_probs_new = move_probs + legal_mask
    move_probs_new = torch.softmax(move_probs_new / temperature, dim=-1)

    return move_probs_new

def play_game(current_model: PolicyNetwork, 
              opponent_model: PolicyNetwork, 
              max_step: int, board_size: int, 
              device: torch.device, sgf_filepath: str = None):
    """执行一盘自我对弈，并返回对局数据。"""
    game_state = GameState.new_game(board_size)
    moves = []
    game_history = deque([], maxlen=13)
    training_losses_abs = []

    if random.random() < 0.5:
        players = {Player.white: current_model, 
                   Player.black: opponent_model}
        current_model_color = Player.white
    else:
        players = {Player.black: current_model, 
                   Player.white: opponent_model}
        current_model_color = Player.black

    pbar = tqdm(total=max_step, desc="Game Progress", leave=False, 
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} steps') # 进度条
    
    for step_count in range(max_step):
        current_board = np.zeros((board_size, board_size), dtype=np.int8)
        # 获得当下局面黑白棋子位置 并加入history中
        for r in range(1, board_size + 1):
            for c in range(1, board_size + 1):
                stone = game_state.board.get(Point(row=r, col=c))
                if stone == Player.black: current_board[r - 1, c - 1] = 1
                elif stone == Player.white: current_board[r - 1, c - 1] = 2
        game_history.append(current_board)

        active_model = players[game_state.next_player]
        requires_grad = (active_model == current_model) # 当前模型需要计算梯度
        move_probs = get_model_prediction(active_model, game_state, 
                                          game_history, board_size, 
                                          requires_grad, device)
        if move_probs is None: # 返回none说明没有合法的走法
            print("没有合法走法，终止对弈。")
            break
        move_idx = torch.multinomial(move_probs, 1).item() # 采样落子位置，增加棋局多样性

        # 计算该步的负对数似然loss
        if requires_grad:
            loss_abs = -torch.log(move_probs[move_idx] + 1e-8)
            training_losses_abs.append(loss_abs)

        point = Point(row=move_idx // board_size + 1, col=move_idx % board_size + 1)
        move = Move.play(point)
        moves.append(move)
        game_state = game_state.apply_move(move)
        pbar.update(1)
    
    pbar.close()

    game_result = compute_game_result(game_state) # 获得对局结果
    current_model_won = (game_result.winner == current_model_color) # 当前模型是否获胜
    z_t = 1.0 if current_model_won else -1.0
    final_loss = torch.stack(training_losses_abs).mean() * z_t # 计算最终loss
    if sgf_filepath: save_game_to_sgf(moves, game_result.winner, 
                                      sgf_filepath, current_model_color) # 保存对局为sgf文件

    return final_loss, current_model_won


def play_game_worker(args_bundle):
    """工作函数，在自己的GPU上创建模型并进行对弈。"""
    args, current_model_state, opponent_model_state, game_idx, total_games, iter_num, device_id = args_bundle
    device = torch.device(f"cuda:{device_id}")

    # 为每个子进程设置随机种子
    seed = os.getpid() + game_idx ** 2
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # 在子进程的GPU上创建模型
    current_model = create_model(args, device)
    current_model.load_state_dict(current_model_state)
    current_model.eval() # 一定要加！

    opponent_model = create_model(args, device)
    opponent_model.load_state_dict(opponent_model_state)
    opponent_model.eval()

    sgf_dir = os.path.join("GoDataset", "self_play", f"iter_{iter_num}")
    sgf_filepath = os.path.join(sgf_dir, f"game_{game_idx+1}.sgf")
    final_loss, won = play_game(current_model, opponent_model, args.max_step, args.board_size, device, sgf_filepath)
    
    current_model.zero_grad()
    final_loss.backward()
    
    # 将计算出的梯度打包并返回
    grads = [param.grad.cpu() for param in current_model.parameters() if param.grad is not None]
    return grads, final_loss.item(), won