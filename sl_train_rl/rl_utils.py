import numpy as np
import torch
from collections import deque
from dlgo.gotypes import Player
from dlgo.goboard_slow import Move
import os

def game_state_to_tensor(next_player: Player, game_history: deque, board_size: int = 19) -> torch.Tensor:
    """
    将GameState和历史状态转换为模型输入张量。
    """
    # 假设模型需要12个历史状态，每个状态2个通道，加上当前状态的3个通道 = 27
    input_tensor = np.zeros((27, board_size, board_size), dtype=np.float32)
    if not game_history: return torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0)

    # 第1-3维：当前状态 (T)
    current_board = game_history[-1]
    if next_player == Player.black:
        input_tensor[0] = (current_board == 1).astype(np.float32)
        input_tensor[1] = (current_board == 2).astype(np.float32)
        input_tensor[2] = np.ones((board_size, board_size), dtype=np.float32)
    else:
        input_tensor[0] = (current_board == 2).astype(np.float32)
        input_tensor[1] = (current_board == 1).astype(np.float32)
        input_tensor[2] = np.zeros((board_size, board_size), dtype=np.float32)
    
    # 第4-27维：历史状态 (T-1 到 T-12)
    for i in range(min(12, len(game_history) - 1)):
        hist_idx = -i - 2  # 从-2开始，即倒数第二个元素 (T-1)
        hist_board = game_history[hist_idx]
        if next_player == Player.black:
            input_tensor[3 + i*2] = (hist_board == 1).astype(np.float32)  
            input_tensor[4 + i*2] = (hist_board == 2).astype(np.float32)
        else:
            input_tensor[3 + i*2] = (hist_board == 2).astype(np.float32)
            input_tensor[4 + i*2] = (hist_board == 1).astype(np.float32)
    
    return torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0)

def move_to_sgf_coord(move: Move) -> str:
    """将 Move 对象转换为 SGF 坐标字符串，例如 'qd'。"""
    if move.is_pass or move.is_resign:
        return ""
    col_str = "abcdefghijklmnopqrstuvwxy"[move.point.col - 1]
    row_str = "abcdefghijklmnopqrstuvwxy"[move.point.row - 1]
    return col_str + row_str

def save_game_to_sgf(moves: list, winner: Player, sgf_filepath: str, current_model_color: Player):
    """将对局保存为 SGF 文件。"""
    
    sgf_content = "(;FF[4]CA[UTF-8]GM[1]SZ[19]\n"
    result = "B+R" if winner == Player.black else "W+R"
    sgf_content += f"RE[{result}]\n"
    if current_model_color == Player.black: sgf_content += "PB[current_model]\nPW[opponent_model]\n"
    else: sgf_content += "PB[opponent_model]\nPW[current_model]\n"

    # 写入着法
    for i, move in enumerate(moves):
        player_str = "B" if (i % 2 == 0) else "W"
        coord_str = move_to_sgf_coord(move)
        sgf_content += f";{player_str}[{coord_str}]"
    
    sgf_content += ")\n"

    # 确保目录存在并保存文件
    os.makedirs(os.path.dirname(sgf_filepath), exist_ok=True)
    with open(sgf_filepath, "w") as f: f.write(sgf_content)