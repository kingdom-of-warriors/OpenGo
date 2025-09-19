import torch
import os
import random
from tqdm import tqdm
from typing import Dict, List, Tuple, Any

class GoDataset(torch.utils.data.Dataset):
    """
    围棋数据集类。
    - 支持从预处理的.pt文件中加载数据。
    - 支持高效、统一、简洁且正确的数据增强。
    """
    def __init__(self, data_dirs: List[str], data_num: int = 2, board_size: int = 19, enable_augmentation: bool = False):
        self.board_size = board_size
        self.enable_augmentation = enable_augmentation
        self.board_size = board_size
        self.data = self._load_data(data_dirs, data_num)

    # 懒加载修改？
    def _load_data(self, data_dirs: List[str], data_num: int) -> List[Dict[str, Any]]:
        """从指定路径加载围棋数据集"""
        all_data = []
        for data_dir in data_dirs:
            for i in range(data_num):
                boards_file = os.path.join(data_dir, f"boards_batch_{i}.pt")
                if not os.path.exists(boards_file):
                    print(f"警告: 文件 {boards_file} 不存在，已跳过。")
                    continue
                colors_file = os.path.join(data_dir, f"colors_batch_{i}.pt")
                moves_file = os.path.join(data_dir, f"moves_batch_{i}.pt")
                boards, colors, moves = torch.load(boards_file), torch.load(colors_file), torch.load(moves_file)
                for j in range(len(boards)):
                    all_data.append({'board': boards[j], 'color': colors[j], 'move': moves[j].item()})
        return all_data

    def _apply_augmentation(self, board: torch.Tensor, move: int) -> Tuple[torch.Tensor, int]:
        """
        对单个样本（棋盘和落子）应用随机的对称旋转变换。
        """
        row, col = move // self.board_size, move % self.board_size
        if random.random() < 0.5: # 随机水平翻转
            board = torch.flip(board, dims=[2])
            col = self.board_size - 1 - col

        k = random.randint(0, 3) # 随机90度旋转
        if k > 0:
            board = torch.rot90(board, k, dims=[1, 2])
            for _ in range(k): row, col = self.board_size - 1 - col, row
        new_move = row * self.board_size + col

        return board, new_move

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        board, move, color = sample['board'], sample['move'], sample['color']
        
        if self.enable_augmentation:
            board, move = self._apply_augmentation(board, move)
        
        return {
            'board': board.float(),
            'move': move,
            'color': color,
        }
    
class WinnerDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, data_num: int = 2, enable_augmentation: bool = False):
        self.enable_augmentation = enable_augmentation
        self.data = self._load_data(data_dir, data_num)
        self.board_size = 19
    def _load_data(self, data_dir: str, data_num: int) -> List[Dict[str, Any]]:
        """从指定路径加载围棋数据集"""
        all_data = []
        for i in range(data_num):
            boards_file = os.path.join(data_dir, f"boards_batch_{i}.pt")
            winners_file = os.path.join(data_dir, f"winner_batch_{i}.pt")
            boards, winners = torch.load(boards_file), torch.load(winners_file)
            for j in range(len(boards)):
                all_data.append({'board': boards[j], 'winner': winners[j].item()})
        return all_data
    
    def __len__(self) -> int: return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        board, winner = sample['board'], sample['winner']
        
        if self.enable_augmentation:
            board, _ = GoDataset._apply_augmentation(self, board, 0)
        return {'board': board.float(), 'move': winner}