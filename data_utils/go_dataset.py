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
    def __init__(self, data_dir: str, data_num: int = 2, board_size: int = 19, enable_augmentation: bool = False):
        self.board_size = board_size
        self.enable_augmentation = enable_augmentation
        
        # 加载数据（保持您原来的方式）
        self.data = self._load_data(data_dir, data_num)

    def _load_data(self, data_dir: str, data_num: int) -> List[Dict[str, Any]]:
        """从指定路径加载围棋数据集"""
        all_data = []
        for i in range(data_num):
            boards_file = os.path.join(data_dir, f"boards_batch_{i}.pt")
            colors_file = os.path.join(data_dir, f"colors_batch_{i}.pt")
            moves_file = os.path.join(data_dir, f"moves_batch_{i}.pt")
            if not all(os.path.exists(f) for f in [boards_file, colors_file, moves_file]):
                print(f"警告: 批次 {i} 的文件不完整，跳过。")
                continue
            boards, colors, moves = torch.load(boards_file), torch.load(colors_file), torch.load(moves_file)
            for j in range(len(boards)):
                all_data.append({'board': boards[j], 'color': colors[j], 'move': moves[j].item()})
        return all_data

    def _apply_augmentation(self, board: torch.Tensor, move: int) -> Tuple[torch.Tensor, int]:
        """
        对单个样本（棋盘和落子）应用随机的对称变换。
        这个版本简洁、高效且逻辑正确。
        """
        # 1. 将一维的move索引分解为二维坐标
        row, col = move // self.board_size, move % self.board_size

        # 2. 随机决定是否进行水平翻转 (50%概率)
        if random.random() < 0.5:
            # 翻转棋盘：在宽度维度(dim=2)上翻转
            board = torch.flip(board, dims=[2])
            # 翻转坐标：列坐标关于中心线对称
            col = self.board_size - 1 - col

        # 3. 随机决定旋转次数 (0, 1, 2, 3 次)
        k = random.randint(0, 3)
        if k > 0:
            # 旋转棋盘k次 (每次90度，逆时针)
            board = torch.rot90(board, k, dims=[1, 2])
            # 旋转坐标k次
            for _ in range(k): # 应用一次逆时针90度旋转的数学公式
                row, col = self.board_size - 1 - col, row
        
        # 4. 将变换后的二维坐标转换回一维索引
        new_move = row * self.board_size + col

        return board, new_move

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        board, move, color = sample['board'], sample['move'], sample['color']
        
        if self.enable_augmentation:
            # 应用数据增强
            board, move = self._apply_augmentation(board, move)
        
        return {
            'board': board.float(),
            'move': torch.tensor(move, dtype=torch.long),
            'color': color,
        }