import torch
import re
import json
import os
import glob
from typing import List, Tuple, Dict, Any, Union, Optional
from tqdm import tqdm
from dataclasses import dataclass
import argparse
from utils import cal_liberity, find_valid
import numpy as np

from dlgo.goboard_slow import GameState, Move
from dlgo.gotypes import Player, Point

@dataclass
class FeatureConfig:
    """特征配置类"""
    enable_history: bool = True         # 历史信息（必须启用）
    enable_liberty: bool = False         # 气数特征
    enable_valid_moves: bool = False     # 可落子位置特征
    enable_winner: bool = False          # 胜负特征（如果启用，则只能启用这个特征）
    history_length: int = 8             # 历史长度

class FeatureExtractor:
    """特征提取器基类"""
    
    def extract(self, 
                black_board: torch.Tensor, 
                white_board: torch.Tensor, 
                current_color: str, 
                board_history: List[Dict[str, Union[torch.Tensor, str]]],
                **kwargs) -> torch.Tensor:
        """
        提取特征
        
        Returns:
            torch.Tensor: 特征张量，shape为 [channels, 19, 19]
        """
        raise NotImplementedError
    
    @property
    def channels(self) -> int:
        """返回该特征的通道数"""
        raise NotImplementedError
    
    @property
    def name(self) -> str:
        """返回特征名称"""
        raise NotImplementedError

class WinnerFeatureExtractor(FeatureExtractor):
    """胜负特征提取器 - 只提取最后一步的棋盘状态"""
    
    def extract(self, 
                black_board: torch.Tensor, 
                white_board: torch.Tensor, 
                current_color: str, 
                board_history: List[Dict[str, Union[torch.Tensor, str]]],
                **kwargs) -> torch.Tensor:
        """
        提取最后一步的棋盘状态
        Returns:
            torch.Tensor: shape [2, 19, 19] - 第0维为黑棋，第1维为白棋
        """
        # 直接返回黑棋和白棋的最终位置
        features = torch.zeros(2, 19, 19)
        features[0] = black_board.float()  # 黑棋位置
        features[1] = white_board.float()  # 白棋位置
        return features
    
    @property
    def channels(self) -> int: return 2
    
    @property
    def name(self) -> str: return "winner"

class HistoryFeatureExtractor(FeatureExtractor):
    """历史信息特征提取器"""
    
    def __init__(self, history_length: int = 8):
        self.history_length = history_length
    
    def extract(self, 
                black_board: torch.Tensor, 
                white_board: torch.Tensor, 
                current_color: str, 
                board_history: List[Dict[str, Union[torch.Tensor, str]]]) -> torch.Tensor:
        
        total_channels = 3 + self.history_length * 2
        features = torch.zeros(total_channels, 19, 19)
        
        if current_color == 'B':
            # 当前视角是黑棋
            features[0] = black_board
            features[1] = white_board
            features[2] = torch.ones(19, 19)
            
            for t in range(self.history_length):
                history_idx = len(board_history) - 1 - t
                if history_idx >= 0:
                    hist_state = board_history[history_idx]
                    features[3 + t * 2]     = hist_state['black']
                    features[3 + t * 2 + 1] = hist_state['white']
                
        else:
            # 当前视角是白棋
            features[0] = white_board
            features[1] = black_board
            features[2] = torch.zeros(19, 19)
            
            for t in range(self.history_length):
                history_idx = len(board_history) - 1 - t
                if history_idx >= 0:
                    hist_state = board_history[history_idx]
                    features[3 + t * 2]     = hist_state['white']
                    features[3 + t * 2 + 1] = hist_state['black']

        return features
    
    @property
    def channels(self) -> int:
        return 3 + self.history_length * 2
    
    @property
    def name(self) -> str:
        return f"history_{self.history_length}"

class LibertyFeatureExtractor(FeatureExtractor):
    """气数特征提取器"""
    
    def extract(self, 
                black_board: torch.Tensor, 
                white_board: torch.Tensor, 
                current_color: str, 
                board_history: List[Dict[str, Union[torch.Tensor, str]]],
                **kwargs) -> torch.Tensor:
        
        # 构建当前状态的棋盘（从历史特征的前两个通道获取）
        if current_color == 'B':
            current_board = torch.stack([black_board, white_board])
        else:
            current_board = torch.stack([white_board, black_board])
        
        liberty_features = cal_liberity(current_board)  # shape: [16, 19, 19]
        return liberty_features
    
    @property
    def channels(self) -> int:
        return 16
    
    @property
    def name(self) -> str:
        return "liberty"

class ValidMovesFeatureExtractor(FeatureExtractor):
    """可落子位置特征提取器"""
    
    def extract(self, 
                black_board: torch.Tensor, 
                white_board: torch.Tensor, 
                current_color: str, 
                board_history: List[Dict[str, Union[torch.Tensor, str]]],
                **kwargs) -> torch.Tensor:
        
        # 构建当前状态的棋盘
        if current_color == 'B':
            current_board = torch.stack([black_board, white_board])
        else:
            current_board = torch.stack([white_board, black_board])
        
        valid_positions = find_valid(current_board)  # shape: [19, 19]
        return valid_positions.float().unsqueeze(0)  # 添加通道维度: [1, 19, 19]
    
    @property
    def channels(self) -> int:
        return 1
    
    @property
    def name(self) -> str:
        return "valid_moves"

class SGFParser:
    def __init__(self, feature_config: FeatureConfig = None) -> None:
        self.board_size: int = 19
        self.feature_config = feature_config or FeatureConfig()
        self.feature_extractors: List[FeatureExtractor] = []
        self._setup_feature_extractors()
        self.total_channels = sum(extractor.channels for extractor in self.feature_extractors)
        
    def _setup_feature_extractors(self) -> None:
        """设置特征提取器"""
        # 如果启用胜负特征，则只使用胜负特征
        if self.feature_config.enable_winner:
            self.feature_extractors.append(WinnerFeatureExtractor())
            return
        
        # 历史信息
        if self.feature_config.enable_history:
            self.feature_extractors.append(
                HistoryFeatureExtractor(self.feature_config.history_length)
            )
        
        # 气数特征
        if self.feature_config.enable_liberty:
            self.feature_extractors.append(LibertyFeatureExtractor())
        
        # 可落子位置特征
        if self.feature_config.enable_valid_moves:
            self.feature_extractors.append(ValidMovesFeatureExtractor())
    
    def add_feature_extractor(self, extractor: FeatureExtractor) -> None:
        """添加自定义特征提取器"""
        self.feature_extractors.append(extractor)
        self.total_channels = sum(ext.channels for ext in self.feature_extractors)
        
    def coord_to_index(self, coord: str) -> Optional[int]:
        """将SGF坐标转换为索引"""
        if len(coord) != 2:
            return None
        col = ord(coord[0]) - ord('a')
        row = ord(coord[1]) - ord('a')
        if 0 <= col < self.board_size and 0 <= row < self.board_size:
            return row * self.board_size + col
        return None
    
    def extract_game_result(self, sgf_content: str) -> Optional[str]:
        """提取游戏结果"""
        # 匹配 RE[结果] 格式
        result_pattern = r'RE\[([^\]]+)\]'
        match = re.search(result_pattern, sgf_content)
        if match:
            return match.group(1)
        return None
    
    def parse_winner_data(self, sgf_path: str) -> Tuple[Optional[torch.Tensor], Optional[int]]:
        """解析SGF文件的胜负数据 - 只返回最后一步的棋盘状态和胜负标签"""
        try:
            # 优先尝试使用 utf-8 编码打开
            with open(sgf_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                # 如果 utf-8 失败，则尝试使用 gbk 编码
                with open(sgf_path, 'r', encoding='gbk') as f:
                    content = f.read()
            except Exception:
                # 如果两种编码都失败，则放弃该文件
                return None, None
            
        if content.count('(') > 1:
            return None, None
        game_result = self.extract_game_result(content)
        if game_result is None:
            return None, None
        
        winner_label = None
        if game_result.startswith('B+') or game_result.startswith('b+'): winner_label = 1  # 黑胜
        elif game_result.startswith('W+') or game_result.startswith('w+'): winner_label = 0  # 白胜
        else: return None, None  # 跳过和棋或其他结果
        
        # 提取所有着法
        moves = self.extract_moves(content)
        game_state = GameState.new_game(19)
        for i, (color, coord) in enumerate(moves):
            move_idx = self.coord_to_index(coord)
            row, col = move_idx // 19, move_idx % 19
            move = Move.play(Point(row + 1, col + 1))
            try:
                game_state = game_state.apply_move(move)
            except Exception:
                return None, None
        
        current_board = np.zeros((19, 19), dtype=int)
        for r in range(1, 19 + 1):
            for c in range(1, 19 + 1):
                stone = game_state.board.get(Point(r, c))
                if stone == Player.black: current_board[r - 1, c - 1] = 1
                elif stone == Player.white: current_board[r - 1, c - 1] = 2
        black_board = torch.tensor((current_board == 1).astype(np.float32))
        white_board = torch.tensor((current_board == 2).astype(np.float32))
        final_board = torch.zeros(2, 19, 19)
        final_board[0] = black_board; final_board[1] = white_board
        
        return final_board, winner_label
    
    def parse_sgf_file(self, sgf_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """解析SGF文件"""
        try:
            # 优先尝试使用 utf-8 编码打开
            with open(sgf_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                # 如果 utf-8 失败，则尝试使用 gbk 编码
                with open(sgf_path, 'r', encoding='gbk') as f:
                    content = f.read()
            except Exception:
                # 如果两种编码都失败，则放弃该文件
                return None, None, None
        if content.count('(') > 1: return None, None, None # 放弃分支棋谱
        
        moves = self.extract_moves(content)
        boards, move_indices, colors = self.generate_training_data_with_features(moves)
        return boards, move_indices, colors
    
    def extract_moves(self, sgf_content: str) -> List[Tuple[str, str]]:
        """从SGF内容中提取着法"""
        pattern = r';([BW])\[([a-s]{2})\]'
        matches = re.findall(pattern, sgf_content)
        moves = []
        for color, coord in matches: moves.append((color, coord))
        
        return moves
    
    def generate_training_data_with_features(self, moves: List[Tuple[str, str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """生成包含可配置特征的训练数据"""
        boards = []
        move_indices = []
        colors = []
        board_history: List[Dict[str, Union[torch.Tensor, str]]] = []
        game_state = GameState.new_game(19)

        for i, (color, coord) in enumerate(moves):
            move_idx = self.coord_to_index(coord)
            row, col = move_idx // 19, move_idx % 19
            current_board = np.zeros((19, 19), dtype=int)
            for r in range(1, 19 + 1):
                for c in range(1, 19 + 1):
                    stone = game_state.board.get(Point(r, c))
                    if stone == Player.black: current_board[r - 1, c - 1] = 1
                    elif stone == Player.white: current_board[r - 1, c - 1] = 2
            black_board = torch.tensor((current_board == 1).astype(np.float32))
            white_board = torch.tensor((current_board == 2).astype(np.float32))
            current_features = self.build_features(black_board, white_board, color, board_history)
            move = Move.play(Point(row + 1, col + 1))
            game_state = game_state.apply_move(move)
            boards.append(current_features)
            move_indices.append(move_idx)
            colors.append(1 if color == 'B' else 0)
            
            # 保存当前步骤后的棋盘状态到历史中
            board_history.append({
                'black': black_board.clone(),
                'white': white_board.clone(),
                'move_color': color
            })
        
        return torch.stack(boards), torch.tensor(move_indices), torch.tensor(colors)
    
    def build_features(
        self, 
        black_board: torch.Tensor, 
        white_board: torch.Tensor, 
        current_color: str, 
        board_history: List[Dict[str, Union[torch.Tensor, str]]]
    ) -> torch.Tensor:
        """使用配置的特征提取器构建特征张量"""
        feature_list = []
        
        for extractor in self.feature_extractors:
            features = extractor.extract(black_board, white_board, current_color, board_history)
            feature_list.append(features)
        
        # 合并所有特征
        combined_features = torch.cat(feature_list, dim=0)
        return combined_features
    
    def get_feature_info(self) -> Dict[str, Any]:
        """获取特征信息"""
        feature_info = {}
        channel_start = 0
        
        for extractor in self.feature_extractors:
            channels = extractor.channels
            feature_info[extractor.name] = {"channels": channels}
            channel_start += channels
        
        return {
            "total_channels": self.total_channels,
            "features": feature_info,
            "config": {
                "enable_history": self.feature_config.enable_history,
                "enable_liberty": self.feature_config.enable_liberty,
                "enable_valid_moves": self.feature_config.enable_valid_moves,
                "enable_winner": self.feature_config.enable_winner,
                "history_length": self.feature_config.history_length
            }
        }
    
    def save_parsed_data(self, sgf_path: str, output_dir: str) -> None:
        """解析SGF并保存为训练数据格式"""
        boards, moves, colors = self.parse_sgf_file(sgf_path)
        
        os.makedirs(output_dir, exist_ok=True)
        
        torch.save(boards, os.path.join(output_dir, "boards.pt"))
        torch.save(moves, os.path.join(output_dir, "moves.pt"))
        torch.save(colors, os.path.join(output_dir, "colors.pt"))
        
        # 保存详细的元数据
        metadata: Dict[str, Any] = {
            "total_positions": len(boards),
            "board_shape": list(boards.shape),
            "source_sgf": sgf_path,
            **self.get_feature_info()
        }
        
        with open(os.path.join(output_dir, "metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"解析完成！")
        print(f"棋盘形状: {boards.shape}")
        print(f"总通道数: {self.total_channels}")
        print(f"数据已保存到: {output_dir}")

def parse_multiple_sgf_files_batched(
    sgf_dir: str, 
    output_dir: str, 
    feature_config: FeatureConfig = None,
    batch_size: int = 1000, 
    batch_num: int = 50
) -> None:
    """分批处理SGF文件，支持可配置特征"""
    parser = SGFParser(feature_config)
    
    # 递归搜索所有子目录中的SGF文件
    sgf_files: List[str] = glob.glob(os.path.join(sgf_dir, "**", "*.sgf"), recursive=True) + \
                          glob.glob(os.path.join(sgf_dir, "**", "*.SGF"), recursive=True)
    
    os.makedirs(output_dir, exist_ok=True)
    
    total_positions: int = 0
    batch_count: int = 0
    iter_count = min(len(sgf_files), batch_num * batch_size)
    
    # 如果启用了胜负特征，使用特殊的处理流程
    if feature_config and feature_config.enable_winner:
        for i in range(0, iter_count, batch_size):
            batch_files: List[str] = sgf_files[i:i + batch_size]
            batch_boards: List[torch.Tensor] = []
            batch_winners: List[int] = []
            
            print(f"处理胜负数据批次 {batch_count + 1}/{(iter_count + batch_size - 1) // batch_size}")
            
            for sgf_file in tqdm(batch_files, desc="解析胜负数据"):
                board, winner = parser.parse_winner_data(sgf_file)
                if board is not None and winner is not None:
                    batch_boards.append(board)
                    batch_winners.append(winner)
            
            if batch_boards:
                combined_boards = torch.stack(batch_boards).to(torch.uint8)
                combined_winners = torch.tensor(batch_winners, dtype=torch.long)
                
                torch.save(combined_boards, os.path.join(output_dir, f"boards_batch_{batch_count}.pt"))
                torch.save(combined_winners, os.path.join(output_dir, f"winner_batch_{batch_count}.pt"))
                
                total_positions += len(combined_boards)
                batch_count += 1
                
                # 清理内存
                del batch_boards, batch_winners
                del combined_boards, combined_winners
    else:
        for i in range(0, iter_count, batch_size):
            batch_files: List[str] = sgf_files[i:i + batch_size]
            batch_boards: List[torch.Tensor] = []
            batch_moves: List[torch.Tensor] = []
            batch_colors: List[torch.Tensor] = []
            
            print(f"处理批次 {batch_count + 1}/{(iter_count + batch_size - 1) // batch_size}")
            
            for sgf_file in tqdm(batch_files, desc="解析文件"):
                try:
                    boards, moves, colors = parser.parse_sgf_file(sgf_file)
                    if boards is not None and moves is not None and colors is not None:
                        batch_boards.append(boards)
                        batch_moves.append(moves)
                        batch_colors.append(colors)
                except Exception as e:
                    print(f"解析失败 {sgf_file}: {e}")
            
            if batch_boards:
                combined_boards = torch.cat(batch_boards, dim=0).to(torch.uint8)
                combined_moves = torch.cat(batch_moves, dim=0)
                combined_colors = torch.cat(batch_colors, dim=0)
                torch.save(combined_boards, os.path.join(output_dir, f"boards_batch_{batch_count}.pt"))
                torch.save(combined_moves, os.path.join(output_dir, f"moves_batch_{batch_count}.pt"))
                torch.save(combined_colors, os.path.join(output_dir, f"colors_batch_{batch_count}.pt"))
                
                total_positions += len(combined_boards)
                batch_count += 1
                
                # 清理内存
                del batch_boards, batch_moves, batch_colors
                del combined_boards, combined_moves, combined_colors
    
    # 保存元数据
    metadata: Dict[str, Any] = {
        "total_positions": total_positions,
        "total_files": len(sgf_files),
        "batch_count": batch_count,
        **parser.get_feature_info()
    }
    
    with open(os.path.join(output_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\n批量解析完成！")
    print(f"处理文件数: {len(sgf_files)}")
    print(f"总位置数: {total_positions}")
    print(f"保存批次数: {batch_count}")
 
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="SGF文件批量处理工具")
    
    parser.add_argument("--sgf_dir", type=str, required=True,
                       help="SGF文件目录路径")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="输出目录路径")
    parser.add_argument("--enable_history", action="store_true", default=True,
                       help="启用历史特征 (默认: True)")
    parser.add_argument("--enable_liberty", action="store_true", default=False,
                       help="启用气数特征 (默认: False)")
    parser.add_argument("--enable_valid_moves", action="store_true", default=False,
                       help="启用可落子位置特征 (默认: False)")
    parser.add_argument("--enable_winner", action="store_true", default=False,
                       help="启用胜负特征 (默认: False，如果启用则只使用此特征)")
    parser.add_argument("--history_length", type=int, default=12,
                       help="历史长度 (默认: 12)")
    parser.add_argument("--batch_size", type=int, default=1000,
                       help="批次大小 (默认: 1000)")
    parser.add_argument("--batch_num", type=int, default=50,
                       help="处理批次数量 (默认: 50)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 根据参数创建特征配置
    feature_config = FeatureConfig(
        enable_history=args.enable_history,
        enable_liberty=args.enable_liberty,
        enable_valid_moves=args.enable_valid_moves,
        enable_winner=args.enable_winner,
        history_length=args.history_length
    )
    
    parse_multiple_sgf_files_batched(
        sgf_dir=args.sgf_dir,
        output_dir=args.output_dir,
        feature_config=feature_config,
        batch_size=args.batch_size,
        batch_num=args.batch_num
    )

if __name__ == "__main__":
    main()