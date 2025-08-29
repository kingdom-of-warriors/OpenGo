import torch
import re
import json
import os
import glob
from typing import List, Tuple, Dict, Any, Union, Optional
from tqdm import tqdm
from dataclasses import dataclass

from utils import cal_liberity, find_valid

@dataclass
class FeatureConfig:
    """特征配置类"""
    enable_history: bool = True         # 历史信息（必须启用）
    enable_liberty: bool = True         # 气数特征
    enable_valid_moves: bool = True     # 可落子位置特征
    history_length: int = 8             # 历史长度
    
    def __post_init__(self):
        """验证配置"""
        if not self.enable_history:
            raise ValueError("历史信息是必需的特征，不能禁用")
        if self.history_length <= 0:
            raise ValueError("历史长度必须大于0")

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

class HistoryFeatureExtractor(FeatureExtractor):
    """历史信息特征提取器"""
    
    def __init__(self, history_length: int = 8):
        self.history_length = history_length
    
    def extract(self, 
                black_board: torch.Tensor, 
                white_board: torch.Tensor, 
                current_color: str, 
                board_history: List[Dict[str, Union[torch.Tensor, str]]],
                **kwargs) -> torch.Tensor:
        
        # 当前棋盘状态 (3个通道) + 历史状态 (history_length * 2个通道)
        total_channels = 3 + self.history_length * 2
        features = torch.zeros(total_channels, 19, 19)
        
        # 当前棋盘状态 (前3个通道)
        if current_color == 'B':
            features[0] = black_board    # 我方棋子（黑棋）
            features[1] = white_board    # 敌方棋子（白棋）
            features[2] = torch.ones(19, 19)  # 下一手标识（黑方下棋）
        else:
            features[0] = white_board    # 我方棋子（白棋）
            features[1] = black_board    # 敌方棋子（黑棋）
            features[2] = torch.zeros(19, 19)  # 下一手标识（白方下棋）
        
        # 历史状态 (通道3开始，共history_length * 2个通道)
        for t in range(self.history_length):
            history_idx = len(board_history) - 1 - t
            channel_idx_player = 3 + t * 2
            channel_idx_opponent = 3 + t * 2 + 1
            
            if history_idx >= 0:
                hist_black = board_history[history_idx]['black']
                hist_white = board_history[history_idx]['white']
                hist_color = board_history[history_idx]['move_color']
                
                if current_color == 'B':
                    if hist_color == 'B':
                        features[channel_idx_player] = hist_black
                        features[channel_idx_opponent] = hist_white
                    else:
                        features[channel_idx_player] = hist_white
                        features[channel_idx_opponent] = hist_black
                else:
                    if hist_color == 'W':
                        features[channel_idx_player] = hist_white
                        features[channel_idx_opponent] = hist_black
                    else:
                        features[channel_idx_player] = hist_black
                        features[channel_idx_opponent] = hist_white
            else:
                features[channel_idx_player] = torch.zeros((19, 19), dtype=torch.uint8)
                features[channel_idx_opponent] = torch.zeros((19, 19), dtype=torch.uint8)
        
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
        
        # 计算总通道数
        self.total_channels = sum(extractor.channels for extractor in self.feature_extractors)
        
    def _setup_feature_extractors(self) -> None:
        """设置特征提取器"""
        # 历史信息（必需）
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
    
    def parse_sgf_file(self, sgf_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """解析SGF文件"""
        with open(sgf_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取着法序列
        moves = self.extract_moves(content)
        
        # 生成训练数据
        boards, move_indices, colors = self.generate_training_data_with_features(moves)
        return boards, move_indices, colors
    
    def extract_moves(self, sgf_content: str) -> List[Tuple[str, str]]:
        """从SGF内容中提取着法"""
        pattern = r';([BW])\[([a-s]{2})\]'
        matches = re.findall(pattern, sgf_content)
        
        moves = []
        for color, coord in matches:
            moves.append((color, coord))
        
        return moves
    
    def generate_training_data_with_features(self, moves: List[Tuple[str, str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """生成包含可配置特征的训练数据"""
        boards = []
        move_indices = []
        colors = []
        board_history: List[Dict[str, Union[torch.Tensor, str]]] = []
        black_board = torch.zeros((19, 19), dtype=torch.uint8)
        white_board = torch.zeros((19, 19), dtype=torch.uint8)
        
        for i, (color, coord) in enumerate(moves):
            move_idx = self.coord_to_index(coord)
            if move_idx is None: continue
        
            current_features = self.build_features(black_board, white_board, color, board_history)
            boards.append(current_features)
            move_indices.append(move_idx)
            colors.append(1 if color == 'B' else 0)
            
            row, col = move_idx // 19, move_idx % 19
            if color == 'B':
                black_board[row, col] = 1
            else:
                white_board[row, col] = 1
            
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
            feature_info[extractor.name] = {
                "channels": channels,
                "start_channel": channel_start,
                "end_channel": channel_start + channels - 1
            }
            channel_start += channels
        
        return {
            "total_channels": self.total_channels,
            "features": feature_info,
            "config": {
                "enable_history": self.feature_config.enable_history,
                "enable_liberty": self.feature_config.enable_liberty,
                "enable_valid_moves": self.feature_config.enable_valid_moves,
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
    batch_num: int = 40
) -> None:
    """分批处理SGF文件，支持可配置特征"""
    parser = SGFParser(feature_config)
    
    # 递归搜索所有子目录中的SGF文件
    sgf_files: List[str] = glob.glob(os.path.join(sgf_dir, "**", "*.sgf"), recursive=True) + \
                          glob.glob(os.path.join(sgf_dir, "**", "*.SGF"), recursive=True)
    
    os.makedirs(output_dir, exist_ok=True)
    
    total_positions: int = 0
    batch_count: int = 0
    
    # 分批处理
    iter_count = min(len(sgf_files), batch_num * batch_size)
    for i in range(0, iter_count, batch_size):
        batch_files: List[str] = sgf_files[i:i + batch_size]
        batch_boards: List[torch.Tensor] = []
        batch_moves: List[torch.Tensor] = []
        batch_colors: List[torch.Tensor] = []
        
        print(f"处理批次 {batch_count + 1}/{(iter_count + batch_size - 1) // batch_size}")
        
        for sgf_file in tqdm(batch_files, desc="解析文件"):
            try:
                boards, moves, colors = parser.parse_sgf_file(sgf_file)
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

def main():
    """主函数示例 - 展示如何使用不同的特征配置"""
    print("--- 开始数据预处理 ---")
    
    # 示例1: 使用所有特征
    config_all = FeatureConfig(
        enable_history=True,
        enable_liberty=True,
        enable_valid_moves=True,
        history_length=8
    )
    
    # 示例2: 只使用历史和气数特征
    config_no_valid = FeatureConfig(
        enable_history=True,
        enable_liberty=True,
        enable_valid_moves=False,
        history_length=8
    )
    
    # 示例3: 只使用历史特征
    config_history_12 = FeatureConfig(
        enable_history=True,
        enable_liberty=False,
        enable_valid_moves=False,
        history_length=12
    )
    
    # 使用配置1处理数据
    parse_multiple_sgf_files_batched(
        sgf_dir="/mnt/petrelfs/lijiarui/OpenGo/GoDataset/AI", 
        output_dir="/mnt/petrelfs/lijiarui/OpenGo/GoDataset/AI_pt",
        feature_config=config_history_12,
        batch_size=1000,
        batch_num=50
    )
    
    print("--- 数据预处理完成 ---")

if __name__ == "__main__":
    main()