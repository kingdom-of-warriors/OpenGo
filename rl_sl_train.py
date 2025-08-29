import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque
import copy
from tqdm import tqdm
import argparse
from typing import List, Dict, Tuple

# --- 1. 导入所需模块 ---
from dlgo.goboard_slow import GameState, Move
from dlgo.gotypes import Player, Point
from dlgo.scoring import compute_game_result
from models.policy_networks import PolicyNetwork_resnet

# --- 2. 特征构造器 (已修正) ---
def build_feature_tensor(game_state: GameState, player: Player, history_length: int = 12) -> torch.Tensor:
    board_size = game_state.board.num_rows
    features = torch.zeros(27, board_size, board_size)

    # 1. 创建一个空的二维Numpy数组来表示棋盘
    dense_grid = np.zeros((board_size, board_size), dtype=np.int8)

    # 2. 遍历字典，将棋子信息填充到二维数组中
    for point, go_string in game_state.board._grid.items():
        # 防御性检查：确保 go_string 不是 None，以防止意外的错误
        if go_string is not None:
            row, col = point.row - 1, point.col - 1
            dense_grid[row, col] = go_string.color.value

    # 3. 现在 dense_grid 是一个完整的二维数组，可以进行后续操作
    black_stones = (dense_grid == Player.black.value)
    white_stones = (dense_grid == Player.white.value)

    if player == Player.black:
        features[0], features[1] = torch.from_numpy(black_stones), torch.from_numpy(white_stones)
    else:
        features[0], features[1] = torch.from_numpy(white_stones), torch.from_numpy(black_stones)

    if game_state.next_player == Player.white:
        features[2] = torch.ones(board_size, board_size)

    # 4. 对历史棋盘状态也应用同样的防御性逻辑
    temp_state = game_state.previous_state
    for t in range(history_length):
        if temp_state is None: break
        
        dense_history_grid = np.zeros((board_size, board_size), dtype=np.int8)
        # 同样进行防御性检查
        for point, go_string in temp_state.board._grid.items():
            if go_string is not None:
                row, col = point.row - 1, point.col - 1
                dense_history_grid[row, col] = go_string.color.value
        
        hist_black = (dense_history_grid == Player.black.value)
        hist_white = (dense_history_grid == Player.white.value)

        if player == Player.black:
            features[3 + t * 2], features[4 + t * 2] = torch.from_numpy(hist_black), torch.from_numpy(hist_white)
        else:
            features[3 + t * 2], features[4 + t * 2] = torch.from_numpy(hist_white), torch.from_numpy(hist_black)
        
        temp_state = temp_state.previous_state

    return features

# --- 3. 训练器类 (已重构和优化) ---
class RLTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 初始化当前策略网络和对手网络
        self.policy_net = self._load_model(args.sl_model_path)
        self.opponent_net = self._load_model(args.sl_model_path)
        self.opponent_net.eval()

        # 初始化对手池
        self.opponent_pool = deque(maxlen=args.opponent_pool_size)
        self.opponent_pool.append(copy.deepcopy(self.policy_net.state_dict()))

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=args.lr)
        self.win_history = deque(maxlen=args.update_opponent_every)

    def _load_model(self, model_path):
        model = PolicyNetwork_resnet(input_channels=27, num_filters=self.args.resnet_filters, num_residual_blocks=self.args.resnet_blocks).to(self.device)
        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)['model_state_dict']
            model.load_state_dict(state_dict)
            print(f"从 {model_path} 加载模型权重成功！")
        else:
            print(f"警告: 未找到模型 {model_path}, 将使用随机初始化的权重。")
        return model

    def run_self_play_game(self) -> Tuple[Player, List[Dict]]:
        game = GameState.new_game(19)
        board_size = game.board.num_rows # <--- 动态获取棋盘尺寸
        trajectory = []
        
        for _ in range(self.args.max_moves_per_game):
            if game.is_over(): break
                
            features = build_feature_tensor(game, game.next_player, history_length=12)
            input_tensor = features.unsqueeze(0).to(self.device).float()
            
            # 根据下棋方，决定是否需要计算梯度
            if game.next_player == Player.black:
                logits = self.policy_net(input_tensor)
            else:
                with torch.no_grad(): logits = self.opponent_net(input_tensor)

            mask = torch.full_like(logits, -1e9)
            for move in game.legal_moves():
                if move.is_play:
                    move_idx = board_size * (move.point.row - 1) + (move.point.col - 1)
                    mask[0, move_idx] = 0.0
                    
            masked_logits = logits + mask
            probs = F.softmax(masked_logits, dim=1)
            move_idx = torch.multinomial(probs, num_samples=1).squeeze().item()
            
            if game.next_player == Player.black:
                log_probs = F.log_softmax(masked_logits, dim=1)
                action_log_prob = log_probs[0, move_idx]
                trajectory.append({'action_log_prob': action_log_prob})
                
            row, col = move_idx // board_size, move_idx % board_size
            game = game.apply_move(Move.play(Point(row=row + 1, col=col + 1)))
            
        game_result = compute_game_result(game)
        return game_result.winner, trajectory

    def train_on_batch(self, batch_trajectories: List[Tuple[Player, List[Dict]]]):
        """根据一批次游戏的结果，执行一次策略梯度更新"""
        self.optimizer.zero_grad()
        
        batch_loss_terms = []
        for winner, trajectory in batch_trajectories:
            if not trajectory:
                continue

            # 我们的模型是黑棋，所以当黑棋赢时，回报为+1
            reward = 1.0 if winner == Player.black else -1.0
            
            # 根据论文，第一轮训练时，基线v(s)设为0
            baseline_value = 0.0
            advantage = reward - baseline_value

            for transition in trajectory:
                # loss = -log(π(a|s)) * (z - v(s))
                #      = -log(π(a|s)) * Advantage
                batch_loss_terms.append(-transition['action_log_prob'] * advantage)
        
        if not batch_loss_terms:
            print("警告：当前批次没有有效的训练数据。")
            return

        # 将整个批次所有步数的损失加起来，然后进行一次反向传播
        total_loss = torch.stack(batch_loss_terms).sum()
        total_loss.backward()
        self.optimizer.step()

    def run(self):
        """主训练循环，现在按批次进行"""
        num_batches = self.args.num_games // self.args.games_per_batch
        
        for batch_num in tqdm(range(1, num_batches + 1), desc="RL训练批次"):
            self.policy_net.train()
            
            # 收集一个批次的游戏数据
            batch_trajectories = []
            for _ in range(self.args.games_per_batch):
                # 从对手池中随机选择一个对手
                if len(self.opponent_pool) > 0:
                    opponent_state_dict = random.choice(self.opponent_pool)
                    self.opponent_net.load_state_dict(opponent_state_dict)
                
                winner, trajectory = self.run_self_play_game()
                self.win_history.append(1 if winner == Player.black else 0)
                batch_trajectories.append((winner, trajectory))

            # 对整个批次的数据进行一次训练
            self.train_on_batch(batch_trajectories)

            # 定期更新对手池和保存模型
            if batch_num * self.args.games_per_batch % self.args.update_opponent_every == 0:
                self.opponent_pool.append(copy.deepcopy(self.policy_net.state_dict()))
                win_rate = np.mean(list(self.win_history)) * 100
                print(f"\n游戏 {batch_num * self.args.games_per_batch}: 最近{len(self.win_history)}盘棋胜率: {win_rate:.1f}%. 模型已更新至对手池。")
                torch.save({'model_state_dict': self.policy_net.state_dict(), 'args': self.args}, self.args.rl_model_save_path)

        print(f"强化学习训练完成！最终模型已保存至 {self.args.rl_model_save_path}")


def parse_rl_args():
    parser = argparse.ArgumentParser(description='围棋策略网络强化学习训练')
    parser.add_argument('--sl_model_path', type=str, required=True, help='预训练的SL模型路径')
    parser.add_argument('--rl_model_save_path', type=str, default='/ailab/user/yangliujia/codes/genesis/Weiqi/checkpoints/rl_policy_model.pth', help='强化学习后模型的保存路径')
    parser.add_argument('--num_games', type=int, default=25600, help='自我对弈的总局数')
    parser.add_argument('--games_per_batch', type=int, default=1, help='每个训练批次包含的游戏局数')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率')
    parser.add_argument('--opponent_pool_size', type=int, default=20, help='对手池的大小')
    parser.add_argument('--update_opponent_every', type=int, default=500, help='每隔多少局更新一次对手池')
    parser.add_argument('--max_moves_per_game', type=int, default=300, help='每盘棋的最大步数上限')
    parser.add_argument('--resnet_blocks', type=int, default=8, help='ResNet残差块数量')
    parser.add_argument('--resnet_filters', type=int, default=128, help='ResNet滤波器数量')
    return parser.parse_args()

if __name__ == '__main__':
    # 为了在非命令行环境测试，可以手动创建参数
    # args = parse_rl_args() 
    class Args:
        sl_model_path = 'checkpoints/resnet/best_model_ddp_steplr_150_datanum_50_lr_1e-4.pth' # <--- 请务必替换为你的真实模型路径
        rl_model_save_path = '/ailab/user/yangliujia/codes/genesis/Weiqi/checkpoints/rl_policy_model.pth'
        num_games = 128
        games_per_batch = 32
        lr = 1e-5
        opponent_pool_size = 20
        update_opponent_every = 16
        max_moves_per_game = 300
        resnet_blocks = 8
        resnet_filters = 128
    
    # 在实际运行时，请使用下面这行来解析命令行参数
    # args = parse_rl_args()
    
    # 这里我们用上面定义的类来模拟参数，方便测试
    # 在你实际运行脚本时，请注释掉下面这行和上面的Args类，并取消注释 `args = parse_rl_args()`
    args = Args()

    if not os.path.exists(args.sl_model_path):
        print(f"错误：找不到SL模型文件 '{args.sl_model_path}'。")
        print("请确保路径正确，或先训练一个SL模型。为了让代码能运行，将创建一个随机初始化的模型。")
        # 如果找不到预训练模型，保存一个随机初始化的，以便代码能继续运行
        temp_model = PolicyNetwork_resnet(input_channels=27, num_filters=args.resnet_filters, num_residual_blocks=args.resnet_blocks)
        torch.save({'model_state_dict': temp_model.state_dict(), 'args': args}, args.sl_model_path)
        print(f"已在 '{args.sl_model_path}' 创建一个随机模型。")


    trainer = RLTrainer(args)
    trainer.run()