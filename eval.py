import torch
import numpy as np
import os
import random

from models.policy_networks import create_model
from config import parse_args

# 使用dlgo库
from dlgo.gotypes import Player, Point
from dlgo.goboard_slow import GameState, Move
from dlgo.scoring import compute_game_result

device = "cuda" if torch.cuda.is_available() else "cpu"
args = parse_args()
model = create_model(args, device)
ckpt_path = "ckpt/resnet/test1.pth"
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

class GoGameEvaluator:
    def __init__(self, model, device, board_size=19):
        self.model = model
        self.device = device
        self.board_size = board_size
        self.game_history = []  # 存储历史棋盘状态
        
    def print_board(self, board):
        """美观地打印棋盘"""
        print('   A B C D E F G H J K L M N O P Q R S T')
        for row in range(19, 0, -1):
            line = []
            for col in range(1, 20):
                stone = board.get(Point(row=row, col=col))
                if stone == Player.black:
                    line.append('X')  # 黑棋
                elif stone == Player.white:
                    line.append('O')  # 白棋
                else:
                    line.append('.')  # 空点
            print(f'{row:2d} {" ".join(line)} {row:2d}')
        print('   A B C D E F G H J K L M N O P Q R S T')
    
    def game_state_to_tensor(self, game_state):
        """将GameState转换为模型输入张量 (27, 19, 19)"""
        input_tensor = np.zeros((27, self.board_size, self.board_size), dtype=np.float32)
        
        # 获取当前棋盘状态
        current_board = np.zeros((self.board_size, self.board_size), dtype=int)
        for row in range(1, self.board_size + 1):
            for col in range(1, self.board_size + 1):
                stone = game_state.board.get(Point(row=row, col=col))
                if stone == Player.black:
                    current_board[row-1, col-1] = 1
                elif stone == Player.white:
                    current_board[row-1, col-1] = 2
        
        # 第1-3维：当前状态
        if game_state.next_player == Player.black:
            # 黑棋回合
            input_tensor[0] = (current_board == 1).astype(np.float32)  # 我方（黑棋）
            input_tensor[1] = (current_board == 2).astype(np.float32)  # 敌方（白棋）
            input_tensor[2] = np.ones((self.board_size, self.board_size), dtype=np.float32)  # 黑棋回合
        else:
            # 白棋回合
            input_tensor[0] = (current_board == 2).astype(np.float32)  # 我方（白棋）
            input_tensor[1] = (current_board == 1).astype(np.float32)  # 敌方（黑棋）
            input_tensor[2] = np.zeros((self.board_size, self.board_size), dtype=np.float32)  # 白棋回合
        
        # 第4-27维：历史状态 (t-1 到 t-12)
        for i in range(min(12, len(self.game_history))):
            hist_idx = len(self.game_history) - 1 - i
            hist_board = self.game_history[hist_idx]
            
            if game_state.next_player == Player.black:
                # 黑棋回合
                input_tensor[3 + i*2] = (hist_board == 1).astype(np.float32)     # 我方历史
                input_tensor[4 + i*2] = (hist_board == 2).astype(np.float32)     # 敌方历史
            else:
                # 白棋回合
                input_tensor[3 + i*2] = (hist_board == 2).astype(np.float32)     # 我方历史
                input_tensor[4 + i*2] = (hist_board == 1).astype(np.float32)     # 敌方历史
        
        return torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0).to(self.device)
    
    def get_model_move(self, game_state):
        """获取模型的落子建议"""
        # 保存当前状态到历史
        current_board = np.zeros((self.board_size, self.board_size), dtype=int)
        for row in range(1, self.board_size + 1):
            for col in range(1, self.board_size + 1):
                stone = game_state.board.get(Point(row=row, col=col))
                if stone == Player.black:
                    current_board[row-1, col-1] = 1
                elif stone == Player.white:
                    current_board[row-1, col-1] = 2
        self.game_history.append(current_board.copy())
        
        # 获取模型输入
        input_tensor = self.game_state_to_tensor(game_state)
        
        with torch.no_grad():
            # 模型预测
            policy_logits = self.model(input_tensor)  # (1, 361)
            policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        
        # 将361维输出转换为19x19坐标
        policy_2d = policy_probs.reshape(19, 19)
        
        # 获取合法落子位置
        legal_moves = []
        for row in range(1, self.board_size + 1):
            for col in range(1, self.board_size + 1):
                move = Move.play(Point(row=row, col=col))
                if game_state.is_valid_move(move):
                    legal_moves.append((row, col, policy_2d[row-1, col-1]))
        
        if not legal_moves:
            return Move.pass_turn()
        
        # 选择概率最高的合法落子
        legal_moves.sort(key=lambda x: x[2], reverse=True)
        best_row, best_col, prob = legal_moves[0]
        
        print(f"AI选择落子: {chr(ord('A') + best_col - 1)}{20 - best_row} (概率: {prob:.3f})")
        return Move.play(Point(row=best_row, col=best_col))
    
    def parse_human_move(self, move_str):
        """解析人类输入的落子格式，如 'D4' 或 'pass'"""
        move_str = move_str.strip().upper()
        
        if move_str == 'PASS':
            return Move.pass_turn()
        
        if len(move_str) < 2:
            return None
            
        try:
            col_char = move_str[0]
            row_str = move_str[1:]
            
            # 跳过 'I' 列
            if col_char >= 'I':
                col = ord(col_char) - ord('A')
            else:
                col = ord(col_char) - ord('A') + 1
                
            row = 20 - int(row_str)  # 转换为dlgo坐标系
            
            if 1 <= row <= 19 and 1 <= col <= 19:
                return Move.play(Point(row=row, col=col))
            else:
                return None
        except:
            return None
    
    def play_human_vs_ai(self):
        """人机对弈主函数"""
        print("=== 围棋人机对弈 ===")
        print("输入格式: 如 'D4', 'K10' 等，或输入 'pass' 跳过")
        print("输入 'quit' 退出游戏")
        print()
        
        # 询问人类执什么颜色
        while True:
            color_choice = input("请选择你的颜色 (black/white): ").strip().lower()
            if color_choice in ['black', 'b']:
                human_player = Player.black
                ai_player = Player.white
                break
            elif color_choice in ['white', 'w']:
                human_player = Player.white
                ai_player = Player.black
                break
            else:
                print("请输入 'black' 或 'white'")
        
        # 初始化游戏
        game = GameState.new_game(19)
        print(f"\n你执{'黑棋' if human_player == Player.black else '白棋'}，AI执{'白棋' if ai_player == Player.white else '黑棋'}")
        print("\n初始棋盘:")
        self.print_board(game.board)
        
        while not game.is_over():
            print(f"\n轮到{'黑棋' if game.next_player == Player.black else '白棋'}:")
            
            if game.next_player == human_player:
                # 人类回合
                while True:
                    move_input = input("请输入你的落子 (如D4, 或pass): ").strip()
                    
                    if move_input.lower() == 'quit':
                        print("游戏结束!")
                        return
                    
                    move = self.parse_human_move(move_input)
                    if move is None:
                        print("输入格式错误，请重新输入!")
                        continue
                    
                    if game.is_valid_move(move):
                        game = game.apply_move(move)
                        break
                    else:
                        print("非法落子，请重新输入!")
            else:
                # AI回合
                print("AI思考中...")
                move = self.get_model_move(game)
                game = game.apply_move(move)
                
                if move.is_pass:
                    print("AI选择 PASS")
            
            print("\n当前棋盘:")
            self.print_board(game.board)
        
        # 游戏结束，计算得分
        print("\n=== 游戏结束 ===")
        try:
            game_result = compute_game_result(game)
            winner_str = "黑棋" if game_result.winner == Player.black else "白棋"
            
            if game_result.winner == human_player:
                print(f"🎉 恭喜！你获胜了！({winner_str} 胜 {game_result.winning_margin} 目)")
            else:
                print(f"😔 AI获胜了！({winner_str} 胜 {game_result.winning_margin} 目)")
        except:
            print("无法计算得分，游戏结束。")

def main():
    evaluator = GoGameEvaluator(model, device)
    evaluator.play_human_vs_ai()

if __name__ == "__main__":
    main()