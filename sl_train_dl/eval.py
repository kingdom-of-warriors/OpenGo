import torch
import numpy as np
import os
from datetime import datetime
import sys
from dlgo.utils import print_board

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.policy_networks import create_model, PolicyNetwork_resnet
from sl_train_dl.config import parse_args
from dlgo.gotypes import Player, Point
from dlgo.goboard_slow import GameState, Move

device = "cuda" if torch.cuda.is_available() else "cpu"
args = parse_args()
model = create_model(args, device)
ckpt_path = args.ckpt_path
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

class GoGameEvaluator:
    def __init__(self, model: PolicyNetwork_resnet, device, board_size=19):
        self.model = model
        self.device = device
        self.board_size = board_size
        self.game_history = []  # 存储历史棋盘状态
        self.move_history = []  # 存储着法历史，用于生成SGF
    
    def point_to_sgf_coord(self, point):
        """将Point坐标转换为SGF格式坐标"""
        if point is None: return ""
        col_chars = "abcdefghijklmnopqrs"
        sgf_col = col_chars[point.col - 1]
        sgf_row = col_chars[19 - point.row]
        return sgf_col + sgf_row
    
    def save_sgf(self, human_player, game_result=None, filename=None):
        """保存对局为SGF文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"game_{timestamp}.sgf"
        
        # 确保保存目录存在
        os.makedirs("saved_games", exist_ok=True)
        filepath = os.path.join("saved_games", filename)
        
        sgf_content = []
        # SGF头信息
        sgf_content.append("(;GM[1]FF[4]SZ[19]")
        sgf_content.append(f"DT[{datetime.now().strftime('%Y-%m-%d')}]")
        sgf_content.append("AP[OpenGo:1.0]")
        sgf_content.append("RU[Chinese]")
        sgf_content.append("KM[7.5]")
        if human_player == Player.black:
            sgf_content.append("PB[Human]")
            sgf_content.append("PW[OpenGo AI]")
        else:
            sgf_content.append("PB[OpenGo AI]")
            sgf_content.append("PW[Human]")
        
        # 对局结果
        if game_result:
            if game_result.winner == Player.black:
                result_str = f"B+{game_result.winning_margin:.1f}"
            else:
                result_str = f"W+{game_result.winning_margin:.1f}"
            sgf_content.append(f"RE[{result_str}]")
        
        # 着法序列
        sgf_moves = []
        for player, move in self.move_history:
            if move.is_pass:
                move_str = f";{player.name[0].upper()}[]"
            else:
                coord = self.point_to_sgf_coord(move.point)
                move_str = f";{player.name[0].upper()}[{coord}]"
            sgf_moves.append(move_str)
        
        # 组合完整的SGF
        full_sgf = "".join(sgf_content) + "".join(sgf_moves) + ")"
        with open(filepath, 'w', encoding='utf-8') as f: f.write(full_sgf)
        print(f"✅ 棋谱已保存到: {filepath}")
        return filepath
    
    def game_state_to_tensor(self, game_state):
        """将GameState转换为模型输入张量 (27, 19, 19)"""
        input_tensor = np.zeros((27, self.board_size, self.board_size), dtype=np.float32)
        
        # 获取当前棋盘状态
        current_board = np.zeros((self.board_size, self.board_size), dtype=int)
        for row in range(1, self.board_size + 1):
            for col in range(1, self.board_size + 1):
                stone = game_state.board.get(Point(row=row, col=col))
                if stone == Player.black: current_board[row-1, col-1] = 1
                elif stone == Player.white: current_board[row-1, col-1] = 2
        
        # 第1-3维：当前状态
        if game_state.next_player == Player.black:
            input_tensor[0] = (current_board == 1).astype(np.float32)  # 我方（黑棋）
            input_tensor[1] = (current_board == 2).astype(np.float32)  # 敌方（白棋）
            input_tensor[2] = np.ones((self.board_size, self.board_size), dtype=np.float32)  # 黑棋回合
        else:
            input_tensor[0] = (current_board == 2).astype(np.float32)  # 我方（白棋）
            input_tensor[1] = (current_board == 1).astype(np.float32)  # 敌方（黑棋）
            input_tensor[2] = np.zeros((self.board_size, self.board_size), dtype=np.float32)  # 白棋回合
        
        # 第4-27维：历史状态 (t-1 到 t-12)
        for i in range(min(12, len(self.game_history))):
            hist_idx = len(self.game_history) - 1 - i
            hist_board = self.game_history[hist_idx]
            
            if game_state.next_player == Player.black:
                input_tensor[3 + i*2] = (hist_board == 1).astype(np.float32)     # 我方历史
                input_tensor[4 + i*2] = (hist_board == 2).astype(np.float32)     # 敌方历史
            else:
                input_tensor[3 + i*2] = (hist_board == 2).astype(np.float32)     # 我方历史
                input_tensor[4 + i*2] = (hist_board == 1).astype(np.float32)     # 敌方历史
        
        return torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0).to(self.device)
    
    def get_model_move(self, game_state: GameState):
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
        
        # 获取模型合法输出
        input_tensor = self.game_state_to_tensor(game_state)
        policy_logits = self.model.sample(input_tensor, requires_grad=False)
        policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        policy_2d = policy_probs.reshape(19, 19)
        
        legal_moves = []
        for row in range(1, self.board_size + 1):
            for col in range(1, self.board_size + 1):
                move = Move.play(Point(row=row, col=col))
                if game_state.is_valid_move(move):
                    legal_moves.append((row, col, policy_2d[row-1, col-1]))
        if not legal_moves: return Move.pass_turn()
        legal_moves.sort(key=lambda x: x[2], reverse=True)
        best_row, best_col, prob = legal_moves[0]

        def col_to_display_char(col):
            if col <= 8: return chr(ord('A') + col - 1) # A-H
            else: return chr(ord('A') + col) # J-T
        
        display_col = col_to_display_char(best_col)
        print(f"AI选择落子: {display_col}{best_row} (概率: {prob:.3f})")
        return Move.play(Point(row=best_row, col=best_col))
    
    def parse_human_move(self, move_str):
        """解析人类输入的落子格式，如 'D4' 或 'pass'"""
        move_str = move_str.strip().upper() # 变成大写
        if move_str == 'PASS': return Move.pass_turn()
        if len(move_str) < 2: return None
            
        col_char = move_str[0]
        row_str = move_str[1:]
        
        # 跳过 'I' 列
        if col_char >= 'I': col = ord(col_char) - ord('A')
        else: col = ord(col_char) - ord('A') + 1
        row = int(row_str)
        
        if 1 <= row <= 19 and 1 <= col <= 19:
            return Move.play(Point(row=row, col=col))
        else:
            return None
    
    def play_human_vs_ai(self):
        """人机对弈主函数"""
        print("=== 围棋人机对弈 ===")
        print("输入格式: 如 'D4', 'K10' 等，或输入 'pass' 跳过")
        print("输入 'quit' 退出游戏，输入 'save' 保存当前棋谱")
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
        
        # 初始化游戏
        game = GameState.new_game(19)
        self.move_history = []
        
        print(f"\n你执{'黑棋' if human_player == Player.black else '白棋'}，AI执{'白棋' if ai_player == Player.white else '黑棋'}")
        print("\n初始棋盘:")
        print_board(game.board)
        
        game_ended_manually = False
        
        while not game.is_over():
            print(f"\n轮到{'黑棋' if game.next_player == Player.black else '白棋'}:")
            
            if game.next_player == human_player:
                # 人类回合
                while True:
                    move_input = input("请输入你的落子 (如D4, pass, quit): ").strip()
                    
                    if move_input.lower() == 'quit':
                        game_ended_manually = True
                        break
                    
                    move = self.parse_human_move(move_input)
                    if move is None:
                        print("输入格式错误，请重新输入!")
                        continue
                    
                    if game.is_valid_move(move):
                        self.move_history.append((game.next_player, move))
                        game = game.apply_move(move)
                        break
                    else:
                        print("非法落子，请重新输入!")
                
                if game_ended_manually:
                    break
                    
            else:
                move = self.get_model_move(game)
                self.move_history.append((game.next_player, move))
                game = game.apply_move(move)
                if move.is_pass: print("AI选择 PASS")
            
            print("\n当前棋盘:")
            print_board(game.board)
        print("\n=== 游戏结束 ===")
        
        game_result = None
        if self.move_history:
            while True:
                save_choice = input("\n是否保存棋谱到SGF文件? (y/n): ").strip().lower()
                if save_choice in ['y', 'yes']:
                    filename_input = input("请输入文件名(留空使用默认名称): ").strip()
                    filename = filename_input if filename_input else None
                    
                    saved_path = self.save_sgf(human_player, game_result, filename)
                    if saved_path:
                        print(f"棋谱已保存!")
                    break
                elif save_choice in ['n', 'no']:
                    print("棋谱未保存")
                    break
                else:
                    print("请输入 'y' 或 'n'")

def main():
    evaluator = GoGameEvaluator(model, device)
    evaluator.play_human_vs_ai()

if __name__ == "__main__":
    main()