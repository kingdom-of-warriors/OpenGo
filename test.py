# test.py

import random
from dlgo.gotypes import Player, Point
# Move 类和 GameState 类都在 goboard_slow 模块中
from dlgo.goboard_slow import GameState, Move
# 计分函数在 scoring 模块中
from dlgo.scoring import compute_game_result

# 这是一个辅助函数，用于在终端更美观地打印棋盘
def print_board(board):
    print('   A B C D E F G H J K L M N O P Q R S T') # 棋盘列坐标
    for row in range(19, 0, -1):
        line = []
        for col in range(1, 20):
            stone = board.get(Point(row=row, col=col))
            if stone == Player.black:
                line.append('X') # 黑棋
            elif stone == Player.white:
                line.append('O') # 白棋
            else:
                line.append('.') # 空点
        print(f'{row:2d} {" ".join(line)} {row:2d}')
    print('   A B C D E F G H J K L M N O P Q R S T')

def main():
    # --- 1. 初始化游戏 ---
    print("--- 1. 初始化一个19x19的空棋盘 ---")
    game = GameState.new_game(19)
    print_board(game.board)
    print(f"初始状态，轮到: {game.next_player}")

    # --- 2. 下几步棋 ---
    print("\n--- 2. 下几步棋 ---")
    # 注意：dlgo的坐标是从1开始的，所以(3,3)是C17
    move = Move.play(Point(row=3, col=3))
    game = game.apply_move(move) # apply_move 会返回一个新的 GameState
    print(f"黑方落子在 C17 (3,3)")
    print_board(game.board)
    print(f"现在轮到: {game.next_player}")
    
    move = Move.play(Point(row=17, col=17))
    game = game.apply_move(move)
    print(f"\n白方落子在 R4 (17,17)")
    print_board(game.board)
    print(f"现在轮到: {game.next_player}")

    # --- 3. 检查合法性 ---
    print("\n--- 3. 演示落子合法性判断 ---")
    
    # a. 尝试在已有棋子的位置落子
    print("\n尝试在(3,3)再次落子...")
    illegal_move = Move.play(Point(row=3, col=3))
    if not game.is_valid_move(illegal_move):
        print("判断成功：不能在已有棋子的位置重复落子！")

    # b. 演示“打劫”（Ko）规则
    print("\n创建一个简单的“打劫”局面...")
    # B: (2,2), W: (1,2), W: (3,2), W: (2,1)
    game = game.apply_move(Move.play(Point(2,2))) # B at B18
    game = game.apply_move(Move.play(Point(1,2))) # W at B19
    game = game.apply_move(Move.play(Point(10,10))) # B at K10 (填劫)
    game = game.apply_move(Move.play(Point(3,2))) # W at B17
    game = game.apply_move(Move.play(Point(11,11))) # B at L9 (填劫)
    game = game.apply_move(Move.play(Point(2,1))) # W at A18
    print("当前棋盘，白棋可以提掉(2,2)处的黑子：")
    print_board(game.board)

    # 黑方提劫
    ko_capture_move_B = Move.play(Point(2,3))
    game = game.apply_move(ko_capture_move_B)
    print(f"\n黑方在 C18 (2,3) 提劫！")
    print_board(game.board)
    
    # 此时，白方不能立即在(2,2)提回
    immediate_recapture_W = Move.play(Point(2,2))
    print(f"\n现在轮到白方，检查是否可以立即在(2,2)提回...")
    if not game.is_valid_move(immediate_recapture_W):
        print("判断成功：白方不能立即提回，违反了“打劫”规则！")
    
    # --- 4. 游戏结束与计分 ---
    print("\n--- 4. 模拟游戏结束与计分 ---")
    # 快速下一些随机的棋步来填充棋盘
    for _ in range(50):
        if game.is_over():
            break
        legal_moves = game.legal_moves()
        if not legal_moves:
            game = game.apply_move(Move.pass_turn())
        else:
            game = game.apply_move(random.choice(legal_moves))

    print("经过50手快速随机对局后的棋盘：")
    print_board(game.board)

    # 模拟双方连续PASS来结束游戏
    print("\n模拟双方连续PASS...")
    game = game.apply_move(Move.pass_turn())
    game = game.apply_move(Move.pass_turn())
    
    if game.is_over():
        print("棋局已确认结束！")
    
        print("\n开始终局计分（中国规则）...")
        # 使用dlgo的计分函数
        game_result = compute_game_result(game)
        
        winner_str = "黑方" if game_result.winner == Player.black else "白方"
        print(f"计分结果 -> 获胜方: {winner_str}, 领先目数: {game_result.winning_margin}")

if __name__ == '__main__':
    main()