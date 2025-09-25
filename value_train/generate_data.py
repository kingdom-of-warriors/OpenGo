import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from tqdm import tqdm
from collections import deque
import sys
import multiprocessing as mp

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.policy_networks import PolicyNetwork, create_model
from config import parse_args
from rl_utils import game_state_to_tensor
from dlgo.goboard_slow import GameState, Move
from dlgo.gotypes import Player, Point
from dlgo.scoring import compute_game_result

# --- 数据生成 ---
def generate_one_game_sample(sl_model: PolicyNetwork, 
                             rl_model: PolicyNetwork, 
                             device, board_size=19, 
                             history_length=12, T=360):
    """
    为价值网络生成一个训练样本
    返回: (s_U+1 的特征张量, 最终胜负 z)
    """
    game_state = GameState.new_game(board_size)
    game_history = deque(maxlen=history_length + 1) # 当前状态也要保存
    
    # 随机选择一个时间步 U
    U = random.randint(1, T - 20)
    U_state = None

    # 整个下棋过程
    input_tensor = torch.zeros((27, board_size, board_size)).unsqueeze(0).to(device)
    for t in tqdm(range(1, T + 1), leave=False):
        # 使用 SL 模型采样
        move_probs = sl_model.sample(input_tensor, False).squeeze(0)
        invalid_move = 0
        legal_moves = []
        legal_mask = torch.zeros_like(move_probs)
        for row in range(board_size):
            for col in range(board_size):
                point = Point(row + 1, col + 1)
                if not game_state.is_valid_move(Move.play(point)):
                    legal_mask[row * board_size + col] = -float('inf')
                    invalid_move += 1 # 计算不合法的步数
                else:
                    legal_moves.append(row * board_size + col)
        # 从合法走法中采样
        if invalid_move == board_size * board_size: # 没有合法的位置
            return None, None
        if t < U: # 从概率中采样
            move_probs = move_probs + legal_mask
            move_probs = torch.softmax(move_probs / 0.5, dim=-1) # 需要增加温度参数
            move_idx = torch.multinomial(move_probs, 1).item()
        elif t == U: # 第U步随机走
            move_idx = random.choice(legal_moves)
        elif t > U: # 选择最强变化
            move_probs = move_probs + legal_mask
            move_idx = torch.argmax(move_probs).item()
        point = Point(row=move_idx // board_size + 1, col=move_idx % board_size + 1)
        game_state = game_state.apply_move(Move.play(point))

        current_board = np.zeros((board_size, board_size), dtype=np.int8)
        # 获得当下局面的input_tensor
        for r in range(1, board_size + 1):
            for c in range(1, board_size + 1):
                stone = game_state.board.get(Point(row=r, col=c))
                if stone == Player.black: current_board[r - 1, c - 1] = 1
                elif stone == Player.white: current_board[r - 1, c - 1] = 2
        game_history.append(current_board)
        input_tensor = game_state_to_tensor(game_state.next_player, game_history).to(device)
        if t == U: # 保存下来value网络的输入
            U_state = input_tensor

    # 确定最终胜负
    game_result = compute_game_result(game_state)
    U_player = Player.black if U % 2 == 1 else Player.white
    z = 1 if game_result.winner == U_player else -1

    return U_state, z

def generate_sample_worker(args_bundle):
    args, sl_model_state, rl_model_state, worker_id, device_id = args_bundle
    device = torch.device(f"cuda:{device_id}")
    # 为每个子进程设置独立的随机种子
    seed = os.getpid() + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    args.model = 'policy'
    # 在子进程的GPU上创建模型
    sl_model = create_model(args, device)
    sl_model.load_state_dict(sl_model_state)
    sl_model.eval()

    rl_model = create_model(args, device)
    rl_model.load_state_dict(rl_model_state)
    rl_model.eval()

    # 循环直到生成一个有效的样本
    while True:
        state, outcome = generate_one_game_sample(
            sl_model,
            rl_model,
            device,
        )
        if state is not None: # 将结果移动到CPU后返回
            return state.cpu(), outcome

# --- 训练主循环 ---
def generate_data(args, device, sl_model, rl_model):
    """价值网络的主训练循环"""
    num_steps = 1000
    batch_size = args.batch_size
    num_parallel_games = args.num_parallel # 从参数获取并行数量
    output_dir = "GoDataset/Value"
    os.makedirs(output_dir, exist_ok=True)
    print(f"数据将被保存在: {os.path.abspath(output_dir)}")

    mp.set_start_method('spawn', force=True)
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs. Using {num_parallel_games} parallel generators.")
    sl_model.to('cpu')
    rl_model.to('cpu')
    sl_model_state = sl_model.state_dict()
    rl_model_state = rl_model.state_dict()

    for step in range(num_steps):
        batch_states = []
        batch_outcomes = []
        # --- 使用进程池并行生成数据 ---
        task_args = [
            (args, sl_model_state, rl_model_state, i, i % num_gpus)
            for i in range(batch_size)
        ]

        with mp.Pool(processes=num_parallel_games) as pool:
            results_iterator = pool.imap_unordered(generate_sample_worker, task_args)
            gen_pbar = tqdm(results_iterator, total=batch_size, desc=f"Step {step+1}/{num_steps} Generating Data")
            
            for state, outcome in gen_pbar:
                batch_states.append(state)
                batch_outcomes.append(outcome)

        # --- 执行保存步骤 ---
        states_tensor = torch.stack(batch_states).squeeze(1).to(device) # [bs, 27, 19, 19]
        outcomes_tensor = torch.tensor(batch_outcomes, dtype=torch.float32).unsqueeze(1).to(device) # [bs, 1]

        file_path = os.path.join(output_dir, f"value_batch_{step}.pt")
        torch.save({'states': states_tensor,
                    'outcomes': outcomes_tensor}, file_path)
        tqdm.write(f"批次 {step+1} 已保存至: {file_path} (包含 {len(states_tensor)} 个样本)")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    args = parse_args()
    
    # 加载sl_model
    sl_model = create_model(args, device)
    sl_model_path = "ckpt/AI_12_192.pth" # 假设这是你的SL模型路径
    print(f"加载SL模型: {sl_model_path}")
    sl_ckpt = torch.load(sl_model_path, map_location=device)
    sl_model.load_state_dict(sl_ckpt['model_state_dict'])
    sl_model.eval()

    # 加载rl_model
    rl_model = create_model(args, device)
    rl_model_path = "ckpt/AI_12_192.pth" # 假设这是你的RL模型路径
    print(f"加载RL模型: {rl_model_path}")
    rl_ckpt = torch.load(rl_model_path, map_location=device)
    rl_model.load_state_dict(rl_ckpt['model_state_dict'])
    rl_model.eval()
    args.batch_size = 500
    # --- 开始生成数据 ---
    generate_data(args, device, sl_model, rl_model)
