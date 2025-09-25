import torch
import glob
import random
import os
from tqdm import tqdm
import multiprocessing as mp

from .game_player import play_game_worker


def load_random_opponent(args):
    """加载对手模型状态字典到CPU。"""
    opponent_files = glob.glob(os.path.join(args.enemies_ckpt_dir, "*.pth"))
    opponent_ckpt_path = random.choice(opponent_files)
    print(f"Loading opponent: {opponent_ckpt_path}")
    checkpoint = torch.load(opponent_ckpt_path, map_location='cpu')
    return checkpoint['model_state_dict']

def train(model, args, device):
    """强化学习主训练循环（GPU并行版，支持梯度累积）。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.rl_lr, weight_decay=1e-4)
    mp.set_start_method('spawn', force=True)
    print("multiprocessing start method set to 'spawn'.")
    
    # 获取可用的GPU数量
    num_gpus = torch.cuda.device_count(); print(f"Found {num_gpus} GPUs.")
    num_parallel = args.num_parallel; print(f"Hardware parallelism set to {num_parallel} games.")
    total_games_per_round = num_gpus * num_parallel; print(f"Total games per round are {total_games_per_round}.") # 每轮并行对弈的总数量
    accumulation_rounds = args.minibatch // total_games_per_round # 运行多少轮才能够一个minibatch 

    for i in range(args.num_iterations):
        print(f"--- Iteration {i + 1}/{args.num_iterations} ---")
        model.to('cpu')
        current_model_state = model.state_dict()
        opponent_model_state = load_random_opponent(args)

        total_wins = 0
        total_loss_val = 0.0
        grad_batches = []
        games_played = 0

        # --- 内循环：分批进行并行对弈，直到收集够一个minibatch ---
        for round_num in range(accumulation_rounds):
            print(f"  > Starting accumulation round {round_num + 1}/{accumulation_rounds}...")
            
            # 准备当前这一小批并行任务的参数，并将任务分配到不同的GPU
            task_args = [
                (args, current_model_state, opponent_model_state, games_played + j, args.minibatch, i + 1, j % num_gpus)
                for j in range(total_games_per_round)
            ]

            # --- ipdb调试时使用的代码 ---
            # results = []
            # for single_task_args in tqdm(task_args, desc=f"  Playing Round {round_num+1} (Debug Mode)"):
            #     results.append(play_game_worker(single_task_args))
            # -------------------------

            with mp.Pool(processes=total_games_per_round) as pool:
                results = list(tqdm(pool.imap_unordered(play_game_worker, task_args), total=total_games_per_round, desc=f"  Playing Round {round_num+1}"))

            for grads, loss_val, won in results:
                grad_batches.append(grads)
                total_loss_val += loss_val
                if won: total_wins += 1
            
            games_played += total_games_per_round

        model.to(device)
        optimizer.zero_grad()
        # 将所有收集到的梯度平均
        for i_param, param in enumerate(model.parameters()):
            if param.requires_grad:
                grad_sum = torch.stack([batch[i_param] for batch in grad_batches if i_param < len(batch)]).sum(dim=0)
                param.grad = (grad_sum / args.minibatch).to(device)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        avg_loss = total_loss_val / args.minibatch
        print(f"Iteration {i + 1} complete:")
        print(f"  Win rate across {games_played} games: {total_wins}/{games_played} = {total_wins/games_played:.2%}")
        print(f"  Average loss: {avg_loss:.4f}")

        # 模型保存
        model.to('cpu')
        if (i + 1) % args.save_enemy == 0:
            opponent_save_path = os.path.join(args.enemies_ckpt_dir, f"opponent_iter_{i + 1}.pth")
            torch.save({'model_state_dict': model.state_dict(), 'iteration': i + 1}, opponent_save_path)
            print(f"New opponent model saved to: {opponent_save_path}")
            
        if (i + 1) % args.save_model == 0:
            checkpoint_path = os.path.join(args.ckpt_dir, f"AI_RL_iter_{i + 1}.pth")
            torch.save({'model_state_dict': model.state_dict(), 'iteration': i + 1}, checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")