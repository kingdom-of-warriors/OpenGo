import torch
import torch.nn.functional as F

@torch.jit.script
def find_valid(board: torch.Tensor) -> torch.Tensor:
    """
    使用向量化和卷积操作找到可落子点（已修正逻辑），兼容JIT。
    
    Args:
        board: torch.Tensor, shape (2, 19, 19)
               board[0] 为自己的棋子 (1=有子, 0=无子)
               board[1] 为对手的棋子 (1=有子, 0=无子)
    
    Returns:
        valid_mask: torch.Tensor, shape (19, 19), dtype=torch.uint8
                    1表示可以落子，0表示不能落子
    """
    # 已经有棋子的点不能再下
    occupied = board[0] + board[1]
    valid_mask = (occupied == 0).to(torch.uint8)

    # 找出被对方棋子完全包围的“禁入点”
    kernel = torch.tensor([[0, 1, 0], 
                           [1, 0, 1], 
                           [0, 1, 0]], dtype=torch.float32, device=board.device)
    kernel = kernel.view(1, 1, 3, 3)
    ones_board = torch.ones(1, 1, 19, 19, dtype=torch.float32, device=board.device)
    total_neighbor_count = F.conv2d(ones_board, kernel, padding=1).squeeze()
    opponent_board = board[1].unsqueeze(0).unsqueeze(0).float()
    opponent_neighbor_count = F.conv2d(opponent_board, kernel, padding=1).squeeze()

    # 一个点的邻居总数 == 它的对手邻居数
    forbidden_points = (total_neighbor_count == opponent_neighbor_count)
    invalid_positions = (valid_mask == 1) & forbidden_points
    valid_mask[invalid_positions] = 0
    
    return valid_mask


def cal_liberity(board: torch.Tensor):
    """
    计算围棋棋盘上每个棋子的气数
    
    Args:
        board: torch.Tensor, shape [2, 19, 19]
               board[0]: 我方棋子 (1表示有子，0表示无子)
               board[1]: 敌方棋子 (1表示有子，0表示无子)
    Returns:
        torch.Tensor, shape [16, 19, 19]
        前8个通道 (0-7): 我方棋子的1-8气 (res[i][r][c]=1 表示我方棋子(r,c)有i+1气)
        后8个通道 (8-15): 敌方棋子的1-8气 (res[i][r][c]=1 表示敌方棋子(r,c)有i-7气)
    """
    result = torch.zeros(16, 19, 19)
    
    player_stones = board[0]  # 我方棋子
    opponent_stones = board[1]  # 敌方棋子
    all_stones = player_stones + opponent_stones  # 所有棋子
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def get_group_and_liberties(stones, start_r, start_c, visited):
        """使用BFS获取连通的棋子组和其气数"""
        if visited[start_r, start_c] or stones[start_r, start_c] == 0:
            return set(), set()
        
        group, liberties = set(), set()
        queue = [(start_r, start_c)]
        visited[start_r, start_c] = True
        
        while queue:
            r, c = queue.pop(0)
            group.add((r, c))
            
            # 检查四个方向
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                # 边界检查
                if 0 <= nr < 19 and 0 <= nc < 19:
                    if all_stones[nr, nc] == 0: liberties.add((nr, nc)) # 空位，计为气
                    elif stones[nr, nc] == 1 and not visited[nr, nc]:  # 同色棋子
                        visited[nr, nc] = True
                        queue.append((nr, nc))
        
        return group, liberties
    
    # 处理我方棋子
    visited_player = torch.zeros(19, 19, dtype=torch.bool)
    for r in range(19):
        for c in range(19):
            if player_stones[r, c] == 1 and not visited_player[r, c]:
                group, liberties = get_group_and_liberties(player_stones, r, c, visited_player)
                liberty_count = len(liberties)
                
                # 限制气数在1-8范围内
                liberty_count = min(max(liberty_count, 1), 8)
                
                # 为该组的所有棋子设置气数
                for gr, gc in group:
                    result[liberty_count - 1, gr, gc] = 1  # 前8个通道表示我方1-8气
    
    # 处理敌方棋子
    visited_opponent = torch.zeros(19, 19, dtype=torch.bool)
    for r in range(19):
        for c in range(19):
            if opponent_stones[r, c] == 1 and not visited_opponent[r, c]:
                group, liberties = get_group_and_liberties(opponent_stones, r, c, visited_opponent)
                liberty_count = len(liberties)
                
                # 限制气数在1-8范围内
                liberty_count = min(max(liberty_count, 1), 8)
                
                for gr, gc in group:
                    result[8 + liberty_count - 1, gr, gc] = 1  # 后8个通道表示敌方1-8气
    
    return result