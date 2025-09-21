import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义8种对称变换和逆变换
TFMs = { 
    'identity': lambda t: t,
    'r90':      lambda t: torch.rot90(t, 1, [2, 3]),
    'r180':     lambda t: torch.rot90(t, 2, [2, 3]),
    'r270':     lambda t: torch.rot90(t, 3, [2, 3]),
    'f':        lambda t: torch.flip(t, [3]),
    'f_r90':    lambda t: torch.rot90(torch.flip(t, [3]), 1, [2, 3]),
    'f_r180':   lambda t: torch.rot90(torch.flip(t, [3]), 2, [2, 3]),
    'f_r270':   lambda t: torch.rot90(torch.flip(t, [3]), 3, [2, 3]),
}

INV_TFMs = {
    'identity': lambda p: p,
    'r90':      lambda p: torch.rot90(p, -1, [1, 2]),
    'r180':     lambda p: torch.rot90(p, -2, [1, 2]),
    'r270':     lambda p: torch.rot90(p, -3, [1, 2]),
    'f':        lambda p: torch.flip(p, [2]),
    'f_r90':    lambda p: torch.flip(torch.rot90(p, -1, [1, 2]), [2]),
    'f_r180':   lambda p: torch.flip(torch.rot90(p, -2, [1, 2]), [2]),
    'f_r270':   lambda p: torch.flip(torch.rot90(p, -3, [1, 2]), [2]),
}

class ResidualBlock(nn.Module):
    """
    单个残差块的实现。
    输入 -> Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU -> 输出
    """
    def __init__(self, num_filters: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=num_filters, 
            out_channels=num_filters, 
            kernel_size=3, 
            padding='same'
        )
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            padding='same'
        )
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        
        return out
    
class PolicyNetwork(nn.Module):
    """基于ResNet的策略网络。"""
    def __init__(self, input_channels: int = 19, num_residual_blocks: int = 12, num_filters: int = 192):
        super(PolicyNetwork, self).__init__()
        self.conv_head = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding='same')
        self.bn_head = nn.BatchNorm2d(num_filters)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_residual_blocks)]
        )
        
        self.conv_policy = nn.Conv2d(num_filters, 2, kernel_size=1, padding='same')
        self.bn_policy = nn.BatchNorm2d(2)
        self.fc_policy = nn.Linear(2 * 19 * 19, 19 * 19)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn_head(self.conv_head(x)))
        out = self.res_blocks(out)
        policy = F.relu(self.bn_policy(self.conv_policy(out)))
        policy = torch.flatten(policy, start_dim=1)
        policy_logits = self.fc_policy(policy)

        return policy_logits

    def sample_rl(self, x: torch.Tensor):
        """
        利用8种对称性来增强预测并输出概率
        支持批处理输入 (N, C, H, W)。
        """
        n, c, h, w = x.shape
        symmetries = torch.stack([tfm(x) for tfm in TFMs.values()]) # [8, N, C, H, W]
        batch_x = symmetries.view(-1, c, h, w) # [8*N, C, H, W]
        batch_logits = self.forward(batch_x)
        batch_probs = F.softmax(batch_logits, dim=1)
        probs_by_symmetry = batch_probs.view(8, n, -1) # [8, N, H*W]

        untransformed_probs = []
        for i, inv_tfm in enumerate(INV_TFMs.values()):
            prob_maps = probs_by_symmetry[i].view(n, h, w) # [N, H*W] -> [N, H, W]
            untransformed_maps = inv_tfm(prob_maps)
            untransformed_probs.append(untransformed_maps.reshape(n, -1))
        avg_probs = torch.mean(torch.stack(untransformed_probs), dim=0)
            
        return avg_probs
    
    def sample(self, x: torch.Tensor, requires_grad: bool = False):
        """根据requires_grad决定使用哪种采样方法。"""
        if requires_grad: 
            return self.sample_rl(x)
        else: 
            with torch.no_grad(): return self.sample_rl(x)


class ValueNetwork(nn.Module):
    """基于ResNet的价值网络"""
    def __init__(self, input_channels: int = 19, num_residual_blocks: int = 12, num_filters: int = 192, board_size: int = 19):
        super(ValueNetwork, self).__init__()
        self.conv_head = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn_head = nn.BatchNorm2d(num_filters)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_residual_blocks)]
        )
        ##################### 以上与策略网络相同  下面是价值头#############################
        self.conv_value = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False) # 转化为1通道
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value1 = nn.Linear(1 * board_size * board_size, 256)
        self.fc_value2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。
        输入: x - 棋盘状态和历史记录的张量, shape: [N, C, H, W]
        输出: value - 局面的评估值, shape: [N, 1]
        """
        # 通过共享的躯干网络
        out = F.relu(self.bn_head(self.conv_head(x)))
        out = self.res_blocks(out)
        value = F.relu(self.bn_value(self.conv_value(out)))
        value = torch.flatten(value, start_dim=1)
        value = F.relu(self.fc_value1(value))
        value = torch.tanh(self.fc_value2(value)) # 缩放到 [-1, 1]
        
        return value


def create_model(args, device):
    """根据参数创建模型"""
    if args.model == 'policy':
        model = PolicyNetwork(
            input_channels=args.input_channels, 
            num_residual_blocks=args.resnet_blocks, 
            num_filters=args.resnet_filters
        )
    elif args.model == 'value':
        model = ValueNetwork(
            input_channels=args.input_channels, 
            num_residual_blocks=args.resnet_blocks, 
            num_filters=args.resnet_filters
        )

    else:
        raise ValueError(f"不支持的模型类型: {args.model}")
    
    return model.to(device)
