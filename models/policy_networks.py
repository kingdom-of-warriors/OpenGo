import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    单个残差块的实现。
    输入 -> Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU -> 输出
    """
    def __init__(self, num_filters: int):
        super(ResidualBlock, self).__init__()
        # 定义残差块中的两个卷积层和批归一化层
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
    
class PolicyNetwork_resnet(nn.Module):
    """
    基于ResNet的策略网络。
    """
    def __init__(self, input_channels: int = 19, num_residual_blocks: int = 12, num_filters: int = 192):
        super(PolicyNetwork_resnet, self).__init__()
        
        # --- 1. 头部：初始卷积块 ---
        self.conv_head = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding='same')
        self.bn_head = nn.BatchNorm2d(num_filters)

        # --- 2. 身体：残差块堆叠 ---
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_residual_blocks)]
        )
        
        # --- 3. 尾部：策略头 ---
        self.conv_policy = nn.Conv2d(num_filters, 2, kernel_size=1, padding='same')
        self.bn_policy = nn.BatchNorm2d(2)
        
        self.fc_policy = nn.Linear(2 * 19 * 19, 19 * 19)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn_head(self.conv_head(x)))
        out = self.res_blocks(out)
        policy = F.relu(self.bn_policy(self.conv_policy(out)))
        
        # 压平 (Flatten) 以送入全连接层
        policy = torch.flatten(policy, start_dim=1)
        policy_logits = self.fc_policy(policy)

        return policy_logits

    def sample_rl(self, x: torch.Tensor):
        """
        利用8种对称性（旋转、翻转）来增强预测并采样。
        支持批处理输入 (N, C, H, W)。
        """
        n, c, h, w = x.shape
        
        # 定义8种对称变换
        transforms = {
            'identity': lambda t: t,
            'r90':      lambda t: torch.rot90(t, 1, [2, 3]),
            'r180':     lambda t: torch.rot90(t, 2, [2, 3]),
            'r270':     lambda t: torch.rot90(t, 3, [2, 3]),
            'f':        lambda t: torch.flip(t, [3]),
            'f_r90':    lambda t: torch.rot90(torch.flip(t, [3]), 1, [2, 3]),
            'f_r180':   lambda t: torch.rot90(torch.flip(t, [3]), 2, [2, 3]),
            'f_r270':   lambda t: torch.rot90(torch.flip(t, [3]), 3, [2, 3]),
        }
        symmetries = torch.stack([transform(x) for transform in transforms.values()]) # [8, N, C, H, W]
        batch_x = symmetries.view(-1, c, h, w) # [8*N, C, H, W]

        batch_logits = self.forward(batch_x)
        batch_probs = F.softmax(batch_logits, dim=1)
        probs_by_symmetry = batch_probs.view(8, n, -1) # [8, N, H*W]
        
        # 定义对应的8种逆变换
        inverse_transforms = {
            'identity': lambda p: p,
            'r90':      lambda p: torch.rot90(p, -1, [1, 2]),
            'r180':     lambda p: torch.rot90(p, -2, [1, 2]),
            'r270':     lambda p: torch.rot90(p, -3, [1, 2]),
            'f':        lambda p: torch.flip(p, [2]),
            'f_r90':    lambda p: torch.flip(torch.rot90(p, -1, [1, 2]), [2]),
            'f_r180':   lambda p: torch.flip(torch.rot90(p, -2, [1, 2]), [2]),
            'f_r270':   lambda p: torch.flip(torch.rot90(p, -3, [1, 2]), [2]),
        }

        untransformed_probs = []
        for i, inv_transform in enumerate(inverse_transforms.values()):
            prob_maps = probs_by_symmetry[i].view(n, h, w) # [N, H*W] -> [N, H, W]
            untransformed_maps = inv_transform(prob_maps)
            untransformed_probs.append(untransformed_maps.reshape(n, -1))
        avg_probs = torch.mean(torch.stack(untransformed_probs), dim=0)
            
        return avg_probs
    
    def sample_inference(self, x: torch.Tensor):
        with torch.no_grad(): return self.sample_rl(x)

def create_model(args, device):
    """根据参数创建模型"""
    if args.model == 'resnet':
        model = PolicyNetwork_resnet(
            input_channels=args.input_channels, 
            num_residual_blocks=args.resnet_blocks, 
            num_filters=args.resnet_filters
        )
    else:
        raise ValueError(f"不支持的模型类型: {args.model}")
    
    return model.to(device)
