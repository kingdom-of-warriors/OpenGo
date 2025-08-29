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


class PolicyNetwork_transformer(nn.Module):
    """
    基于Transformer架构的围棋策略网络。
    """
    def __init__(self, 
                 input_channels: int = 19, 
                 board_size: int = 19,
                 d_model: int = 128,          # 嵌入向量的维度
                 n_head: int = 8,             # 多头注意力的头数
                 num_encoder_layers: int = 6, # Transformer编码器的层数
                 dim_feedforward: int = 512,  # 前馈网络中间层的维度
                 dropout: float = 0.1):
        
        super(PolicyNetwork_transformer, self).__init__()
        
        self.board_size = board_size
        self.num_tokens = board_size * board_size # 19 * 19 = 361
        
        # 线性投射层，将19维的输入特征嵌入到d_model维
        self.embedding = nn.Linear(input_channels, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.num_tokens, d_model))
        self.pre_ln = nn.LayerNorm(d_model) # before transformer
        self.post_ln = nn.LayerNorm(d_model) # after transformer
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        # 将多个编码器层堆叠起来
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # --- 3. 尾部：策略头 ---
        # 将经过Transformer处理的d_model维特征向量，映射回一个代表logit的标量
        self.policy_head = nn.Linear(d_model, 1)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        定义并应用权重初始化方案。
        """
        # 如果是线性层
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # 如果是层归一化层
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x 的形状: [Batch, Channels, Height, Width] -> [B, 19, 19, 19]
        bs = x.shape[0]

        x = x.view(bs, x.size(1), -1).permute(0, 2, 1)
        # c. 线性嵌入
        x = self.embedding(x)
        x = x + self.positional_encoding

        # 归一化
        x = self.pre_ln(x)
        x = self.dropout(x)

        # [B, 361, d_model] -> [B, 361, d_model]
        x = self.transformer_encoder(x)
        
        # [B, 361, d_model] -> [B, 361, 1] -> [B, 361]
        x = self.post_ln(x)
        policy_logits = self.policy_head(x).squeeze(-1)
        
        return policy_logits
    

def create_model(args, device):
    """根据参数创建模型"""
    if args.model == 'resnet':
        model = PolicyNetwork_resnet(
            input_channels=args.input_channels, 
            num_residual_blocks=args.resnet_blocks, 
            num_filters=args.resnet_filters
        )
    elif args.model == 'transformer':
        model = PolicyNetwork_transformer(
            input_channels=19,
            board_size=19,
            d_model=args.d_model,
            n_head=args.n_head,
            num_encoder_layers=args.transformer_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"不支持的模型类型: {args.model}")
    
    return model.to(device)
