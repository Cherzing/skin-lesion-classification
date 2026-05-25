"""
主模型结构定义。

"""

import torch
import torch.nn as nn # 导入 PyTorch 的神经网络模块，包含各种层、损失函数和工具。
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


def extract_prefixed_state_dict(state_dict, prefix):
    """从checkpoint中提取指定前缀的权重，例如只提取backbone参数。"""
    return {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}


class ChannelAttention(nn.Module):
    """CBAM 的通道注意力分支，用于判断哪些特征通道更重要。"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8) # reduction 过大可能导致信息瓶颈，过小可能参数过多，经验值通常在 8-64 之间。
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # 全局平均池化得到通道描述符，大小为 (batch_size, channels, 1, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 共享 MLP，输入是通道描述符，输出是每个通道的权重。两层全连接，隐藏层使用 ReLU 激活，输出层不使用激活函数，最后通过 sigmoid 输出 0-1 之间的权重。
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False), # 1x1 卷积实现全连接，参数更少且适合卷积特征图输入
            nn.ReLU(inplace=True), # 激活函数引入非线性，inplace=True 节省内存
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False), # 输出通道数恢复为输入通道数，得到每个通道的权重
        )
        self.sigmoid = nn.Sigmoid() # 输出通道权重，范围在 0-1 之间

    def forward(self, x):
        # 平均池化关注整体响应，最大池化关注最强响应，两者相加得到通道权重。
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """CBAM的空间注意力分支，用于判断图像中哪些位置更重要。"""

    def __init__(self, kernel_size=7):
        super().__init__()
        # 保持输入输出尺寸不变，常用奇数核大小如 3、5、7
        padding = kernel_size // 2
        # 输入是两通道的平均图和最大图，输出是单通道的空间权重图。
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        # 输出空间权重，范围在0-1之间，乘以特征图实现空间位置的动态调整。
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿通道维度压缩为平均图和最大图，再学习空间位置权重。
        avg_out = torch.mean(x, dim=1, keepdim=True) # 平均图关注整体响应，最大图关注局部强响应，两者拼接得到空间权重输入。
        max_out, _ = torch.max(x, dim=1, keepdim=True)# 最大图关注局部强响应，两者拼接得到空间权重输入。
        attention = torch.cat([avg_out, max_out], dim=1) # 拼接得到空间权重输入，大小为 (batch_size, 2, height, width)
        return self.sigmoid(self.conv(attention))


class CBAM(nn.Module):
    """卷积注意力模块：先做通道注意力，再做空间注意力。"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class ResidualAdapter(nn.Module):
    """轻量残差适配器，让原图流和去毛发流拥有少量分支特异参数。"""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False) # 1x1 卷积实现通道变换，参数较少且适合卷积特征图输入
        self.bn = nn.BatchNorm2d(channels)# 批归一化稳定训练，适应不同分布的特征图输入
        
        # 零初始化使适配器初始近似恒等映射，降低训练初期扰动。
        nn.init.zeros_(self.conv.weight)
        nn.init.ones_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

    def forward(self, x):
        """输入特征图 x 经过适配器变换后与原图相加，形成残差连接。"""
        return x + self.bn(self.conv(x))


class ArcMarginHead(nn.Module):
    """ArcFace 风格的角度间隔分类头，用于增强类别间可分性。"""

    def __init__(self, in_features, num_classes, scale=30.0, margin=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, labels=None):
        # 特征和类别权重都做 L2 归一化，线性层输出即余弦相似度。
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        if labels is None:
            return cosine * self.scale

        # 训练时仅对真实类别增加角度间隔margin。
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cosine)
        target_logits = torch.cos(theta + self.margin)
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        logits = one_hot * target_logits + (1.0 - one_hot) * cosine
        return logits * self.scale


class DifferenceGuidance(nn.Module):
    """差分图引导模块，输出双流融合权重和空间引导图。"""

    def __init__(self, feature_channels):
        super().__init__()
        # 输入是差分图，经过三层卷积提取引导特征，再分别输出融合权重和空间引导图。
        # 融合权重通过 softmax 归一化为两路特征的动态融合系数，空间引导图通过 sigmoid 输出 0-1 之间的权重，用于调整融合特征的空间响应。
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.weight_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )
        self.spatial_head = nn.Sequential(nn.Conv2d(128, feature_channels, kernel_size=1, bias=False), nn.Sigmoid())
        # 初始时偏向原图流，空间引导接近中性，避免训练一开始过度扰动特征。
        nn.init.zeros_(self.weight_head[-1].weight) # 输出层权重初始化为零，使初始输出为零向量，softmax 后即 0.5/0.5 融合两路特征。
        self.weight_head[-1].bias.data = torch.tensor([1.0, 0.0], dtype=self.weight_head[-1].bias.dtype)
        nn.init.zeros_(self.spatial_head[0].weight) # 空间引导卷积权重初始化为零，使初始输出为 0.5 的中性引导图，避免训练初期过度扰动特征。

    def forward(self, difference, target_size):
        """输入差分图，输出双流融合权重和空间引导图。"""
        diff_feature = self.encoder(difference)
        fusion_weights = torch.softmax(self.weight_head(diff_feature), dim=1)
        spatial = self.spatial_head(diff_feature)
        spatial = F.interpolate(spatial, size=target_size, mode="bilinear", align_corners=False)
        return fusion_weights, spatial


class HairSuppressionDualStreamMetricNet(nn.Module):
    """毛发抑制双流度量学习网络。"""

    def __init__(
        self,
        num_classes=7,
        embedding_dim=256,
        pretrained=False,
        use_hair_removed_input=True,
        use_dual_stream=True,
        use_cbam=True,
        use_metric_head=True,
        use_difference_guidance=True,
    ):
        super().__init__()
        self.backbone = EfficientNet.from_name("efficientnet-b0")
        self.feature_channels = 1280
        self.use_hair_removed_input = use_hair_removed_input
        self.use_dual_stream = use_dual_stream
        self.use_cbam = use_cbam
        self.use_metric_head = use_metric_head
        self.use_difference_guidance = use_difference_guidance

        self.raw_adapter = ResidualAdapter(self.feature_channels)
        self.clean_adapter = ResidualAdapter(self.feature_channels)
        self.guidance = DifferenceGuidance(self.feature_channels)
        self.cbam = CBAM(self.feature_channels)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Linear(self.feature_channels, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.metric_head = ArcMarginHead(embedding_dim, num_classes)

        if pretrained:
            # 当前项目不联网下载权重；如需迁移学习，请在配置中填写本地 backbone_init_checkpoint。
            raise ValueError("当前离线实验环境不支持 pretrained=True，请改用本地 checkpoint。")

    def extract_feature_map(self, image):
        """调用 EfficientNet-B0 提取最后一层卷积特征图。"""
        return self.backbone.extract_features(image)

    def load_backbone_from_checkpoint(self, checkpoint_path):
        """从历史 checkpoint 中加载 backbone 参数，用于可选 warm start。"""
        # 加载 checkpoint，map_location="cpu" 确保在没有 GPU 的环境下也能加载。
        checkpoint = torch.load(checkpoint_path, map_location="cpu") 
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        backbone_state = extract_prefixed_state_dict(checkpoint, "backbone.") # 提取backbone参数，假设checkpoint中backbone 参数以 "backbone." 为前缀。
        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
        return {"loaded_keys": len(backbone_state), "missing_keys": list(missing), "unexpected_keys": list(unexpected)}

    def forward(self, image, hair_removed, difference, labels=None):
        """前向传播，返回分类 logits、嵌入向量和融合信息。"""
        clean_input = hair_removed if self.use_hair_removed_input else image
        raw_feature = self.raw_adapter(self.extract_feature_map(image))
        clean_feature = self.clean_adapter(self.extract_feature_map(clean_input))

        batch_size = image.size(0)
        if self.use_dual_stream:
            if self.use_difference_guidance:
                fusion_weights, spatial_guidance = self.guidance(difference, raw_feature.shape[-2:])
            else:
                # 消融实验：关闭差分图引导时，固定 0.5/0.5 融合两路特征。
                fusion_weights = torch.full((batch_size, 2), 0.5, dtype=raw_feature.dtype, device=raw_feature.device)
                spatial_guidance = torch.zeros_like(raw_feature)
            raw_weight = fusion_weights[:, 0].view(-1, 1, 1, 1)
            clean_weight = fusion_weights[:, 1].view(-1, 1, 1, 1)
            fused = raw_weight * raw_feature + clean_weight * clean_feature
            if self.use_difference_guidance:
                fused = fused * (1.0 + (spatial_guidance - 0.5))
        else:
            # 消融实验：关闭双流结构时，只保留原图流。
            fusion_weights = torch.tensor([[1.0, 0.0]], dtype=raw_feature.dtype, device=raw_feature.device).repeat(batch_size, 1)
            fused = raw_feature

        if self.use_cbam:
            fused = self.cbam(fused)
        pooled = self.global_pool(fused).flatten(1)
        embeddings = self.embedding(pooled)
        if self.use_metric_head:
            # 训练时使用 ArcMarginHead 计算带 margin 的 logits，推理时直接输出余弦相似度。
            logits = self.metric_head(embeddings, labels)
        else:
            # 消融实验：关闭度量约束时，退化为普通归一化线性分类。
            logits = F.linear(embeddings, F.normalize(self.metric_head.weight))
        return {"logits": logits, "embeddings": embeddings, "fusion_weights": fusion_weights, "feature_map": fused}
