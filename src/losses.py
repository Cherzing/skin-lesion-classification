"""
损失函数模块。

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """多分类 Focal Loss。"""

    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """根据 logits 和真实标签计算 Focal Loss。"""
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce) # pt 是模型对正确类别的预测概率，ce 是对应的交叉熵损失。Focal Loss 通过 (1 - pt) ** gamma 调整样本权重，难分类样本（pt 小）权重更大，易分类样本（pt 大）权重更小。
        loss = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            # alpha 为类别权重，按每个样本的真实类别取出对应权重。
            alpha = self.alpha.to(logits.device)[targets]
            loss = alpha * loss
        return loss.mean()


class CombinedClassificationLoss(nn.Module):
    """交叉熵与 Focal Loss 的加权组合。"""

    def __init__(self, alpha=None, gamma=2.0, ce_weight=0.4, focal_weight=0.6):
        super().__init__()
        self.alpha = alpha
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.focal = FocalLoss(alpha=alpha, gamma=gamma) # 内部使用 FocalLoss 计算焦点损失，支持类别权重 alpha 和聚焦参数 gamma。

    def forward(self, logits, targets):
        """返回组合分类损失。"""
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha) # 计算交叉熵损失，支持类别权重 alpha。交叉熵损失关注整体分类性能，Focal Loss 关注难分类样本，两者结合可以兼顾整体性能和对难样本的提升。
        focal_loss = self.focal(logits, targets) 
        return self.ce_weight * ce_loss + self.focal_weight * focal_loss


def compute_class_weights(records, class_to_idx):
    """根据训练集类别频数计算反频率权重。"""
    counts = torch.zeros(len(class_to_idx), dtype=torch.float32)
    for record in records:
        counts[class_to_idx[record["class_name"]]] += 1
    # 防止极端情况下除零；正常训练集中每个类别都应有样本。
    counts = torch.clamp(counts, min=1.0)
    return counts.sum() / (len(counts) * counts)
