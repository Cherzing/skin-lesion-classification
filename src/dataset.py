"""
数据集读取模块。
"""

import csv
from pathlib import Path

import albumentations as A # 图像增强库，提供丰富的变换函数和灵活的组合方式
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def build_transforms(image_size, is_train):
    """构建训练或验证/测试阶段的数据增强流程。"""
    normalize = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) # ImageNet 预训练模型常用的归一化参数
    if is_train:
        # 训练阶段使用轻量增强，提高模型对方向、颜色和局部扰动的鲁棒性。
        transforms = [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=30, p=0.4),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.3),
            normalize,
            ToTensorV2(),
        ]
    else:
        # 验证和测试阶段只做确定性处理，保证指标可复现。
        transforms = [A.Resize(image_size, image_size), normalize, ToTensorV2()]
    # 同步增强三路图像，保持空间位置一致，避免差分图和原图错位。
    return A.Compose(transforms, additional_targets={"hair_removed": "image", "difference": "image"})


class HairGuidedDataset(Dataset):
    """论文主模型使用的数据集类，返回三路图像、标签和样本元信息。"""

    def __init__(self, manifest_path, raw_root, hair_removed_root, diff_root, split, image_size=160, is_train=False):
        self.manifest_path = Path(manifest_path)
        self.raw_root = Path(raw_root)
        self.hair_removed_root = Path(hair_removed_root)
        self.diff_root = Path(diff_root)
        self.split = split
        self.transform = build_transforms(image_size=image_size, is_train=is_train)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"未找到数据清单文件: {self.manifest_path}")
        self.records = self._load_records()
        if not self.records:
            raise ValueError(f"manifest 中没有 split={self.split!r} 的样本: {self.manifest_path}")
        # 类别按名称排序，确保训练、验证、测试的 label 编号一致。
        self.class_names = sorted({record["class_name"] for record in self.records})
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

    def _load_records(self):
        """从 CSV manifest 中读取指定 split 的样本记录。"""
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return [row for row in rows if row["split"] == self.split]

    def __len__(self):
        """返回当前 split 的样本数量。"""
        return len(self.records)

    def _read_image(self, path):
        """读取 RGB 图像；np.fromfile 能兼容中文路径。"""
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            # raise 是 Python 中主动抛出异常的关键字。
            raise FileNotFoundError(f"无法读取图像字节: {path}")
        image = cv2.imdecode(data, cv2.IMREAD_COLOR) # 以 BGR 格式解码图像
        if image is None:
            # raise 是 Python 中主动抛出异常的关键字。
            raise FileNotFoundError(f"图像解码失败: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 转为 RGB 格式，符合 PyTorch 模型输入习惯

    def __getitem__(self, index):
        """按索引读取样本，并返回模型训练所需的数据字典。"""
        # 从 manifest 中获取样本记录，包含 split、class_name、file_name、relative_path、hair_removed 和 difference 字段
        record = self.records[index]
        image = self._read_image(self.raw_root / Path(record["relative_path"]))
        hair_removed = self._read_image(self.hair_removed_root / Path(record["hair_removed"]))
        difference = self._read_image(self.diff_root / Path(record["difference"]))

        # 同步增强三路图像，避免差分图和原图空间位置错位。
        transformed = self.transform(image=image, hair_removed=hair_removed, difference=difference)
        # 转为整数标签，适合分类任务的交叉熵损失函数使用
        label = torch.tensor(self.class_to_idx[record["class_name"]], dtype=torch.long)
        return {
            "image": transformed["image"],
            "hair_removed": transformed["hair_removed"],
            "difference": transformed["difference"],
            "label": label,
            "meta": {
                "split": record["split"],
                "class_name": record["class_name"],
                "file_name": record["file_name"],
                "relative_path": record["relative_path"],
            },
        }
