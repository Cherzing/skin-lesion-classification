"""
HAM10000 数据集预处理脚本。

运行示例：
    python scripts/prepare_dataset.py --source-root /path/to/HAM10000 --output-root data/
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
DEFAULT_SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def parse_args():
    parser = argparse.ArgumentParser(description="划分 HAM10000 数据集并生成去毛发图、差分图和 manifest 文件。")
    parser.add_argument("--source-root", type=Path, required=True, help="原始 HAM10000 分类目录。")
    parser.add_argument("--output-root", type=Path, required=True, help="预处理结果输出目录。")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO, help="训练集比例。")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help="验证集比例。")
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO, help="测试集比例。")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子，保证划分可复现。")
    return parser.parse_args()


def read_image_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image_unicode(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"图像编码失败: {path}")
    encoded.tofile(str(path))


def dullrazor_remove_hair(image):
    """DullRazor 去毛发算法。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    blurred = cv2.GaussianBlur(blackhat, (3, 3), cv2.BORDER_DEFAULT)
    _, mask = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(image, mask, 6, cv2.INPAINT_TELEA)


def compute_difference(original, cleaned):
    return cv2.absdiff(original, cleaned)


def gather_images(source_root):
    class_images = defaultdict(list)
    for cls in CLASS_NAMES:
        cls_dir = source_root / cls
        if not cls_dir.is_dir():
            print(f"  警告：类别目录不存在，已跳过: {cls_dir}")
            continue
        for file_path in sorted(cls_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                class_images[cls].append(file_path)
    return class_images


def stratified_split(class_images, train_ratio, val_ratio, test_ratio):
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(f"train/val/test 比例之和必须为 1.0，当前为 {ratio_sum:.4f}")

    splits = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}
    stats = {}
    for cls in CLASS_NAMES:
        images = sorted(class_images.get(cls, []))
        random.shuffle(images)
        n = len(images)
        if n == 0:
            stats[cls] = {"train": 0, "val": 0, "test": 0, "total": 0}
            continue

        n_train = max(1, round(n * train_ratio))
        n_val = max(1, round(n * val_ratio)) if n >= 3 else max(0, n - n_train)
        n_test = n - n_train - n_val
        if n >= 3 and n_test < 1:
            n_val = max(1, n - n_train - 1)
            n_test = n - n_train - n_val

        splits["train"][cls] = images[:n_train]
        splits["val"][cls] = images[n_train:n_train + n_val]
        splits["test"][cls] = images[n_train + n_val:]
        stats[cls] = {"train": len(splits["train"][cls]), "val": len(splits["val"][cls]), "test": len(splits["test"][cls]), "total": n}
    return splits, stats


def process_dataset(splits, raw_root, hair_root, diff_root, manifest_dir):
    records = []
    total = sum(len(files) for split in splits.values() for files in split.values())
    for split_name in ["train", "val", "test"]:
        for cls in CLASS_NAMES:
            (raw_root / split_name / cls).mkdir(parents=True, exist_ok=True)
            (hair_root / split_name / cls).mkdir(parents=True, exist_ok=True)
            (diff_root / split_name / cls).mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for split_name in ["train", "val", "test"]:
        for cls in CLASS_NAMES:
            for src_path in splits[split_name][cls]:
                file_name = src_path.name
                rel_path = f"{split_name}/{cls}/{file_name}"
                image = read_image_unicode(src_path)
                if image is None:
                    print(f"  跳过：图像读取失败: {src_path}")
                    continue

                write_image_unicode(raw_root / rel_path, image)
                cleaned = dullrazor_remove_hair(image)
                write_image_unicode(hair_root / rel_path, cleaned)
                write_image_unicode(diff_root / rel_path, compute_difference(image, cleaned))

                records.append({
                    "split": split_name,
                    "class_name": cls,
                    "file_name": file_name,
                    "relative_path": rel_path,
                    "hair_removed": rel_path,
                    "difference": rel_path,
                })
                processed += 1
                if processed % 500 == 0:
                    print(f"  处理进度: {processed}/{total}")
    return records


def write_manifest(records, manifest_path, summary_path):
    fieldnames = ["split", "class_name", "file_name", "relative_path", "hair_removed", "difference"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    summary = {"total_images": len(records), "splits": {}}
    for record in records:
        split_name = record["split"]
        class_name = record["class_name"]
        summary["splits"].setdefault(split_name, {})
        summary["splits"][split_name][class_name] = summary["splits"][split_name].get(class_name, 0) + 1
    for split_name in summary["splits"]:
        summary["splits"][split_name]["_total"] = sum(summary["splits"][split_name].values())
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    raw_root = output_root / "raw"
    hair_root = output_root / "hair_removed"
    diff_root = output_root / "difference"
    manifest_dir = output_root / "manifests"

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("HAM10000 数据集预处理")
    print("=" * 60)
    print(f"源数据目录: {source_root}")
    print(f"输出目录:   {output_root}")
    if not source_root.is_dir():
        raise FileNotFoundError(f"未找到源数据目录: {source_root}")

    print("\n[1/4] 收集各类别图像 ...")
    class_images = gather_images(source_root)
    print(f"  共找到 {sum(len(paths) for paths in class_images.values())} 张图像")
    for cls in CLASS_NAMES:
        print(f"    {cls}: {len(class_images[cls])}")

    print("\n[2/4] 分层划分数据集 ...")
    splits, stats = stratified_split(class_images, args.train_ratio, args.val_ratio, args.test_ratio)
    for cls in CLASS_NAMES:
        row = stats[cls]
        print(f"    {cls}: train={row['train']}, val={row['val']}, test={row['test']}")

    print("\n[3/4] 生成 raw、hair_removed 和 difference 图像 ...")
    records = process_dataset(splits, raw_root, hair_root, diff_root, manifest_dir)
    print(f"  完成，共生成 {len(records)} 条记录。")

    print("\n[4/4] 写出 manifest 与统计摘要 ...")
    summary = write_manifest(records, manifest_dir / "dataset_manifest.csv", manifest_dir / "dataset_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n完成！")


if __name__ == "__main__":
    main()
