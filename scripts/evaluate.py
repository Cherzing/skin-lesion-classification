"""
模型测试与指标生成入口。

运行示例：
    python scripts/evaluate.py --config configs/main_experiment.json

若不传 `--checkpoint`，默认读取配置中 output.checkpoints_dir/best_model.pth。
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, precision_recall_fscore_support, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import HairGuidedDataset
from src.model import HairSuppressionDualStreamMetricNet

PATH_FIELDS = {
    "data": ["manifest_path", "raw_root", "hair_removed_root", "diff_root"],
    "output": ["checkpoints_dir", "logs_dir", "results_dir", "figures_dir", "training_report_path", "test_report_path"],
    "model": ["backbone_init_checkpoint"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="在测试集上评估毛发抑制双流模型。")
    parser.add_argument("--config", type=Path, required=True, help="实验配置 JSON 文件路径。")
    parser.add_argument("--checkpoint", type=Path, default=None, help="可选：指定待评估 checkpoint。")
    parser.add_argument("--limit-test-batches", type=int, default=None, help="限制测试 batch 数，用于快速检查。")
    return parser.parse_args()


def resolve_path(value, base_dir):
    if value in (None, ""):
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def resolve_config_paths(config, config_path):
    resolved = copy.deepcopy(config)
    base_dir = Path(config_path).resolve().parent
    for section, fields in PATH_FIELDS.items():
        if section not in resolved:
            continue
        for field in fields:
            if field in resolved[section]:
                resolved[section][field] = resolve_path(resolved[section][field], base_dir)
    return resolved


def load_config(path):
    config_path = Path(path).resolve()
    return resolve_config_paths(json.loads(config_path.read_text(encoding="utf-8")), config_path)


def build_test_loader(config):
    data_cfg = config["data"]
    dataset = HairGuidedDataset(data_cfg["manifest_path"], data_cfg["raw_root"], data_cfg["hair_removed_root"], data_cfg["diff_root"], "test", config["model"]["image_size"], False)
    loader = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=False, num_workers=config["train"]["num_workers"], pin_memory=torch.cuda.is_available())
    return dataset, loader


def save_confusion_matrix(matrix, class_names, output_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200); plt.close()


def save_roc_curve(labels, probabilities, class_names, output_path):
    labels_one_hot = label_binarize(labels, classes=list(range(len(class_names))))
    probabilities = np.asarray(probabilities)
    fpr = {}; tpr = {}; roc_auc = {}
    for i, class_name in enumerate(class_names):
        fpr[i], tpr[i], _ = roc_curve(labels_one_hot[:, i], probabilities[:, i])
        roc_auc[class_name] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(class_names))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(class_names)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(class_names)
    macro_auc = auc(all_fpr, mean_tpr)

    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(class_names):
        plt.plot(fpr[i], tpr[i], linewidth=1.5, label=f"{class_name} (AUC={roc_auc[class_name]:.3f})")
    plt.plot(all_fpr, mean_tpr, linestyle="--", linewidth=2.5, label=f"macro-average (AUC={macro_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle=":", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curves")
    plt.legend(loc="lower right", fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200); plt.close()
    return {"macro_auc": macro_auc, "per_class_auc": roc_auc}


def compute_metrics(labels, preds, class_names):
    label_ids = list(range(len(class_names)))
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(labels, preds, labels=label_ids, average="macro", zero_division=0)
    metrics = {"accuracy": accuracy_score(labels, preds), "macro_precision": macro_precision, "macro_recall": macro_recall, "macro_f1": macro_f1}
    class_report = classification_report(labels, preds, labels=label_ids, target_names=class_names, digits=4, zero_division=0)
    matrix = confusion_matrix(labels, preds, labels=label_ids)
    return metrics, class_report, matrix


def write_markdown_report(report_path, metrics, class_report, confusion_path, roc_path, checkpoint_path):
    content = f"""# 主模型测试结果

## 测试配置

- checkpoint：`{checkpoint_path}`
- 混淆矩阵：`{confusion_path}`
- ROC 曲线：`{roc_path}`

## 总体指标

| 指标 | 数值 |
| --- | --- |
| Accuracy | {metrics['accuracy']:.4f} |
| Macro Precision | {metrics['macro_precision']:.4f} |
| Macro Recall | {metrics['macro_recall']:.4f} |
| Macro F1 | {metrics['macro_f1']:.4f} |
| Macro AUC | {metrics['macro_auc']:.4f} |

## 分类别指标

```text
{class_report}
```
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, loader = build_test_loader(config)
    checkpoint_path = Path(args.checkpoint or Path(config["output"]["checkpoints_dir"]) / "best_model.pth").resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = HairSuppressionDualStreamMetricNet(
        num_classes=len(dataset.class_names),
        embedding_dim=config["model"]["embedding_dim"],
        pretrained=False,
        use_hair_removed_input=config["model"].get("use_hair_removed_input", True),
        use_dual_stream=config["model"].get("use_dual_stream", True),
        use_cbam=config["model"].get("use_cbam", True),
        use_metric_head=config["model"].get("use_metric_head", True),
        use_difference_guidance=config["model"].get("use_difference_guidance", True),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_labels = []; all_preds = []; all_probs = []
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            if args.limit_test_batches is not None and step > args.limit_test_batches:
                break
            image = batch["image"].to(device); hair_removed = batch["hair_removed"].to(device); difference = batch["difference"].to(device); labels = batch["label"].to(device)
            outputs = model(image, hair_removed, difference)
            preds = outputs["logits"].argmax(dim=1)
            probs = torch.softmax(outputs["logits"], dim=1)
            all_labels.extend(labels.cpu().numpy().tolist()); all_preds.extend(preds.cpu().numpy().tolist()); all_probs.extend(probs.cpu().numpy().tolist())

    metrics, class_report, matrix = compute_metrics(all_labels, all_preds, dataset.class_names)
    figures_dir = Path(config["output"]["figures_dir"]); results_dir = Path(config["output"]["results_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True); results_dir.mkdir(parents=True, exist_ok=True)
    confusion_path = figures_dir / "confusion_matrix.png"; roc_path = figures_dir / "roc_curve.png"
    save_confusion_matrix(matrix, dataset.class_names, confusion_path)
    metrics.update(save_roc_curve(all_labels, all_probs, dataset.class_names, roc_path))
    (results_dir / "test_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(Path(config["output"]["test_report_path"]), metrics, class_report, confusion_path, roc_path, checkpoint_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
