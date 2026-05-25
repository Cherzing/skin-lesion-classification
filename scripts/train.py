"""
主模型训练入口。

运行示例：
    python scripts/train.py --config configs/main_experiment.json

配置文件中的相对路径会按"配置文件所在目录"解析，避免依赖当前工作目录。
"""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# 将项目根目录加入 sys.path，确保 src 包可被导入。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import HairGuidedDataset
from src.losses import CombinedClassificationLoss, compute_class_weights
from src.model import HairSuppressionDualStreamMetricNet

PATH_FIELDS = {
    "data": ["manifest_path", "raw_root", "hair_removed_root", "diff_root"],
    "output": ["checkpoints_dir", "logs_dir", "results_dir", "figures_dir", "training_report_path", "test_report_path"],
    "model": ["backbone_init_checkpoint"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="训练毛发抑制双流度量学习主模型。")
    parser.add_argument("--config", type=Path, required=True, help="训练配置 JSON 文件路径。")
    parser.add_argument("--max-epochs", type=int, default=None, help="临时覆盖训练轮数，用于快速测试。")
    parser.add_argument("--limit-train-batches", type=int, default=None, help="限制每轮训练 batch 数。")
    parser.add_argument("--limit-val-batches", type=int, default=None, help="限制每轮验证 batch 数。")
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
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return resolve_config_paths(config, config_path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    return {
        "image": batch["image"].to(device),
        "hair_removed": batch["hair_removed"].to(device),
        "difference": batch["difference"].to(device),
        "label": batch["label"].to(device),
        "meta": batch["meta"],
    }


def run_epoch(model, loader, criterion, optimizer, device, limit_batches=None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for step, batch in enumerate(loader, start=1):
        if limit_batches is not None and step > limit_batches:
            break
        batch = to_device(batch, device)
        optimizer.zero_grad()
        outputs = model(batch["image"], batch["hair_removed"], batch["difference"], batch["label"])
        loss = criterion(outputs["logits"], batch["label"])
        loss.backward()
        optimizer.step()

        predictions = outputs["logits"].argmax(dim=1)
        total_loss += loss.item() * batch["label"].size(0)
        total_correct += (predictions == batch["label"]).sum().item()
        total_samples += batch["label"].size(0)
    return {"loss": total_loss / max(total_samples, 1), "accuracy": total_correct / max(total_samples, 1)}


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device, limit_batches=None):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels = []
    all_preds = []
    for step, batch in enumerate(loader, start=1):
        if limit_batches is not None and step > limit_batches:
            break
        batch = to_device(batch, device)
        outputs = model(batch["image"], batch["hair_removed"], batch["difference"], batch["label"])
        loss = criterion(outputs["logits"], batch["label"])
        predictions = outputs["logits"].argmax(dim=1)
        total_loss += loss.item() * batch["label"].size(0)
        total_correct += (predictions == batch["label"]).sum().item()
        total_samples += batch["label"].size(0)
        all_labels.extend(batch["label"].cpu().tolist())
        all_preds.extend(predictions.cpu().tolist())
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def build_dataloaders(config):
    data_cfg = config["data"]
    train_dataset = HairGuidedDataset(data_cfg["manifest_path"], data_cfg["raw_root"], data_cfg["hair_removed_root"], data_cfg["diff_root"], "train", config["model"]["image_size"], True)
    val_dataset = HairGuidedDataset(data_cfg["manifest_path"], data_cfg["raw_root"], data_cfg["hair_removed_root"], data_cfg["diff_root"], "val", config["model"]["image_size"], False)
    train_loader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True, num_workers=config["train"]["num_workers"], pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=config["train"]["batch_size"], shuffle=False, num_workers=config["train"]["num_workers"], pin_memory=torch.cuda.is_available())
    return train_dataset, val_dataset, train_loader, val_loader


def write_training_markdown(output_path, metrics, config_path, best_checkpoint):
    content = f"""# 主模型训练记录

## 训练环境

- 配置文件：`{config_path}`
- 最优模型：`{best_checkpoint}`
- 训练曲线：`{output_path.parent / 'figures' / 'training_curves.png'}`

## 训练摘要

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val Macro F1 |
| --- | --- | --- | --- | --- | --- |
"""
    for row in metrics:
        content += f"| {row['epoch']} | {row['train_loss']:.4f} | {row['train_acc']:.4f} | {row['val_loss']:.4f} | {row['val_acc']:.4f} | {row['val_macro_f1']:.4f} |\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def save_training_curves(metrics, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in metrics]
    train_loss = [row["train_loss"] for row in metrics]
    val_loss = [row["val_loss"] for row in metrics]
    train_acc = [row["train_acc"] for row in metrics]
    val_acc = [row["val_acc"] for row in metrics]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker="o", label="Train Loss")
    plt.plot(epochs, val_loss, marker="s", label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training and Validation Loss")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout(); plt.savefig(output_dir / "loss_curve.png", dpi=200); plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_acc, marker="o", label="Train Accuracy")
    plt.plot(epochs, val_acc, marker="s", label="Val Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Training and Validation Accuracy")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout(); plt.savefig(output_dir / "accuracy_curve.png", dpi=200); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, train_loss, marker="o", label="Train Loss")
    axes[0].plot(epochs, val_loss, marker="s", label="Val Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Loss Curve"); axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].plot(epochs, train_acc, marker="o", label="Train Accuracy")
    axes[1].plot(epochs, val_acc, marker="s", label="Val Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].set_title("Accuracy Curve"); axes[1].grid(alpha=0.3); axes[1].legend()
    fig.tight_layout(); fig.savefig(output_dir / "training_curves.png", dpi=200); plt.close(fig)


def main():
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, _, train_loader, val_loader = build_dataloaders(config)

    model = HairSuppressionDualStreamMetricNet(
        num_classes=len(train_dataset.class_names),
        embedding_dim=config["model"]["embedding_dim"],
        pretrained=config["model"]["pretrained"],
        use_hair_removed_input=config["model"].get("use_hair_removed_input", True),
        use_dual_stream=config["model"].get("use_dual_stream", True),
        use_cbam=config["model"].get("use_cbam", True),
        use_metric_head=config["model"].get("use_metric_head", True),
        use_difference_guidance=config["model"].get("use_difference_guidance", True),
    ).to(device)

    warm_start_info = None
    backbone_init_checkpoint = config["model"].get("backbone_init_checkpoint")
    if backbone_init_checkpoint:
        warm_start_info = model.load_backbone_from_checkpoint(backbone_init_checkpoint)

    class_weights = compute_class_weights(train_dataset.records, train_dataset.class_to_idx).to(device)
    criterion = CombinedClassificationLoss(alpha=class_weights)
    optimizer = AdamW(model.parameters(), lr=config["train"]["learning_rate"], weight_decay=config["train"]["weight_decay"])
    epochs = args.max_epochs or config["train"]["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    patience = config["train"].get("early_stopping_patience")

    output_cfg = config["output"]
    checkpoints_dir = Path(output_cfg["checkpoints_dir"]); logs_dir = Path(output_cfg["logs_dir"]); results_dir = Path(output_cfg["results_dir"])
    checkpoints_dir.mkdir(parents=True, exist_ok=True); logs_dir.mkdir(parents=True, exist_ok=True); results_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = checkpoints_dir / "best_model.pth"
    best_val_acc = -1.0; best_val_macro_f1 = -1.0; best_epoch = 0; epochs_without_improvement = 0; metrics_log = []

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, args.limit_train_batches)
        val_metrics = evaluate_epoch(model, val_loader, criterion, device, args.limit_val_batches)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_metrics["loss"], "train_acc": train_metrics["accuracy"], "val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"], "val_macro_f1": val_metrics["macro_f1"]}
        metrics_log.append(row)
        print(f"[Epoch {epoch:3d}] train_loss={row['train_loss']:.4f} train_acc={row['train_acc']:.4f} val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.4f} val_f1={row['val_macro_f1']:.4f}", flush=True)
        (logs_dir / "train_metrics.json").write_text(json.dumps(metrics_log, ensure_ascii=False, indent=2), encoding="utf-8")

        if row["val_acc"] > best_val_acc:
            best_val_acc = row["val_acc"]; best_val_macro_f1 = row["val_macro_f1"]; best_epoch = epoch; epochs_without_improvement = 0
            torch.save({"model_state_dict": model.state_dict(), "class_names": train_dataset.class_names, "config": config, "best_val_acc": best_val_acc, "best_val_macro_f1": best_val_macro_f1}, best_checkpoint)
        else:
            epochs_without_improvement += 1
        if patience is not None and epochs_without_improvement >= patience:
            break

    save_training_curves(metrics_log, Path(output_cfg["figures_dir"]))
    write_training_markdown(Path(output_cfg["training_report_path"]), metrics_log, args.config, best_checkpoint)
    summary = {"device": str(device), "best_val_acc": best_val_acc, "best_val_macro_f1": best_val_macro_f1, "best_epoch": best_epoch, "epochs": epochs, "executed_epochs": len(metrics_log), "best_checkpoint": str(best_checkpoint), "warm_start_info": warm_start_info}
    (results_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
