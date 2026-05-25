"""
消融实验批量运行入口。

脚本会依次训练并测试六个消融变体，最后汇总测试指标。

运行示例：
    python scripts/run_ablations.py --variants full_model no_cbam --max-epochs 5
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
ABLATION_DIR = PROJECT_ROOT / "experiments" / "ablation"
TRAIN_SCRIPT = SCRIPTS_DIR / "train.py"
EVAL_SCRIPT = SCRIPTS_DIR / "evaluate.py"

VARIANTS = ["full_model", "no_hair_guidance", "no_dual_stream", "no_cbam", "no_metric_constraint", "no_difference_guidance"]


def parse_args():
    parser = argparse.ArgumentParser(description="按顺序运行全部或部分消融实验。")
    parser.add_argument("--variants", nargs="*", default=VARIANTS, help="要运行的变体名称，默认运行全部。")
    parser.add_argument("--max-epochs", type=int, default=None, help="临时覆盖训练轮数。")
    parser.add_argument("--limit-train-batches", type=int, default=None, help="限制训练 batch 数。")
    parser.add_argument("--limit-val-batches", type=int, default=None, help="限制验证 batch 数。")
    parser.add_argument("--limit-test-batches", type=int, default=None, help="限制测试 batch 数。")
    return parser.parse_args()


def run_cmd(cmd, desc):
    print(f"\n{'=' * 60}")
    print(f"  {desc}")
    print(f"  CMD: {' '.join(map(str, cmd))}")
    print(f"{'=' * 60}", flush=True)
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  失败，返回码: {result.returncode}", flush=True)
        return False
    print("  完成", flush=True)
    return True


def append_optional_arg(cmd, name, value):
    if value is not None:
        cmd.extend([name, str(value)])


def main():
    args = parse_args()
    results = {}
    for name in args.variants:
        variant_dir = ABLATION_DIR / "configs" / name
        config_path = variant_dir / "config.json"
        if not config_path.is_file():
            print(f"跳过 {name}: 未找到配置文件 {config_path}", flush=True)
            continue

        train_cmd = [sys.executable, "-u", str(TRAIN_SCRIPT), "--config", str(config_path)]
        append_optional_arg(train_cmd, "--max-epochs", args.max_epochs)
        append_optional_arg(train_cmd, "--limit-train-batches", args.limit_train_batches)
        append_optional_arg(train_cmd, "--limit-val-batches", args.limit_val_batches)
        if not run_cmd(train_cmd, f"训练: {name}"):
            print(f"{name} 训练失败，跳过测试", flush=True)
            continue

        checkpoint = variant_dir / "checkpoints" / "best_model.pth"
        eval_cmd = [sys.executable, "-u", str(EVAL_SCRIPT), "--config", str(config_path), "--checkpoint", str(checkpoint)]
        append_optional_arg(eval_cmd, "--limit-test-batches", args.limit_test_batches)
        if not run_cmd(eval_cmd, f"测试: {name}"):
            print(f"{name} 测试失败，跳过指标汇总", flush=True)
            continue

        test_metrics_path = variant_dir / "results" / "test_metrics.json"
        if test_metrics_path.exists():
            metrics = json.loads(test_metrics_path.read_text(encoding="utf-8"))
            results[name] = {
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "macro_auc": metrics.get("macro_auc", 0),
            }
            print(f"  {name}: Acc={metrics['accuracy']:.4f}, F1={metrics['macro_f1']:.4f}", flush=True)

    summary_path = ABLATION_DIR / "results" / "ablation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'=' * 60}")
    print("消融实验汇总:")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"已保存到 {summary_path}")


if __name__ == "__main__":
    main()
