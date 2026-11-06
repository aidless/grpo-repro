"""绘图：GRPO 训练曲线（含滑动均值与零方差比例）+ 四模型评测对比。

只画真实跑出来的数据：所有曲线都从 experiments/logs/*.jsonl 读取，
没有跑过的 run 不会出现在图里。
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .common import load_config, project_path, read_jsonl  # noqa: E402

# 长程实验里需要画的 run：tag -> 图例
LONG_RUNS = {
    "long_base_s42": "full prompts (seed 42)",
    "long_base_s42_filt": "variance-filtered (seed 42)",
    "long_base_s7": "full prompts (seed 7)",
    "long_base_s7_filt": "variance-filtered (seed 7)",
}


def rolling(xs: list[float], window: int) -> list[float]:
    out, acc = [], []
    for x in xs:
        acc.append(x)
        if len(acc) > window:
            acc.pop(0)
        out.append(sum(acc) / len(acc))
    return out


def series(rows: list[dict], keys: list[str]) -> tuple[list[int], list[float], str | None]:
    for key in keys:
        pts = [(r["step"], r[key]) for r in rows if key in r]
        if pts:
            return [p[0] for p in pts], [p[1] for p in pts], key
    return [], [], None


def load_long() -> dict[str, list[dict]]:
    data = {}
    for tag in LONG_RUNS:
        path = project_path("experiments/logs", f"{tag}.jsonl")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            data[tag] = read_jsonl(path)
    return data


def plot_long_runs(window: int = 20) -> list[str]:
    """三面板：奖励（原始 + 滑动均值）/ 零方差比例 / 每步有梯度组数。"""
    data = load_long()
    if not data:
        print("[plot] 没有长程日志，跳过长程曲线")
        return []
    made = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    for tag, rows in data.items():
        xs, ys, _ = series(rows, ["rewards/correctness_reward/mean", "reward"])
        if xs:
            axes[0].plot(xs, ys, alpha=0.25, linewidth=0.8, color=f"C{list(data).index(tag)}")
            axes[0].plot(xs, rolling(ys, window), label=LONG_RUNS[tag],
                         color=f"C{list(data).index(tag)}", linewidth=1.8)
        xs, zs, _ = series(rows, ["frac_reward_zero_std"])
        if xs:
            axes[1].plot(xs, rolling(zs, window), label=LONG_RUNS[tag], color=f"C{list(data).index(tag)}")
        xs, ls, _ = series(rows, ["filter/learnable_groups"])
        if xs:
            axes[2].plot(xs, rolling(ls, window), label=LONG_RUNS[tag], color=f"C{list(data).index(tag)}")
        else:
            # 全量版没有 filter/* 字段：用 4 组 × (1 - 零方差比例) 复算可学习组数
            xs, zs, _ = series(rows, ["frac_reward_zero_std"])
            if xs:
                axes[2].plot(xs, rolling([(1 - z) * 4 for z in zs], window),
                             label=LONG_RUNS[tag], color=f"C{list(data).index(tag)}")

    axes[0].set_title("correctness reward (raw + rolling mean)")
    axes[1].set_title("frac_reward_zero_std (rolling mean)")
    axes[2].set_title("groups with non-zero advantage per step")
    for ax in axes:
        ax.set_xlabel("GRPO step")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    out = project_path("experiments/curves/long_reward_zero_std.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    made.append(out)

    # 单独的「过滤 vs 全量」对比图（同 seed）
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    pairs = [("long_base_s42", "long_base_s42_filt"), ("long_base_s7", "long_base_s7_filt")]
    for full, filt in pairs:
        for tag, style in ((full, "-"), (filt, "--")):
            if tag not in data:
                continue
            xs, ys, _ = series(data[tag], ["rewards/correctness_reward/mean", "reward"])
            if xs:
                axes[0].plot(xs, rolling(ys, window), style, label=LONG_RUNS[tag])
            xs, zs, _ = series(data[tag], ["frac_reward_zero_std"])
            if xs:
                axes[1].plot(xs, rolling(zs, window), style, label=LONG_RUNS[tag])
    axes[0].set_title("reward: full vs filtered")
    axes[1].set_title("zero-variance fraction: full vs filtered")
    for ax in axes:
        ax.set_xlabel("GRPO step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = project_path("experiments/curves/filter_compare.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    made.append(out)
    return made


def plot_reward_curves() -> list[str]:
    made = []
    runs = {"base": project_path("experiments/logs/grpo_base.jsonl"),
            "format": project_path("experiments/logs/grpo_format.jsonl")}
    data = {}
    for name, path in runs.items():
        if os.path.exists(path):
            data[name] = read_jsonl(path)
    if not data:
        print("[plot] 没有 GRPO 日志，跳过训练曲线")
        return made

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = [
        (["reward", "rewards/total/mean"], "mean total reward"),
        (["rewards/correctness_reward/mean", "correctness_reward/mean", "reward"], "correctness reward"),
        (["rewards/format_reward/mean", "format_reward/mean"], "format reward"),
    ]
    for ax, (keys, title) in zip(axes, panels):
        for name, rows in data.items():
            xs, ys, key = series(rows, keys)
            if xs:
                ax.plot(xs, ys, label=f"{name} ({key})", marker="o", markersize=3)
        ax.set_title(title)
        ax.set_xlabel("GRPO step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("GRPO 60-step pilot: reward curves (real logs)")
    fig.tight_layout()
    out = project_path("experiments/curves/grpo_reward.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    made.append(out)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, rows in data.items():
        xs, ys, key = series(rows, ["loss", "train_loss"])
        if xs:
            ax.plot(xs, ys, label=f"{name} ({key})")
    ax.set_title("GRPO pilot loss (real logs)")
    ax.set_xlabel("GRPO step")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = project_path("experiments/curves/grpo_loss.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    made.append(out)
    return made


def plot_eval_bars() -> list[str]:
    files = sorted(glob.glob(project_path("experiments/eval/*.json")))
    if not files:
        print("[plot] 没有评测结果，跳过对比图")
        return []
    # 只画正式评测（排除 verify_* 快速检查结果），并包含长程 run
    metrics = {}
    for path in files:
        name = os.path.basename(path)[:-5]
        if name.startswith("verify_") or name.startswith("smoke_"):
            continue
        with open(path, "r", encoding="utf-8") as f:
            metrics[name] = json.load(f)
    if not metrics:
        return []
    order = ["base", "sft", "grpo_base", "grpo_format", "long_base_s42",
             "long_base_s42_filt", "long_base_s7", "long_base_s7_filt"]
    names = [n for n in order if n in metrics] + [n for n in metrics if n not in order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    accs = [metrics[n]["accuracy"] * 100 for n in names]
    fmts = [metrics[n]["format_compliance_rate"] * 100 for n in names]
    axes[0].bar(names, accs, color="#4C72B0")
    axes[0].set_title(f"GSM8K accuracy (%, n={metrics[names[0]]['num_samples']})")
    axes[0].set_ylim(0, max(accs + [10]) * 1.25)
    for i, v in enumerate(accs):
        axes[0].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    axes[1].bar(names, fmts, color="#DD8452")
    axes[1].set_title("format compliance rate (%)")
    axes[1].set_ylim(0, 105)
    for i, v in enumerate(fmts):
        axes[1].text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=25, labelsize=8)
    fig.tight_layout()
    out = project_path("experiments/curves/eval_compare.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return [out]


if __name__ == "__main__":
    cfg = load_config(project_path("configs/full.yaml"))
    print(plot_reward_curves())
    print(plot_long_runs())
    print(plot_eval_bars())
