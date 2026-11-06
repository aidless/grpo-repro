"""把「训练奖励」和「评测准确率」画在同一张图上（双 y 轴，x = 步数）。

数据来源：
- 训练奖励：experiments/logs/<tag>.jsonl（每步一条）
- 评测准确率：experiments/eval/<tag>_step<N>.json（每 100 步一个 checkpoint）

输出：experiments/curves/reward_vs_eval.png
"""
from __future__ import annotations

import json
import os
import glob
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = {
    "ep900_base": ("seed42 full prompts", "C0"),
    "ep900_base_filt": ("seed42 zero-var filter", "C1"),
    "ep900_s7_base": ("seed7 full prompts", "C2"),
    "ep900_s7_filt": ("seed7 zero-var filter", "C4"),
}
SFT_ACC = 0.385  # experiments/eval/sft.json


def rolling(values: list[float], window: int = 20) -> list[float]:
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(float(np.mean(values[lo : i + 1])))
    return out


def eval_points(tag: str) -> list[tuple[int, float, int]]:
    """(step, accuracy, n)：优先读 *_ckpt.jsonl（每 100 步一点，含样本量）。"""
    jp = os.path.join(ROOT, "experiments", "eval", f"{tag}_ckpt.jsonl")
    if os.path.exists(jp):
        rows = [json.loads(l) for l in open(jp, encoding="utf-8")]
        # 同一步可能有 n=100 与 n=200 两条，取 n 大的那条
        best = {}
        for r in rows:
            s, a, n = int(r["step"]), float(r["accuracy"]), int(r["n"])
            if s not in best or n > best[s][1]:
                best[s] = (a, n)
        return sorted((s, a, n) for s, (a, n) in best.items())
    pts = []
    for p in glob.glob(os.path.join(ROOT, "experiments", "eval", f"{tag}_step*.json")):
        m = json.load(open(p, encoding="utf-8"))
        sm = re.search(r"step(\d+)\.json$", p)
        if sm:
            pts.append((int(sm.group(1)), m["accuracy"], int(m["num_samples"])))
    return sorted(pts)


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax2 = ax.twinx()

    plotted = False
    for tag, (label, color) in RUNS.items():
        log_path = os.path.join(ROOT, "experiments", "logs", f"{tag}.jsonl")
        if not os.path.exists(log_path):
            print(f"[plot] skip {tag}: no log")
            continue
        rows = [json.loads(l) for l in open(log_path, encoding="utf-8")]
        steps = [r["step"] for r in rows]
        rewards = [r["rewards/correctness_reward/mean"] for r in rows]
        ax.plot(steps, rewards, color=color, alpha=0.18, lw=0.8)
        ax.plot(steps, rolling(rewards), color=color, lw=2, label=f"{label}: train reward (20-step mean)")

        pts = eval_points(tag)
        if not pts:
            print(f"[plot] {tag}: 还没有 checkpoint 评测")
            continue
        ax2.plot([p[0] for p in pts], [p[1] for p in pts], color=color, marker="o", ls="--", lw=1.5,
                 label=f"{label}: eval accuracy (every 100 steps)")
        # 末点用方块标出并注明样本量（中间点 n=100，末点 n=200）
        last = pts[-1]
        ax2.plot([last[0]], [last[1]], color=color, marker="s", ms=7, ls="none")
        ax2.annotate(f"{last[1]*100:.1f}% (n={last[2]})", (last[0], last[1]), textcoords="offset points",
                     xytext=(-8, 8), ha="right", fontsize=8, color=color)

    ax2.axhline(SFT_ACC, color="gray", ls=":", lw=1)
    ax2.text(5, SFT_ACC + 0.006, f"SFT n=200: {SFT_ACC:.3f}", color="gray", fontsize=8)
    # 中间 checkpoint 用的是同一批测试集的前 100 题，SFT 在该子集上是 0.370，需同口径参考
    ax2.axhline(0.370, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax2.text(5, 0.370 - 0.020, "SFT first-100 (n=100): 0.370", color="gray", fontsize=8)

    ax.set_xlabel("GRPO step")
    ax.set_ylabel("training correctness reward")
    ax2.set_ylabel("eval accuracy (greedy; n=100 per checkpoint, final n=200)")
    ax.set_title("Reward vs eval: 900 steps on a 1,100-prompt pool (3.3 epochs), 2 configs × 2 seeds")
    ax.grid(alpha=0.25)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left", framealpha=0.9)

    out = os.path.join(ROOT, "experiments", "curves", "reward_vs_eval.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"[plot] {out}")


if __name__ == "__main__":
    main()
