"""长程实验汇总：从 experiments/logs/*.jsonl 复算所有数字，输出 experiments/long_summary.md。

复算内容（全部来自真实训练日志，不做任何估算）：
- 每步组内零方差比例 frac_reward_zero_std 的均值
- 有效梯度密度 = 1 - frac_reward_zero_std（每步真正贡献梯度的组占比）
- 奖励前 50 步 / 后 50 步滑动均值，判断是否还在上升
- 过滤版相对全量版的梯度密度提升（同 seed 配对）
"""
from __future__ import annotations

import json
import os
import statistics
import sys

from .common import project_path, read_jsonl

RUNS = [
    ("long_base_s42", "full prompts", 42, 0),
    ("long_base_s42_filt", "variance-filtered", 42, 2),
    ("long_base_s7", "full prompts", 7, 0),
    ("long_base_s7_filt", "variance-filtered", 7, 2),
]


def mean_of(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if key in r and isinstance(r[key], (int, float))]
    return statistics.mean(vals) if vals else None


def window_mean(rows: list[dict], key: str, lo: int, hi: int) -> float | None:
    vals = [r[key] for r in rows if key in r and lo < r["step"] <= hi]
    return statistics.mean(vals) if vals else None


def summarize(tag: str) -> dict | None:
    path = project_path("experiments/logs", f"{tag}.jsonl")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    rows = read_jsonl(path)
    meta_path = project_path("experiments/logs", f"{tag}.meta.json")
    meta = json.loads(open(meta_path, encoding="utf-8").read()) if os.path.exists(meta_path) else {}
    steps = rows[-1]["step"] if rows else 0
    target = meta.get("max_steps")
    return {
        "tag": tag,
        "steps": steps,
        "target": target,
        "complete": bool(meta) and (target is None or steps >= target),
        "reward_mean": mean_of(rows, "rewards/correctness_reward/mean"),
        "reward_first50": window_mean(rows, "rewards/correctness_reward/mean", 0, 50),
        "reward_last50": window_mean(rows, "rewards/correctness_reward/mean", max(0, steps - 50), steps),
        "zero_std": mean_of(rows, "frac_reward_zero_std"),
        "learnable": mean_of(rows, "filter/learnable_groups"),
        "grad_density": (1 - mean_of(rows, "frac_reward_zero_std")) if mean_of(rows, "frac_reward_zero_std") is not None else None,
        "rounds": mean_of(rows, "filter/rounds"),
        "entropy": mean_of(rows, "entropy"),
        "seconds": meta.get("train_seconds"),
        "peak_gib": meta.get("peak_gpu_gib"),
        "prompt_visits": meta.get("prompt_visits"),
        "pool": meta.get("pool_size"),
        "batch": meta.get("batch_size"),
        "gen": meta.get("num_generations"),
    }


def pct(x: float | None) -> str:
    return "未验证" if x is None else f"{x*100:.1f}%"


def num(x, fmt: str) -> str:
    return "未验证" if x is None else fmt.format(x)


def steps_label(r: dict) -> str:
    if r["complete"]:
        return str(r["steps"])
    if r["target"]:
        return f"{r['steps']}（未完成，目标 {r['target']}）"
    return f"{r['steps']}（未完成）"


def main() -> None:
    rows = [s for s in (summarize(tag) for tag, *_ in RUNS) if s]
    if not rows:
        print("没有长程日志，无法汇总", file=sys.stderr)
        sys.exit(1)

    lines = ["# 长程 GRPO 汇总（由 src/analyze_long.py 从真实日志复算）", ""]
    lines += [
        "| run | steps | prompts seen | reward 全期均值 | reward 前50步 | reward 后50步 | 零方差比例 | 有效梯度密度 | 重采样轮数/步 | 训练秒 | 峰值显存 GiB |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        rounds = "—" if r["rounds"] is None else f"{r['rounds']:.2f}"
        secs = "未验证" if r["seconds"] is None else f"{r['seconds']:.0f}"
        peak = "未验证" if r["peak_gib"] is None else f"{r['peak_gib']:.2f}"
        lines.append(
            f"| {r['tag']} | {steps_label(r)} | {num(r['prompt_visits'], '{:d}')} | {pct(r['reward_mean'])} | "
            f"{pct(r['reward_first50'])} | {pct(r['reward_last50'])} | {pct(r['zero_std'])} | "
            f"{pct(r['grad_density'])} | {rounds} | {secs} | {peak} |"
        )

    lines += ["", "## 过滤带来的有效梯度密度提升（同 seed 配对）", "",
              "| seed | 全量版梯度密度 | 过滤版梯度密度 | 绝对提升 | 相对提升 |",
              "|---|---|---|---|---|"]
    by_seed = {}
    for r in rows:
        seed = 42 if "s42" in r["tag"] else 7
        key = "filt" if "filt" in r["tag"] else "full"
        by_seed.setdefault(seed, {})[key] = r
    for seed, d in sorted(by_seed.items()):
        if "full" in d and "filt" in d and d["full"]["grad_density"] and d["filt"]["grad_density"]:
            a, b = d["full"]["grad_density"], d["filt"]["grad_density"]
            lines.append(f"| {seed} | {pct(a)} | {pct(b)} | +{(b-a)*100:.1f} pp | +{(b-a)/a*100:.1f}% |")
        else:
            lines.append(f"| {seed} | 未验证 | 未验证 | 未验证 | 未验证 |")

    text = "\n".join(lines)
    print(text)
    out = project_path("experiments/long_summary.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n[analyze_long] 写入 {out}")


if __name__ == "__main__":
    main()
