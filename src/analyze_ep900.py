"""多 epoch（1,100 题库 / 900 步）实验汇总：梯度密度 + 评测 + 配对检验。

输出 experiments/ep900_summary.md，全部数字从 experiments/logs 与 experiments/eval 复算。
"""
from __future__ import annotations

import glob
import json
import math
import os
import re

from .analyze_eval import binom_two_sided, load_samples, wilson
from .common import project_path

RUNS = {
    "ep900_base": "seed42 全量",
    "ep900_base_filt": "seed42 过滤",
    "ep900_s7_base": "seed7 全量",
    "ep900_s7_filt": "seed7 过滤",
}
POOL = 1100
PROMPTS_PER_STEP = 4
FILTER_PAIRS = [("ep900_base", "ep900_base_filt", "seed 42"), ("ep900_s7_base", "ep900_s7_filt", "seed 7")]


def rolling(vals: list[float], w: int = 20) -> list[float]:
    return [sum(vals[max(0, i - w + 1) : i + 1]) / len(vals[max(0, i - w + 1) : i + 1]) for i in range(len(vals))]


def slope(xs: list[int], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs) * 100


def ckpt_evals(tag: str) -> list[tuple[int, float, int]]:
    """返回 (step, accuracy, n)。优先读 *_ckpt.jsonl（含每点样本量），否则退回旧的文件命名。"""
    jp = project_path(f"experiments/eval/{tag}_ckpt.jsonl")
    if os.path.exists(jp):
        rows = [json.loads(l) for l in open(jp, encoding="utf-8")]
        return sorted((int(r["step"]), float(r["accuracy"]), int(r["n"])) for r in rows)
    pts = []
    for p in glob.glob(str(project_path(f"experiments/eval/{tag}_step*.json"))):
        m = re.search(r"step(\d+)\.json$", p)
        if not m:
            continue
        d = json.load(open(p, encoding="utf-8"))
        pts.append((int(m.group(1)), d["accuracy"], d["num_samples"]))
    return sorted(pts)


def final_point(tag: str):
    """最终评测点：step 最大，同 step 内取 n 最大（900 步同时有 n=100 与 n=200 两条）。"""
    ev = ckpt_evals(tag)
    if not ev:
        return None
    step = ev[-1][0]
    return max([e for e in ev if e[0] == step], key=lambda e: e[2])


def main() -> None:
    logs, metas = {}, {}
    for tag in RUNS:
        log = project_path(f"experiments/logs/{tag}.jsonl")
        if not os.path.exists(log):
            continue
        logs[tag] = [json.loads(l) for l in open(log, encoding="utf-8")]
        mp = project_path(f"experiments/logs/{tag}.meta.json")
        metas[tag] = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else {}

    partial = [label for tag, label in RUNS.items() if tag in logs and not metas.get(tag)]
    lines = ["# 多 epoch 实验（题库 1,100，900 步 ≈ 3.3 epoch）", ""]
    if partial:
        lines += [f"> **状态：部分 run 仍在训练中（{ '、'.join(partial) }）——这些行是中途快照，不是最终结论。**", ""]
    lines += ["| run | steps | epochs seen | reward 全期 | 后 50 步 | 奖励斜率/100 步 | 零方差 | 梯度密度 | 训练秒 | 最终准确率 (n=200) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    finals = {}
    for tag, label in RUNS.items():
        if tag not in logs:
            lines.append(f"| {label} | 未验证 | 未验证 | 未验证 | 未验证 | 未验证 | 未验证 | 未验证 | 未验证 | 未验证 |")
            continue
        r = logs[tag]
        steps = r[-1]["step"]
        done = bool(metas.get(tag))
        rew = [x["rewards/correctness_reward/mean"] for x in r]
        zero = [x["frac_reward_zero_std"] for x in r]
        fp = final_point(tag)
        if fp is not None:
            finals[tag] = fp[0]
        lines.append(
            f"| {label} | {steps if done else f'进行中（step {steps}）'} | "
            f"{f'{steps * PROMPTS_PER_STEP / POOL:.2f}' if done else '—'} | {sum(rew)/len(rew)*100:.1f}% | "
            f"{sum(rew[-50:])/50*100:.1f}% | {slope([x['step'] for x in r], rew):+.2f} pp | "
            f"{sum(zero)/len(zero)*100:.1f}% | {(1-sum(zero)/len(zero))*100:.1f}% | "
            f"{metas.get(tag, {}).get('train_seconds', '进行中')} | "
            f"{'未验证' if fp is None else f'{fp[1]*100:.1f}% (n={fp[2]})'} |"
        )

    lines += ["", "## 评测曲线（每 100 步一个 checkpoint；中间点 n=100，末点 n=200）", "",
              "| run | 步数 | 准确率 | 95% CI | n |", "|---|---|---|---|---|"]
    for tag, label in RUNS.items():
        for step, acc, n in ckpt_evals(tag):
            lo, hi = wilson(int(round(acc * n)), n)
            lines.append(f"| {label} | {step} | {acc*100:.1f}% | [{lo*100:.1f}, {hi*100:.1f}] | {n} |")

    # 过滤省不省步数：每个 seed 内对比（奖励口径 + 评测口径）
    lines += ["", "## 过滤买到了什么（900 步尺度，按 seed）", ""]
    for base_tag, filt_tag, seed_label in FILTER_PAIRS:
        if base_tag not in logs or filt_tag not in logs:
            lines.append(f"- **{seed_label}**：{base_tag} / {filt_tag} 未跑齐 → 未验证。")
            continue
        base_rew = [x["rewards/correctness_reward/mean"] for x in logs[base_tag]]
        target = sum(base_rew[-20:]) / 20
        filt = logs[filt_tag]
        frew = rolling([x["rewards/correctness_reward/mean"] for x in filt])
        reach = next((filt[i]["step"] for i, v in enumerate(frew) if v >= target), None)

        # 奖励口径对过滤版偏高（重采样后挑的是"有对有错"的组），所以再给一个评测口径：
        # 过滤版最早在哪个 checkpoint 达到同 seed 全量版最终 checkpoint 的准确率。
        fe = ckpt_evals(filt_tag)
        bfp = final_point(base_tag)
        acc_target = bfp[1] if bfp else None
        reach_acc = next((s for s, a, _ in fe if a >= acc_target), None) if (acc_target is not None and fe) else None

        lines += [f"- **{seed_label}**：全量版最后 20 步奖励 = {target*100:.1f}%；过滤版达到该水平 = "
                  f"{'未达到' if reach is None else f'第 {reach} 步'}（奖励口径，过滤版因重采样偏高，仅供参考）。",
                  f"  - 全量版最终 checkpoint 准确率 = "
                  f"{'未验证' if bfp is None else f'{acc_target*100:.1f}% (n={bfp[2]})'}；过滤版最早达到该准确率 = "
                  f"{'未达到/未验证' if reach_acc is None else f'step {reach_acc}'}（评测口径，两版可比；中间点 n=100）。"]
    lines.append("")

    # 配对检验（最终 checkpoint 对 SFT / 对 base）
    pairs = [("sft", tag) for tag in RUNS] + [(b, f) for b, f, _ in FILTER_PAIRS] + [
        ("ep900_base", "ep900_s7_base"), ("ep900_base_filt", "ep900_s7_filt")]
    if finals:
        data = {n: load_samples(n) for n in ["sft"] if os.path.exists(project_path("experiments/eval/sft_samples.jsonl"))}
        for tag, step in finals.items():
            if os.path.exists(project_path(f"experiments/eval/{tag}_step{step}_samples.jsonl")):
                data[tag] = load_samples(f"{tag}_step{step}")
        lines += ["| 对比 | 前者对/后者错 | 前者错/后者对 | 净差 | McNemar p |",
                  "|---|---|---|---|---|"]
        effects = {}
        for a, b in pairs:
            if a not in data or b not in data:
                lines.append(f"| {a} → {b} | 未验证 | 未验证 | 未验证 | 未验证 |")
                continue
            keys = sorted(set(data[a]) & set(data[b]))
            only_a = sum(1 for k in keys if data[a][k]["correct"] and not data[b][k]["correct"])
            only_b = sum(1 for k in keys if not data[a][k]["correct"] and data[b][k]["correct"])
            p = binom_two_sided(min(only_a, only_b), only_a + only_b)
            lines.append(f"| {a} → {b} | {only_a} | {only_b} | {only_b-only_a:+d} | {p:.3f} |")
            effects[(a, b)] = (only_b - only_a, p)

        # 过滤主效应：相对同 seed 全量版的净差（题数），两 seed 是否同向
        lines += ["", "## 过滤效应（相对同 seed 全量版，n=200）", ""]
        dirs, details = [], []
        for base_tag, filt_tag, seed_label in FILTER_PAIRS:
            e = effects.get((base_tag, filt_tag))
            if e is None:
                details.append(f"- **{seed_label}**：未验证")
                continue
            net, p = e
            dirs.append(net)
            details.append(f"- **{seed_label}**：过滤 − 全量 = {net:+d} 题，McNemar p = {p:.3f}")
        lines += details
        if len(dirs) == 2:
            if (dirs[0] > 0) != (dirs[1] > 0) and dirs[0] != 0 and dirs[1] != 0:
                lines.append("- **两 seed 方向相反 → 过滤效应不可靠**（单 seed 的 +4.5 pp 不能推广）。")
            elif dirs[0] == 0 and dirs[1] == 0:
                lines.append("- 两 seed 净差均为 0 题 → 过滤不改变最终准确率。")
            else:
                lines.append("- 两 seed 方向相同，但每个 seed 的 p 值见上；仍受 n=200 的统计功效限制。")
        elif len(dirs) == 1:
            lines.append("- 只有一个 seed 有过滤对照，过滤效应仍是单 seed 观察。")

    out = project_path("experiments/ep900_summary.md")
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[done] {out}")


if __name__ == "__main__":
    main()
