"""评测结果分析：Wilson 置信区间 + 配对 McNemar 精确检验（同一测试子集，同判分口径）。

只使用标准库，避免额外依赖。
"""
from __future__ import annotations

import json
import math
import os

from .common import project_path

ORDER = ["base", "sft", "grpo_base", "grpo_format",
         "long_base_s42", "long_base_s42_filt", "long_base_s7", "long_base_s7_filt",
         "ep900_base_step900"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def binom_two_sided(k: int, n: int) -> float:
    """McNemar 精确检验：在 b+c 个不一致样本中，观察 b 个的概率（双侧）。"""
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) / 2**n for i in range(n + 1)]
    obs = probs[k]
    return min(1.0, sum(p for p in probs if p <= obs + 1e-12))


def load_samples(name: str) -> dict[int, dict]:
    path = project_path(f"experiments/eval/{name}_samples.jsonl")
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    return {r["test_index"]: r for r in rows}


def main() -> None:
    present = [n for n in ORDER if os.path.exists(project_path(f"experiments/eval/{n}_samples.jsonl"))]
    data = {n: load_samples(n) for n in present}

    lines = ["| model | n | accuracy (95% CI) | format compliance | parse fail | 平均输出 token |",
             "|---|---|---|---|---|---|"]
    for n in present:
        rows = list(data[n].values())
        k = sum(r["correct"] for r in rows)
        lo, hi = wilson(k, len(rows))
        fmt = sum(r["format_compliant"] for r in rows) / len(rows)
        pf = sum(r["source"] == "none" for r in rows) / len(rows)
        tok = sum(r["completion_tokens"] for r in rows) / len(rows)
        lines.append(
            f"| {n} | {len(rows)} | {k/len(rows)*100:.1f}% ({lo*100:.1f}-{hi*100:.1f}) | "
            f"{fmt*100:.1f}% | {pf*100:.1f}% | {tok:.1f} |"
        )

    lines += ["", "配对 McNemar 精确检验（同一批测试样本）：", "",
              "| 对比 (A → B) | A 对/B 错 | A 错/B 对 | 净变化 | McNemar p |", "|---|---|---|---|---|"]
    pairs = [("base", "sft"), ("sft", "grpo_base"), ("sft", "grpo_format"),
             ("sft", "long_base_s42"), ("sft", "long_base_s42_filt"),
             ("sft", "long_base_s7"), ("sft", "long_base_s7_filt"),
             ("sft", "ep900_base_step900"),             # 3.27 epoch：主结果
             ("long_base_s42", "ep900_base_step900"),   # 同一配置，0.27 → 3.27 epoch
             ("long_base_s42", "long_base_s42_filt"),   # 同 seed：过滤 vs 全量
             ("long_base_s42", "long_base_s7")]         # 同配置不同 seed：噪声尺度
    for a, b in pairs:
        if a not in data or b not in data:
            continue
        common = sorted(set(data[a]) & set(data[b]))
        b_only = sum(1 for i in common if data[a][i]["correct"] and not data[b][i]["correct"])
        b_better = sum(1 for i in common if not data[a][i]["correct"] and data[b][i]["correct"])
        p = binom_two_sided(min(b_only, b_better), b_only + b_better)
        lines.append(f"| {a} → {b} | {b_only} | {b_better} | {b_better - b_only:+d} | {p:.3f} |")

    text = "\n".join(lines)
    print(text)
    out = project_path("experiments/eval/compare.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("# 评测对比（GSM8K test 固定子集）\n\n" + text + "\n")


if __name__ == "__main__":
    main()
