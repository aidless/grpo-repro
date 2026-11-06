"""对训练过程中每 100 步存下的 adapter 逐个评测（固定测试子集）。

用法:
  # 每 100 步的 checkpoint 在 n=100 上评测，最终 adapter 另在 n=200 上评测
  python3 scripts/eval_checkpoints.py --config configs/epoch900.yaml --tag ep900_base \
      --limit 100 --final-limit 200 --out-jsonl experiments/eval/ep900_base_ckpt.jsonl

  # 全部用同一个 n（旧行为）
  python3 scripts/eval_checkpoints.py --config configs/epoch900.yaml --tag ep900_base --limit 100
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def step_of(path: str) -> int:
    m = re.search(r"checkpoint-(\d+)$", path)
    return int(m.group(1)) if m else 0


def run_eval(name: str, model_path: str, config: str, limit: int | None) -> dict:
    """评测并返回 metrics；文件已存在则直接读，保证可断点续跑。"""
    out = os.path.join(ROOT, "experiments", "eval", f"{name}.json")
    if os.path.exists(out):
        print(f"[eval-ckpt] skip {name}（已存在）", flush=True)
        return json.load(open(out, encoding="utf-8"))
    cmd = [sys.executable, "-m", "src.evaluate", "--config", config, "--name", name, "--model-path", model_path]
    if limit:
        cmd += ["--limit", str(limit)]
    print(f"[eval-ckpt] {name} (n={limit or 'full'})", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    return json.load(open(out, encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--limit", type=int, default=None, help="每个 checkpoint 的评测条数")
    parser.add_argument("--final-limit", type=int, default=None, help="最终 adapter 的评测条数；默认同 --limit")
    parser.add_argument("--out-jsonl", default=None, help="把每个 checkpoint 的指标汇总成 jsonl")
    args = parser.parse_args()

    ckpt_root = os.path.join(ROOT, "experiments", "checkpoints", args.tag)
    ckpts = sorted(glob.glob(os.path.join(ckpt_root, "checkpoint-*")), key=step_of)
    if not ckpts:
        print(f"[eval-ckpt] {ckpt_root} 下没有 checkpoint-*，退出", flush=True)
        return

    meta_path = os.path.join(ROOT, "experiments", "logs", f"{args.tag}.meta.json")
    total = None
    if os.path.exists(meta_path):
        total = json.load(open(meta_path, encoding="utf-8")).get("max_steps")

    records = []
    # 中间 checkpoint：加 _n<limit> 后缀，避免和最终 n=200 的文件重名
    for cp in ckpts:
        if not os.path.exists(os.path.join(cp, "adapter_config.json")):
            print(f"[eval-ckpt] skip {cp}（缺 adapter_config.json）", flush=True)
            continue
        step = step_of(cp)
        suffix = f"_n{args.limit}" if args.limit else ""
        name = f"{args.tag}_step{step}{suffix}"
        m = run_eval(name, cp, args.config, args.limit)
        records.append({"step": step, "n": m.get("num_samples"), "accuracy": m["accuracy"],
                        "format_compliance_rate": m.get("format_compliance_rate"),
                        "parse_fail_rate": m.get("parse_fail_rate")})

    # 最终 adapter（训练结束后 save_model 写的那份）
    final_limit = args.final_limit if args.final_limit is not None else args.limit
    if total and os.path.exists(os.path.join(ckpt_root, "adapter_config.json")):
        name = f"{args.tag}_step{total}"
        m = run_eval(name, ckpt_root, args.config, final_limit)
        records.append({"step": total, "n": m.get("num_samples"), "accuracy": m["accuracy"],
                        "format_compliance_rate": m.get("format_compliance_rate"),
                        "parse_fail_rate": m.get("parse_fail_rate")})

    if args.out_jsonl:
        out = args.out_jsonl if os.path.isabs(args.out_jsonl) else os.path.join(ROOT, args.out_jsonl)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for r in sorted(records, key=lambda x: x["step"]):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[eval-ckpt] 汇总 {len(records)} 个点 → {out}", flush=True)


if __name__ == "__main__":
    main()
