"""Run the four evaluations and print one markdown table.

Usage:
    python scripts/verify.py --config configs/full.yaml --limit 40   # fast check
    python scripts/verify.py --config configs/full.yaml              # full 200

Outputs are written to experiments/eval/verify_<name>.json so the published
200-sample results in experiments/eval/<name>.json are never overwritten.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import get_model_path, load_config  # noqa: E402

STAGES = ["base", "sft", "grpo_base", "grpo_format"]


def model_path_for(stage: str, cfg: dict) -> str:
    if stage == "base":
        return get_model_path(cfg)
    mapping = {
        "sft": "sft_merged",
        "grpo_base": "grpo_base_out",
        "grpo_format": "grpo_format_out",
    }
    if stage in mapping:
        return str(ROOT / cfg[mapping[stage]])
    # 长程 run：检查点在 experiments/checkpoints/<tag>
    return str(ROOT / "experiments" / "checkpoints" / stage)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stages", default=None, help="逗号分隔；默认 base,sft,grpo_base,grpo_format")
    ap.add_argument("--extra-tags", default=None, help="追加评测的长程 run 标签，逗号分隔")
    args = ap.parse_args()
    cfg = load_config(str(ROOT / args.config))
    if args.stages is None:
        stages = list(STAGES)
    else:
        stages = [s for s in args.stages.split(",") if s]
    if args.extra_tags:
        stages += [t for t in args.extra_tags.split(",") if t]

    for stage in stages:
        path = model_path_for(stage, cfg)
        if not (Path(path) / "adapter_config.json").exists() and not (Path(path) / "config.json").exists():
            # 没跑过的 run（例如预留的第二个 seed，或训练中被中断的空目录）明确跳过
            print(f"[skip] {stage}: no checkpoint at {path}", flush=True)
            continue
        cmd = [
            sys.executable, "-m", "src.evaluate",
            "--config", args.config,
            "--name", f"verify_{stage}",
            "--model-path", path,
        ]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)

    print()
    print(f"{'model':<14}{'n':>4}{'acc':>8}{'fmt':>8}{'parse_fail':>12}{'avg_tok':>9}")
    for stage in stages:
        path = ROOT / "experiments" / "eval" / f"verify_{stage}.json"
        if not path.exists():
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        print(f"{stage:<14}{m['num_samples']:>4}{m['accuracy']:>8.3f}"
              f"{m['format_compliance_rate']:>8.3f}{m['parse_fail_rate']:>12.3f}"
              f"{m['avg_completion_tokens']:>9.1f}")


if __name__ == "__main__":
    main()
