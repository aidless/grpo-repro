"""检查「评测集 vs 训练集」是否重叠：索引级 + 文本级。

用法：
  python3 scripts/check_overlap.py --config configs/epoch900.yaml

检查项：
  1. data/<dir>/eval.jsonl 的 test_index / question 是否与按 config 重新推导的 test[0:eval_n] 一致；
  2. data/<dir>/grpo_train.jsonl 的 question 是否与 train[sft_n:] 抽样结果逐条一致；
  3. data/<dir>/sft.jsonl 是否只含 train[0:sft_n] 的题目；
  4. 三份数据的题目文本两两求交集，必须为空。
任何一项不通过，退出码非 0。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import yaml
from datasets import load_dataset


def norm(q: str) -> str:
    return " ".join(q.lower().split())


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    root = os.path.dirname(os.path.dirname(os.path.abspath(args.config)))
    data_dir = os.path.join(root, cfg["data_dir"])

    ds = load_dataset("openai/gsm8k", "main")
    train_ds, test_ds = ds["train"], ds["test"]

    # 按 config 的 seed 重新推导切分（与 src/prepare_data.py 完全一致的逻辑）
    train_idx = list(range(len(train_ds)))
    random.Random(cfg["seed"]).shuffle(train_idx)
    sft_idx = train_idx[: cfg["sft_n"]]
    remain = train_idx[cfg["sft_n"] :]
    grpo_n = cfg["grpo_n"]
    grpo_seed = cfg.get("grpo_sample_seed")
    grpo_idx = sorted(random.Random(int(grpo_seed)).sample(remain, grpo_n)) if grpo_seed else remain[:grpo_n]

    test_idx = list(range(len(test_ds)))
    random.Random(cfg["seed"] + 1).shuffle(test_idx)
    eval_idx = test_idx[: cfg["eval_n"]]

    ev = read_jsonl(os.path.join(data_dir, "eval.jsonl"))
    grpo = read_jsonl(os.path.join(data_dir, "grpo_train.jsonl"))
    sft = read_jsonl(os.path.join(data_dir, "sft.jsonl"))

    failures: list[str] = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failures.append(name)

    # 1) eval.jsonl 与 test 切分逐条一致
    idx_ok = [r["test_index"] for r in ev] == eval_idx
    q_ok = [norm(r["question"]) for r in ev] == [norm(test_ds[i]["question"]) for i in eval_idx]
    check("eval 索引来自 test 切分", idx_ok, f"n={len(ev)}, 与重新推导的 test_index 序列一致={idx_ok}")
    check("eval 题目文本与 test 切分一致", q_ok, f"逐条比对 {len(ev)} 题，一致={q_ok}")

    # 2) grpo_train.jsonl 与 train[sft_n:] 抽样结果逐条一致
    gq = [norm(r["question"]) for r in grpo]
    exq = [norm(train_ds[i]["question"]) for i in grpo_idx]
    check("grpo 题目来自 train 切分且顺序一致", gq == exq, f"n={len(grpo)}, 一致={gq == exq}")
    ga = [r["answer"] for r in grpo]
    exa = [train_ds[i]["answer"].split("####")[-1].strip() for i in grpo_idx]
    check("grpo 标准答案一致", ga == exa, f"一致={ga == exa}")

    # 3) sft.jsonl 只覆盖 train[0:sft_n]
    sft_q = [norm(train_ds[i]["question"]) for i in sft_idx]
    text_blob = "\n".join(r["text"] for r in sft)
    missing = sum(1 for q in sft_q if q not in norm(text_blob))
    check("sft 文本覆盖 train[0:sft_n]", missing == 0, f"n={len(sft)}, 未在文本中找到的题数={missing}")

    # 4) 三份数据题目文本两两交集必须为空
    def dup(pool: list[str]) -> int:
        return len(pool) - len(set(pool))

    for a, b, na, nb in [
        ([norm(r["question"]) for r in ev], gq, "eval", "grpo"),
        ([norm(r["question"]) for r in ev], sft_q, "eval", "sft"),
        (gq, sft_q, "grpo", "sft"),
    ]:
        inter = set(a) & set(b)
        check(f"{na} ∩ {nb} 为空", not inter, f"交集={len(inter)} 题, n_{na}={len(a)}, n_{nb}={len(b)}")
    check("eval 内部无重复题", dup([norm(r["question"]) for r in ev]) == 0, f"重复={dup([norm(r['question']) for r in ev])}")
    check("grpo 池内部无重复题", dup(gq) == 0, f"重复={dup(gq)}")

    # 5) 同时与 long/full 的评测集一致（保证跨实验可比）
    for other in ("data/long/eval.jsonl", "data/full/eval.jsonl"):
        p = os.path.join(root, other)
        if os.path.exists(p):
            same = [r["test_index"] for r in read_jsonl(p)] == [r["test_index"] for r in ev]
            check(f"{other} 与本评测集 test_index 一致", same, f"一致={same}")

    print("\n结果:", "全部通过（无重叠）" if not failures else f"{len(failures)} 项失败: {failures}")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
