"""数据准备：GSM8K → SFT 文本 / GRPO prompt / 固定评测子集。

划分（全部固定 seed）：
- SFT 训练集   : train[0 : sft_n]
- GRPO 训练集  : train[sft_n : sft_n + grpo_n]（与 SFT 不重叠，排除背题）
- 评测集       : test[0 : eval_n]（GSM8K 官方测试集，SFT/GRPO 都未见过）
"""
from __future__ import annotations

import argparse
import os
import random

from datasets import load_dataset
from transformers import AutoTokenizer

from .common import build_messages, clean_solution, get_model_path, load_config, project_path, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    seed = cfg["seed"]
    data_dir = project_path(cfg["data_dir"])
    os.makedirs(data_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(get_model_path(cfg))
    ds = load_dataset("openai/gsm8k", "main")

    rng = random.Random(seed)
    train_idx = list(range(len(ds["train"])))
    rng.shuffle(train_idx)
    test_idx = list(range(len(ds["test"])))
    random.Random(seed + 1).shuffle(test_idx)

    sft_idx = train_idx[: cfg["sft_n"]]
    remain = train_idx[cfg["sft_n"] :]
    grpo_n = cfg["grpo_n"]
    if grpo_n <= 0:  # -1 表示「SFT 之外的全部剩题」
        grpo_n = len(remain)
    # grpo_sample_seed 存在时：从剩余题库里固定 seed 抽样（用于「缩小题库、多 epoch」的实验）
    grpo_seed = cfg.get("grpo_sample_seed")
    if grpo_seed is None:
        grpo_idx = remain[:grpo_n]
    else:
        grpo_idx = sorted(random.Random(int(grpo_seed)).sample(remain, grpo_n))

    sft_rows, grpo_rows = [], []
    for i in sft_idx:
        ex = ds["train"][i]
        messages = build_messages(ex["question"])
        text = tok.apply_chat_template(
            messages + [{"role": "assistant", "content": clean_solution(ex["answer"])}],
            tokenize=False,
            add_generation_prompt=False,
        )
        sft_rows.append({"text": text, "answer": ex["answer"].split("####")[-1].strip()})

    for i in grpo_idx:
        ex = ds["train"][i]
        gold = ex["answer"].split("####")[-1].strip()
        grpo_rows.append({"prompt": build_messages(ex["question"]), "answer": gold, "question": ex["question"]})

    eval_rows = []
    for i in test_idx[: cfg["eval_n"]]:
        ex = ds["test"][i]
        gold = ex["answer"].split("####")[-1].strip()
        eval_rows.append(
            {"prompt": build_messages(ex["question"]), "answer": gold, "question": ex["question"], "test_index": i}
        )

    write_jsonl(os.path.join(data_dir, "sft.jsonl"), sft_rows)
    write_jsonl(os.path.join(data_dir, "grpo_train.jsonl"), grpo_rows)
    write_jsonl(os.path.join(data_dir, "eval.jsonl"), eval_rows)

    # 自检：打印一条样例，确认 prompt 模板与答案抽取都正常
    from .common import answers_match, extract_answer

    sample = clean_solution(ds["train"][sft_idx[0]]["answer"])
    pred, source = extract_answer(sample)
    gold = ds["train"][sft_idx[0]]["answer"].split("####")[-1].strip()
    print(f"[data] dir={data_dir}")
    print(f"[data] sft={len(sft_rows)} grpo={len(grpo_rows)} eval={len(eval_rows)}")
    print(f"[data] 自检 抽取答案={pred} 来源={source} 标准答案={gold} 判分={answers_match(pred, gold)}")
    print(f"[data] SFT 样例前 200 字: {sft_rows[0]['text'][:200]!r}")


if __name__ == "__main__":
    main()
