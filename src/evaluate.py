"""统一评测：同一个固定测试子集 + 同一套判分口径，对比 SFT 前后 / GRPO 前后。

用法：
  python -m src.evaluate --config configs/smoke.yaml --name base --model-path <dir>
  name 可为 base / sft / grpo_base / grpo_format
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import (
    answers_match,
    completion_to_text,
    extract_answer,
    get_model_path,
    is_format_compliant,
    load_config,
    project_path,
    read_jsonl,
)


def load_model(model_path: str, base_path: str):
    """adapter 目录先合并再评测，避免逐 token 的 LoRA 开销。统一放 GPU 上跑。"""
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        base = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.bfloat16, device_map="cuda")
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ev = cfg["eval"]

    base_path = project_path(cfg["sft_merged"])
    if not os.path.isdir(base_path):
        base_path = get_model_path(cfg)

    tok = AutoTokenizer.from_pretrained(base_path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = load_model(args.model_path, base_path)
    rows = read_jsonl(project_path(cfg["data_dir"], "eval.jsonl"))
    if args.limit:
        rows = rows[: args.limit]

    max_new = args.max_new_tokens or ev["max_new_tokens"]
    batch_size = ev["batch_size"]

    details = []
    t0 = time.time()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        texts = [tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True) for r in batch]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=False,
                repetition_penalty=1.0,
                eos_token_id=[151645, 151643],  # <|im_end|> 与 <|endoftext|>
                pad_token_id=tok.pad_token_id,
            )
        prompt_len = enc["input_ids"].shape[1]
        completions = tok.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
        for row, completion in zip(batch, completions):
            pred, source = extract_answer(completion)
            correct = answers_match(pred, row["answer"])
            details.append(
                {
                    "test_index": row.get("test_index"),
                    "question": row["question"],
                    "gold": row["answer"],
                    "pred": pred,
                    "source": source,
                    "correct": correct,
                    "format_compliant": is_format_compliant(completion),
                    "completion_tokens": int((out[:, prompt_len:].ne(tok.pad_token_id)).sum().item() / len(batch)),
                    "completion": completion,
                }
            )
        done = min(start + batch_size, len(rows))
        print(f"[eval:{args.name}] {done}/{len(rows)} acc={sum(d['correct'] for d in details) / len(details):.3f}")

    n = len(details)
    metrics = {
        "name": args.name,
        "model_path": args.model_path,
        "num_samples": n,
        "accuracy": sum(d["correct"] for d in details) / n,
        "format_compliance_rate": sum(d["format_compliant"] for d in details) / n,
        "parse_fail_rate": sum(d["source"] == "none" for d in details) / n,
        "avg_completion_tokens": sum(d["completion_tokens"] for d in details) / n,
        "elapsed_seconds": time.time() - t0,
    }

    out_dir = project_path("experiments/eval")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{args.name}.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, f"{args.name}_samples.jsonl"), "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"[eval:{args.name}] 完成 acc={metrics['accuracy']:.4f} fmt={metrics['format_compliance_rate']:.4f} "
          f"parse_fail={metrics['parse_fail_rate']:.4f} avg_tok={metrics['avg_completion_tokens']:.1f} "
          f"用时={metrics['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
