"""LoRA SFT 打底：让 0.5B 模型先学会「分步推理 + 末行 \\boxed{}」的输出习惯。

产出：
- experiments/checkpoints/sft           (LoRA adapter)
- experiments/checkpoints/sft_merged    (合并后的完整权重，供 GRPO 起点与评测)
"""
from __future__ import annotations

import argparse
import gc
import os
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from .common import get_model_path, load_config, project_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    sft_cfg = cfg["sft"]

    model_path = get_model_path(cfg)
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_ds = load_dataset("json", data_files=project_path(cfg["data_dir"], "sft.jsonl"), split="train")
    print(f"[sft] 样本数={len(train_ds)}")

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    model.config.use_cache = False

    lora = LoraConfig(
        r=sft_cfg["lora_r"],
        lora_alpha=sft_cfg["lora_alpha"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    out_dir = project_path(cfg["sft_out"])
    sft_args = SFTConfig(
        output_dir=out_dir,
        max_length=sft_cfg["max_length"],
        per_device_train_batch_size=sft_cfg["batch_size"],
        gradient_accumulation_steps=sft_cfg["grad_accum"],
        learning_rate=sft_cfg["lr"],
        num_train_epochs=sft_cfg["epochs"],
        max_steps=sft_cfg.get("max_steps", -1),
        logging_steps=sft_cfg["logging_steps"],
        save_strategy="no",
        report_to="none",
        bf16=True,
        gradient_checkpointing=True,
        dataset_text_field="text",
        warmup_steps=sft_cfg.get("warmup_steps", 10),
        lr_scheduler_type="cosine",
        seed=cfg["seed"],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        processing_class=tok,
        peft_config=lora,
    )
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"[sft] 训练完成 用时={elapsed:.1f}s")

    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)

    # 合并 LoRA 到基础权重，产出 GRPO 的起点模型
    merged_dir = project_path(cfg["sft_merged"])
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    base = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, out_dir)
    merged = merged.merge_and_unload()
    os.makedirs(merged_dir, exist_ok=True)
    merged.save_pretrained(merged_dir)
    tok.save_pretrained(merged_dir)
    print(f"[sft] 合并权重已保存 → {merged_dir}")

    os.makedirs(project_path("experiments/logs"), exist_ok=True)
    with open(project_path("experiments/logs/sft_time.txt"), "w", encoding="utf-8") as f:
        f.write(f"train_seconds={elapsed:.1f}\n")


if __name__ == "__main__":
    main()
