"""GRPO 训练（LoRA），单变量对照：只改奖励函数 / 采样策略 / 随机种子。

- base   : reward = correctness (0/1)
- format : reward = correctness + R1 式格式分（合规 +0.5 / 不合规 -0.5）

零方差过滤（--zero-var-rounds K）：
  每一步先正常生成 P 个 prompt × G 条 completion。组内奖励全等（既有全对也有全错）时，
  该组的组内优势恒为 0，对梯度没有任何贡献（trl 内部 advantages 会被置 0）。
  过滤版对零方差组做至多 K 轮重采样：从题库里换一个新 prompt 重新生成，直到拿到
  有方差的组为止；仍拿不到的组保留但优势为 0（不会污染梯度）。
  这样「每个 optimizer step 里带梯度信号的组数」上升，量化指标见输出 jsonl 的
  filter/rounds、filter/zero_std_after、filter/learnable_groups。

所有指标逐行落盘到 experiments/logs/<tag>.jsonl，可被 make report 复算。
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from .common import get_model_path, load_config, project_path
from .rewards import correctness_reward, format_reward


class MetricDumpCallback(TrainerCallback):
    """把每个 logging step 的指标逐行落盘，保证训练曲线有据可查。"""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")

    def on_log(self, args, state, control, logs=None, **kwargs):
        # 末尾的 train_runtime 汇总行没有奖励字段，跳过，保证 jsonl 一行一步
        if logs and state.global_step > 0 and any(k in logs for k in ("reward", "rewards/correctness_reward/mean")):
            row = {"step": state.global_step}
            row.update({k: v for k, v in logs.items() if isinstance(v, (int, float))})
            self.f.write(json.dumps(row) + "\n")
            self.f.flush()

    def on_train_end(self, args, state, control, **kwargs):
        self.f.close()


class ZeroVarFilterGRPOTrainer(GRPOTrainer):
    """trl 1.17 没有内置 dynamic sampling，这里在生成阶段做「换题重采样」。

    零方差组的 advantages 本来就恒为 0（不产生梯度），所以过滤的意义不是「改变梯度方向」，
    而是把本来会浪费掉的采样名额换成有信号的题，从而提升每步的有效梯度密度。
    """

    def __init__(self, *args, pool: list[dict] | None = None, max_rounds: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool = pool or []
        self.max_rounds = int(max_rounds)
        self._cursor = 0
        self._stats = {"rounds": 0.0, "replaced_groups": 0.0, "zero_std_after": 0.0, "learnable": 0.0}

    def _draw_prompts(self, n_groups: int) -> list[dict]:
        """取 n_groups 个新 prompt，并按 trl 的约定每行重复 num_generations 次（每行生成 1 条 completion）。"""
        rows = []
        for _ in range(n_groups):
            row = dict(self.pool[self._cursor % len(self.pool)])
            self._cursor += 1
            rows.extend([row] * self.num_generations)
        return rows

    def _snapshot(self):
        # 只快照 metrics；self._logs 是定长 deque（仅用于 log_completions 表格），这里不动它
        return {k: len(v) for k, v in self._metrics.get("train", {}).items()}

    def _restore(self, snap):
        for k in list(self._metrics["train"]):
            if k not in snap:
                self._metrics["train"][k].clear()
                continue
            del self._metrics["train"][k][snap[k]:]

    def _generate_and_score_completions(self, inputs):
        if not self.model.training or self.max_rounds <= 0:
            return super()._generate_and_score_completions(inputs)

        mode = "train"
        snap = self._snapshot()
        out = super()._generate_and_score_completions(inputs)

        G = self.num_generations
        n_groups = out["advantages"].numel() // G
        answers = [str(x.get("answer", "")) for x in inputs]
        replaced = 0
        rounds = 0

        for _ in range(self.max_rounds):
            adv = out["advantages"].view(n_groups, G)
            zero = (adv.abs().sum(dim=1) == 0).nonzero().flatten().tolist()
            if not zero:
                break
            rounds += 1
            repl = self._draw_prompts(len(zero))
            new_out = super()._generate_and_score_completions(repl)
            new_adv = new_out["advantages"].view(-1, G)
            good = (new_adv.abs().sum(dim=1) > 0).tolist()
            for j, gi in enumerate(zero):
                if not good[j]:
                    continue
                for key in ("prompt_ids", "prompt_mask", "completion_ids", "completion_mask", "advantages"):
                    if key not in out or key not in new_out:
                        continue
                    src = new_out[key][j * G : (j + 1) * G]
                    if src.ndim > 1 and src.shape[1] != out[key].shape[1]:
                        dst_cols = out[key].shape[1]
                        if src.shape[1] < dst_cols:  # 两次调用的 padding 宽度可能不同
                            pad = src.new_zeros((src.shape[0], dst_cols - src.shape[1]))
                            src = torch.cat([src, pad], dim=1)
                        else:
                            src = src[:, :dst_cols]
                    out[key][gi * G : (gi + 1) * G] = src
                answers[gi * G : (gi + 1) * G] = [str(repl[j].get("answer", ""))] * G
                replaced += 1

        # 子调用会重复写 metrics / logs：恢复后再按最终 batch 重算，避免数字被稀释
        self._restore(snap)
        texts = self.processing_class.batch_decode(out["completion_ids"], skip_special_tokens=True)
        per_func: dict[str, list[float]] = {}
        total = [0.0] * len(texts)
        for i, fn in enumerate(self.reward_funcs):
            name = getattr(fn, "__name__", f"reward_{i}")
            vals = fn(completions=texts, answer=answers)
            per_func[name] = [float(v) if v is not None else 0.0 for v in vals]
            for j, v in enumerate(per_func[name]):
                total[j] += v
        r = torch.tensor(total, dtype=torch.float32)
        if os.environ.get("GRPO_DEBUG"):
            print(f"[dbg] n_groups={n_groups} rounds={rounds} replaced={replaced} "
                  f"rewards={[round(x,2) for x in total]}", flush=True)
        group_std_zero = (r.view(n_groups, G).std(dim=1) < 1e-6)
        learnable = int((~group_std_zero).sum().item())

        met = self._metrics[mode]
        met.setdefault("reward", []).append(float(r.mean()))
        met.setdefault("reward_std", []).append(float(r.std()) if r.numel() > 1 else 0.0)
        met.setdefault("frac_reward_zero_std", []).append(float(group_std_zero.float().mean()))
        met.setdefault("filter/rounds", []).append(float(rounds))
        met.setdefault("filter/replaced_groups", []).append(float(replaced))
        met.setdefault("filter/learnable_groups", []).append(float(learnable))
        for name, vals in per_func.items():
            met.setdefault(f"rewards/{name}/mean", []).append(float(sum(vals) / len(vals)))
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--reward-set", choices=["base", "format"], required=True)
    parser.add_argument("--seed", type=int, default=None, help="覆盖 config 里的 seed")
    parser.add_argument("--tag", default=None, help="实验标签，决定日志/输出目录名")
    parser.add_argument("--zero-var-rounds", type=int, default=None, help="零方差组重采样轮数")
    parser.add_argument("--max-steps", type=int, default=None, help="覆盖 max_steps")
    args = parser.parse_args()
    cfg = load_config(args.config)
    g = cfg["grpo"]
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    rounds = args.zero_var_rounds if args.zero_var_rounds is not None else int(g.get("zero_var_rounds", 0))
    max_steps = args.max_steps if args.max_steps is not None else int(g["max_steps"])
    tag = args.tag or f"{args.reward_set}_s{seed}" + ("_filt" if rounds > 0 else "")

    model_dir = project_path(cfg["sft_merged"])
    if not os.path.isdir(model_dir):
        model_dir = get_model_path(cfg)
        print("[grpo] 警告：未找到 SFT 合并权重，直接以基座模型为起点")

    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset("json", data_files=project_path(cfg["data_dir"], "grpo_train.jsonl"), split="train")
    pool = [dict(row) for row in ds]
    print(f"[grpo:{tag}] 题库={len(pool)} seed={seed} zero_var_rounds={rounds} max_steps={max_steps}")

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    model.config.use_cache = False

    lora = LoraConfig(
        r=g["lora_r"],
        lora_alpha=g["lora_alpha"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    reward_funcs = [correctness_reward] if args.reward_set == "base" else [correctness_reward, format_reward]
    out_dir = project_path("experiments/checkpoints", tag)

    grpo_args = GRPOConfig(
        output_dir=out_dir,
        per_device_train_batch_size=g["batch_size"],
        gradient_accumulation_steps=1,
        num_generations=g["num_generations"],
        max_completion_length=g["max_completion_length"],
        learning_rate=g["lr"],
        max_steps=max_steps,
        temperature=g["temperature"],
        beta=g.get("beta", 0.0),
        loss_type=g.get("loss_type", "grpo"),
        scale_rewards="group",
        logging_steps=1,
        save_strategy="steps",
        save_steps=int(g.get("save_steps", 100)),
        save_total_limit=None,
        save_only_model=True,
        report_to="none",
        bf16=True,
        gradient_checkpointing=True,
        seed=seed,
        shuffle_dataset=True,
        log_completions=False,
    )

    log_path = project_path("experiments/logs", f"{tag}.jsonl")
    trainer = ZeroVarFilterGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=grpo_args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=lora,
        pool=pool,
        max_rounds=rounds,
        callbacks=[MetricDumpCallback(log_path)],
    )

    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer.train()
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    print(f"[grpo:{tag}] 训练完成 用时={elapsed:.1f}s 峰值显存={peak:.2f}GiB")

    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    meta = {
        "tag": tag,
        "seed": seed,
        "reward_set": args.reward_set,
        "zero_var_rounds": rounds,
        "max_steps": max_steps,
        "batch_size": g["batch_size"],
        "num_generations": g["num_generations"],
        "max_completion_length": g["max_completion_length"],
        "pool_size": len(pool),
        "prompt_visits": max_steps * g["batch_size"] // g["num_generations"],
        "train_seconds": round(elapsed, 1),
        "peak_gpu_gib": round(peak, 3),
    }
    with open(project_path("experiments/logs", f"{tag}.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(project_path("experiments/logs", f"{tag}_time.txt"), "w", encoding="utf-8") as f:
        f.write(f"train_seconds={elapsed:.1f}\nreward_set={args.reward_set}\n")


if __name__ == "__main__":
    main()
