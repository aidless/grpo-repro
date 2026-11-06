#!/usr/bin/env bash
# 一键复现（RTX 3060 6GB 实测约 1.5-2 小时）。无 GPU 时把 CFG 换成 configs/smoke.yaml 即可跑通链路。
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"   # 国内网络走镜像，海外可去掉
CFG="${CFG:-configs/full.yaml}"
M="${GRPO_MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"

echo "== 0. 环境自检 =="
python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY

echo "== 1. 数据准备 =="
python3 -m src.prepare_data --config "$CFG"

echo "== 2. LoRA SFT 打底 =="
python3 -m src.train_sft --config "$CFG"

echo "== 3. 评测：基座 / SFT 后 =="
python3 -m src.evaluate --config "$CFG" --name base --model-path "$M"
python3 -m src.evaluate --config "$CFG" --name sft --model-path experiments/checkpoints/sft_merged

echo "== 4. GRPO 组 A：只奖励 correctness =="
python3 -m src.train_grpo --config "$CFG" --reward-set base

echo "== 5. GRPO 组 B：correctness + R1 格式约束 =="
python3 -m src.train_grpo --config "$CFG" --reward-set format

echo "== 6. 评测：GRPO 两组 =="
python3 -m src.evaluate --config "$CFG" --name grpo_base --model-path experiments/checkpoints/grpo_base
python3 -m src.evaluate --config "$CFG" --name grpo_format --model-path experiments/checkpoints/grpo_format

echo "== 7. 画图 =="
python3 -m src.plot_curves

echo "== 完成，结果在 experiments/ =="
