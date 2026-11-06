#!/usr/bin/env bash
# seed 7 · 全量 prompt：训练 900 步 → 9×n=100 + 最终 n=200 评测 → 汇总 + 画图。
# 训练已在别的进程占卡时不要跑；本脚本只负责这一条 run。
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

TAG=ep900_s7_base
if [ -f "experiments/logs/${TAG}.meta.json" ]; then
  echo "[s7] ${TAG} 已完成，跳过训练"
else
  echo "== [1/2] 训练 ${TAG}（seed 7，全量 prompt） $(date +%H:%M:%S) =="
  if ! python3 -m src.train_grpo --config configs/epoch900.yaml --reward-set base \
      --seed 7 --tag "${TAG}" > "experiments/logs/${TAG}.out" 2>&1; then
    echo "[s7] 训练失败，见 experiments/logs/${TAG}.out；不执行评测"
    exit 1
  fi
fi

echo "== [2/2] 评测 ${TAG}：9 个 checkpoint n=100 + 最终 n=200 $(date +%H:%M:%S) =="
python3 scripts/eval_checkpoints.py --config configs/epoch900.yaml --tag "${TAG}" \
  --limit 100 --final-limit 200 --out-jsonl "experiments/eval/${TAG}_ckpt.jsonl" \
  || { echo "[s7] 评测失败"; exit 1; }

python3 -m src.analyze_ep900 > experiments/logs/ep900_summary.out 2>&1 || echo "[s7] analyze 失败"
python3 -m src.plot_reward_eval || echo "[s7] plot 失败"
echo "== ${TAG} 完成 $(date +%H:%M:%S) =="
