#!/usr/bin/env bash
# 抢显存窗口 → 冒烟校验 → 长程四连跑。
# 背景执行：bash scripts/bootstrap_long.sh > experiments/logs/bootstrap.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

wait_gpu() {  # 等到显存低于 1.2GB
  for i in $(seq 1 720); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$u" -lt 1200 ]; then echo "[gpu] free ${u}MiB $(date +%H:%M:%S)"; return 0; fi
    [ $((i % 6)) -eq 0 ] && echo "[gpu] busy ${u}MiB $(date +%H:%M:%S)"
    sleep 10
  done
  return 1
}

wait_gpu || exit 4

# 1) 5 步冒烟：验证过滤实现不会崩（base 通路 + 过滤通路）
python3 -u -m src.train_grpo --config configs/long.yaml --reward-set base \
  --tag smoke_5step_off --max-steps 5 > experiments/logs/smoke_5step_off.out 2>&1
echo "[smoke] off rc=$?"
python3 -u -m src.train_grpo --config configs/long.yaml --reward-set base \
  --tag smoke_5step_filt --max-steps 5 --zero-var-rounds 2 > experiments/logs/smoke_5step_filt.out 2>&1
echo "[smoke] filt rc=$?"
grep -E "训练完成|Error|Traceback" experiments/logs/smoke_5step_filt.out | tail -3

# 2) 长程四连跑
STEPS="${STEPS:-300}" SEEDS="${SEEDS:-42 7}" ROUNDS="${ROUNDS:-2}" bash scripts/run_long.sh
