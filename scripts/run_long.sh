#!/usr/bin/env bash
# 长程实验批量执行：全量 prompt vs 零方差过滤，双 seed。
# 默认复现已交付的 3 条曲线（s42 全量 / s42 过滤 / s7 全量）；
# 想补第二个 seed 的过滤组，用 FILTER_SEEDS="42 7" bash scripts/run_long.sh（约多 2 小时）。
# 每一步都可单独复算：所有 run 的日志都在 experiments/logs/<tag>.jsonl。
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

CFG="${CFG:-configs/long.yaml}"
STEPS="${STEPS:-300}"
SEEDS="${SEEDS:-42 7}"
FILTER_SEEDS="${FILTER_SEEDS:-42}"
ROUNDS="${ROUNDS:-2}"

wait_gpu() {  # 阈值 3.0GB：本实验峰值仅 ~2.2GB，可与小模型共存
  for _ in $(seq 1 240); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$u" -lt "${GPU_FREE_MIB:-3000}" ]; then return 0; fi
    echo "[wait] GPU 占用 ${u}MiB，等待 10s..."
    sleep 10
  done
  echo "[wait] GPU 一直不空闲，放弃"; return 1
}

run_one() {  # $1=seed $2=rounds $3=tag
  local seed=$1 rounds=$2 tag=$3
  if [ -f "experiments/logs/${tag}.meta.json" ]; then
    echo "[skip] $tag 已完成"
    return 0
  fi
  for attempt in 1 2; do
    wait_gpu || return 1
    echo "== $tag seed=$seed zero_var_rounds=$rounds steps=$STEPS attempt=$attempt $(date +%H:%M:%S) =="
    python3 -u -m src.train_grpo --config "$CFG" --reward-set base \
      --seed "$seed" --tag "$tag" --max-steps "$STEPS" --zero-var-rounds "$rounds" \
      2>&1 | tee "experiments/logs/${tag}.out"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 0 ] && [ -f "experiments/logs/${tag}.meta.json" ]; then
      echo "[ok] $tag"
      return 0
    fi
    echo "[fail] $tag rc=$rc attempt=$attempt"
  done
  return 0
}

for seed in $SEEDS; do
  run_one "$seed" 0 "long_base_s${seed}"
  case " $FILTER_SEEDS " in
    *" $seed "*) run_one "$seed" "$ROUNDS" "long_base_s${seed}_filt" ;;
  esac
done

echo "== 全部 run 结束 $(date +%H:%M:%S) =="
