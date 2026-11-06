#!/usr/bin/env bash
# 900 步实验收口：先补 base 的 checkpoint 评测（便宜、马上有 base 支曲线），
# 再跑过滤组 900 步，再评测过滤组，最后汇总 + 双轴图。
# 每条评测按文件名跳过，可断点续跑；训练失败就明确停在该步，不静默重试。
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

wait_for_run() {  # $1=tag $2=上限小时（默认 8h）
  local tag="$1" cap_h="${2:-8}"
  local deadline=$(( $(date +%s) + cap_h * 3600 ))
  while true; do
    if [ -f "experiments/logs/${tag}.meta.json" ]; then
      echo "[chain] ${tag} meta 已就绪 $(date +%H:%M:%S)"
      return 0
    fi
    if ! pgrep -f "tag ${tag}" > /dev/null; then
      sleep 30   # 进程退出后给 meta 落盘留时间
      if [ -f "experiments/logs/${tag}.meta.json" ]; then
        echo "[chain] ${tag} meta 已就绪（进程退出后） $(date +%H:%M:%S)"
        return 0
      fi
      echo "[chain] ${tag} 进程消失但没有 meta，视为异常退出"
      return 1
    fi
    if [ "$(date +%s)" -ge "${deadline}" ]; then
      echo "[chain] ${tag} 等待超过 ${cap_h}h，停止"
      return 1
    fi
    sleep 30
  done
}

echo "== [1/4] 评测 ep900_base：9 个 checkpoint n=100 + 最终 n=200 $(date +%H:%M:%S) =="
python3 scripts/eval_checkpoints.py --config configs/epoch900.yaml --tag ep900_base \
  --limit 100 --final-limit 200 --out-jsonl experiments/eval/ep900_base_ckpt.jsonl \
  || { echo "[chain] base checkpoint 评测失败"; exit 1; }

if [ -f experiments/logs/ep900_base_filt.meta.json ]; then
  echo "== [2/4] ep900_base_filt 已完成，跳过训练 $(date +%H:%M:%S) =="
else
  echo "== [2/4] 训练 ep900_base_filt：900 步，零方差过滤 2 轮 $(date +%H:%M:%S) =="
  if ! python3 -m src.train_grpo --config configs/epoch900.yaml --reward-set base \
      --tag ep900_base_filt --zero-var-rounds 2 > experiments/logs/ep900_base_filt.out 2>&1; then
    echo "[chain] 过滤组训练失败，见 experiments/logs/ep900_base_filt.out；后续评测不执行"
    exit 1
  fi
  if [ ! -f experiments/logs/ep900_base_filt.meta.json ]; then
    echo "[chain] 过滤组训练结束但没有 meta，视为未完成；后续评测不执行"
    exit 1
  fi
fi

echo "== [3/4] 评测 ep900_base_filt：9 个 checkpoint n=100 + 最终 n=200 $(date +%H:%M:%S) =="
python3 scripts/eval_checkpoints.py --config configs/epoch900.yaml --tag ep900_base_filt \
  --limit 100 --final-limit 200 --out-jsonl experiments/eval/ep900_base_filt_ckpt.jsonl \
  || { echo "[chain] 过滤组 checkpoint 评测失败"; exit 1; }

echo "== [4/4] 汇总与画图 $(date +%H:%M:%S) =="
python3 -m src.analyze_ep900 > experiments/logs/ep900_summary.out 2>&1 || { echo "[chain] analyze_ep900 失败"; exit 1; }
python3 -m src.plot_reward_eval || { echo "[chain] plot_reward_eval 失败"; exit 1; }
echo "== 全部完成 $(date +%H:%M:%S) =="
