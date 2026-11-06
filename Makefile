PY ?= python3
CFG ?= configs/full.yaml
LONG_CFG ?= configs/long.yaml
LONG_TAGS ?= long_base_s42,long_base_s42_filt,long_base_s7,long_base_s7_filt
EP900_CFG ?= configs/epoch900.yaml
EP900_TAGS ?= ep900_base,ep900_base_filt,ep900_s7_base,ep900_s7_filt

.DEFAULT_GOAL := verify

.PHONY: verify verify-long verify-ep900 overlap eval report summary summary-ep900 plot-ep900 long test repro

# 快速复核（默认 40 条，约 3 分钟）：base / sft / 60步 GRPO 两组
verify:
	$(PY) scripts/verify.py --config $(CFG) --limit 40

# 快速复核长程实验的 checkpoint（40 条）
verify-long:
	$(PY) scripts/verify.py --config $(LONG_CFG) --limit 40 --stages "" --extra-tags $(LONG_TAGS)

# 快速复核多 epoch（1,100 题库 / 900 步）实验；可直接覆写：
#   make verify-long LONG_CFG=$(EP900_CFG) LONG_TAGS=$(EP900_TAGS)
verify-ep900:
	$(PY) scripts/verify.py --config $(EP900_CFG) --limit 40 --stages "" --extra-tags $(EP900_TAGS)

# 训练/评测是否重叠（索引级 + 文本级），不通过则退出码非 0
overlap:
	$(PY) scripts/check_overlap.py --config $(EP900_CFG)

# 正式评测：全量测试子集（200 条），含长程 run
eval:
	$(PY) scripts/verify.py --config $(CFG)
	$(PY) scripts/verify.py --config $(LONG_CFG) --stages "" --extra-tags $(LONG_TAGS)

# 从真实日志重算长程汇总表（零方差比例 / 有效梯度密度 / 过滤提升）
summary:
	$(PY) -m src.analyze_long

# 从真实日志重算 900 步 / 3.3 epoch 汇总表（含每 100 步评测与 McNemar）
summary-ep900:
	$(PY) -m src.analyze_ep900

# 奖励（左轴）与评测准确率（右轴）同图：experiments/curves/reward_vs_eval.png
plot-ep900:
	$(PY) -m src.plot_reward_eval

# 从真实日志重生所有图
report:
	$(PY) -m src.plot_curves

# 长程实验批量跑：默认 3 条（s42 全量 / s42 过滤 / s7 全量）；FILTER_SEEDS="42 7" 可补第四个
long:
	bash scripts/run_long.sh

# CPU-only 离线单测
test:
	$(PY) -m pytest -q

# 一轮完整复现（SFT 3000 → GRPO 60 步两组 → 绘图）
repro:
	bash scripts/run_all.sh
