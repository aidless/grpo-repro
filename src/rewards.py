"""可验证奖励函数（verifier-based rewards）。

- correctness_reward: 解析模型输出最终答案，与 GSM8K 标准答案比对，对=1 错=0。
- format_reward: R1 式格式约束，答案必须落在末行 \\boxed{} 内，合规 +0.5 / 不合规 -0.5。

trl 会以 reward = sum(各函数返回值) 作为总奖励（见 GRPOConfig.reward_weights）：
- 组 A（基准）：只用 correctness_reward  → 奖励 ∈ {0, 1}
- 组 B（格式约束）：correctness + format → 奖励 ∈ {-0.5, 0.5, 1.0, 1.5}
"""
from __future__ import annotations

from .common import answers_match, completion_to_text, extract_answer, format_score

# 训练过程中累积的统计，便于确认奖励确实在算而不是全 0
_STATS = {"total": 0, "correct": 0, "compliant": 0, "parse_fail": 0}


def reset_stats() -> None:
    for k in _STATS:
        _STATS[k] = 0


def get_stats() -> dict:
    return dict(_STATS)


def correctness_reward(completions, answer=None, **kwargs) -> list[float]:
    """对每条 completion 计 1/0。answer 为数据集里的标准答案列。"""
    rewards = []
    for i, completion in enumerate(completions):
        text = completion_to_text(completion)
        gold = answer[i] if answer is not None else ""
        pred, source = extract_answer(text)
        ok = answers_match(pred, str(gold))
        _STATS["total"] += 1
        _STATS["correct"] += int(ok)
        _STATS["parse_fail"] += int(source == "none")
        rewards.append(1.0 if ok else 0.0)
    return rewards


def format_reward(completions, **kwargs) -> list[float]:
    rewards = []
    for i, completion in enumerate(completions):
        text = completion_to_text(completion)
        score = format_score(text)
        _STATS["compliant"] += int(score > 0)
        rewards.append(score)
    return rewards
