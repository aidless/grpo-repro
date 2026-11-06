"""公共工具：配置读取、提示词构造、答案抽取与判分。

判分口径（训练奖励与评测共用同一套函数，保证两者一致）：
1. 从模型输出中抽取最终答案：优先 \\boxed{...}，其次 "#### xxx"，再次 "answer is xxx"，最后回退到末行最后一个数字。
2. 数值归一化后与 GSM8K 标准答案比较（支持整数/小数/分数/百分号/千分位逗号）。
"""
from __future__ import annotations

import json
import os
import re
from fractions import Fraction

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SYSTEM_PROMPT = (
    "You are a math problem solver. Solve the problem step by step in English. "
    "Put the final numeric answer inside \\boxed{} on the last line."
)

BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
HASH_RE = re.compile(r"####\s*([^\n]*)")
ANSWER_IS_RE = re.compile(r"answer\s*(?:is|=|:)\s*([^\n]*)", re.IGNORECASE)
NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*(?:/\d+)?")
CALC_RE = re.compile(r"<<[^>]*>>")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = os.path.abspath(path)
    return cfg


def project_path(*parts: str) -> str:
    return os.path.join(PROJECT_ROOT, *parts)


def get_model_path(cfg: dict | None = None) -> str:
    """优先使用环境变量 GRPO_MODEL_PATH；否则从 HF（可用 HF_ENDPOINT 镜像）拉取。"""
    env = os.environ.get("GRPO_MODEL_PATH")
    if env:
        return env
    if cfg and cfg.get("model_path") and os.path.isdir(cfg["model_path"]):
        return cfg["model_path"]
    from huggingface_hub import snapshot_download

    name = (cfg or {}).get("model", "Qwen/Qwen2.5-0.5B-Instruct")
    return snapshot_download(
        name,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt", "tokenizer*", "vocab*"],
    )


def build_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def clean_solution(gsm8k_answer: str) -> str:
    """把 GSM8K 的解答转成 SFT 目标文本：去掉计算器标注 <<...>>，#### 换成 \\boxed{}。"""
    text = CALC_RE.sub("", gsm8k_answer).strip()
    if "####" in text:
        reasoning, final = text.rsplit("####", 1)
        final = final.strip()
        text = f"{reasoning.strip()}\n\n\\boxed{{{final}}}"
    return text


def completion_to_text(completion) -> str:
    """统一把 completion 还原成纯文本。

    trl 在 conversational prompt 下会把生成结果包成 [{"role": "assistant", "content": ...}]，
    评测时则是普通字符串；两者都必须走同一套判分，否则格式检查会永远失败。
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, (list, tuple)):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(completion)


def extract_answer(text: str) -> tuple[str | None, str]:
    """返回 (答案字符串, 抽取来源)。来源用于统计解析失败率。"""
    matches = BOXED_RE.findall(text)
    if matches:
        return matches[-1].strip(), "boxed"

    m = HASH_RE.search(text)
    if m and m.group(1).strip():
        return m.group(1).strip(), "hash"

    matches = ANSWER_IS_RE.findall(text)
    if matches:
        cleaned = matches[-1].strip().rstrip(".")
        if cleaned:
            return cleaned, "answer_is"

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if lines:
        nums = NUM_RE.findall(lines[-1])
        if nums:
            return nums[-1], "fallback_last_number"
    return None, "none"


def _to_fraction(value: str) -> Fraction | None:
    s = value.strip().replace(",", "").replace("$", "").replace("%", "").strip()
    s = s.rstrip(".").strip()
    if not s:
        return None
    try:
        if "/" in s and re.fullmatch(r"[-+]?\d+\s*/\s*\d+", s):
            return Fraction(s.replace(" ", ""))
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        pass
    # 兜底：答案文本里混了单位（如 "1234 dollars"）时，取第一个数字
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", s)
    if m:
        try:
            return Fraction(m.group(0))
        except (ValueError, ZeroDivisionError):
            return None
    return None


def answers_match(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    p = _to_fraction(pred)
    g = _to_fraction(gold)
    if p is not None and g is not None:
        if p == g:
            return True
        # 浮点容差（如 0.3333 vs 1/3 这类不相等时不做近似，保持严格）
        try:
            return abs(float(p) - float(g)) <= 1e-4 * max(1.0, abs(float(g)))
        except (OverflowError, ZeroDivisionError):
            return False
    return pred.strip().lower() == gold.strip().lower()


def is_format_compliant(text: str) -> bool:
    """格式合规：最后一个非空行必须恰好是 \\boxed{...} 形式。"""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    last = lines[-1].rstrip(".").strip()
    return bool(re.fullmatch(r"\\boxed\{[^{}]+\}", last))


def format_score(text: str) -> float:
    """R1 式格式分：合规 +0.5，不合规 -0.5（与 correctness 相加构成总奖励）。"""
    return 0.5 if is_format_compliant(text) else -0.5


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
