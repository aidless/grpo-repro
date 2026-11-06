"""Offline unit tests for the shared verifier (no GPU, no model download).

These pin the exact judging semantics that both training rewards and the final
metrics depend on, so a future edit cannot silently move the reported numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (  # noqa: E402
    answers_match,
    clean_solution,
    completion_to_text,
    extract_answer,
    is_format_compliant,
)


class TestExtractAnswer:
    def test_prefers_last_boxed(self):
        text = "first \\boxed{3}, then \\boxed{7}"
        assert extract_answer(text) == ("7", "boxed")

    def test_hash_fallback(self):
        assert extract_answer("work\n#### 42") == ("42", "hash")

    def test_answer_is_fallback(self):
        assert extract_answer("so the answer is 18.")[0] == "18"

    def test_fallback_last_number(self):
        assert extract_answer("I get 12 apples") == ("12", "fallback_last_number")

    def test_none_when_no_number(self):
        assert extract_answer("no idea") == (None, "none")


class TestAnswersMatch:
    @pytest.mark.parametrize(
        "pred,gold",
        [
            ("490", "490"),
            ("1/3", "1/3"),
            ("0.50", "1/2"),
            ("$1,234", "1234"),
            ("45%", "45"),
            ("12 dollars", "12"),
            (" 7 ", "7"),
        ],
    )
    def test_equivalent_forms(self, pred, gold):
        assert answers_match(pred, gold)

    @pytest.mark.parametrize("pred,gold", [("290", "490"), ("1/3", "0.333"), (None, "5")])
    def test_non_equivalent(self, pred, gold):
        assert not answers_match(pred, gold)


class TestFormatCompliance:
    def test_requires_boxed_last_line(self):
        assert is_format_compliant("reasoning\n\\boxed{5}")

    def test_rejects_text_after_box(self):
        assert not is_format_compliant("\\boxed{5}\nmore text")

    def test_rejects_unboxed_number(self):
        assert not is_format_compliant("the answer is 5")


class TestCompletionToText:
    def test_plain_string(self):
        assert completion_to_text("hi") == "hi"

    def test_conversational_list(self):
        assert completion_to_text([{"role": "assistant", "content": "hi"}]) == "hi"

    def test_dict(self):
        assert completion_to_text({"content": "hi"}) == "hi"


def test_clean_solution_strips_calculator_and_boxes_answer():
    raw = "She earns 10 <<10*2=20>> dollars per day.\n#### 20"
    cleaned = clean_solution(raw)
    assert "<<" not in cleaned
    assert cleaned.endswith("\\boxed{20}")
