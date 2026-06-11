"""GPU-free unit tests for the prompt/answer logic.

Run with:  pytest -q
These exercise the pure functions that the whole pipeline depends on, so a
regression in formatting or answer parsing is caught without a training run.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import (  # noqa: E402
    LETTERS,
    build_messages,
    build_question_block,
    build_target,
    extract_answer_letter,
    merge_system_into_user,
    normalize_example,
)

RAW_DICT_EXAMPLE = {
    "question": "Which vitamin deficiency causes scurvy?",
    "options": {"A": "Vitamin A", "B": "Vitamin C", "C": "Vitamin D", "D": "Vitamin K"},
    "answer": "Vitamin C",
    "answer_idx": "B",
}


def test_normalize_dict_options():
    norm = normalize_example(RAW_DICT_EXAMPLE)
    assert norm["answer_letter"] == "B"
    assert norm["answer_text"] == "Vitamin C"
    assert norm["options"]["C"] == "Vitamin D"


def test_normalize_recovers_letter_from_text_when_idx_missing():
    ex = {k: v for k, v in RAW_DICT_EXAMPLE.items() if k != "answer_idx"}
    norm = normalize_example(ex)
    assert norm["answer_letter"] == "B"


def test_normalize_list_options():
    ex = {
        "question": "Q?",
        "options": ["first", "second", "third", "fourth"],
        "answer": "third",
    }
    norm = normalize_example(ex)
    assert norm["options"]["A"] == "first"
    assert norm["answer_letter"] == "C"  # matched by text


def test_question_block_lists_all_options_and_instruction():
    block = build_question_block(RAW_DICT_EXAMPLE["question"], RAW_DICT_EXAMPLE["options"])
    for letter in ["A", "B", "C", "D"]:
        assert f"({letter})" in block
    assert "The answer is" in block  # instruction present


def test_build_target_format():
    assert build_target("B", "Vitamin C") == "The answer is (B) Vitamin C"


def test_build_messages_training_includes_assistant():
    msgs = build_messages(RAW_DICT_EXAMPLE, include_answer=True)
    assert msgs[-1]["role"] == "assistant"
    assert "(B)" in msgs[-1]["content"]


def test_build_messages_eval_omits_assistant():
    msgs = build_messages(RAW_DICT_EXAMPLE, include_answer=False)
    assert all(m["role"] != "assistant" for m in msgs)


# ---- answer extraction ------------------------------------------------

import pytest  # noqa: E402


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The answer is (C) Vitamin D", "C"),
        ("The answer is B because ...", "B"),
        ("answer: A", "A"),
        ("(D) potassium", "D"),
        ("C. This is the best option", "C"),
        ("e) something lowercase", "E"),
        ("I am not sure about this one", None),
        ("", None),
    ],
)
def test_extract_answer_letter(text, expected):
    assert extract_answer_letter(text) == expected


def test_extract_prefers_explicit_answer_phrase_over_stray_letter():
    # A stray "A" appears first, but the explicit phrase should win.
    text = "A patient presents... The answer is (D) heart failure."
    assert extract_answer_letter(text) == "D"


def test_letters_constant():
    assert LETTERS[:4] == ["A", "B", "C", "D"]


# ---- system-message merging (Mistral compatibility) -------------------

def test_merge_system_into_user_folds_into_following_user():
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
    ]
    merged = merge_system_into_user(msgs)
    assert all(m["role"] != "system" for m in merged)
    assert merged[0]["role"] == "user"
    assert merged[0]["content"] == "SYS\n\nQ"
    assert merged[1] == {"role": "assistant", "content": "A"}


def test_merge_system_into_user_noop_without_system():
    msgs = [{"role": "user", "content": "Q"}]
    assert merge_system_into_user(msgs) == msgs


def test_merge_system_into_user_does_not_mutate_input():
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "Q"}]
    merge_system_into_user(msgs)
    assert msgs[0]["role"] == "system"  # original list untouched
