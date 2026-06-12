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
    apply_chat_template,
    build_cot_target,
    build_messages,
    build_question_block,
    build_target,
    clean_explanation,
    extract_answer_letter,
    merge_system_into_user,
    normalize_example,
    normalize_medmcqa,
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


# ---- apply_chat_template output normalization (transformers drift) ----

class _StubTokenizer:
    """Mimics apply_chat_template, returning a configurable shape so we can
    verify normalization without downloading a real tokenizer."""

    def __init__(self, mode):
        self.mode = mode

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        if not tokenize:
            return "RENDERED_PROMPT"
        ids = [1, 2, 3]
        if self.mode == "list":
            return ids
        if self.mode == "batched":
            return [ids]
        if self.mode == "dict":
            return {"input_ids": ids, "attention_mask": [1, 1, 1]}
        if self.mode == "batchencoding":
            class BE:
                input_ids = ids
            return BE()
        raise AssertionError(self.mode)


@pytest.mark.parametrize("mode", ["list", "batched", "dict", "batchencoding"])
def test_apply_chat_template_normalizes_to_flat_list(mode):
    tok = _StubTokenizer(mode)
    out = apply_chat_template(tok, [{"role": "user", "content": "Q"}],
                              tokenize=True, add_generation_prompt=True)
    assert out == [1, 2, 3]
    # critically, it must support list concatenation (the bug we hit)
    assert out + [99] == [1, 2, 3, 99]


def test_apply_chat_template_passes_string_through_when_not_tokenizing():
    tok = _StubTokenizer("list")
    out = apply_chat_template(tok, [{"role": "user", "content": "Q"}],
                              tokenize=False, add_generation_prompt=True)
    assert out == "RENDERED_PROMPT"


# ---- MedMCQA normalization (0-indexed cop) ----------------------------

RAW_MEDMCQA = {
    "question": "Scurvy is caused by deficiency of?",
    "opa": "Vitamin A", "opb": "Vitamin C", "opc": "Vitamin D", "opd": "Vitamin K",
    "cop": 1,  # 0-indexed -> B
    "exp": "Ans-b. Vitamin C deficiency causes scurvy via impaired collagen synthesis.",
    "choice_type": "single",
}


def test_normalize_medmcqa_cop_is_zero_indexed():
    norm = normalize_medmcqa(RAW_MEDMCQA)
    assert norm["answer_letter"] == "B"          # cop=1 -> B, NOT C
    assert norm["answer_text"] == "Vitamin C"
    assert norm["options"]["A"] == "Vitamin A"
    assert norm["choice_type"] == "single"
    assert "collagen" in norm["explanation"]


def test_normalize_medmcqa_cop_zero_is_A():
    ex = dict(RAW_MEDMCQA, cop=0)
    assert normalize_medmcqa(ex)["answer_letter"] == "A"


def test_normalize_medmcqa_bad_cop_returns_none():
    ex = dict(RAW_MEDMCQA, cop=None)
    assert normalize_medmcqa(ex)["answer_letter"] is None


# ---- CoT target building + explanation cleaning -----------------------

def test_clean_explanation_strips_leading_answer_prefix():
    assert clean_explanation("Ans-b. Vitamin C is correct.").lower().startswith("vitamin c")
    assert clean_explanation("Answer: C - because reasons").lower().startswith("because")


def test_build_cot_target_appends_canonical_answer_line():
    target = build_cot_target("Vitamin C prevents scurvy.", "B", "Vitamin C")
    assert target.startswith("Vitamin C prevents scurvy.")
    assert target.strip().endswith("The answer is (B) Vitamin C")


def test_build_cot_target_without_explanation_is_just_answer_line():
    assert build_cot_target("", "B", "Vitamin C") == "The answer is (B) Vitamin C"


def test_build_messages_cot_uses_cot_instruction_and_reasoning():
    msgs = build_messages(normalize_medmcqa(RAW_MEDMCQA), include_answer=True, cot=True)
    assert "step by step" in msgs[1]["content"].lower()
    # assistant target carries reasoning then the answer line
    assert "collagen" in msgs[-1]["content"]
    assert msgs[-1]["content"].strip().endswith("The answer is (B) Vitamin C")


def test_extract_answer_letter_takes_last_in_cot():
    cot = ("Option A is unlikely. The answer is (B) seems plausible but on "
           "reflection, the answer is (C) potassium.")
    assert extract_answer_letter(cot) == "C"
