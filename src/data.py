"""Dataset loading and prompt formatting for MedQA.

The pure string functions here (build_question_block, build_target,
extract_answer_letter, normalize_example) have **no GPU/network
dependency** and are covered by tests/test_data.py so the project's core
logic is verifiable without a training run.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

LETTERS = ["A", "B", "C", "D", "E"]

SYSTEM_PROMPT = (
    "You are a medical expert answering USMLE-style multiple-choice "
    "questions. Read the question and options, then give the single best "
    "answer."
)

ANSWER_INSTRUCTION = (
    'Respond with the correct option in the form "The answer is (X)" '
    "followed by a one-sentence justification."
)


def normalize_example(example: Dict) -> Dict:
    """Coerce a raw MedQA row into a stable schema.

    The HF dataset stores ``options`` as a dict mapping letter -> text and
    the gold answer under ``answer_idx`` (a letter). We defend against minor
    schema variations (list options, missing answer_idx) so the rest of the
    pipeline can assume:

        {"question": str, "options": {"A": str, ...}, "answer_letter": "C"}
    """
    question = example["question"].strip()

    raw_options = example["options"]
    if isinstance(raw_options, dict):
        options = {k.strip().upper(): v.strip() for k, v in raw_options.items()}
    else:  # list-like -> assign letters positionally
        options = {LETTERS[i]: str(v).strip() for i, v in enumerate(raw_options)}

    answer_letter = example.get("answer_idx")
    if answer_letter is None:
        # Fall back: match the gold answer *text* to an option.
        gold_text = str(example.get("answer", "")).strip()
        answer_letter = next(
            (letter for letter, text in options.items() if text == gold_text),
            None,
        )
    if answer_letter is not None:
        answer_letter = answer_letter.strip().upper()

    return {
        "question": question,
        "options": options,
        "answer_letter": answer_letter,
        "answer_text": options.get(answer_letter, "") if answer_letter else "",
    }


def build_question_block(question: str, options: Dict[str, str]) -> str:
    """Render the user-facing question + lettered options + instruction."""
    lines = [question, ""]
    for letter in LETTERS:
        if letter in options:
            lines.append(f"({letter}) {options[letter]}")
    lines.append("")
    lines.append(ANSWER_INSTRUCTION)
    return "\n".join(lines)


def build_target(answer_letter: str, answer_text: str) -> str:
    """The assistant's gold completion we train the model to produce."""
    return f"The answer is ({answer_letter}) {answer_text}".strip()


def build_messages(example: Dict, include_answer: bool) -> List[Dict[str, str]]:
    """Chat-format messages for a normalized example.

    ``include_answer=True`` appends the gold assistant turn (training);
    ``False`` leaves it open for generation (evaluation/inference).
    """
    norm = example if "answer_letter" in example and "options" in example else normalize_example(example)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_question_block(norm["question"], norm["options"])},
    ]
    if include_answer:
        messages.append(
            {"role": "assistant", "content": build_target(norm["answer_letter"], norm["answer_text"])}
        )
    return messages


_LETTER_RE = re.compile(r"\(?\s*([A-E])\s*\)?", re.IGNORECASE)


def extract_answer_letter(text: str) -> Optional[str]:
    """Pull the predicted option letter out of a model generation.

    Strategy, in priority order:
      1. The phrase "answer is (X)" / "answer: X".
      2. The first standalone A-E letter near the start of the text.
    Returns an uppercase letter or None if nothing parseable is found.
    """
    if not text:
        return None

    # 1) Explicit "answer is X" pattern.
    m = re.search(r"answer\s*(?:is|:)?\s*\(?\s*([A-E])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 2) First "(X)" style token.
    m = re.search(r"\(\s*([A-E])\s*\)", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 3) A leading bare letter, e.g. "C. Because ...".
    m = re.match(r"\s*([A-E])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return None


def load_medqa(config) -> "DatasetDict":  # noqa: F821 - datasets imported lazily
    """Load MedQA and normalize every split. Imports `datasets` lazily so
    importing this module (e.g. for tests) does not require the heavy dep."""
    from datasets import load_dataset

    raw = load_dataset(config.dataset_name)
    normalized = raw.map(normalize_example, remove_columns=raw["train"].column_names)
    # Drop any rows we could not assign a gold letter to.
    normalized = normalized.filter(lambda ex: ex["answer_letter"] in LETTERS)
    return normalized
