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

# --- Chain-of-thought (CoT) variants ----------------------------------
COT_SYSTEM_PROMPT = (
    "You are a medical expert answering multiple-choice questions. Reason "
    "carefully through the clinical details before deciding."
)

COT_INSTRUCTION = (
    "Briefly explain your reasoning in 2-4 sentences, then write a final "
    'line in the exact form "The answer is (X)" where X is the correct '
    "option letter."
)

# Keep CoT targets concise so the model learns to reach its answer within a
# small generation budget (long targets -> slow eval + truncated answers).
COT_MAX_SENTENCES = 3
COT_MAX_CHARS = 400


def normalize_medmcqa(example: Dict) -> Dict:
    """Coerce a raw MedMCQA row into the same schema as MedQA.

    MedMCQA stores options as separate ``opa..opd`` fields and the gold
    answer as ``cop``, a **0-indexed** integer (0->A, 1->B, ...), verified
    empirically. It also ships an ``exp`` explanation we use as the CoT
    rationale. Output schema adds ``explanation`` and ``choice_type``:

        {"question", "options"{A..D}, "answer_letter", "answer_text",
         "explanation", "choice_type"}
    """
    options = {
        "A": (example.get("opa") or "").strip(),
        "B": (example.get("opb") or "").strip(),
        "C": (example.get("opc") or "").strip(),
        "D": (example.get("opd") or "").strip(),
    }
    cop = example.get("cop")
    answer_letter = LETTERS[cop] if isinstance(cop, int) and 0 <= cop < 4 else None
    return {
        "question": (example.get("question") or "").strip(),
        "options": options,
        "answer_letter": answer_letter,
        "answer_text": options.get(answer_letter, "") if answer_letter else "",
        "explanation": (example.get("exp") or "").strip(),
        "choice_type": example.get("choice_type", "single"),
    }


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


def build_cot_question_block(question: str, options: Dict[str, str]) -> str:
    """Question + options + the *chain-of-thought* instruction."""
    lines = [question, ""]
    for letter in LETTERS:
        if letter in options:
            lines.append(f"({letter}) {options[letter]}")
    lines.append("")
    lines.append(COT_INSTRUCTION)
    return "\n".join(lines)


def build_target(answer_letter: str, answer_text: str) -> str:
    """The assistant's gold completion we train the model to produce."""
    return f"The answer is ({answer_letter}) {answer_text}".strip()


def clean_explanation(explanation: str) -> str:
    """Tidy a raw MedMCQA ``exp`` rationale for use as a CoT trace.

    Strips a leading "Ans-a"/"Answer: B"-style prefix (the gold letter is
    appended separately as the final line) and collapses whitespace.
    """
    text = (explanation or "").strip()
    text = re.sub(r"^\s*(ans(?:wer)?[\s\-:.]*\(?[a-eA-E]\)?[\s\-:.]*)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _truncate_reasoning(text: str, max_sentences: int = COT_MAX_SENTENCES,
                        max_chars: int = COT_MAX_CHARS) -> str:
    """Keep CoT reasoning short: first few sentences, hard char cap."""
    text = (text or "").strip()
    if not text:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out = " ".join(sentences[:max_sentences]).strip()
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0].rstrip()
    return out


def build_cot_target(explanation: str, answer_letter: str, answer_text: str) -> str:
    """Gold CoT completion: concise reasoning then the canonical answer line."""
    reasoning = _truncate_reasoning(clean_explanation(explanation))
    answer_line = f"The answer is ({answer_letter}) {answer_text}".strip()
    return f"{reasoning}\n\n{answer_line}".strip() if reasoning else answer_line


def build_messages(example: Dict, include_answer: bool, cot: bool = False) -> List[Dict[str, str]]:
    """Chat-format messages for a normalized example.

    ``include_answer=True`` appends the gold assistant turn (training);
    ``False`` leaves it open for generation (evaluation/inference).
    ``cot=True`` uses the chain-of-thought system prompt, instruction, and
    (for training) a reasoning-augmented target built from ``explanation``.
    """
    norm = example if "answer_letter" in example and "options" in example else normalize_example(example)
    if cot:
        system = COT_SYSTEM_PROMPT
        user = build_cot_question_block(norm["question"], norm["options"])
    else:
        system = SYSTEM_PROMPT
        user = build_question_block(norm["question"], norm["options"])
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if include_answer:
        if cot:
            content = build_cot_target(
                norm.get("explanation", ""), norm["answer_letter"], norm["answer_text"]
            )
        else:
            content = build_target(norm["answer_letter"], norm["answer_text"])
        messages.append({"role": "assistant", "content": content})
    return messages


def merge_system_into_user(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Fold any system message into the following user turn.

    Some chat templates (notably Mistral-7B-Instruct) reject a ``system``
    role. This produces an equivalent message list without one, so the same
    prompts work across Mistral / Llama / Qwen.
    """
    system_txt = ""
    out: List[Dict[str, str]] = []
    for m in messages:
        if m["role"] == "system":
            system_txt = m["content"]
            continue
        if m["role"] == "user" and system_txt:
            out.append({"role": "user", "content": f"{system_txt}\n\n{m['content']}"})
            system_txt = ""
        else:
            out.append(dict(m))
    if system_txt:  # no user turn followed it; keep as a leading user message
        out.insert(0, {"role": "user", "content": system_txt})
    return out


def apply_chat_template(tokenizer, messages, tokenize: bool, add_generation_prompt: bool):
    """Apply the tokenizer's chat template, retrying without a system role
    if the template rejects one. Centralizes template handling so train,
    eval, and inference all behave identically across model families.

    When ``tokenize=True`` the result is normalized to a flat ``list[int]``:
    some transformers versions return a ``BatchEncoding``/dict (or a batched
    ``[[...]]``) here, which would break downstream list operations."""
    try:
        out = tokenizer.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
        )
    except Exception:
        out = tokenizer.apply_chat_template(
            merge_system_into_user(messages),
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
    if tokenize:
        return _to_token_id_list(out)
    return out


def _to_token_id_list(out) -> List[int]:
    """Coerce assorted apply_chat_template(tokenize=True) return shapes into a
    flat list of token ids: list[int], BatchEncoding/dict, or batched [[...]]."""
    # dict / BatchEncoding -> pull input_ids
    if hasattr(out, "input_ids"):
        out = out.input_ids
    elif isinstance(out, dict) and "input_ids" in out:
        out = out["input_ids"]
    # tensors / arrays -> python list
    if hasattr(out, "tolist"):
        out = out.tolist()
    # un-batch a single [[...]] sequence
    if out and isinstance(out, list) and isinstance(out[0], list):
        out = out[0]
    return list(out)


_LETTER_RE = re.compile(r"\(?\s*([A-E])\s*\)?", re.IGNORECASE)


def extract_answer_letter(text: str) -> Optional[str]:
    """Pull the predicted option letter out of a model generation.

    Strategy, in priority order:
      1. The phrase "answer is (X)" / "answer: X" — for chain-of-thought
         output we take the **last** such phrase (the conclusion), so stray
         "option B is wrong" mentions mid-reasoning don't mislead us.
      2. The first standalone "(X)" token.
      3. A leading bare letter, e.g. "C. Because ...".
    Returns an uppercase letter or None if nothing parseable is found.
    """
    if not text:
        return None

    # 1) Explicit "answer is X" pattern — last match wins (CoT conclusion).
    matches = re.findall(r"answer\s*(?:is|:)?\s*\(?\s*([A-E])\b", text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()

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


def load_medmcqa(dataset_name: str = "openlifescienceai/medmcqa", require_exp: bool = False):
    """Load MedMCQA and normalize every split to the shared schema.

    ``require_exp=True`` keeps only single-answer rows with a non-empty
    explanation (for building CoT *training* targets). Eval splits should
    pass ``require_exp=False`` to keep every labeled question.

    Note: MedMCQA's ``test`` answers are hidden, so use ``validation`` for
    labeled in-distribution evaluation.
    """
    from datasets import load_dataset

    raw = load_dataset(dataset_name)
    normalized = raw.map(normalize_medmcqa, remove_columns=raw["train"].column_names)

    def keep(ex):
        if ex["answer_letter"] not in LETTERS:
            return False
        if ex.get("choice_type", "single") != "single":
            return False
        if require_exp and not ex.get("explanation"):
            return False
        return True

    return normalized.filter(keep)
