"""Interactive / single-shot inference with the fine-tuned adapter.

    python -m src.inference --adapter outputs/medqa-qlora/adapter \
        --question "A 45-year-old man ..." --options "A) ... B) ... C) ... D) ..."

Or import `answer_question` from a notebook for ad-hoc demos.
"""
from __future__ import annotations

import argparse
from typing import Dict, Optional

from .config import Config, CONFIG
from .data import SYSTEM_PROMPT, apply_chat_template, build_question_block, extract_answer_letter
from .train import load_tokenizer


def _load(adapter_dir: Optional[str], config: Config):
    from .train import load_base_model
    tokenizer = load_tokenizer(config)
    model = load_base_model(config, for_training=False)
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def answer_question(model, tokenizer, question: str, options: Dict[str, str], config: Config = CONFIG) -> Dict:
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_question_block(question, options)},
    ]
    prompt = apply_chat_template(tokenizer, messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=config.max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return {"answer_letter": extract_answer_letter(text), "raw": text.strip()}


def _parse_options(raw: str) -> Dict[str, str]:
    """Parse 'A) foo B) bar C) baz' into {'A': 'foo', ...}."""
    import re
    parts = re.split(r"\b([A-E])\)\s*", raw)
    options: Dict[str, str] = {}
    # parts: ['', 'A', 'foo ', 'B', 'bar ', ...]
    for i in range(1, len(parts) - 1, 2):
        options[parts[i].upper()] = parts[i + 1].strip()
    return options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="outputs/medqa-qlora/adapter")
    parser.add_argument("--question", required=True)
    parser.add_argument("--options", required=True, help='e.g. "A) ... B) ... C) ... D) ..."')
    args = parser.parse_args()

    model, tokenizer = _load(args.adapter, CONFIG)
    result = answer_question(model, tokenizer, args.question, _parse_options(args.options))
    print(f"\nPredicted: ({result['answer_letter']})")
    print(f"Model:     {result['raw']}")


if __name__ == "__main__":
    main()
