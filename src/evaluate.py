"""Evaluate exact-match accuracy on MedQA, comparing base vs fine-tuned.

This is the headline metric of the project. We generate an answer for each
held-out test question, parse the predicted option letter, and compare it
to the gold letter. Running it for the base model *and* the adapter-loaded
model produces the before/after numbers reported in the README.

    python -m src.evaluate --adapter outputs/medqa-qlora/adapter
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

from .config import Config, CONFIG
from .data import build_messages, extract_answer_letter, load_medqa
from .train import load_base_model, load_tokenizer, _subset


def _generate_batch(model, tokenizer, prompts: List[str], config: Config) -> List[str]:
    import torch

    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=config.max_seq_length,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,  # greedy -> deterministic, reproducible eval
            pad_token_id=tokenizer.pad_token_id,
        )
    # Slice off the prompt tokens so we only decode the completion.
    gen = out[:, inputs["input_ids"].shape[1]:]
    return tokenizer.batch_decode(gen, skip_special_tokens=True)


def evaluate_model(model, tokenizer, dataset, config: Config, batch_size: int = 8) -> Dict:
    """Return accuracy + per-example records for an already-loaded model."""
    # Left padding is required for correct batched generation alignment.
    tokenizer.padding_side = "left"

    records: List[Dict] = []
    correct = 0
    parsed = 0

    for start in range(0, len(dataset), batch_size):
        batch = dataset.select(range(start, min(start + batch_size, len(dataset))))
        prompts = [
            tokenizer.apply_chat_template(
                build_messages(ex, include_answer=False),
                tokenize=False, add_generation_prompt=True,
            )
            for ex in batch
        ]
        generations = _generate_batch(model, tokenizer, prompts, config)

        for ex, gen in zip(batch, generations):
            pred = extract_answer_letter(gen)
            gold = ex["answer_letter"]
            is_correct = pred is not None and pred == gold
            correct += int(is_correct)
            parsed += int(pred is not None)
            records.append(
                {"gold": gold, "pred": pred, "correct": is_correct, "generation": gen.strip()}
            )
        print(f"  {min(start + batch_size, len(dataset))}/{len(dataset)} "
              f"running acc={correct / len(records):.3f}", end="\r")

    print()
    n = len(records)
    return {
        "accuracy": correct / n if n else 0.0,
        "parse_rate": parsed / n if n else 0.0,
        "n": n,
        "records": records,
    }


def load_adapter_model(adapter_dir: str, config: Config):
    """Load the 4-bit base model and apply the trained LoRA adapter."""
    from peft import PeftModel

    base = load_base_model(config, for_training=False)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model


def run(adapter_dir: Optional[str], config: Config = CONFIG, eval_base: bool = True) -> Dict:
    tokenizer = load_tokenizer(config)
    ds = load_medqa(config)
    test_ds = _subset(ds["test"], config.max_eval_samples)
    print(f"Evaluating on {len(test_ds)} MedQA test questions.\n")

    results: Dict[str, Dict] = {}

    if eval_base:
        print("== Base model ==")
        base_model = load_base_model(config, for_training=False)
        base_model.eval()
        results["base"] = evaluate_model(base_model, tokenizer, test_ds, config)
        print(f"Base accuracy:       {results['base']['accuracy']:.3f}")
        del base_model
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()

    if adapter_dir:
        print("\n== Fine-tuned (LoRA) model ==")
        ft_model = load_adapter_model(adapter_dir, config)
        results["finetuned"] = evaluate_model(ft_model, tokenizer, test_ds, config)
        print(f"Fine-tuned accuracy: {results['finetuned']['accuracy']:.3f}")

    if "base" in results and "finetuned" in results:
        delta = results["finetuned"]["accuracy"] - results["base"]["accuracy"]
        print(f"\nAbsolute improvement: {delta:+.3f} "
              f"({delta * 100:+.1f} percentage points)")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="outputs/medqa-qlora/adapter")
    parser.add_argument("--no-base", action="store_true", help="skip base-model eval")
    parser.add_argument("--out", default="outputs/eval_results.json")
    args = parser.parse_args()

    results = run(args.adapter, eval_base=not args.no_base)

    # Persist a compact summary (drop the verbose per-example records).
    summary = {
        k: {kk: vv for kk, vv in v.items() if kk != "records"}
        for k, v in results.items()
    }
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {args.out}")


if __name__ == "__main__":
    main()
