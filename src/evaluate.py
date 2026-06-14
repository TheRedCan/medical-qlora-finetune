"""Paired base-vs-fine-tuned evaluation with significance testing.

For each held-out question we prompt the model (chain-of-thought when
``config.use_cot``), parse the predicted option letter, and compare to gold.
Base and fine-tuned models answer the *same* questions, so we run a paired
**McNemar** test (see ``src/stats.py``) — the headline "is the gain real?"
number — plus a CI on the accuracy difference.

Inference defaults to fp16 with the LoRA adapter merged in: bitsandbytes
4-bit generation is slow, and a 3B model fits a T4 comfortably in fp16.

    python -m src.evaluate --adapter outputs/medqa-qlora/adapter
"""
from __future__ import annotations

import argparse
import gc
import json
from typing import Dict, List, Optional

from .config import Config, CONFIG
from .data import apply_chat_template, build_messages, build_plain_prompt, extract_answer_letter
from .stats import mcnemar
from .train import load_base_model, load_tokenizer, _subset, _compute_dtype


# --------------------------------------------------------------------------
# Model loading (fast fp16 path + 4-bit fallback)
# --------------------------------------------------------------------------
def load_base_for_eval(config: Config):
    if not config.eval_in_fp16:
        m = load_base_model(config, for_training=False)
        m.eval()
        return m
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model, torch_dtype=_compute_dtype(), device_map="auto"
    )
    model.eval()
    return model


def load_finetuned_for_eval(adapter_dir: str, config: Config):
    from peft import PeftModel

    if not config.eval_in_fp16:
        base = load_base_model(config, for_training=False)
        model = PeftModel.from_pretrained(base, adapter_dir)
        model.eval()
        return model

    base = load_base_for_eval(config)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.merge_and_unload()  # fold LoRA in for fast fp16 inference
    model.eval()
    return model


# Backwards-compatible alias used by older callers/notebook.
def load_adapter_model(adapter_dir: str, config: Config):
    return load_finetuned_for_eval(adapter_dir, config)


def _free(model):
    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# Generation + scoring
# --------------------------------------------------------------------------
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
    gen = out[:, inputs["input_ids"].shape[1]:]
    return tokenizer.batch_decode(gen, skip_special_tokens=True)


def evaluate_model(model, tokenizer, dataset, config: Config, batch_size: Optional[int] = None) -> Dict:
    """Generate + score over a dataset; returns accuracy, parse_rate, and the
    aligned per-example correctness list used for the paired test."""
    batch_size = batch_size or config.eval_batch_size
    tokenizer.padding_side = "left"  # required for correct batched generation

    records: List[Dict] = []
    correct = parsed = 0

    for start in range(0, len(dataset), batch_size):
        batch = dataset.select(range(start, min(start + batch_size, len(dataset))))
        if config.use_chat_template:
            prompts = [
                apply_chat_template(
                    tokenizer, build_messages(ex, include_answer=False, cot=config.use_cot),
                    tokenize=False, add_generation_prompt=True,
                )
                for ex in batch
            ]
        else:  # base model: plain-text prompt
            prompts = [build_plain_prompt(ex, include_answer=False, cot=config.use_cot) for ex in batch]
        generations = _generate_batch(model, tokenizer, prompts, config)
        for ex, gen in zip(batch, generations):
            pred = extract_answer_letter(gen)
            is_correct = pred is not None and pred == ex["answer_letter"]
            correct += int(is_correct)
            parsed += int(pred is not None)
            records.append({"gold": ex["answer_letter"], "pred": pred,
                            "correct": is_correct, "generation": gen.strip()})
        print(f"  {len(records)}/{len(dataset)} running acc={correct / len(records):.3f}", end="\r")

    print()
    n = len(records)
    return {
        "accuracy": correct / n if n else 0.0,
        "parse_rate": parsed / n if n else 0.0,
        "n": n,
        "correct_flags": [r["correct"] for r in records],
        "records": records,
    }


def paired_compare(adapter_dir: str, dataset, config: Config, tokenizer=None) -> Dict:
    """Evaluate base then fine-tuned on the SAME dataset (one model in memory
    at a time) and run the paired McNemar significance test."""
    tokenizer = tokenizer or load_tokenizer(config)

    print("  [base] loading + evaluating ...")
    base_model = load_base_for_eval(config)
    base = evaluate_model(base_model, tokenizer, dataset, config)
    _free(base_model)

    print("  [fine-tuned] loading + evaluating ...")
    ft_model = load_finetuned_for_eval(adapter_dir, config)
    ft = evaluate_model(ft_model, tokenizer, dataset, config)
    _free(ft_model)

    test = mcnemar(base["correct_flags"], ft["correct_flags"])
    return {
        "n": base["n"],
        "base_accuracy": round(base["accuracy"], 4),
        "finetuned_accuracy": round(ft["accuracy"], 4),
        "base_parse_rate": round(base["parse_rate"], 4),
        "finetuned_parse_rate": round(ft["parse_rate"], 4),
        "significance": test.as_dict(),
    }


def _print_summary(name: str, res: Dict):
    s = res["significance"]
    print(f"\n=== {name} (n={res['n']}) ===")
    print(f"  base       : {res['base_accuracy']:.4f}")
    print(f"  fine-tuned : {res['finetuned_accuracy']:.4f}")
    print(f"  delta      : {s['accuracy_delta']:+.4f}  "
          f"95% CI [{s['delta_ci_low']:+.4f}, {s['delta_ci_high']:+.4f}]")
    print(f"  McNemar    : b={s['ft_right_base_wrong']} c={s['ft_wrong_base_right']} "
          f"p={s['p_value']:.4g}  -> {'SIGNIFICANT' if s['significant_05'] else 'not significant'} (a=0.05)")


def run(adapter_dir: str, config: Config = CONFIG) -> Dict:
    """Single-dataset paired eval on the configured dataset's test/validation."""
    from .data import load_medmcqa, load_medqa

    if "medmcqa" in config.dataset_name.lower():
        split = load_medmcqa(config.dataset_name)["validation"]
    else:
        split = load_medqa(config)["test"]
    dataset = _subset(split, config.max_eval_samples)
    res = paired_compare(adapter_dir, dataset, config)
    _print_summary(config.dataset_name, res)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="outputs/medqa-qlora/adapter")
    parser.add_argument("--out", default="outputs/eval_results.json")
    args = parser.parse_args()

    res = run(args.adapter)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSummary written to {args.out}")


if __name__ == "__main__":
    main()
