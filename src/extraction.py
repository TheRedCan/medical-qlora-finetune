"""Medical structured-extraction task: clinical text -> JSON of diseases.

This is the task SFT is *good* at — format/behavior, not knowledge. A base
(non-instruct) model can't reliably emit the strict JSON schema zero-shot;
fine-tuning teaches it, so JSON-validity and entity-F1 jump dramatically.

Pure functions (bio_to_entities, build_extraction_prompt, parse_diseases,
score_example) are unit-tested with no GPU/network dependency.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple

SYSTEM_PROMPT = (
    "You are a clinical NLP system that extracts disease mentions from "
    "biomedical text."
)
INSTRUCTION = (
    'Extract every disease mention. Respond with ONLY a JSON object of the '
    'form {"diseases": ["...", "..."]} and nothing else.'
)


def bio_to_entities(tokens: List[str], tags: List[int]) -> List[str]:
    """Convert BIO tag ids (0=O, 1=B-Disease, 2=I-Disease) into entity
    strings. Deterministic and tested."""
    entities, current = [], []
    for tok, tag in zip(tokens, tags):
        if tag == 1:                      # B: start new entity
            if current:
                entities.append(" ".join(current))
            current = [tok]
        elif tag == 2 and current:        # I: continue
            current.append(tok)
        elif tag == 2 and not current:    # I with no B: treat as start
            current = [tok]
        else:                             # O: close any open entity
            if current:
                entities.append(" ".join(current))
                current = []
    if current:
        entities.append(" ".join(current))
    return entities


def normalize_ner_example(example: Dict) -> Dict:
    """Raw {tokens, tags} -> {text, diseases}. Deduplicates entities while
    preserving order."""
    tokens = example["tokens"]
    entities = bio_to_entities(tokens, example["tags"])
    seen, deduped = set(), []
    for e in entities:
        key = e.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(e)
    return {"text": " ".join(tokens), "diseases": deduped}


def build_target(diseases: List[str]) -> str:
    return json.dumps({"diseases": diseases}, ensure_ascii=False)


def build_extraction_prompt(example: Dict, include_answer: bool) -> str:
    """Plain-text prompt for a base model, ending in 'JSON:'."""
    text = example["text"]
    prompt = f"{SYSTEM_PROMPT}\n\nText: {text}\n\n{INSTRUCTION}\nJSON:"
    if include_answer:
        prompt = f"{prompt} {build_target(example['diseases'])}"
    return prompt


_JSON_OBJ_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_diseases(generation: str) -> Tuple[List[str], bool]:
    """Parse a model generation into (diseases, json_valid).

    Finds the first JSON object, loads it, and pulls a string list out of
    "diseases". Returns ([], False) when nothing schema-valid is found.
    """
    if not generation:
        return [], False
    m = _JSON_OBJ_RE.search(generation)
    if not m:
        return [], False
    try:
        obj = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return [], False
    if not isinstance(obj, dict) or "diseases" not in obj:
        return [], False
    vals = obj["diseases"]
    if not isinstance(vals, list):
        return [], False
    diseases = [str(v).strip() for v in vals if str(v).strip()]
    return diseases, True


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def score_example(pred: List[str], gold: List[str]) -> Dict:
    """Set-based TP/FP/FN (case/space-insensitive) + exact-set-match flag."""
    pred_set = {_norm(p) for p in pred}
    gold_set = {_norm(g) for g in gold}
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return {"tp": tp, "fp": fp, "fn": fn, "exact": pred_set == gold_set}


# --------------------------------------------------------------------------
# Training tokenization (prompt-masked, plain text for a base model)
# --------------------------------------------------------------------------
def tokenize_extraction(example: Dict, tokenizer, config) -> Dict:
    prompt_str = build_extraction_prompt(example, include_answer=False)
    full_str = build_extraction_prompt(example, include_answer=True)
    prompt_ids = tokenizer(prompt_str, add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(full_str, add_special_tokens=True)["input_ids"]
    if tokenizer.eos_token_id is not None and (not full_ids or full_ids[-1] != tokenizer.eos_token_id):
        full_ids = full_ids + [tokenizer.eos_token_id]
    full_ids = full_ids[: config.max_seq_length]
    labels = list(full_ids)
    for i in range(min(len(prompt_ids), len(full_ids))):
        labels[i] = -100
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def load_ner(dataset_name: str = "rjac/biobert-ner-diseases-dataset"):
    from datasets import load_dataset
    raw = load_dataset(dataset_name)
    return raw.map(normalize_ner_example, remove_columns=raw["train"].column_names)


# --------------------------------------------------------------------------
# Evaluation (paired base vs fine-tuned)
# --------------------------------------------------------------------------
def evaluate_extraction(model, tokenizer, dataset, config) -> Dict:
    from .evaluate import _generate_batch

    tokenizer.padding_side = "left"
    bs = config.eval_batch_size
    counts, exacts, valids, preds = [], [], [], []
    for start in range(0, len(dataset), bs):
        batch = dataset.select(range(start, min(start + bs, len(dataset))))
        prompts = [build_extraction_prompt(ex, include_answer=False) for ex in batch]
        gens = _generate_batch(model, tokenizer, prompts, config)
        for ex, gen in zip(batch, gens):
            pred, valid = parse_diseases(gen)
            sc = score_example(pred, ex["diseases"])
            counts.append((sc["tp"], sc["fp"], sc["fn"]))
            exacts.append(sc["exact"])
            valids.append(valid)
            preds.append(pred)
        f1 = _f1(counts)
        print(f"  {len(counts)}/{len(dataset)} micro-F1={f1:.3f}", end="\r")
    print()
    return {"counts": counts, "exact_flags": exacts,
            "json_valid_rate": sum(valids) / len(valids), "preds": preds}


def micro_f1_on(model, tokenizer, dataset, config) -> float:
    """Convenience: micro-F1 of a single model on a dataset (used for the
    train-vs-test generalization-gap overfitting check)."""
    from .stats import micro_f1
    return micro_f1(evaluate_extraction(model, tokenizer, dataset, config)["counts"])


def _f1(counts) -> float:
    from .stats import micro_f1
    return micro_f1(counts)


def paired_compare_extraction(adapter_dir: str, dataset, config, tokenizer=None) -> Dict:
    from .train import load_tokenizer
    from .evaluate import load_base_for_eval, load_finetuned_for_eval, _free
    from .stats import mcnemar, bootstrap_f1_diff

    tokenizer = tokenizer or load_tokenizer(config)

    print("  [base] evaluating ...")
    base_model = load_base_for_eval(config)
    base = evaluate_extraction(base_model, tokenizer, dataset, config)
    _free(base_model)

    print("  [fine-tuned] evaluating ...")
    ft_model = load_finetuned_for_eval(adapter_dir, config)
    ft = evaluate_extraction(ft_model, tokenizer, dataset, config)
    _free(ft_model)

    boot = bootstrap_f1_diff(base["counts"], ft["counts"])
    exact = mcnemar(base["exact_flags"], ft["exact_flags"])

    # qualitative examples for eyeballing (text / gold / base / fine-tuned)
    n_ex = min(15, len(dataset))
    examples = [
        {"text": dataset[i]["text"], "gold": dataset[i]["diseases"],
         "base_pred": base["preds"][i], "finetuned_pred": ft["preds"][i]}
        for i in range(n_ex)
    ]
    return {
        "n": len(dataset),
        "examples": examples,
        "base_micro_f1": boot["f1_base"],
        "finetuned_micro_f1": boot["f1_ft"],
        "f1_significance": boot,
        "base_json_valid_rate": round(base["json_valid_rate"], 4),
        "finetuned_json_valid_rate": round(ft["json_valid_rate"], 4),
        "base_exact_match": round(sum(base["exact_flags"]) / len(dataset), 4),
        "finetuned_exact_match": round(sum(ft["exact_flags"]) / len(dataset), 4),
        "exact_match_significance": exact.as_dict(),
    }


def print_summary(name: str, res: Dict):
    b = res["f1_significance"]
    print(f"\n=== {name} (n={res['n']}) ===")
    print(f"  micro-F1     : base {res['base_micro_f1']:.3f} -> ft {res['finetuned_micro_f1']:.3f} "
          f"(Δ {b['f1_delta']:+.3f}, 95% CI [{b['ci_low']:+.3f},{b['ci_high']:+.3f}], p={b['p_value']:.4g})")
    print(f"  JSON-valid   : base {res['base_json_valid_rate']:.3f} -> ft {res['finetuned_json_valid_rate']:.3f}")
    print(f"  exact-match  : base {res['base_exact_match']:.3f} -> ft {res['finetuned_exact_match']:.3f} "
          f"(McNemar p={res['exact_match_significance']['p_value']:.4g})")
    sig = b["significant_05"] and b["f1_delta"] > 0
    print(f"  => F1 gain {'SIGNIFICANT' if sig else 'not significant'} (a=0.05)")
