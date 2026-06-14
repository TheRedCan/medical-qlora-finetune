"""Orchestration for the medical structured-extraction experiment.

We pivoted here after four runs confirmed SFT can't add the *knowledge* that
MCQA accuracy needs. Extraction is a format/behavior task SFT is good at:

  * Task  : clinical text -> JSON list of disease mentions.
  * Model : Qwen2.5-1.5B (base) -> fails strict JSON zero-shot (the headroom).
  * Data  : rjac/biobert-ner-diseases-dataset (disease NER, train/test).
  * Eval  : entity micro-F1 (paired bootstrap CI) + JSON-valid + exact-match
            (paired McNemar), base vs fine-tuned, on THREE fronts:
              1. in-distribution test set,
              2. train-vs-test gap (overfitting check),
              3. out-of-distribution BC5CDR disease set (different corpus).

Flip PILOT via env PILOT=1 for a fast smoke test.
"""
import json
import os
import sys

WORK = "/kaggle/working"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

PILOT = os.environ.get("PILOT", "0") == "1"  # default FULL (pilot proved out)

# --- config via env (read by src/config.py) ---------------------------
os.environ["TASK"] = "extraction"
os.environ.setdefault("BASE_MODEL", "Qwen/Qwen2.5-1.5B")  # BASE model
os.environ["USE_CHAT_TEMPLATE"] = "0"        # base model -> plain-text prompts
os.environ["DATASET_NAME"] = "rjac/biobert-ner-diseases-dataset"
os.environ["EVAL_IN_FP16"] = "1"
os.environ["GRAD_CHECKPOINT"] = "0"
os.environ["MAX_SEQ_LENGTH"] = "512"         # short sentences + short JSON
os.environ["EVAL_BATCH_SIZE"] = "32"
os.environ["MAX_NEW_TOKENS"] = "96"          # enough for a JSON list of diseases
os.environ["LEARNING_RATE"] = "2e-4"
os.environ["OUTPUT_DIR"] = os.path.join(WORK, "outputs", "disease-extraction-qlora")
if PILOT:
    os.environ["MAX_TRAIN_SAMPLES"] = "1500"
    os.environ["MAX_EVAL_SAMPLES"] = "300"
    os.environ["NUM_TRAIN_EPOCHS"] = "1"
else:
    os.environ["MAX_TRAIN_SAMPLES"] = "8000"
    os.environ["MAX_EVAL_SAMPLES"] = "1500"
    os.environ["NUM_TRAIN_EPOCHS"] = "1"     # 1 epoch = sweet spot (less overfit)

import torch  # noqa: E402
print("MODE:", "PILOT" if PILOT else "FULL")
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")

from src.config import Config                       # noqa: E402
from src.train import train, _subset               # noqa: E402
from src import evaluate, extraction               # noqa: E402

cfg = Config()
print("Config:", json.dumps(cfg.as_dict(), indent=2))

# --- sanity: eyeball one formatted training example -------------------
ner = extraction.load_ner(cfg.dataset_name)
_s = ner["train"][0]
print("\n--- sample training prompt+target (sanity check) ---")
print(extraction.build_extraction_prompt(_s, include_answer=True)[:500])
print("gold diseases:", _s["diseases"])

# --- train -------------------------------------------------------------
print("\n===== TRAINING (Qwen2.5-1.5B base QLoRA, disease extraction) =====", flush=True)
adapter_dir = train(cfg)

tokenizer = evaluate.load_tokenizer(cfg)

# --- 1) in-distribution eval: held-out test set -----------------------
print("\n===== EVAL: disease NER test set (in-distribution) =====", flush=True)
test_ds = _subset(ner["test"], cfg.max_eval_samples)
res = extraction.paired_compare_extraction(adapter_dir, test_ds, cfg, tokenizer=tokenizer)
extraction.print_summary("Disease extraction (held-out test)", res)

# --- 2) overfitting check: train-vs-test F1 gap -----------------------
# Sample from the *same* shuffled order the trainer used, so these rows were
# genuinely trained on. Small (train F1 - test F1) gap = healthy generalization.
print("\n===== VALIDATION: train-vs-test F1 gap (overfitting check) =====", flush=True)
from src.evaluate import load_finetuned_for_eval, _free  # noqa: E402
train_seen = _subset(ner["train"].shuffle(seed=cfg.seed), 600)
ft_model = load_finetuned_for_eval(adapter_dir, cfg)
train_f1 = extraction.micro_f1_on(ft_model, tokenizer, train_seen, cfg)
_free(ft_model)
test_f1 = res["finetuned_micro_f1"]
gen_gap = round(train_f1 - test_f1, 4)
print(f"  fine-tuned F1 trained-on={train_f1:.4f}  held-out test={test_f1:.4f}  "
      f"gap={gen_gap:+.4f} ({'healthy' if gen_gap < 0.1 else 'mild overfit'})")

# --- 3) out-of-distribution eval: BC5CDR disease (different corpus) ----
print("\n===== OOD EVAL: BC5CDR disease (leakage-filtered) =====", flush=True)
rjac_texts = set()
for split in ner:
    rjac_texts.update(extraction._norm_sentence(t) for t in ner[split]["text"])
ood = _subset(extraction.load_bc5cdr_disease_ood(exclude_texts=rjac_texts), cfg.max_eval_samples)
print(f"OOD: {len(ood)} BC5CDR-disease sentences NOT in the training corpus")
res_ood = extraction.paired_compare_extraction(adapter_dir, ood, cfg, tokenizer=tokenizer)
extraction.print_summary("BC5CDR disease (out-of-distribution)", res_ood)

# --- persist -----------------------------------------------------------
summary = {
    "mode": "pilot" if PILOT else "full",
    "task": "disease structured extraction (text -> JSON)",
    "base_model": cfg.base_model,
    "dataset": cfg.dataset_name,
    "train_samples": cfg.max_train_samples,
    "num_train_epochs": cfg.num_train_epochs,
    "in_distribution": res,
    "overfitting_check": {
        "finetuned_train_f1": round(train_f1, 4),
        "finetuned_test_f1": test_f1,
        "generalization_gap": gen_gap,
    },
    "out_of_distribution_bc5cdr": res_ood,
}
out_path = os.path.join(WORK, "eval_results.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nWROTE {out_path}")
print(json.dumps({k: v for k, v in summary.items()
                  if k not in ("in_distribution", "out_of_distribution_bc5cdr")}, indent=2))
