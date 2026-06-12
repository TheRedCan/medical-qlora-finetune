"""Orchestration for the CoT QLoRA experiment (iterate this via GitHub).

Pipeline:
  1. QLoRA fine-tune on MedMCQA with chain-of-thought targets.
  2. Paired base-vs-fine-tuned eval (McNemar) on:
       - MedMCQA validation  (in-distribution -> the significance result)
       - MedQA / USMLE test   (transfer -> the headline generalization result)
  3. Write /kaggle/working/eval_results_cot.json.

Flip PILOT via the env var PILOT=0 for the full run. Pilot is a fast
end-to-end smoke test (~20 min) to catch bugs before the multi-hour full run.
"""
import json
import os
import sys

WORK = "/kaggle/working"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
# Reduce allocator fragmentation (suggested by the OOM error) and pin to a
# single T4 so device_map doesn't imbalance activations across 2 GPUs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

PILOT = os.environ.get("PILOT", "1") == "1"

# --- config via env (read by src/config.py) ---------------------------
os.environ.setdefault("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
os.environ["USE_COT"] = "1"
os.environ["DATASET_NAME"] = "openlifescienceai/medmcqa"
os.environ["EVAL_IN_FP16"] = "1"          # fast eval (big win, separate phase)
os.environ["GRAD_CHECKPOINT"] = "1"       # memory-safe training (proven config)
os.environ["MAX_SEQ_LENGTH"] = "768"      # MedMCQA CoT fits; faster than 1024
os.environ["EVAL_BATCH_SIZE"] = "12"
# Enough room for the *base* model's verbose CoT to reach its answer line;
# at 200 it was truncated -> unparseable -> unfairly scored wrong.
os.environ["MAX_NEW_TOKENS"] = "512"
os.environ["OUTPUT_DIR"] = os.path.join(WORK, "outputs", "medmcqa-cot-qlora")
if PILOT:
    os.environ["MAX_TRAIN_SAMPLES"] = "400"
    os.environ["MAX_EVAL_SAMPLES"] = "150"
    os.environ["NUM_TRAIN_EPOCHS"] = "1"
else:
    os.environ["MAX_TRAIN_SAMPLES"] = "5000"
    os.environ["MAX_EVAL_SAMPLES"] = "1000"
    os.environ["NUM_TRAIN_EPOCHS"] = "1"

import torch  # noqa: E402
print("MODE:", "PILOT" if PILOT else "FULL")
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")

from src.config import Config                       # noqa: E402
from src.train import train, _subset               # noqa: E402
from src import evaluate                            # noqa: E402
from src.data import load_medmcqa, normalize_example, LETTERS, build_messages  # noqa: E402

cfg = Config()
print("Config:", json.dumps(cfg.as_dict(), indent=2))

# --- sanity: eyeball one formatted CoT training example ----------------
_sample = load_medmcqa(cfg.dataset_name, require_exp=True)["train"][0]
print("\n--- sample CoT training target (sanity check labels/format) ---")
for m in build_messages(_sample, include_answer=True, cot=True):
    print(f"[{m['role']}] {m['content'][:300]}")
print("gold letter:", _sample["answer_letter"], "| text:", _sample["answer_text"][:60])

# --- train -------------------------------------------------------------
print("\n===== TRAINING (CoT QLoRA on MedMCQA) =====", flush=True)
adapter_dir = train(cfg)

tokenizer = evaluate.load_tokenizer(cfg)

# --- in-distribution eval: MedMCQA validation --------------------------
print("\n===== EVAL: MedMCQA validation (in-distribution) =====", flush=True)
medmcqa_val = _subset(load_medmcqa(cfg.dataset_name)["validation"], cfg.max_eval_samples)
res_indist = evaluate.paired_compare(adapter_dir, medmcqa_val, cfg, tokenizer=tokenizer)
evaluate._print_summary("MedMCQA validation (in-distribution)", res_indist)

# --- transfer eval: MedQA / USMLE test ---------------------------------
print("\n===== EVAL: MedQA USMLE test (transfer) =====", flush=True)
from datasets import load_dataset  # noqa: E402
medqa_raw = load_dataset("GBaker/MedQA-USMLE-4-options")["test"]
medqa_test = medqa_raw.map(normalize_example, remove_columns=medqa_raw.column_names)
medqa_test = medqa_test.filter(lambda e: e["answer_letter"] in LETTERS)
medqa_test = _subset(medqa_test, cfg.max_eval_samples)
res_transfer = evaluate.paired_compare(adapter_dir, medqa_test, cfg, tokenizer=tokenizer)
evaluate._print_summary("MedQA USMLE test (transfer)", res_transfer)

# --- persist -----------------------------------------------------------
summary = {
    "mode": "pilot" if PILOT else "full",
    "base_model": cfg.base_model,
    "method": "QLoRA 4-bit NF4 + LoRA, chain-of-thought SFT on MedMCQA",
    "train_samples": cfg.max_train_samples,
    "num_train_epochs": cfg.num_train_epochs,
    "in_distribution_medmcqa_val": res_indist,
    "transfer_medqa_test": res_transfer,
}
out_path = os.path.join(WORK, "eval_results_cot.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nWROTE {out_path}")
print(json.dumps(summary, indent=2))
