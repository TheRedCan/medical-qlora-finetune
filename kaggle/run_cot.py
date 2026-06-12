"""Orchestration for the small-model QLoRA experiment (iterate via GitHub).

Findings so far: Qwen2.5-3B (instruct *or* base) is already strong at medical
MCQA, so QLoRA SFT can't improve it (no headroom). This run uses a genuinely
smaller model with real headroom, on cleaner data:

  * Model : Qwen2.5-0.5B (base, non-instruct) -> plain-text prompts.
  * Data  : MedQA / USMLE (cleaner than noisy MedMCQA), direct answer targets.
  * Eval  : MedQA test  (in-distribution -> the significance result)
            MedMCQA val (transfer -> generalization)
  * Test  : paired McNemar + CI (src/stats.py).

Flip PILOT via env PILOT=0 for the full run.
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

PILOT = os.environ.get("PILOT", "1") == "1"

# --- config via env (read by src/config.py) ---------------------------
os.environ.setdefault("BASE_MODEL", "Qwen/Qwen2.5-0.5B")  # small BASE model
os.environ["USE_COT"] = "0"                 # direct answer-format SFT
os.environ["USE_CHAT_TEMPLATE"] = "0"       # base model -> plain-text prompts
os.environ["DATASET_NAME"] = "GBaker/MedQA-USMLE-4-options"  # clean USMLE data
os.environ["EVAL_IN_FP16"] = "1"
os.environ["GRAD_CHECKPOINT"] = "0"         # 0.5B is tiny -> no checkpointing
os.environ["MAX_SEQ_LENGTH"] = "1024"       # USMLE vignettes can be long
os.environ["EVAL_BATCH_SIZE"] = "32"
os.environ["MAX_NEW_TOKENS"] = "48"         # only need "The answer is (X)"
os.environ["LEARNING_RATE"] = "2e-4"
os.environ["OUTPUT_DIR"] = os.path.join(WORK, "outputs", "medqa-small-qlora")
if PILOT:
    os.environ["MAX_TRAIN_SAMPLES"] = "1000"
    os.environ["MAX_EVAL_SAMPLES"] = "250"
    os.environ["NUM_TRAIN_EPOCHS"] = "1"
else:
    os.environ["MAX_TRAIN_SAMPLES"] = "0"   # all ~10k MedQA train
    os.environ["MAX_EVAL_SAMPLES"] = "1000"
    os.environ["NUM_TRAIN_EPOCHS"] = "3"

import torch  # noqa: E402
print("MODE:", "PILOT" if PILOT else "FULL")
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")

from src.config import Config                       # noqa: E402
from src.train import train, _subset               # noqa: E402
from src import evaluate                            # noqa: E402
from src.data import load_medqa, load_medmcqa, build_plain_prompt  # noqa: E402

cfg = Config()
print("Config:", json.dumps(cfg.as_dict(), indent=2))

# --- sanity: eyeball one formatted training example -------------------
medqa = load_medqa(cfg)
_sample = medqa["train"][0]
print("\n--- sample training prompt+target (sanity check labels/format) ---")
print(build_plain_prompt(_sample, include_answer=True, cot=cfg.use_cot)[:600])
print("gold letter:", _sample["answer_letter"], "| text:", _sample["answer_text"][:60])

# --- train -------------------------------------------------------------
print("\n===== TRAINING (Qwen2.5-0.5B base QLoRA on MedQA) =====", flush=True)
adapter_dir = train(cfg)

tokenizer = evaluate.load_tokenizer(cfg)

# --- in-distribution eval: MedQA test ----------------------------------
print("\n===== EVAL: MedQA USMLE test (in-distribution) =====", flush=True)
medqa_test = _subset(medqa["test"], cfg.max_eval_samples)
res_indist = evaluate.paired_compare(adapter_dir, medqa_test, cfg, tokenizer=tokenizer)
evaluate._print_summary("MedQA USMLE test (in-distribution)", res_indist)

# --- transfer eval: MedMCQA validation ---------------------------------
print("\n===== EVAL: MedMCQA validation (transfer) =====", flush=True)
medmcqa_val = _subset(load_medmcqa("openlifescienceai/medmcqa")["validation"], cfg.max_eval_samples)
res_transfer = evaluate.paired_compare(adapter_dir, medmcqa_val, cfg, tokenizer=tokenizer)
evaluate._print_summary("MedMCQA validation (transfer)", res_transfer)

# --- persist -----------------------------------------------------------
summary = {
    "mode": "pilot" if PILOT else "full",
    "base_model": cfg.base_model,
    "method": f"QLoRA 4-bit NF4 + LoRA; direct SFT on MedQA; base model (plain prompts)",
    "train_samples": cfg.max_train_samples,
    "num_train_epochs": cfg.num_train_epochs,
    "in_distribution_medqa_test": res_indist,
    "transfer_medmcqa_val": res_transfer,
}
out_path = os.path.join(WORK, "eval_results_cot.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nWROTE {out_path}")
print(json.dumps(summary, indent=2))
