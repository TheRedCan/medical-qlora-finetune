"""Orchestration for the medical structured-extraction experiment.

We pivoted here after four runs confirmed SFT can't add the *knowledge* that
MCQA accuracy needs. Extraction is a format/behavior task SFT is good at:

  * Task  : clinical text -> JSON list of disease mentions.
  * Model : Qwen2.5-1.5B (base) -> fails strict JSON zero-shot (the headroom).
  * Data  : rjac/biobert-ner-diseases-dataset (disease NER, train/test).
  * Eval  : entity micro-F1 (paired bootstrap CI) + JSON-valid rate +
            exact-set-match (paired McNemar), base vs fine-tuned.

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
    os.environ["NUM_TRAIN_EPOCHS"] = "2"

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

# --- eval: held-out test set ------------------------------------------
print("\n===== EVAL: disease NER test set =====", flush=True)
test_ds = _subset(ner["test"], cfg.max_eval_samples)
res = extraction.paired_compare_extraction(adapter_dir, test_ds, cfg, tokenizer=tokenizer)
extraction.print_summary("Disease extraction (held-out test)", res)

# --- persist -----------------------------------------------------------
summary = {
    "mode": "pilot" if PILOT else "full",
    "task": "disease structured extraction (text -> JSON)",
    "base_model": cfg.base_model,
    "dataset": cfg.dataset_name,
    "train_samples": cfg.max_train_samples,
    "num_train_epochs": cfg.num_train_epochs,
    "result": res,
}
out_path = os.path.join(WORK, "eval_results_cot.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nWROTE {out_path}")
print(json.dumps(summary, indent=2))
