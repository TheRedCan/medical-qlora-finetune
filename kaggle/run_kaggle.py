"""Kaggle kernel entry point — runs the full QLoRA pipeline headlessly.

Pushed to Kaggle via `kaggle kernels push -p kaggle/`. It clones the public
repo (single source of truth for src/), installs deps, evaluates the base
model, fine-tunes with QLoRA, re-evaluates, and writes results to
/kaggle/working so they can be downloaded with `kaggle kernels output`.
"""
import json
import os
import subprocess
import sys

REPO = "https://github.com/TheRedCan/medical-qlora-finetune.git"
WORK = "/kaggle/working"


def sh(cmd):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


# --- 1. get the code ---------------------------------------------------
repo_dir = os.path.join(WORK, "repo")
if not os.path.exists(repo_dir):
    sh(["git", "clone", "--depth", "1", REPO, repo_dir])
os.chdir(repo_dir)
sys.path.insert(0, repo_dir)

# --- 2. deps (torch is preinstalled on Kaggle GPU images) --------------
sh([sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.45", "peft>=0.13", "bitsandbytes>=0.43",
    "accelerate>=0.34", "datasets>=3.0", "sentencepiece>=0.2"])

# --- 3. config via env (ungated model => no HF token needed) -----------
os.environ["BASE_MODEL"] = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
os.environ["MAX_TRAIN_SAMPLES"] = os.environ.get("MAX_TRAIN_SAMPLES", "4000")
os.environ["MAX_EVAL_SAMPLES"] = os.environ.get("MAX_EVAL_SAMPLES", "300")
os.environ["NUM_TRAIN_EPOCHS"] = os.environ.get("NUM_TRAIN_EPOCHS", "1")
os.environ["OUTPUT_DIR"] = os.path.join(WORK, "outputs", "medqa-qlora")

import torch  # noqa: E402
print("\nGPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")

from src.config import Config            # noqa: E402
from src.train import train, load_base_model, load_tokenizer, _subset  # noqa: E402
from src.data import load_medqa          # noqa: E402
from src import evaluate                 # noqa: E402

cfg = Config()
print("Config:", json.dumps(cfg.as_dict(), indent=2))

# --- 4. baseline eval --------------------------------------------------
tokenizer = load_tokenizer(cfg)
test_ds = _subset(load_medqa(cfg)["test"], cfg.max_eval_samples)

print("\n===== BASELINE (base model) =====", flush=True)
base_model = load_base_model(cfg, for_training=False)
base_model.eval()
base_res = evaluate.evaluate_model(base_model, tokenizer, test_ds, cfg)
print(f"BASE_ACCURACY={base_res['accuracy']:.4f} PARSE_RATE={base_res['parse_rate']:.4f}")

import gc  # noqa: E402
del base_model
gc.collect()
torch.cuda.empty_cache()

# --- 5. fine-tune ------------------------------------------------------
print("\n===== TRAINING (QLoRA) =====", flush=True)
adapter_dir = train(cfg)

# --- 6. fine-tuned eval ------------------------------------------------
print("\n===== FINE-TUNED EVAL =====", flush=True)
ft_model = evaluate.load_adapter_model(adapter_dir, cfg)
ft_res = evaluate.evaluate_model(ft_model, tokenizer, test_ds, cfg)
print(f"FT_ACCURACY={ft_res['accuracy']:.4f} PARSE_RATE={ft_res['parse_rate']:.4f}")

delta = ft_res["accuracy"] - base_res["accuracy"]
print(f"\nDELTA={delta:+.4f}  ({delta*100:+.1f} percentage points)")

# --- 7. persist a clean summary ---------------------------------------
summary = {
    "base_model": cfg.base_model,
    "dataset": cfg.dataset_name,
    "n_eval": base_res["n"],
    "base_accuracy": round(base_res["accuracy"], 4),
    "base_parse_rate": round(base_res["parse_rate"], 4),
    "finetuned_accuracy": round(ft_res["accuracy"], 4),
    "finetuned_parse_rate": round(ft_res["parse_rate"], 4),
    "absolute_improvement": round(delta, 4),
}
with open(os.path.join(WORK, "eval_results.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\nWROTE /kaggle/working/eval_results.json")
print(json.dumps(summary, indent=2))
