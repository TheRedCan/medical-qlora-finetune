"""Stable Kaggle kernel entry point (thin loader).

Deliberately minimal so it almost never changes: it clones the repo fresh,
installs deps, and hands off to the repo's orchestration script
(``kaggle/run_cot.py``). All experiment logic lives in that script, so it can
be iterated via ordinary GitHub pushes WITHOUT re-pushing the kernel (which
would reset the accelerator to P100 and trigger a wasted run).
"""
import os
import shutil
import subprocess
import sys

REPO = "https://github.com/TheRedCan/medical-qlora-finetune.git"
WORK = "/kaggle/working"
repo_dir = os.path.join(WORK, "repo")

# Always clone fresh so we run the latest main.
if os.path.exists(repo_dir):
    shutil.rmtree(repo_dir, ignore_errors=True)
subprocess.run(["git", "clone", "--depth", "1", REPO, repo_dir], check=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.45", "peft>=0.13", "bitsandbytes>=0.43",
                "accelerate>=0.34", "datasets>=3.0", "sentencepiece>=0.2"], check=True)

subprocess.run([sys.executable, os.path.join(repo_dir, "kaggle", "run_cot.py")],
               check=True, cwd=repo_dir)
