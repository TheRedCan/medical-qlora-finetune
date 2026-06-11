"""Central configuration for the medical QLoRA fine-tuning project.

Everything tunable lives here so the notebook, training script, and eval
script all read from one source of truth. Override any field from the
environment (e.g. on Colab) without editing code, e.g.:

    import os; os.environ["BASE_MODEL"] = "Qwen/Qwen2.5-3B-Instruct"
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import List


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


@dataclass
class Config:
    # --- Model -----------------------------------------------------------
    # Default matches the brief (Mistral). It is a *gated* HF model: a
    # reviewer must accept the license at huggingface.co/mistralai and set
    # HF_TOKEN. For a frictionless ungated run, set:
    #   BASE_MODEL=Qwen/Qwen2.5-3B-Instruct   (Apache-2.0, no gating)
    base_model: str = field(default_factory=lambda: _env("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"))

    # --- Dataset ---------------------------------------------------------
    # MedQA (USMLE) with 4 answer options -> clean exact-match accuracy.
    dataset_name: str = field(default_factory=lambda: _env("DATASET_NAME", "GBaker/MedQA-USMLE-4-options"))
    # Cap examples so a free-tier T4 finishes in a sensible time. Set to 0
    # (or a negative number) to use the full split.
    max_train_samples: int = field(default_factory=lambda: _env_int("MAX_TRAIN_SAMPLES", 4000))
    max_eval_samples: int = field(default_factory=lambda: _env_int("MAX_EVAL_SAMPLES", 300))
    max_seq_length: int = field(default_factory=lambda: _env_int("MAX_SEQ_LENGTH", 1024))

    # --- LoRA / QLoRA ----------------------------------------------------
    lora_r: int = field(default_factory=lambda: _env_int("LORA_R", 16))
    lora_alpha: int = field(default_factory=lambda: _env_int("LORA_ALPHA", 32))
    lora_dropout: float = field(default_factory=lambda: _env_float("LORA_DROPOUT", 0.05))
    # Attention + MLP projections. Covers Mistral/Llama/Qwen architectures.
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # --- Training --------------------------------------------------------
    output_dir: str = field(default_factory=lambda: _env("OUTPUT_DIR", "outputs/medqa-qlora"))
    num_train_epochs: float = field(default_factory=lambda: _env_float("NUM_TRAIN_EPOCHS", 1.0))
    per_device_train_batch_size: int = field(default_factory=lambda: _env_int("TRAIN_BATCH_SIZE", 2))
    gradient_accumulation_steps: int = field(default_factory=lambda: _env_int("GRAD_ACCUM", 8))
    learning_rate: float = field(default_factory=lambda: _env_float("LEARNING_RATE", 2e-4))
    warmup_ratio: float = field(default_factory=lambda: _env_float("WARMUP_RATIO", 0.03))
    weight_decay: float = field(default_factory=lambda: _env_float("WEIGHT_DECAY", 0.0))
    logging_steps: int = field(default_factory=lambda: _env_int("LOGGING_STEPS", 10))
    save_steps: int = field(default_factory=lambda: _env_int("SAVE_STEPS", 200))
    seed: int = field(default_factory=lambda: _env_int("SEED", 42))

    # --- Generation (eval / inference) -----------------------------------
    max_new_tokens: int = field(default_factory=lambda: _env_int("MAX_NEW_TOKENS", 64))

    def as_dict(self) -> dict:
        return asdict(self)


# Convenience singleton used by scripts; the notebook can build its own.
CONFIG = Config()
