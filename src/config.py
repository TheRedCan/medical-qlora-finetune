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

    # --- Task ------------------------------------------------------------
    # "mcqa": multiple-choice QA. "extraction": clinical text -> JSON of
    # disease mentions (a format/behavior task SFT reliably improves).
    task: str = field(default_factory=lambda: _env("TASK", "mcqa"))

    # --- Dataset ---------------------------------------------------------
    # Training set. MedMCQA ships an `exp` rationale we turn into CoT targets.
    dataset_name: str = field(default_factory=lambda: _env("DATASET_NAME", "openlifescienceai/medmcqa"))
    # Cap examples so a free-tier T4 finishes in a sensible time. Set to 0
    # (or a negative number) to use the full split.
    max_train_samples: int = field(default_factory=lambda: _env_int("MAX_TRAIN_SAMPLES", 8000))
    max_eval_samples: int = field(default_factory=lambda: _env_int("MAX_EVAL_SAMPLES", 1000))
    max_seq_length: int = field(default_factory=lambda: _env_int("MAX_SEQ_LENGTH", 1024))

    # --- Chain-of-thought ------------------------------------------------
    # When True, train on reasoning targets (explanation -> answer line) and
    # prompt the model to think step by step at eval.
    use_cot: bool = field(default_factory=lambda: _env("USE_COT", "1") == "1")

    # Instruct models use their chat template; base (non-instruct) models have
    # none, so we render a plain-text prompt instead. Set USE_CHAT_TEMPLATE=0
    # for base models.
    use_chat_template: bool = field(default_factory=lambda: _env("USE_CHAT_TEMPLATE", "1") == "1")

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
    save_steps: int = field(default_factory=lambda: _env_int("SAVE_STEPS", 500))
    seed: int = field(default_factory=lambda: _env_int("SEED", 42))
    # Gradient checkpointing trades compute for memory. Disabling it is much
    # faster on a T4 when the (small, 4-bit) model already fits.
    gradient_checkpointing: bool = field(default_factory=lambda: _env("GRAD_CHECKPOINT", "0") == "1")

    # --- Generation (eval / inference) -----------------------------------
    # CoT needs room to reason; raise vs. the old answer-only default.
    max_new_tokens: int = field(default_factory=lambda: _env_int("MAX_NEW_TOKENS", 256))
    eval_batch_size: int = field(default_factory=lambda: _env_int("EVAL_BATCH_SIZE", 16))
    # Evaluate in fp16 with the adapter merged: bitsandbytes 4-bit generation
    # is slow, and a 3B model fits a T4 comfortably in fp16.
    eval_in_fp16: bool = field(default_factory=lambda: _env("EVAL_IN_FP16", "1") == "1")

    def as_dict(self) -> dict:
        return asdict(self)


# Convenience singleton used by scripts; the notebook can build its own.
CONFIG = Config()
