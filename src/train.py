"""QLoRA fine-tuning of a small instruct LLM on MedQA.

Run as a script (CLI) or import `train()` from a notebook:

    python -m src.train

Key techniques demonstrated:
  * 4-bit NF4 quantization (QLoRA) via bitsandbytes
  * LoRA adapters via PEFT
  * **prompt-token masking** so the loss is computed only on the answer,
    not on the question (cleaner signal than training on the full text)
"""
from __future__ import annotations

import os
from typing import Dict, List

from .config import Config, CONFIG
from .data import apply_chat_template, build_messages, build_plain_prompt, load_medqa


def _compute_dtype():
    """Pick bf16 on Ampere+ (e.g. A100), fp16 on Turing (e.g. Colab T4)."""
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_tokenizer(config: Config):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.base_model, use_fast=True)
    if tok.pad_token is None:
        # Causal LMs often lack a pad token; reuse EOS for padding.
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_base_model(config: Config, for_training: bool = True):
    """Load the base model in 4-bit. Used for both training and the
    'before' evaluation, so eval and train see identical quantization."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=_compute_dtype(),
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=_compute_dtype(),
    )
    model.config.use_cache = not for_training  # cache off while training
    return model


def tokenize_with_masking(example: Dict, tokenizer, config: Config) -> Dict:
    """Build input_ids + labels where prompt tokens are masked to -100.

    We render the prompt (system+user, with the generation prefix) and the
    full conversation (prompt + gold answer) using the tokenizer's chat
    template, then mask everything up to the answer so loss is computed on
    the completion only.
    """
    if config.use_chat_template:
        prompt_ids = apply_chat_template(
            tokenizer, build_messages(example, include_answer=False, cot=config.use_cot),
            tokenize=True, add_generation_prompt=True,
        )
        full_ids = apply_chat_template(
            tokenizer, build_messages(example, include_answer=True, cot=config.use_cot),
            tokenize=True, add_generation_prompt=False,
        )
    else:  # base model: plain-text prompt, tokenize directly
        prompt_str = build_plain_prompt(example, include_answer=False, cot=config.use_cot)
        full_str = build_plain_prompt(example, include_answer=True, cot=config.use_cot)
        prompt_ids = tokenizer(prompt_str, add_special_tokens=True)["input_ids"]
        full_ids = tokenizer(full_str, add_special_tokens=True)["input_ids"]

    # Append EOS so the model learns to stop.
    if tokenizer.eos_token_id is not None and full_ids[-1] != tokenizer.eos_token_id:
        full_ids = full_ids + [tokenizer.eos_token_id]

    full_ids = full_ids[: config.max_seq_length]
    labels = list(full_ids)
    prompt_len = min(len(prompt_ids), len(full_ids))
    for i in range(prompt_len):
        labels[i] = -100  # mask the prompt

    attention_mask = [1] * len(full_ids)
    return {"input_ids": full_ids, "attention_mask": attention_mask, "labels": labels}


class CausalDataCollator:
    """Pad input_ids/attention_mask/labels to the longest item in a batch.
    Labels are padded with -100 so padding never contributes to the loss."""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features: List[Dict]) -> Dict:
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_id] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
            batch["labels"].append(f["labels"] + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def build_lora_model(model, config: Config):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=config.gradient_checkpointing
    )
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def _subset(dataset, n: int):
    if n and n > 0 and n < len(dataset):
        return dataset.select(range(n))
    return dataset


def load_training_split(config: Config):
    """Load the training split, normalized to the shared schema. MedMCQA is
    used for CoT (it ships explanations); MedQA otherwise."""
    from .data import load_medmcqa

    if "medmcqa" in config.dataset_name.lower():
        ds = load_medmcqa(config.dataset_name, require_exp=config.use_cot)
        return ds["train"]
    return load_medqa(config)["train"]


def train(config: Config = CONFIG):
    from transformers import Trainer, TrainingArguments

    tokenizer = load_tokenizer(config)

    if config.task == "extraction":
        from .extraction import load_ner, tokenize_extraction
        print(f"Loading NER training data ({config.dataset_name}) ...")
        full_train = load_ner(config.dataset_name)["train"].shuffle(seed=config.seed)
        train_ds = _subset(full_train, config.max_train_samples)
        tok_fn = lambda ex: tokenize_extraction(ex, tokenizer, config)  # noqa: E731
    else:
        style = "CoT" if config.use_cot else "direct"
        print(f"Loading + normalizing training data ({config.dataset_name}, {style}) ...")
        full_train = load_training_split(config).shuffle(seed=config.seed)
        train_ds = _subset(full_train, config.max_train_samples)
        tok_fn = lambda ex: tokenize_with_masking(ex, tokenizer, config)  # noqa: E731

    print(f"Tokenizing {len(train_ds)} training examples (prompt-masked) ...")
    tokenized = train_ds.map(tok_fn, remove_columns=train_ds.column_names, desc="tokenizing")

    print("Loading base model in 4-bit + attaching LoRA ...")
    model = load_base_model(config, for_training=True)
    model = build_lora_model(model, config)

    import torch

    args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=2,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",  # memory-friendly optimizer (QLoRA)
        bf16=_compute_dtype() == torch.bfloat16,
        fp16=_compute_dtype() == torch.float16,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if config.gradient_checkpointing else None,
        report_to="none",
        seed=config.seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=CausalDataCollator(tokenizer),
    )

    trainer.train()

    adapter_dir = os.path.join(config.output_dir, "adapter")
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\nLoRA adapter saved to: {adapter_dir}")
    return adapter_dir


if __name__ == "__main__":
    train()
