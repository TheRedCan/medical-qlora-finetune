# Medical-Domain LLM Fine-Tuning with QLoRA

Fine-tune a small open-source instruct LLM (**Mistral-7B-Instruct**, swappable
to Llama / Qwen) on **MedQA (USMLE)** medical multiple-choice questions using
**QLoRA** (4-bit NF4 + LoRA adapters), then measure the improvement with a
clean **exact-match accuracy** benchmark — base model vs. fine-tuned.

> Built to run end-to-end on a **free Google Colab T4 (16 GB)**. The training
> loop also runs locally on an 8 GB GPU if you switch to the 3B model.

---

## Why this project

It demonstrates the full applied-fine-tuning workflow that production LLM work
relies on:

| Skill | Where it shows up |
|-------|-------------------|
| **QLoRA / LoRA** | 4-bit NF4 quantization + low-rank adapters ([`src/train.py`](src/train.py)) |
| **Data engineering** | Schema-normalizing a real HF dataset, chat-template formatting ([`src/data.py`](src/data.py)) |
| **Correct loss masking** | Loss computed on the *answer only*, prompt tokens masked to `-100` |
| **Rigorous evaluation** | Deterministic greedy decoding, before/after accuracy, parse-rate sanity check ([`src/evaluate.py`](src/evaluate.py)) |
| **Reproducibility & testing** | Config-driven, seeded, GPU-free `pytest` suite for the core logic |

---

## Results

QLoRA fine-tune of **Qwen2.5-3B-Instruct** on 4,000 MedQA training examples
(1 epoch), evaluated on a 300-question held-out test sample with greedy
(deterministic) decoding. Run on a single Kaggle T4. Full metrics in
[`results/eval_results.json`](results/eval_results.json).

| Model | MedQA test accuracy | Δ vs base |
|-------|--------------------:|----------:|
| Qwen2.5-3B-Instruct (base, 4-bit) | 47.67% | — |
| **+ QLoRA fine-tune (this repo)** | **50.33%** | **+2.67 pp** |

**Parse rate: 1.00 for both** — every generation yielded a parseable answer
letter, so the accuracy delta reflects genuine answer quality, not formatting
luck. A few points from one epoch of QLoRA is the realistic, honest outcome on
USMLE-level questions; the point of the project is the *methodology* (see below),
and the harness scales straight to more epochs / the full train split / a larger
base model for a bigger gain.

---

## Quickstart

### Option A — Google Colab (recommended)

1. Open [`notebooks/finetune_medical_qlora.ipynb`](notebooks/finetune_medical_qlora.ipynb) in Colab.
2. Set the runtime to **GPU** (T4 is enough).
3. Run all cells. The notebook installs deps, runs a baseline eval, fine-tunes,
   re-evaluates, and shows demo predictions.

> **Mistral is a gated model.** Accept the license at
> [huggingface.co/mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
> and paste an HF token when prompted. To skip gating entirely, set
> `BASE_MODEL=Qwen/Qwen2.5-3B-Instruct` (Apache-2.0, ungated).

### Option B — Local

```bash
pip install -r requirements.txt

# Train (8 GB GPU: use the smaller, ungated model)
export BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
python -m src.train

# Evaluate base vs fine-tuned
python -m src.evaluate --adapter outputs/medqa-qlora/adapter

# Ask a single question
python -m src.inference --adapter outputs/medqa-qlora/adapter \
  --question "A 22-year-old presents with..." \
  --options "A) Option one B) Option two C) Option three D) Option four"
```

---

## How it works

```
raw MedQA row
  └─ normalize_example()      # → {question, options{A..D}, answer_letter}
       └─ build_messages()    # → chat-templated [system, user, (assistant)]
            ├─ training:  prompt + gold answer, prompt tokens masked to -100
            └─ eval:      prompt only → model.generate() → extract_answer_letter()
```

- **Quantization:** `bitsandbytes` 4-bit NF4 with double quantization; compute
  dtype auto-selects bf16 (Ampere+) or fp16 (Colab T4).
- **Adapters:** LoRA on all attention + MLP projections (`r=16`, `alpha=32`).
- **Optimizer:** `paged_adamw_8bit` + gradient checkpointing to fit in memory.
- **Eval is honest:** greedy/deterministic, and reports a **parse rate** so a
  high "accuracy" can't hide behind unparseable generations.

Everything is driven by [`src/config.py`](src/config.py) and overridable via
environment variables (no code edits needed for a different model/dataset).

---

## Tests

```bash
pytest -q          # 17 GPU-free tests covering formatting + answer parsing
```

The pure logic the pipeline depends on (schema normalization, prompt building,
answer-letter extraction) is unit-tested so it stays correct independent of any
training run.

---

## Project layout

```
src/
  config.py      # one source of truth, env-overridable
  data.py        # dataset loading + prompt/answer logic (tested)
  train.py       # QLoRA training w/ prompt-masked loss
  evaluate.py    # base-vs-finetuned accuracy benchmark
  inference.py   # single-question / interactive use
tests/
  test_data.py   # GPU-free unit tests
notebooks/
  finetune_medical_qlora.ipynb   # Colab end-to-end
```

## Notes & limitations

- MedQA (USMLE) is hard; a single-epoch QLoRA on a 7B model yields a modest but
  real gain — the point is the *methodology*, not a SOTA score.
- This is an educational artifact and **not medical advice**.
- For a stronger score: more epochs, the full train split, or a larger base
  model on a bigger GPU.
