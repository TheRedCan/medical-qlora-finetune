# Fine-Tuning a Small LLM for Medical Structured Extraction (QLoRA)

Fine-tune a small open base model (**Qwen2.5-1.5B**) with **QLoRA** to turn
free clinical text into **strict JSON** of disease mentions — and prove the
gain is real with paired significance testing, a leakage check, and an
overfitting diagnostic.

> **Headline result (1,500 held-out sentences):** entity **micro-F1 0.33 → 0.84**
> (+0.51, 95% CI [+0.485, +0.537], p≈0). JSON-valid rate 0.96 → 1.00.
> Exact-set match 0.21 → 0.82 (McNemar 955 vs 32, p≈0).

```
Input:  "...chronic severe heart failure secondary to dilated cardiomyopathy
         with ventricular arrhythmias and QT prolongation..."
Output: {"diseases": ["heart failure", "dilated cardiomyopathy",
                       "ventricular arrhythmias", "QT prolongation", ...]}
```

---

## What this project demonstrates

Beyond "I can run QLoRA," it shows judgment about **when fine-tuning actually
helps** — and the discipline to *measure* it honestly:

| Skill | Where |
|-------|-------|
| **QLoRA / LoRA** | 4-bit NF4 base + LoRA adapters, prompt-masked loss ([`src/train.py`](src/train.py)) |
| **Knowing what SFT can/can't do** | Pivoted from a *knowledge*-bound task (MCQA, where SFT failed) to a *behavior* task (extraction, where it wins) — see below |
| **Rigorous evaluation** | Paired **McNemar** test + **bootstrap F1 CI** ([`src/stats.py`](src/stats.py)), not just point estimates |
| **Validation against artifacts** | Train/test **leakage check**, **overfitting** diagnostic (train-vs-test F1 gap), qualitative review |
| **Reproducibility & testing** | Config-driven, seeded, **62 GPU-free unit tests** for the core logic |

---

## The key insight (and an honest journey)

I first tried the obvious thing — **fine-tune to improve medical multiple-choice
accuracy** (MedMCQA / MedQA). Across **four** configurations (3B instruct, 3B
base, 0.5B base; direct and chain-of-thought) QLoRA SFT was **neutral-to-slightly-negative**
and never significant. That's not a bug — it's a real lesson:

> **MCQA accuracy is *knowledge*-bound.** Modern small models already know the
> answer format (parse rate ≈ 1.0), so SFT has nothing to teach there — and you
> can't reliably *inject medical knowledge* into a small model with a few
> thousand Q&A pairs. Fine-tuning is great at **format and behavior**, weak at
> **facts.**

So I pivoted to a task that plays to SFT's strength: **structured extraction.**
A base (non-instruct) model can't reliably emit a strict JSON schema on command;
fine-tuning teaches the *behavior*, and the gain is large, reliable, and
significant. The negative MCQA results are kept as part of the story — knowing
*why* an approach fails is the point.

---

## Results

Qwen2.5-1.5B (base) + QLoRA, trained on 8k examples (2 epochs) of
[`rjac/biobert-ner-diseases-dataset`](https://huggingface.co/datasets/rjac/biobert-ner-diseases-dataset),
evaluated on 1,500 held-out test sentences. Both models use the **same**
zero-shot prompt (fair comparison). Full metrics:
[`results/extraction_results.json`](results/extraction_results.json).

| Metric | Base | Fine-tuned | Δ | Significance |
|--------|-----:|-----------:|--:|--------------|
| Entity micro-F1 | 0.330 | **0.840** | **+0.510** | bootstrap 95% CI [+0.485, +0.537], **p≈0** |
| Exact-set match | 0.205 | **0.821** | **+0.615** | McNemar χ²=861, **p≈0** |
| JSON-valid rate | 0.956 | **1.000** | +0.044 | — |

**What it learned (from qualitative review):** to exclude drugs (`anorexigens`
→ `[]`), fix entity boundaries, and normalize casing — i.e. the dataset's
genuine annotation conventions, not metric-gaming.

### Validation

- **Leakage:** only 13/5,724 test sentences appear in train (0.2%), all trivial
  fragments (`"case report ."`) with no entities → the held-out test is clean.
- **Overfitting:** fine-tuned F1 is **0.97 on trained-on examples vs 0.84 on
  test** — a 0.13 gap. It generalizes strongly, with *mild* overfitting; the
  1-epoch pilot reached 0.82 test F1 with less overfit, so **1 epoch is the
  sweet spot** (2 epochs mostly raised train F1).
- **Significance is paired** (same sentences, both models) — McNemar/bootstrap,
  not an unpaired CI.

### Honest scope

The test set shares the training corpus's annotation conventions, so 0.84 F1 is
**in-distribution** disease extraction ("learned *this* task well"), not a claim
of universal medical NER. Cross-dataset robustness is future work.

---

## How it works

```
{tokens, BIO tags}                      # rjac disease-NER dataset
  └─ bio_to_entities()                  # -> ["heart failure", ...]   (tested)
       └─ build_extraction_prompt()     # "Text: ...\nJSON:"  + {"diseases":[...]}
            ├─ train: prompt masked to -100, loss only on the JSON target
            └─ eval:  generate -> parse_diseases() -> micro-F1 / exact-match
```

- **Quantization:** bitsandbytes 4-bit NF4 + double quant (compute dtype auto:
  bf16 on Ampere+, fp16 on a T4).
- **Adapters:** LoRA on attention + MLP projections (`r=16`, `α=32`).
- **Base model:** non-instruct, so prompts are plain text (no chat template) —
  this is *why* there's headroom for SFT.
- **Fast eval:** fp16 with the adapter merged (4-bit generation is slow).

---

## Reproduce

The full experiment runs headlessly on a free **Kaggle T4** via
[`kaggle/run_cot.py`](kaggle/run_cot.py) (orchestration) launched by the thin
loader [`kaggle/run_kaggle.py`](kaggle/run_kaggle.py). Locally:

```bash
pip install -r requirements.txt

# Train + evaluate the extraction model (set PILOT=1 in the script for a smoke test)
TASK=extraction BASE_MODEL=Qwen/Qwen2.5-1.5B USE_CHAT_TEMPLATE=0 \
DATASET_NAME=rjac/biobert-ner-diseases-dataset python -m src.train
```

Everything is driven by [`src/config.py`](src/config.py) (env-overridable).

## Tests

```bash
pytest -q          # 62 GPU-free tests: BIO conversion, JSON parsing, scoring,
                   # McNemar, bootstrap, prompt formatting
```

## Project layout

```
src/
  config.py       # one env-overridable source of truth
  data.py         # MCQA loaders + prompt formatting (the exploration phase)
  extraction.py   # the extraction task: BIO->JSON, parse, scoring, paired eval
  stats.py        # McNemar + bootstrap-F1 significance (tested)
  train.py        # QLoRA training (MCQA + extraction), prompt-masked loss
  evaluate.py     # paired base-vs-fine-tuned eval, fp16 fast inference
tests/            # 62 unit tests (test_data / test_stats / test_extraction)
kaggle/           # headless T4 runner (loader + orchestration)
results/          # extraction_results.json (final metrics + examples)
notebooks/        # Colab notebook from the earlier MCQA exploration
```

## Notes & limitations

- Educational artifact, **not** a clinical tool.
- Result is in-distribution disease extraction (see scope above).
- The baseline is zero-shot; a few-shot-prompted base would be a stronger (but
  the comparison here is fair — identical prompt for both models).
