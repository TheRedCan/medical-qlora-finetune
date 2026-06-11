"""Generate notebooks/finetune_medical_qlora.ipynb from source-of-truth cells.

Keeping the notebook in a builder script means it never drifts from the
`src/` modules and stays diff-friendly in git. Run:  python scripts/build_notebook.py
"""
import os
import nbformat as nbf

REPO_URL = "https://github.com/TheRedCan/medical-qlora-finetune.git"

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md(
        "# Medical-Domain LLM Fine-Tuning with QLoRA\n"
        "\n"
        "Fine-tune **Mistral-7B-Instruct** (swappable) on **MedQA (USMLE)** with "
        "**QLoRA**, then benchmark base vs. fine-tuned accuracy.\n"
        "\n"
        "**Runtime → Change runtime type → GPU (T4 is enough).**"
    ),
    md("## 1. Setup"),
    code(
        "# Confirm we have a GPU\n"
        "import torch\n"
        "assert torch.cuda.is_available(), 'No GPU! Runtime > Change runtime type > GPU'\n"
        "print(torch.cuda.get_device_name(0))"
    ),
    code(
        "# Clone the project (skip if you uploaded it manually)\n"
        f"REPO_URL = '{REPO_URL}'\n"
        "import os\n"
        "if not os.path.exists('medical-qlora-finetune'):\n"
        "    !git clone $REPO_URL\n"
        "%cd medical-qlora-finetune"
    ),
    code("!pip install -q -r requirements.txt"),
    md(
        "## 2. Hugging Face login\n"
        "Mistral is **gated** — accept the license at "
        "[huggingface.co/mistralai/Mistral-7B-Instruct-v0.3]"
        "(https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) and paste a "
        "token below. To skip gating, set `BASE_MODEL` to an ungated model in the "
        "next cell and skip this one."
    ),
    code(
        "from huggingface_hub import login\n"
        "login()  # paste your HF token when prompted"
    ),
    md("## 3. Configuration\nOverride anything here before importing the package."),
    code(
        "import os\n"
        "# --- pick your base model ---\n"
        "os.environ['BASE_MODEL'] = 'mistralai/Mistral-7B-Instruct-v0.3'\n"
        "# Ungated alternative (no HF gating, runs on 8GB):\n"
        "# os.environ['BASE_MODEL'] = 'Qwen/Qwen2.5-3B-Instruct'\n"
        "\n"
        "# --- keep the run short for a free T4; raise for better results ---\n"
        "os.environ['MAX_TRAIN_SAMPLES'] = '4000'\n"
        "os.environ['MAX_EVAL_SAMPLES']  = '300'\n"
        "os.environ['NUM_TRAIN_EPOCHS']  = '1'\n"
        "\n"
        "from src.config import Config\n"
        "cfg = Config()\n"
        "cfg.as_dict()"
    ),
    md(
        "## 4. Baseline: how good is the model *before* fine-tuning?\n"
        "We evaluate the raw 4-bit base model first so we have an honest "
        "before/after comparison."
    ),
    code(
        "from src import evaluate\n"
        "from src.train import load_base_model, load_tokenizer, _subset\n"
        "from src.data import load_medqa\n"
        "\n"
        "tokenizer = load_tokenizer(cfg)\n"
        "test_ds = _subset(load_medqa(cfg)['test'], cfg.max_eval_samples)\n"
        "\n"
        "base_model = load_base_model(cfg, for_training=False); base_model.eval()\n"
        "base_results = evaluate.evaluate_model(base_model, tokenizer, test_ds, cfg)\n"
        "print('Base accuracy:', round(base_results['accuracy'], 3),\n"
        "      '| parse rate:', round(base_results['parse_rate'], 3))\n"
        "\n"
        "# free VRAM before training\n"
        "import gc; del base_model; gc.collect(); torch.cuda.empty_cache()"
    ),
    md("## 5. Fine-tune with QLoRA\nTrains LoRA adapters with prompt-masked loss. ~20–40 min on a T4 at these settings."),
    code(
        "from src.train import train\n"
        "adapter_dir = train(cfg)\n"
        "print('Adapter saved to', adapter_dir)"
    ),
    md("## 6. Evaluate the fine-tuned model"),
    code(
        "ft_model = evaluate.load_adapter_model(adapter_dir, cfg)\n"
        "ft_results = evaluate.evaluate_model(ft_model, tokenizer, test_ds, cfg)\n"
        "\n"
        "print('Base       accuracy:', round(base_results['accuracy'], 3))\n"
        "print('Fine-tuned accuracy:', round(ft_results['accuracy'], 3))\n"
        "delta = ft_results['accuracy'] - base_results['accuracy']\n"
        "print(f'Improvement: {delta*100:+.1f} percentage points')"
    ),
    md("## 7. Try it — qualitative demo"),
    code(
        "from src.inference import answer_question\n"
        "\n"
        "q = ('A 25-year-old woman presents with fatigue, weight gain, cold "
        "intolerance, and constipation. Lab shows elevated TSH and low free T4. "
        "What is the most likely diagnosis?')\n"
        "opts = {'A': 'Graves disease', 'B': 'Hashimoto thyroiditis',\n"
        "        'C': 'Pheochromocytoma', 'D': 'Cushing syndrome'}\n"
        "\n"
        "print(answer_question(ft_model, tokenizer, q, opts, cfg))"
    ),
    md(
        "## 8. (Optional) Push the adapter to the Hub\n"
        "```python\n"
        "ft_model.push_to_hub('your-username/mistral-7b-medqa-qlora')\n"
        "tokenizer.push_to_hub('your-username/mistral-7b-medqa-qlora')\n"
        "```"
    ),
]

nb["cells"] = cells
nb["metadata"] = {
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}

out = os.path.join(os.path.dirname(__file__), "..", "notebooks", "finetune_medical_qlora.ipynb")
out = os.path.abspath(out)
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", out)
