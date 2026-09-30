"""
create_colab_notebook.py - Generates notebooks/colab_runner.ipynb
"""

import json
import os

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Dual-View RAG-VLM: IU X-Ray Report Generation Benchmark Runner\n",
                "### Plan 2.1 Rigorous Implementation & SOTA Beating Workflow\n",
                "\n",
                "This notebook executes the end-to-end pipeline designed in `plan2.md`:\n",
                "1. **Dual-View Radiograph Ingestion**: Frontal + Lateral paired inputs.\n",
                "2. **RAD-DINO Vision Encoder**: 518x518 px, pooled to 144 tokens per view (288 total visual tokens).\n",
                "3. **Train-Only FAISS RAG**: L2-normalized global vector retrieval with train-time query isolation (zero data leakage).\n",
                "4. **Clinical Prior Head**: 14-finding classification head and 5-fold OOF training priors.\n",
                "5. **Qwen2.5-3B QLoRA**: 4-bit NF4 fine-tuning with 2-layer MLP projection.\n",
                "6. **Fast MBR Consensus**: N=16 candidate sampling with local LCS proxy and final pycocoevalcap official scoring.\n",
                "\n",
                "**Benchmark Targets to Beat:**\n",
                "* **ROUGE-L > 0.411** (DART CVPR 2025)\n",
                "* **METEOR > 0.316** (LePaX ECCV 2026)\n",
                "* Stretch: BLEU-4 > 0.235 (LePaX 2026), BLEU-1 > 0.531 (LePaX 2026)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 0: Environment Setup & Hardware Verification"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Check GPU availability and specifications\n",
                "!nvidia-smi\n",
                "\n",
                "# Install Java Runtime (mandatory for official Stanford PTB tokenizer & pycocoevalcap METEOR/CIDEr)\n",
                "!apt-get update -qq && apt-get install -y default-jre -qq\n",
                "!java -version\n",
                "\n",
                "# Install Python dependencies\n",
                "!pip install -q pycocoevalcap rouge-score nltk faiss-cpu peft bitsandbytes accelerate torchvision transformers scikit-learn tqdm"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 1: Data Preparation & Integrity Locking"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run R2Gen-compatible data preparation\n",
                "# Discards single-view studies (~174) to match literature benchmark protocol (~2069 train / 296 val / 590 test)\n",
                "!python src/prepare_data.py --data_dir data --out_dir processed"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 2: Constant Normal Baseline (Hurdle Floor E0)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Establish the constant normal report floor with pycocoevalcap\n",
                "!python src/evaluate.py --mode normal --ref processed/test.json --out outputs/normal_baseline.json"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 3: Visual Feature Caching & FAISS Index Creation (E1 Baseline)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Extract frozen RAD-DINO features, build FAISS train index, evaluate E1 retrieval hurdle, and compute 5-fold OOF priors\n",
                "!python src/cache_features.py --batch_size 16 --encoder microsoft/rad-dino"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 4: Two-Stage Training (Stage 1 Warmup + Stage 2 QLoRA SFT)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Stage 1: Warmup MLP projector + classifier head (3 epochs, lr=1e-3, base LLM frozen)\n",
                "# Stage 2: Full SFT with Qwen2.5-3B QLoRA (10 epochs, early stopping patience 3)\n",
                "!python src/train_sft.py --stage 0 --batch_size 4 --grad_accum 4 --epochs_s1 3 --epochs_s2 10"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 5: Fast MBR Consensus Decoding & Official Evaluation"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Generate candidate pool (N=16) and select consensus report with fast Python LCS proxy\n",
                "# Evaluates official headline metrics using pycocoevalcap on the 589 test studies\n",
                "!python src/generate_mbr.py --checkpoint_dir outputs/checkpoints/stage2_best --mode mbr --n_candidates 16 --split test"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 6: Headline Comparison Against Benchmark Literature"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json, pandas as pd\n",
                "\n",
                "with open('outputs/test_mbr_predictions.json') as f:\n",
                "    mbr_res = json.load(f)['headline_metrics']\n",
                "\n",
                "benchmarks = [\n",
                "    {'Model': 'DART (CVPR 2025)', 'BLEU-1': 0.486, 'BLEU-4': 0.208, 'METEOR': 0.205, 'ROUGE-L': 0.411},\n",
                "    {'Model': 'LePaX (ECCV 2026)', 'BLEU-1': 0.531, 'BLEU-4': 0.235, 'METEOR': 0.316, 'ROUGE-L': 0.402},\n",
                "    {'Model': 'MPDRL (Frontiers 2026)', 'BLEU-1': 0.508, 'BLEU-4': 0.185, 'METEOR': 0.231, 'ROUGE-L': 0.383},\n",
                "    {'Model': 'DAMPER (AAAI 2025)', 'BLEU-1': 0.520, 'BLEU-4': 0.225, 'METEOR': 0.284, 'ROUGE-L': 0.397},\n",
                "    {'Model': 'KiUT (CVPR 2023)', 'BLEU-1': 0.525, 'BLEU-4': 0.185, 'METEOR': 0.242, 'ROUGE-L': 0.409},\n",
                "    {'Model': 'Dual-View RAG-VLM (Ours)', 'BLEU-1': mbr_res['BLEU-1'], 'BLEU-4': mbr_res['BLEU-4'], 'METEOR': mbr_res['METEOR'], 'ROUGE-L': mbr_res['ROUGE-L']}\n",
                "]\n",
                "\n",
                "df = pd.DataFrame(benchmarks)\n",
                "print(df.to_markdown(index=False))\n"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"}
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

os.makedirs("notebooks", exist_ok=True)
with open("notebooks/colab_runner.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print("notebooks/colab_runner.ipynb generated successfully!")
