"""
create_colab_notebook.py - Programmatically creates the comprehensive notebooks/colab_runner.ipynb
supporting both Google Drive mounting and GitHub cloning workflows.
"""

import json
import os

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Dual-View RAG-VLM: IU X-Ray Report Generation Runner\n",
                "### Plan 2.1 Rigorous Implementation & SOTA Beating Pipeline\n",
                "\n",
                "This notebook trains and evaluates the **Dual-View RAG-VLM** architecture designed to surpass SOTA on the Indiana University Chest X-Ray (IU X-Ray) benchmark.\n",
                "\n",
                "#### SOTA Targets to Beat:\n",
                "* **ROUGE-L > 0.411** (DART CVPR 2025)\n",
                "* **METEOR > 0.316** (LePaX ECCV 2026)\n",
                "* Stretch: BLEU-4 > 0.235, BLEU-1 > 0.531 (LePaX 2026)\n",
                "\n",
                "#### Empirical Baseline Hurdle Floors (Test Set, N=589):\n",
                "* **E0 (Constant Normal)**: BLEU-1=0.2390, BLEU-4=0.0603, METEOR=0.1455, ROUGE-L=0.2809, CIDEr=0.1855\n",
                "* **E1 (Pure Retrieval Top-1)**: BLEU-1=0.3449, BLEU-4=0.0871, METEOR=0.1514, ROUGE-L=0.2513, CIDEr=0.1970\n",
                "* **Clinical Representation (RAD-DINO Probe)**: Macro-AUC = **0.8135** (Effusion 0.967, Pneumonia 0.941, Cardiomegaly 0.894)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 0: Hardware Verification & Dependencies"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Check GPU allocation (T4 15GB / L4 24GB / A100 40GB)\n",
                "!nvidia-smi\n",
                "\n",
                "# Install Java Runtime (required by Stanford PTBTokenizer for official pycocoevalcap scoring)\n",
                "!apt-get update -qq && apt-get install -y default-jre -qq\n",
                "!java -version\n",
                "\n",
                "# Install core Python dependencies\n",
                "!pip install -q pycocoevalcap rouge-score nltk faiss-cpu peft bitsandbytes accelerate torchvision transformers scikit-learn tqdm pandas tabulate"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 1: Workspace Selection (Option A: Google Drive vs Option B: GitHub Clone)\n",
                "\n",
                "Choose ONE of the two options below depending on how you store your files."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# =====================================================================\n",
                "# OPTION A: Google Drive Mount (Recommended if repo is in Google Drive)\n",
                "# =====================================================================\n",
                "from google.colab import drive\n",
                "drive.mount('/content/drive')\n",
                "\n",
                "# Change directory to your capstone project folder in Drive\n",
                "# (Adjust path if your folder has a different name or location)\n",
                "%cd /content/drive/MyDrive/capstone\n",
                "!ls -la"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# =====================================================================\n",
                "# OPTION B: Git Clone (If you pushed the repo to GitHub)\n",
                "# =====================================================================\n",
                "# Uncomment and run if cloning from GitHub:\n",
                "# !git clone https://github.com/<YOUR_USERNAME>/capstone.git /content/capstone\n",
                "# %cd /content/capstone\n",
                "# !ls -la"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 2: Verify Baseline Floors (E0 & E1)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Verify Constant Normal Baseline (E0)\n",
                "!python src/evaluate.py --mode normal --ref processed/test.json\n",
                "\n",
                "# Inspect E1 Retrieval Baseline & Linear Probe AUC logs\n",
                "import json\n",
                "with open('outputs/e1_retrieval_baseline.json') as f:\n",
                "    print('E1 Retrieval Baseline:', json.load(f)['headline_metrics'])\n",
                "with open('outputs/linear_probe_auc.json') as f:\n",
                "    print('RAD-DINO Linear Probe Macro-AUC:', json.load(f)['macro_auc'])"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 3: Feature Caching (Optional if already cached in `processed/feats/`)\n",
                "\n",
                "If `processed/feats/` is already present (e.g. from Google Drive), you can skip this step.\n",
                "If running in a fresh environment with raw images, this step extracts frozen RAD-DINO features in ~2.5 minutes."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "feats_count = len(os.listdir('processed/feats')) if os.path.exists('processed/feats') else 0\n",
                "print(f'Cached features found: {feats_count} / 2943 studies')\n",
                "\n",
                "if feats_count < 2943:\n",
                "    print('Running feature extraction and caching...')\n",
                "    !python src/cache_features.py --batch_size 16 --encoder microsoft/rad-dino --skip_existing\n",
                "else:\n",
                "    print('All 2,943 studies already pre-computed! Ready for training.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 4: Two-Stage Training (Projector Warmup + QLoRA SFT)\n",
                "\n",
                "- **Stage 1 (3 epochs)**: Warmup 2-layer MLP projector + 14-finding classifier head (lr=1e-3, base LLM frozen).\n",
                "- **Stage 2 (10 epochs)**: Full SFT with Qwen2.5-3B QLoRA (LoRA lr=1e-4, Projector lr=2e-5, Head lr=1e-4) with early stopping on validation ROUGE-L (patience 3)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!python src/train_sft.py \\\n",
                "    --stage 0 \\\n",
                "    --llm_name Qwen/Qwen2.5-3B-Instruct \\\n",
                "    --batch_size 4 \\\n",
                "    --grad_accum 4 \\\n",
                "    --epochs_s1 3 \\\n",
                "    --epochs_s2 10 \\\n",
                "    --seed 42"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Step 5: Fast MBR Consensus Decoding & Official Scoring\n",
                "\n",
                "- Generates candidate pool ($N=16$: 4 beam candidates + 12 diverse stochastic samples at $T=0.7, \\text{top-p}=0.9$).\n",
                "- Selects the consensus report using fast Python LCS proxy (`0.6*ROUGE-L + 0.4*ngram_F1`).\n",
                "- Executes official `pycocoevalcap` scoring exclusively on the final consensus outputs on the **589 test studies**."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run MBR consensus generation and official evaluation\n",
                "!python src/generate_mbr.py \\\n",
                "    --checkpoint_dir outputs/checkpoints/stage2_best \\\n",
                "    --llm_name Qwen/Qwen2.5-3B-Instruct \\\n",
                "    --mode mbr \\\n",
                "    --n_candidates 16 \\\n",
                "    --split test \\\n",
                "    --out_json outputs/test_mbr_predictions.json"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Step 6: Benchmark Comparison Table"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json\n",
                "import pandas as pd\n",
                "\n",
                "with open('outputs/test_mbr_predictions.json') as f:\n",
                "    mbr_data = json.load(f)\n",
                "    mbr_scores = mbr_data['headline_metrics']\n",
                "\n",
                "table = [\n",
                "    {'Model': 'DART / DATR (CVPR 2025)', 'BLEU-1': 0.486, 'BLEU-4': 0.208, 'METEOR': 0.205, 'ROUGE-L': 0.411},\n",
                "    {'Model': 'LePaX (ECCV 2026)', 'BLEU-1': 0.531, 'BLEU-4': 0.235, 'METEOR': 0.316, 'ROUGE-L': 0.402},\n",
                "    {'Model': 'MPDRL (Frontiers 2026)', 'BLEU-1': 0.508, 'BLEU-4': 0.185, 'METEOR': 0.231, 'ROUGE-L': 0.383},\n",
                "    {'Model': 'DAMPER (AAAI 2025)', 'BLEU-1': 0.520, 'BLEU-4': 0.225, 'METEOR': 0.284, 'ROUGE-L': 0.397},\n",
                "    {'Model': 'KiUT (CVPR 2023)', 'BLEU-1': 0.525, 'BLEU-4': 0.185, 'METEOR': 0.242, 'ROUGE-L': 0.409},\n",
                "    {'Model': 'Dual-View RAG-VLM (Ours)', 'BLEU-1': round(mbr_scores['BLEU-1'], 4), 'BLEU-4': round(mbr_scores['BLEU-4'], 4), 'METEOR': round(mbr_scores['METEOR'], 4), 'ROUGE-L': round(mbr_scores['ROUGE-L'], 4)}\n",
                "]\n",
                "\n",
                "df = pd.DataFrame(table)\n",
                "print('=== Benchmark Comparison on IU X-Ray Test Set (N=589) ===')\n",
                "print(df.to_markdown(index=False))\n",
                "\n",
                "# Check win conditions\n",
                "wins = []\n",
                "if mbr_scores['ROUGE-L'] > 0.411:\n",
                "    wins.append(f\"WIN: ROUGE-L ({mbr_scores['ROUGE-L']:.4f}) beats DART (0.411)\")\n",
                "if mbr_scores['METEOR'] > 0.316:\n",
                "    wins.append(f\"WIN: METEOR ({mbr_scores['METEOR']:.4f}) beats LePaX (0.316)\")\n",
                "if mbr_scores['BLEU-4'] > 0.235:\n",
                "    wins.append(f\"WIN: BLEU-4 ({mbr_scores['BLEU-4']:.4f}) beats LePaX (0.235)\")\n",
                "\n",
                "print('\\nOutcome:')\n",
                "for w in wins:\n",
                "    print('  *', w)\n",
                "if not wins:\n",
                "    print('  * Completed evaluation across all headline metrics.')"
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

print("Regenerated notebooks/colab_runner.ipynb with Option A (Drive) and Option B (Git) workflows!")
