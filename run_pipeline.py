"""
run_pipeline.py - Complete End-to-End Pipeline Runner (Converted from colab_runner.ipynb).

Executes the entire Plan 2.1 workflow:
  1. System & Hardware Verification (Auto-detects VRAM and adjusts batch size)
  2. Data Splits & Baseline Floors Check (E0 Normal Baseline & E1 Retrieval Hurdle)
  3. Feature Cache Verification (processed/feats/ 2,943 studies)
  4. Two-Stage Training (Stage 1 Warmup + Stage 2 QLoRA SFT)
  5. Fast MBR Consensus Decoding & Official Scoring (N=16 on 589 test studies)
  6. Final Benchmark Comparison Table vs DART, LePaX, MPDRL, DAMPER, KiUT

Works seamlessly on Linux (Lightning AI / Google Colab) and Windows local machine.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import pandas as pd
import torch


def log_header(title: str):
    print("\n" + "=" * 70)
    print(f" {title.upper()}")
    print("=" * 70)


def step_0_check_system():
    log_header("Step 0: System & Hardware Verification")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")

    vram_gb = 0.0
    if cuda_avail:
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {device_name} ({vram_gb:.2f} GB VRAM)")
    else:
        print("WARNING: CUDA not available. Running on CPU will be extremely slow.")

    # Check Java for pycocoevalcap Stanford PTBTokenizer
    java_installed = shutil.which("java") is not None
    if java_installed:
        try:
            java_ver = subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT, text=True)
            print("Java Runtime: Detected (" + java_ver.splitlines()[0] + ")")
        except Exception:
            print("Java Runtime: Detected")
    else:
        print("WARNING: Java is not installed on PATH! (Required for official METEOR/CIDEr scoring)")

    return vram_gb


def step_1_check_baselines():
    log_header("Step 1: Baseline Floors (E0 & E1)")

    normal_path = "outputs/normal_baseline.json"
    if os.path.exists(normal_path):
        with open(normal_path) as f:
            data = json.load(f)
            e0 = data.get("overall", data.get("headline_metrics", {}))
        print("E0 Constant Normal Report Floor (Test Set, N=589):")
        print(f"  BLEU-1: {e0['BLEU-1']:.4f}  BLEU-4: {e0['BLEU-4']:.4f}  METEOR: {e0['METEOR']:.4f}  ROUGE-L: {e0['ROUGE-L']:.4f}")
    else:
        print("Running E0 normal baseline evaluation...")
        cmd = [sys.executable, "src/evaluate.py", "--mode", "normal", "--ref", "processed/test.json", "--out", normal_path]
        subprocess.run(cmd, check=True)

    e1_path = "outputs/e1_retrieval_baseline.json"
    if os.path.exists(e1_path):
        with open(e1_path) as f:
            data = json.load(f)
            e1 = data.get("headline_metrics", data.get("overall", {}))
        print("E1 Pure Retrieval Top-1 Hurdle (Test Set, N=589):")
        print(f"  BLEU-1: {e1['BLEU-1']:.4f}  BLEU-4: {e1['BLEU-4']:.4f}  METEOR: {e1['METEOR']:.4f}  ROUGE-L: {e1['ROUGE-L']:.4f}")

    probe_path = "outputs/linear_probe_auc.json"
    if os.path.exists(probe_path):
        with open(probe_path) as f:
            auc = json.load(f)
        print(f"RAD-DINO Clinical Linear Probe Macro-AUC: {auc['macro_auc']:.4f}")


def step_2_check_feature_cache():
    log_header("Step 2: Visual Feature Cache Verification")
    feats_dir = "processed/feats"
    if os.path.exists(feats_dir):
        count = len(os.listdir(feats_dir))
        print(f"Found {count} / 2943 cached feature files in '{feats_dir}'.")
        if count >= 2943:
            print("  All visual features pre-computed and ready!")
            return True

    print("Cached features missing or incomplete.")
    print("If you have 'feats.zip', extract it with: unzip -q feats.zip -d processed/feats/")
    return False


def step_3_train(args):
    log_header("Step 3: Two-Stage Training (Projector Warmup + QLoRA SFT)")
    cmd = [
        sys.executable, "src/train_sft.py",
        "--stage", str(args.stage),
        "--llm_name", args.llm_name,
        "--batch_size", str(args.batch_size),
        "--grad_accum", str(args.grad_accum),
        "--epochs_s1", str(args.epochs_s1),
        "--epochs_s2", str(args.epochs_s2),
        "--lr_lora", str(args.lr_lora),
        "--lora_dropout", str(args.lora_dropout),
        "--seed", str(args.seed),
        "--out_dir", args.out_dir,
        "--save_steps", "50",
    ]
    if getattr(args, "fresh_s2", False):
        cmd.append("--fresh_s2")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)


def step_4_evaluate(args):
    log_header("Step 4: MBR Consensus Generation & Official Scoring")
    best_ckpt = os.path.join(args.out_dir, "stage2_best")
    if not os.path.exists(best_ckpt):
        best_ckpt = os.path.join(args.out_dir, "stage1_best")
        print(f"Note: Using stage1 checkpoint at '{best_ckpt}'")

    out_json = f"outputs/test_{args.mode}_predictions.json"
    cmd = [
        sys.executable, "src/generate_mbr.py",
        "--checkpoint_dir", best_ckpt,
        "--llm_name", args.llm_name,
        "--mode", args.mode,
        "--n_candidates", str(args.n_candidates),
        "--split", "test",
        "--out_json", out_json,
    ]
    if getattr(args, "max_samples", None) is not None:
        cmd.extend(["--max_samples", str(args.max_samples)])
    print(f"Executing: {' '.join(cmd)}")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    subprocess.run(cmd, env=env, check=True)

    return out_json


def step_5_benchmark_comparison(pred_json_path: str):
    log_header("Step 5: Final Benchmark Comparison Table")
    if not os.path.exists(pred_json_path):
        print(f"Predictions file not found at {pred_json_path}")
        return

    with open(pred_json_path) as f:
        data = json.load(f)
        scores = data["headline_metrics"]

    table = [
        {"Model": "DART / DATR (CVPR 2025)", "BLEU-1": 0.486, "BLEU-4": 0.208, "METEOR": 0.205, "ROUGE-L": 0.411},
        {"Model": "LePaX (ECCV 2026)", "BLEU-1": 0.531, "BLEU-4": 0.235, "METEOR": 0.316, "ROUGE-L": 0.402},
        {"Model": "MPDRL (Frontiers 2026)", "BLEU-1": 0.508, "BLEU-4": 0.185, "METEOR": 0.231, "ROUGE-L": 0.383},
        {"Model": "DAMPER (AAAI 2025)", "BLEU-1": 0.520, "BLEU-4": 0.225, "METEOR": 0.284, "ROUGE-L": 0.397},
        {"Model": "KiUT (CVPR 2023)", "BLEU-1": 0.525, "BLEU-4": 0.185, "METEOR": 0.242, "ROUGE-L": 0.409},
        {
            "Model": "Dual-View RAG-VLM (Ours)",
            "BLEU-1": round(scores["BLEU-1"], 4),
            "BLEU-4": round(scores["BLEU-4"], 4),
            "METEOR": round(scores["METEOR"], 4),
            "ROUGE-L": round(scores["ROUGE-L"], 4),
        },
    ]

    df = pd.DataFrame(table)
    print("\nBenchmark Scores on IU X-Ray Test Set (N=589):\n")
    print(df.to_string(index=False))

    print("\nOutcome Analysis:")
    wins = []
    if scores["ROUGE-L"] > 0.411:
        wins.append(f"WIN: ROUGE-L ({scores['ROUGE-L']:.4f}) beats DART (0.411)")
    elif scores["ROUGE-L"] > 0.402:
        wins.append(f"WIN: ROUGE-L ({scores['ROUGE-L']:.4f}) beats LePaX (0.402)")
    elif scores["ROUGE-L"] > 0.383:
        wins.append(f"WIN: ROUGE-L ({scores['ROUGE-L']:.4f}) beats MPDRL (0.383)")

    if scores["METEOR"] > 0.316:
        wins.append(f"WIN: METEOR ({scores['METEOR']:.4f}) beats LePaX (0.316)")
    elif scores["METEOR"] > 0.205:
        wins.append(f"WIN: METEOR ({scores['METEOR']:.4f}) beats DART (0.205)")

    if scores["BLEU-4"] > 0.235:
        wins.append(f"WIN: BLEU-4 ({scores['BLEU-4']:.4f}) beats LePaX (0.235)")
    elif scores["BLEU-4"] > 0.185:
        wins.append(f"WIN: BLEU-4 ({scores['BLEU-4']:.4f}) beats KiUT / MPDRL (0.185)")

    if scores.get("BLEU-1", 0) > 0.3449:
        wins.append(f"BEATS HURDLE: BLEU-1 ({scores['BLEU-1']:.4f}) beats E1 Pure Retrieval Hurdle (0.3449)")
    if scores.get("ROUGE-L", 0) > 0.2809:
        wins.append(f"BEATS FLOOR: ROUGE-L ({scores['ROUGE-L']:.4f}) beats E0 Normal Report Floor (0.2809)")

    for w in wins:
        print(f"  * {w}")
    if not wins:
        print("  * Pipeline completed official evaluation.")


def main():
    parser = argparse.ArgumentParser(description="End-to-End IU X-Ray Pipeline Runner")
    parser.add_argument("--stage", type=int, choices=[0, 1, 2], default=0, help="0=Both, 1=Warmup only, 2=SFT only")
    parser.add_argument("--epochs_s1", type=int, default=1, help="Stage 1 warmup epochs (default: 1)")
    parser.add_argument("--epochs_s2", type=int, default=5, help="Stage 2 SFT epochs (default: 5)")
    parser.add_argument("--lr_lora", type=float, default=5e-5, help="LoRA learning rate (default: 5e-5)")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout rate (default: 0.1)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (auto-configured based on VRAM if None)")
    parser.add_argument("--grad_accum", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--llm_name", default="Qwen/Qwen2.5-3B-Instruct", help="Hugging Face base LLM model ID")
    parser.add_argument("--out_dir", default="outputs/checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["mbr", "beam", "sample"], default="mbr", help="Inference decoding mode")
    parser.add_argument("--n_candidates", type=int, default=16, help="Candidate pool size for MBR consensus")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of test samples to evaluate")
    parser.add_argument("--skip_train", action="store_true", help="Skip training and jump straight to MBR evaluation")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation after training")
    parser.add_argument("--fresh_s2", action="store_true", help="Start fresh Stage 2 training from Stage 1 projector")
    args = parser.parse_args()

    # Step 0: System check & automatic batch size configuration
    vram_gb = step_0_check_system()
    if args.batch_size is None:
        if vram_gb > 0 and vram_gb <= 6.5:
            # 6GB GPU (RTX 4050 laptop)
            args.batch_size = 1
            args.grad_accum = 16
            print(f"Auto-configured for {vram_gb:.1f}GB GPU (RTX 4050): batch_size=1, grad_accum=16 (effective batch 16)")
        elif vram_gb > 6.5 and vram_gb <= 16.0:
            # 8GB - 16GB GPU (T4 15GB): batch_size=2, grad_accum=8 prevents Stage 2 LoRA peak VRAM OOM
            args.batch_size = 2
            args.grad_accum = 8
            print(f"Auto-configured for {vram_gb:.1f}GB GPU (T4 safe): batch_size=2, grad_accum=8 (effective batch 16)")
        else:
            args.batch_size = 4
            args.grad_accum = 4
            print(f"Configured for large GPU ({vram_gb:.1f}GB): batch_size=4, grad_accum=4 (effective batch 16)")
    elif args.grad_accum is None:
        args.grad_accum = max(16 // args.batch_size, 1)

    # Step 1: Baseline checks
    step_1_check_baselines()

    # Step 2: Feature cache check
    feats_ok = step_2_check_feature_cache()
    if not feats_ok and not args.skip_train:
        print("\nERROR: Cannot train without visual features in 'processed/feats/'.")
        print("Please extract feats.zip or run src/cache_features.py first.")
        sys.exit(1)

    # Step 3: Training
    if not args.skip_train:
        step_3_train(args)
    else:
        print("\nSkipping training as requested (--skip_train).")

    # Step 4: Generation & Evaluation
    if not args.skip_eval:
        pred_json = step_4_evaluate(args)
        # Step 5: Benchmark table
        step_5_benchmark_comparison(pred_json)

    print("\nPipeline run complete!")


if __name__ == "__main__":
    main()
