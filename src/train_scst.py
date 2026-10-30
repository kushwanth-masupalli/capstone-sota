"""
train_scst.py - Self-Critical Sequence Training (SCST / RL) for Dual-View RAG-VLM.

Directly optimizes for ROUGE-L, METEOR, and BLEU-4 metrics using Policy Gradients (REINFORCE)
with a greedy decoding baseline for variance reduction.

Reference:
  Rennie et al., "Self-Critical Sequence Training for Image Captioning", CVPR 2017.
  Applied in medical report generation (CMM+RL, CMAS-RL, BoxMed-RL).
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from dataset import IUReportDataset, collate_fn, IMAGE_PAD_TOKEN
from evaluate import rouge_l_f1, ngram_f1, truncate
from model import DualViewRAGVLM
from train_sft import load_weights, safe_torch_load


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def approx_bleu4(hyp: str, ref: str) -> float:
    """Fast sentence-level 4-gram precision proxy for training reward."""
    h = hyp.lower().split()
    r = ref.lower().split()
    if not h or not r:
        return 0.0
    from collections import Counter
    precisions = []
    for n in range(1, 5):
        if len(h) < n:
            precisions.append(1.0 / (len(h) + 1))
            continue
        hc = Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
        rc = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
        matched = sum((hc & rc).values())
        precisions.append((matched + 0.1) / (sum(hc.values()) + 0.1))
    bp = 1.0 if len(h) >= len(r) else 2.71828 ** (1.0 - len(r) / len(h))
    geom = (precisions[0] * precisions[1] * precisions[2] * precisions[3]) ** 0.25
    return bp * geom


def approx_meteor(hyp: str, ref: str, alpha: float = 0.85) -> float:
    """Recall-weighted F-measure matching METEOR's alpha=0.85-0.90 prioritization."""
    hw = set(hyp.lower().split())
    rw = set(ref.lower().split())
    if not hw or not rw:
        return 0.0
    common = len(hw & rw)
    p = common / len(hw)
    rec = common / len(rw)
    if p + rec == 0:
        return 0.0
    return (p * rec) / (alpha * p + (1.0 - alpha) * rec)


def compute_reward(hyp: str, ref: str, w_meteor=0.55, w_rouge=0.30, w_bleu=0.15) -> float:
    """Composite clinical reward heavily targeting METEOR (> 0.215) and ROUGE-L."""
    if not hyp.strip() or not ref.strip():
        return 0.0
    r_rouge = rouge_l_f1(hyp, ref)
    r_meteor = approx_meteor(hyp, ref, alpha=0.85)
    r_bleu = approx_bleu4(hyp, ref)
    return w_meteor * r_meteor + w_rouge * r_rouge + w_bleu * r_bleu


def evaluate_val_reward(model: DualViewRAGVLM, val_dataset: IUReportDataset, device: str, max_samples: int = 50) -> Dict[str, float]:
    model.eval()
    rewards = []
    rouges = []
    count = min(len(val_dataset), max_samples)
    with torch.no_grad():
        for i in range(count):
            item = val_dataset[i]
            ref = truncate(item["target_findings"], 60)
            f_feat = item["frontal_feat"].to(device)
            l_feat = item["lateral_feat"].to(device)
            p_ids = item["input_ids"].to(device)

            pred = model.generate_report(
                frontal_feat=f_feat,
                lateral_feat=l_feat,
                prompt_ids=p_ids,
                max_new_tokens=60,
                num_beams=1,
                do_sample=False,
            )[0]
            pred_clean = truncate(pred, 60)
            rewards.append(compute_reward(pred_clean, ref))
            rouges.append(rouge_l_f1(pred_clean, ref))

    return {
        "reward": float(np.mean(rewards)) if rewards else 0.0,
        "rouge_l": float(np.mean(rouges)) if rouges else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Self-Critical Sequence Training (SCST / RL) on Dual-View RAG-VLM")
    parser.add_argument("--sft_checkpoint", default="outputs/checkpoints/stage2_best", help="Base SFT checkpoint to start RL from")
    parser.add_argument("--save_dir", default="outputs/checkpoints/stage3_rl_best")
    parser.add_argument("--data_dir", default="processed")
    parser.add_argument("--feats_dir", default="processed/feats")
    parser.add_argument("--epochs", type=int, default=3, help="RL epochs (typically 2-4 epochs suffice)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for policy gradients")
    parser.add_argument("--grad_accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--max_new_tokens", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--llm_name", default="Qwen/Qwen2.5-3B-Instruct")
    args = parser.parse_args()

    set_seed(args.seed)
    print("=" * 70)
    print("=== STAGE 3: SELF-CRITICAL SEQUENCE TRAINING (RL / SCST) ===")
    print(f"Base SFT Checkpoint: {args.sft_checkpoint}")
    print(f"Target Save Dir:     {args.save_dir}")
    print(f"Learning Rate:       {args.lr}")
    print(f"Epochs:              {args.epochs}")
    print("=" * 70)

    # 1. Initialize model and load Stage 2 SFT weights
    model = DualViewRAGVLM(llm_name_or_path=args.llm_name, load_in_4bit=True)
    load_weights(model, args.sft_checkpoint)

    # Enable gradients for LoRA and Projector
    for name, param in model.llm.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
    for param in model.projector.parameters():
        param.requires_grad = True

    # 2. Datasets
    train_dataset = IUReportDataset(
        split_json_path=os.path.join(args.data_dir, "train.json"),
        feats_dir=args.feats_dir,
        tokenizer=model.tokenizer,
        priors_json_path=os.path.join(args.data_dir, "train_oof_priors.json"),
        faiss_index_path=os.path.join(args.data_dir, "train_faiss.index"),
        faiss_meta_path=os.path.join(args.data_dir, "train_faiss_meta.json"),
        k_retrieval=2,
        is_training=False,  # prompt only
    )

    val_dataset = IUReportDataset(
        split_json_path=os.path.join(args.data_dir, "val.json"),
        feats_dir=args.feats_dir,
        tokenizer=model.tokenizer,
        priors_json_path=os.path.join(args.data_dir, "val_priors.json"),
        faiss_index_path=os.path.join(args.data_dir, "train_faiss.index"),
        faiss_meta_path=os.path.join(args.data_dir, "train_faiss_meta.json"),
        k_retrieval=2,
        is_training=False,
    )

    train_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(train_params, lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_dataset) // args.grad_accum, eta_min=1e-6)

    # Initial validation score
    print("\nEvaluating pre-RL validation reward...")
    init_val = evaluate_val_reward(model, val_dataset, args.device, max_samples=40)
    print(f"Pre-RL Val Reward: {init_val['reward']:.4f} | ROUGE-L: {init_val['rouge_l']:.4f}")

    best_val_reward = init_val["reward"]
    os.makedirs(args.save_dir, exist_ok=True)

    # 3. Training Loop
    indices = list(range(len(train_dataset)))
    total_steps = 0

    for epoch in range(1, args.epochs + 1):
        random.shuffle(indices)
        model.train()
        epoch_rewards = []
        epoch_advantages = []
        optimizer.zero_grad()
        pbar = tqdm(indices, desc=f"RL Epoch {epoch}/{args.epochs}")

        for step, idx in enumerate(pbar):
            item = train_dataset[idx]
            ref = truncate(item["target_findings"], 60)
            if not ref.strip():
                continue

            f_feat = item["frontal_feat"].to(args.device)
            l_feat = item["lateral_feat"].to(args.device)
            p_ids = item["input_ids"].to(args.device)

            # A. Generate Greedy Baseline (model.eval mode, no grad)
            model.eval()
            with torch.no_grad():
                baseline_text = model.generate_report(
                    frontal_feat=f_feat,
                    lateral_feat=l_feat,
                    prompt_ids=p_ids,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                )[0]
                baseline_clean = truncate(baseline_text, 60)
                reward_baseline = compute_reward(baseline_clean, ref)

                # B. Sample Stochastically (temperature=0.7)
                sampled_text = model.generate_report(
                    frontal_feat=f_feat,
                    lateral_feat=l_feat,
                    prompt_ids=p_ids,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=1,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )[0]
                sampled_clean = truncate(sampled_text, 60)
                reward_sample = compute_reward(sampled_clean, ref)

            advantage = reward_sample - reward_baseline
            epoch_rewards.append(reward_sample)
            epoch_advantages.append(advantage)

            # C. Policy Gradient Update on Sampled Hypothesis
            if abs(advantage) > 1e-4:
                model.train()
                # Encode the sampled sequence
                cand_ids = model.tokenizer.encode(f"{sampled_clean}<|im_end|>", add_special_tokens=False)
                if cand_ids:
                    full_input_ids = torch.cat([p_ids, torch.tensor(cand_ids, device=args.device)])
                    # Labels mask: -100 on prompt tokens, target ids on generated tokens
                    full_labels = torch.cat([
                        torch.full((len(p_ids),), -100, dtype=torch.long, device=args.device),
                        torch.tensor(cand_ids, dtype=torch.long, device=args.device)
                    ])

                    # Forward pass
                    out = model(
                        frontal_feats=f_feat.unsqueeze(0),
                        lateral_feats=l_feat.unsqueeze(0),
                        input_ids=full_input_ids.unsqueeze(0),
                        attention_mask=torch.ones((1, len(full_input_ids)), device=args.device),
                        labels=full_labels.unsqueeze(0),
                    )

                    # SCST loss: - advantage * log_prob (normalized by length)
                    rl_loss = - advantage * (out["loss_lm"] / args.grad_accum)
                    rl_loss.backward()

            total_steps += 1
            if total_steps % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(train_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % 20 == 0:
                pbar.set_postfix({
                    "mean_r": f"{np.mean(epoch_rewards[-40:]):.3f}" if epoch_rewards else "0",
                    "mean_adv": f"{np.mean(epoch_advantages[-40:]):.3f}" if epoch_advantages else "0",
                })

        # Epoch Validation
        val_res = evaluate_val_reward(model, val_dataset, args.device, max_samples=60)
        print(f"\n--- Epoch {epoch} Val Reward: {val_res['reward']:.4f} | ROUGE-L: {val_res['rouge_l']:.4f} ---")

        if val_res["reward"] > best_val_reward:
            best_val_reward = val_res["reward"]
            print(f"  [New Best Val Reward: {best_val_reward:.4f}] Saving to {args.save_dir}...")
            # Save custom heads
            custom_state = {
                "projector": model.projector.state_dict(),
                "classifier_head": model.classifier_head.state_dict(),
                "view_embed_frontal": model.view_embed_frontal.data.cpu(),
                "view_embed_lateral": model.view_embed_lateral.data.cpu(),
            }
            torch.save(custom_state, os.path.join(args.save_dir, "custom_heads.pt"))
            # Save LoRA adapter
            model.llm.save_pretrained(os.path.join(args.save_dir, "lora"))

    print("\n" + "=" * 70)
    print(f"Self-Critical Sequence Training complete! Best Checkpoint saved to: {args.save_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

