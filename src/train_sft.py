"""
train_sft.py - Two-Stage Training for Dual-View RAG-VLM (Projector Warmup + QLoRA SFT).

Protocol (Plan 2.1):
  * Loss: L = L_CE(report tokens) + 0.2 * L_BCE(14 findings)
  * Stage 1: Warmup Projector + Head + View Embeddings (LLM frozen, 3 epochs, lr=1e-3)
  * Stage 2: SFT LoRA (lr=1e-4) + Projector (lr=2e-5) + Head (lr=1e-4) for 10 epochs
  * Early stopping on validation ROUGE-L with patience 3
  * Effective batch size 16 (batch_size=4 x grad_accum=4)
  * Peak VRAM tracking & epoch timing
"""

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from dataset import IUReportDataset, collate_fn
from evaluate import rouge_l_f1, truncate
from model import DualViewRAGVLM


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_peak_vram_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def save_checkpoint(model: DualViewRAGVLM, save_dir: str):
    """Save projector, heads, and LoRA adapter weights."""
    os.makedirs(save_dir, exist_ok=True)
    custom_state = {
        "projector": model.projector.state_dict(),
        "classifier_head": model.classifier_head.state_dict(),
        "view_embed_frontal": model.view_embed_frontal.data.cpu(),
        "view_embed_lateral": model.view_embed_lateral.data.cpu(),
    }
    torch.save(custom_state, os.path.join(save_dir, "custom_heads.pt"))

    # Save LoRA adapter
    lora_dir = os.path.join(save_dir, "lora")
    model.llm.save_pretrained(lora_dir)
    print(f"  Checkpoint successfully saved to {save_dir}")


def evaluate_val_rouge(
    model: DualViewRAGVLM,
    val_dataset: IUReportDataset,
    device: str,
    max_eval_samples: int = 100,
) -> float:
    """Fast validation ROUGE-L proxy using local LCS (zero Java overhead)."""
    model.eval()
    scores = []
    eval_count = min(len(val_dataset), max_eval_samples)

    with torch.no_grad():
        for i in range(eval_count):
            item = val_dataset[i]
            ref = truncate(item["target_findings"], 60)
            f_feat = item["frontal_feat"].to(device)
            l_feat = item["lateral_feat"].to(device)
            p_ids = item["input_ids"].to(device)

            gen = model.generate_report(
                frontal_feat=f_feat,
                lateral_feat=l_feat,
                prompt_ids=p_ids,
                max_new_tokens=96,
                num_beams=1,
                do_sample=False,
            )[0]
            gen_trunc = truncate(gen, 60)
            scores.append(rouge_l_f1(gen_trunc, ref))

    return float(np.mean(scores)) if scores else 0.0


def train_stage(
    stage: int,
    model: DualViewRAGVLM,
    train_loader: DataLoader,
    val_dataset: IUReportDataset,
    epochs: int,
    device: str,
    grad_accum_steps: int = 4,
    save_dir: str = "outputs/checkpoints",
    patience: int = 3,
) -> Dict:
    """Train one stage (Stage 1 warmup or Stage 2 SFT)."""
    print(f"\n{'='*25} Starting Stage {stage} ({epochs} epochs) {'='*25}")
    model.set_stage(stage)

    # Configure learning rates per parameter group
    if stage == 1:
        # High LR for new initialized projector and heads
        params = [
            {"params": model.projector.parameters(), "lr": 1e-3},
            {"params": model.classifier_head.parameters(), "lr": 1e-3},
            {"params": [model.view_embed_frontal, model.view_embed_lateral], "lr": 1e-3},
        ]
    else:
        # Differential LRs: lower for base LoRA, lower for tuned projector
        lora_params = [p for n, p in model.llm.named_parameters() if p.requires_grad]
        params = [
            {"params": lora_params, "lr": 1e-4},
            {"params": model.projector.parameters(), "lr": 2e-5},
            {"params": model.classifier_head.parameters(), "lr": 1e-4},
            {"params": [model.view_embed_frontal, model.view_embed_lateral], "lr": 2e-5},
        ]

    optimizer = AdamW(params, weight_decay=0.01)

    total_steps = (len(train_loader) // grad_accum_steps) * epochs
    warmup_steps = max(int(total_steps * 0.1), 1)

    scheduler1 = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps])

    best_val_rouge = -1.0
    patience_counter = 0
    history = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        running_lm = 0.0
        running_cls = 0.0
        step_count = 0

        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Stage {stage} Ep {epoch}/{epochs}")
        for step, batch in enumerate(pbar):
            f_feats = batch["frontal_feats"].to(device)
            l_feats = batch["lateral_feats"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            labels_14 = batch["labels_14"].to(device)

            outputs = model(
                frontal_feats=f_feats,
                lateral_feats=l_feats,
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels=labels,
                labels_14=labels_14,
            )

            loss = outputs["loss"] / grad_accum_steps
            loss.backward()

            running_loss += outputs["loss"].item()
            running_lm += outputs["loss_lm"].item()
            running_cls += outputs["loss_cls"].item()
            step_count += 1

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix({
                "loss": f"{running_loss/max(step_count,1):.4f}",
                "lm": f"{running_lm/max(step_count,1):.4f}",
                "cls": f"{running_cls/max(step_count,1):.4f}",
            })

        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / max(step_count, 1)
        peak_vram = get_peak_vram_mb()

        # Validation evaluation
        print(f"\nEvaluating epoch {epoch} on validation set...")
        val_rouge = evaluate_val_rouge(model, val_dataset, device=device, max_eval_samples=60)
        print(f"Epoch {epoch} ({epoch_time:.1f}s, Peak VRAM: {peak_vram:.0f} MB): Train Loss = {avg_loss:.4f}, Val ROUGE-L = {val_rouge:.4f}")

        history.append({
            "stage": stage,
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "val_rouge_l": round(val_rouge, 4),
            "epoch_sec": round(epoch_time, 1),
            "peak_vram_mb": round(peak_vram, 1),
        })

        # Checkpoint saving & early stopping
        if val_rouge > best_val_rouge:
            best_val_rouge = val_rouge
            patience_counter = 0
            save_checkpoint(model, os.path.join(save_dir, f"stage{stage}_best"))
        else:
            patience_counter += 1
            if stage == 2 and patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
                break

    # Save final checkpoint
    save_checkpoint(model, os.path.join(save_dir, f"stage{stage}_final"))
    return {"best_val_rouge": best_val_rouge, "history": history}


def main():
    parser = argparse.ArgumentParser(description="Train DualViewRAGVLM")
    parser.add_argument("--data_dir", default="processed")
    parser.add_argument("--feats_dir", default="processed/feats")
    parser.add_argument("--llm_name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--out_dir", default="outputs/checkpoints")
    parser.add_argument("--stage", type=int, choices=[1, 2, 0], default=0, help="1=Warmup, 2=SFT, 0=Both")
    parser.add_argument("--epochs_s1", type=int, default=3)
    parser.add_argument("--epochs_s2", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"=== Plan 2.1 Two-Stage Training (Seed={args.seed}) ===")
    print(f"Base LLM: {args.llm_name}")
    print(f"Device:   {args.device}")

    # 1. Initialize model
    model = DualViewRAGVLM(
        llm_name_or_path=args.llm_name,
        load_in_4bit=True,
    )

    # 2. Build datasets and dataloaders
    train_path = os.path.join(args.data_dir, "train.json")
    val_path = os.path.join(args.data_dir, "val.json")
    train_priors = os.path.join(args.data_dir, "train_oof_priors.json")
    val_priors = os.path.join(args.data_dir, "val_priors.json")
    faiss_index = os.path.join(args.data_dir, "train_faiss.index")
    faiss_meta = os.path.join(args.data_dir, "train_faiss_meta.json")

    train_dataset = IUReportDataset(
        split_json_path=train_path,
        feats_dir=args.feats_dir,
        tokenizer=model.tokenizer,
        priors_json_path=train_priors,
        faiss_index_path=faiss_index,
        faiss_meta_path=faiss_meta,
        k_retrieval=2,
        is_training=True,
    )

    val_dataset = IUReportDataset(
        split_json_path=val_path,
        feats_dir=args.feats_dir,
        tokenizer=model.tokenizer,
        priors_json_path=val_priors,
        faiss_index_path=faiss_index,
        faiss_meta_path=faiss_meta,
        k_retrieval=2,
        is_training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id=model.tokenizer.pad_token_id or 151643),
        num_workers=0,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    full_history = []

    # Stage 1: Warmup
    if args.stage in [0, 1]:
        res_s1 = train_stage(
            stage=1,
            model=model,
            train_loader=train_loader,
            val_dataset=val_dataset,
            epochs=args.epochs_s1,
            device=args.device,
            grad_accum_steps=args.grad_accum,
            save_dir=args.out_dir,
        )
        full_history.extend(res_s1["history"])

    # Stage 2: SFT
    if args.stage in [0, 2]:
        res_s2 = train_stage(
            stage=2,
            model=model,
            train_loader=train_loader,
            val_dataset=val_dataset,
            epochs=args.epochs_s2,
            device=args.device,
            grad_accum_steps=args.grad_accum,
            save_dir=args.out_dir,
        )
        full_history.extend(res_s2["history"])

    # Save full training log
    log_path = os.path.join("outputs", f"training_log_seed{args.seed}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(full_history, f, indent=2)
    print(f"\nTraining complete! Log saved to {log_path}")


if __name__ == "__main__":
    main()
