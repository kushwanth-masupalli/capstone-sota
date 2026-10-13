"""
train_sft.py - Two-Stage Training for Dual-View RAG-VLM (Projector Warmup + QLoRA SFT).

Protocol (Plan 2.1):
  * Loss: L = L_CE(report tokens) + 0.2 * L_BCE(14 findings)
  * Stage 1: Warmup Projector + Head + View Embeddings (LLM frozen, lr=2e-4)
  * Stage 2: SFT LoRA (lr=1e-4) + Projector (lr=2e-5) + Head (lr=1e-4)
  * Early stopping on validation ROUGE-L with patience 3
  * Effective batch size 16 (auto-scaled: batch_size=1 x grad_accum=16 on 6GB RTX 4050)
  * Continuous step-level checkpointing & full stateful auto-resume
  * 100% Offline execution once base model weights are cached
"""

import argparse
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import math
import random
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from dataset import IUReportDataset, collate_fn
from download_model import download_base_model, is_model_complete
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


def save_checkpoint(
    model: DualViewRAGVLM,
    save_dir: str,
    stage: int = 1,
    epoch: int = 1,
    step: int = 0,
    global_step: int = 0,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    best_val_rouge: float = -1.0,
    patience_counter: int = 0,
    history: Optional[List] = None,
):
    """Save projector, heads, LoRA adapter weights, and full state for mid-training resume."""
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

    # Save complete training state
    state = {
        "stage": stage,
        "epoch": epoch,
        "step": step,
        "global_step": global_step,
        "best_val_rouge": best_val_rouge,
        "patience_counter": patience_counter,
        "history": history or [],
        "torch_rng": torch.get_rng_state(),
        "np_rng": np.random.get_state(),
        "py_rng": random.getstate(),
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()

    torch.save(state, os.path.join(save_dir, "training_state.pt"))

    meta = {
        "stage": stage,
        "epoch": epoch,
        "step": step,
        "global_step": global_step,
        "best_val_rouge": round(best_val_rouge, 4),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(save_dir, "checkpoint_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"  [Checkpoint Saved] -> {save_dir} (Stage {stage}, Epoch {epoch}, Step {step})")


def safe_torch_load(path: str, map_location="cpu"):
    """Load checkpoint safely in PyTorch 2.6+ (where weights_only defaults to True)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_weights(model: DualViewRAGVLM, ckpt_dir: str):
    """Restore projector, classifier head, and LoRA adapter weights."""
    custom_weights_path = os.path.join(ckpt_dir, "custom_heads.pt")
    if os.path.exists(custom_weights_path):
        print(f"Loading custom heads from {custom_weights_path}...")
        custom_weights = safe_torch_load(custom_weights_path, map_location="cpu")
        model.projector.load_state_dict(custom_weights["projector"])
        model.classifier_head.load_state_dict(custom_weights["classifier_head"])
        model.view_embed_frontal.data.copy_(custom_weights["view_embed_frontal"])
        model.view_embed_lateral.data.copy_(custom_weights["view_embed_lateral"])
        print("  Custom heads restored.")

    lora_dir = os.path.join(ckpt_dir, "lora")
    if os.path.exists(lora_dir):
        print(f"Loading LoRA weights from {lora_dir}...")
        from peft import load_peft_weights, set_peft_model_state_dict
        lora_weights = load_peft_weights(lora_dir)
        set_peft_model_state_dict(model.llm, lora_weights)
        print("  LoRA weights restored.")


def evaluate_val_rouge(
    model: DualViewRAGVLM,
    val_dataset: IUReportDataset,
    device: str,
    max_eval_samples: int = 20,
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
                max_new_tokens=64,
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
    grad_accum_steps: int = 16,
    save_dir: str = "outputs/checkpoints",
    save_steps: int = 50,
    val_samples: int = 20,
    patience: int = 3,
    resume_state: Optional[Dict] = None,
) -> Dict:
    """Train one stage (Stage 1 warmup or Stage 2 SFT) with fault-tolerant checkpointing."""
    print(f"\n{'='*25} Starting Stage {stage} ({epochs} epochs) {'='*25}")
    model.set_stage(stage)

    # Configure learning rates per parameter group
    if stage == 1:
        params = [
            {"params": model.projector.parameters(), "lr": 2e-4},
            {"params": model.classifier_head.parameters(), "lr": 2e-4},
            {"params": [model.view_embed_frontal, model.view_embed_lateral], "lr": 1e-4},
        ]
    else:
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
    start_epoch = 1
    start_step = 0
    global_step = 0

    latest_ckpt_dir = os.path.join(save_dir, "latest_checkpoint")

    # Restore exact training state if resuming this exact stage
    if resume_state and resume_state.get("stage") == stage:
        start_epoch = resume_state.get("epoch", 1)
        start_step = resume_state.get("step", 0)
        global_step = resume_state.get("global_step", 0)
        best_val_rouge = resume_state.get("best_val_rouge", -1.0)
        patience_counter = resume_state.get("patience_counter", 0)
        history = resume_state.get("history", [])

        if "optimizer" in resume_state:
            try:
                optimizer.load_state_dict(resume_state["optimizer"])
                print("  Optimizer state restored.")
            except Exception as e:
                print(f"  Note: Optimizer state reset: {e}")
        if "scheduler" in resume_state:
            try:
                scheduler.load_state_dict(resume_state["scheduler"])
                print("  Scheduler state restored.")
            except Exception as e:
                print(f"  Note: Scheduler state reset: {e}")

        # Check if epoch was already finished
        if start_step >= len(train_loader):
            print(f"  Stage {stage} Epoch {start_epoch} was already completed. Advancing to Epoch {start_epoch + 1}...")
            start_epoch += 1
            start_step = 0

        print(f"  ==> RESUMING Stage {stage} from Epoch {start_epoch}, Step {start_step}/{len(train_loader)}!")

    if start_epoch > epochs:
        print(f"Stage {stage} already completed all {epochs} epochs! Skipping to next stage.")
        return {"best_val_rouge": best_val_rouge, "history": history}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        running_lm = 0.0
        running_cls = 0.0
        step_count = 0

        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Stage {stage} Ep {epoch}/{epochs}")

        for step, batch in enumerate(pbar):
            # Skip steps if resuming mid-epoch
            if epoch == start_epoch and step < start_step:
                continue

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

            # Numerical stability guard against NaNs/Infs
            if torch.isnan(outputs["loss"]) or torch.isinf(outputs["loss"]):
                print(f"Warning: NaN/Inf detected at step {step}, skipping batch...")
                optimizer.zero_grad()
                continue

            loss = outputs["loss"] / grad_accum_steps
            loss.backward()

            running_loss += outputs["loss"].item()
            running_lm += outputs["loss_lm"].item()
            running_cls += outputs["loss_cls"].item()
            step_count += 1
            global_step += 1

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                grad_norm = nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                if not torch.isnan(grad_norm) and not torch.isinf(grad_norm):
                    optimizer.step()
                    scheduler.step()
                else:
                    print(f"Warning: NaN grad_norm at step {step}, skipping step")
                optimizer.zero_grad()

            pbar.set_postfix({
                "loss": f"{running_loss/max(step_count,1):.4f}",
                "lm": f"{running_lm/max(step_count,1):.4f}",
                "cls": f"{running_cls/max(step_count,1):.4f}",
            })

            # Periodic checkpoint for fault tolerance (even if lost power/internet)
            if save_steps > 0 and (step + 1) % save_steps == 0:
                save_checkpoint(
                    model=model,
                    save_dir=latest_ckpt_dir,
                    stage=stage,
                    epoch=epoch,
                    step=step + 1,
                    global_step=global_step,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_val_rouge=best_val_rouge,
                    patience_counter=patience_counter,
                    history=history,
                )

        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / max(step_count, 1)
        peak_vram = get_peak_vram_mb()

        # Reset start_step after the first resumed epoch finishes
        start_step = 0

        # Save latest at epoch end
        save_checkpoint(
            model=model,
            save_dir=latest_ckpt_dir,
            stage=stage,
            epoch=epoch,
            step=len(train_loader),
            global_step=global_step,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_rouge=best_val_rouge,
            patience_counter=patience_counter,
            history=history,
        )

        # Validation evaluation
        print(f"\nEvaluating epoch {epoch} on validation set ({val_samples} samples)...")
        val_rouge = evaluate_val_rouge(model, val_dataset, device=device, max_eval_samples=val_samples)
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
            save_checkpoint(
                model=model,
                save_dir=os.path.join(save_dir, f"stage{stage}_best"),
                stage=stage,
                epoch=epoch,
                step=len(train_loader),
                global_step=global_step,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_rouge=best_val_rouge,
                patience_counter=patience_counter,
                history=history,
            )
        else:
            patience_counter += 1
            if stage == 2 and patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
                break

    # Save final checkpoint for this stage
    save_checkpoint(
        model=model,
        save_dir=os.path.join(save_dir, f"stage{stage}_final"),
        stage=stage,
        epoch=epochs,
        step=len(train_loader),
        global_step=global_step,
        optimizer=optimizer,
        scheduler=scheduler,
        best_val_rouge=best_val_rouge,
        patience_counter=patience_counter,
        history=history,
    )
    return {"best_val_rouge": best_val_rouge, "history": history}


def main():
    parser = argparse.ArgumentParser(description="Train DualViewRAGVLM")
    parser.add_argument("--data_dir", default="processed")
    parser.add_argument("--feats_dir", default="processed/feats")
    parser.add_argument("--llm_name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--out_dir", default="outputs/checkpoints")
    parser.add_argument("--stage", type=int, choices=[1, 2, 0], default=0, help="1=Warmup, 2=SFT, 0=Both")
    parser.add_argument("--epochs_s1", type=int, default=1, help="Stage 1 warmup epochs (default: 1)")
    parser.add_argument("--epochs_s2", type=int, default=3, help="Stage 2 SFT epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (auto-configured if None)")
    parser.add_argument("--grad_accum", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every N steps (default: 50)")
    parser.add_argument("--val_samples", type=int, default=20, help="Validation sample count per epoch (default: 20)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume_from", default=None, help="Directory to resume from")
    parser.add_argument("--no_auto_resume", action="store_true", help="Disable automatic checkpoint resuming")
    args = parser.parse_args()

    set_seed(args.seed)

    # 1. Hardware-aware safe batch size auto-detection
    if args.batch_size is None or args.batch_size <= 0:
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if vram_gb <= 6.5:
                # 6GB GPU (RTX 4050 Laptop): batch_size=1, grad_accum=16 to prevent OOM
                args.batch_size = 1
                args.grad_accum = 16
                print(f"[Auto-Config] Detected {vram_gb:.1f} GB GPU (RTX 4050): batch_size=1, grad_accum=16 (effective batch 16)")
            elif vram_gb <= 8.5:
                args.batch_size = 2
                args.grad_accum = 8
                print(f"[Auto-Config] Detected {vram_gb:.1f} GB GPU: batch_size=2, grad_accum=8 (effective batch 16)")
            else:
                args.batch_size = 4
                args.grad_accum = 4
                print(f"[Auto-Config] Detected {vram_gb:.1f} GB GPU: batch_size=4, grad_accum=4 (effective batch 16)")
        else:
            args.batch_size = 1
            args.grad_accum = 16
    elif args.grad_accum is None:
        args.grad_accum = max(16 // args.batch_size, 1)

    print(f"\n=== Plan 2.1 Two-Stage Training (Seed={args.seed}) ===")
    print(f"Base LLM:        {args.llm_name}")
    print(f"Device:          {args.device}")
    print(f"Batch Size:      {args.batch_size} (Grad Accum: {args.grad_accum}, Effective Batch: {args.batch_size * args.grad_accum})")
    print(f"Save Steps:      every {args.save_steps} steps (Continuous Fault-Tolerant Checkpointing)")

    # 2. Base model cache verification & offline mode
    if is_model_complete(args.llm_name):
        print(f"[Offline Ready] Base LLM '{args.llm_name}' is fully cached. Running 100% OFFLINE without internet dependency.")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        print(f"[Notice] Base LLM '{args.llm_name}' weights not detected in local cache.")
        print(f"Starting fault-tolerant resumable downloader...")
        ok = download_base_model(args.llm_name)
        if ok:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        else:
            print("[Warning] Base model download incomplete. Re-run 'python src/download_model.py' to resume.")

    # 3. Check for existing checkpoints to auto-resume
    os.makedirs(args.out_dir, exist_ok=True)
    latest_ckpt = os.path.join(args.out_dir, "latest_checkpoint")
    if not args.no_auto_resume and args.resume_from is None:
        if os.path.exists(latest_ckpt) and os.path.exists(os.path.join(latest_ckpt, "custom_heads.pt")):
            args.resume_from = latest_ckpt
            print(f"\n[AUTO-RESUME] Found existing progress at '{latest_ckpt}'! Resuming automatically...")
        elif args.stage == 2:
            if os.path.exists(os.path.join(args.out_dir, "stage2_best")):
                args.resume_from = os.path.join(args.out_dir, "stage2_best")
                print(f"[AUTO-RESUME] Resuming Stage 2 from best checkpoint: {args.resume_from}")
            elif os.path.exists(os.path.join(args.out_dir, "stage1_final")):
                args.resume_from = os.path.join(args.out_dir, "stage1_final")
                print(f"[AUTO-RESUME] Loading Stage 1 trained projector from: {args.resume_from}")

    resume_state = None
    if args.resume_from and os.path.exists(args.resume_from):
        state_file = os.path.join(args.resume_from, "training_state.pt")
        if os.path.exists(state_file):
            resume_state = safe_torch_load(state_file, map_location="cpu")
            saved_stage = resume_state.get("stage", 1)
            saved_epoch = resume_state.get("epoch", 1)
            saved_step = resume_state.get("step", 0)
            print(f"  Checkpoint details: Stage {saved_stage}, Epoch {saved_epoch}, Step {saved_step}")

    # 4. Initialize model
    model = DualViewRAGVLM(
        llm_name_or_path=args.llm_name,
        load_in_4bit=True,
    )

    # 5. Restore weights if resuming
    if args.resume_from and os.path.exists(args.resume_from):
        load_weights(model, args.resume_from)

    # 6. Build datasets and dataloaders
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

    full_history = []

    # Determine starting stage from resume_state
    saved_stage = resume_state.get("stage", 1) if resume_state else 1

    # Stage 1: Warmup
    if args.stage in [0, 1] and saved_stage <= 1:
        res_s1 = train_stage(
            stage=1,
            model=model,
            train_loader=train_loader,
            val_dataset=val_dataset,
            epochs=args.epochs_s1,
            device=args.device,
            grad_accum_steps=args.grad_accum,
            save_dir=args.out_dir,
            save_steps=args.save_steps,
            val_samples=args.val_samples,
            resume_state=resume_state if saved_stage == 1 else None,
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
            save_steps=args.save_steps,
            val_samples=args.val_samples,
            resume_state=resume_state if saved_stage == 2 else None,
        )
        full_history.extend(res_s2["history"])

    # Save full training log
    log_path = os.path.join("outputs", f"training_log_seed{args.seed}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(full_history, f, indent=2)
    print(f"\nTraining complete! Log saved to {log_path}")


if __name__ == "__main__":
    main()
