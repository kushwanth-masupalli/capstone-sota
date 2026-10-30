"""
generate_mbr.py - Inference with Beam Search, Stochastic Sampling, and Fast MBR Consensus.

Key specifications (Plan 2.1):
  * Baseline: Beam search (num_beams=4, repetition_penalty=1.05, no_repeat_ngram_size=3, max_new_tokens=128)
  * MBR Decoding: Candidate pool N=16 (beam candidates + sampling at T=0.7, top-p=0.9)
  * Fast Utility: 0.6 * ROUGE-L + 0.4 * (unigram_f1 + bigram_f1)/2 (Python LCS, zero Java overhead)
  * Headline Scoring: pycocoevalcap run only on the final consensus selections
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from dataset import IUReportDataset, collate_fn
from evaluate import coco_scores, fmt, mbr_select, truncate, utility, meteor_utility
from model import DualViewRAGVLM
from cross_check import cross_check_report


def generate_study_candidates(
    model: DualViewRAGVLM,
    frontal_feat: torch.Tensor,
    lateral_feat: torch.Tensor,
    prompt_ids: torch.Tensor,
    device: str,
    n_candidates: int = 16,
    max_new_tokens: int = 128,
    retrieved_candidates: Optional[List[str]] = None,
) -> List[str]:
    """
    Generate a diverse pool of candidate reports for MBR selection.
    Mix of beam search outputs, stochastic samples (T=0.7), and retrieved training exemplars.
    """
    candidates = []

    # 1. High-precision Beam Search candidates (top 4)
    n_beams = min(4, n_candidates)
    beam_cands = model.generate_report(
        frontal_feat=frontal_feat,
        lateral_feat=lateral_feat,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        num_beams=n_beams,
        do_sample=False,
        repetition_penalty=1.05,
        no_repeat_ngram_size=3,
        num_return_sequences=n_beams,
    )
    candidates.extend(beam_cands)

    # 2. Diverse Stochastic Samples (remaining candidates at T=0.7, top_p=0.9)
    n_samples = n_candidates - len(candidates)
    if n_samples > 0:
        sample_cands = model.generate_report(
            frontal_feat=frontal_feat,
            lateral_feat=lateral_feat,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            no_repeat_ngram_size=3,
            num_return_sequences=n_samples,
        )
        candidates.extend(sample_cands)

    # 3. Add retrieved exemplar reports into the MBR pool
    if retrieved_candidates:
        candidates.extend(retrieved_candidates)

    # Filter empty or duplicate candidates while preserving order
    unique_cands = []
    for c in candidates:
        cleaned = truncate(c.strip(), 60)
        if cleaned and cleaned not in unique_cands:
            unique_cands.append(cleaned)

    return unique_cands if unique_cands else [candidates[0]]


def run_inference(
    model: DualViewRAGVLM,
    dataset: IUReportDataset,
    device: str,
    mode: str = "mbr",
    n_candidates: int = 16,
    max_new_tokens: int = 128,
    max_samples: Optional[int] = None,
    include_retrieval: bool = True,
    len_bonus: float = 0.15,
    util: str = "meteor",
    out_json_path: str = "outputs/predictions.json",
) -> Dict:
    """
    Run generation on dataset and evaluate headline metrics.
    """
    total_eval = min(len(dataset), max_samples) if max_samples else len(dataset)
    print(f"\nRunning inference in mode: '{mode}' on {total_eval} / {len(dataset)} studies...")
    model.eval()

    predictions = []
    references_dict = {}
    hypotheses_dict = {}

    start_time = time.time()

    for i in tqdm(range(total_eval), desc=f"Generating ({mode})"):
        item = dataset[i]
        uid = item["uid"]
        ground_truth = truncate(item["target_findings"], 60)

        f_feat = item["frontal_feat"].to(device)
        l_feat = item["lateral_feat"].to(device)
        p_ids = item["input_ids"].to(device)

        if mode == "beam":
            reports = model.generate_report(
                frontal_feat=f_feat,
                lateral_feat=l_feat,
                prompt_ids=p_ids,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                do_sample=False,
                repetition_penalty=1.05,
                no_repeat_ngram_size=3,
                num_return_sequences=1,
            )
            selected_report = truncate(reports[0], 60)
            candidate_pool = reports
        elif mode == "sample":
            reports = model.generate_report(
                frontal_feat=f_feat,
                lateral_feat=l_feat,
                prompt_ids=p_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.05,
                no_repeat_ngram_size=3,
                num_return_sequences=1,
            )
            selected_report = truncate(reports[0], 60)
            candidate_pool = reports
        elif mode == "mbr":
            ret_cands = item.get("similar_cases", []) if include_retrieval else None
            candidate_pool = generate_study_candidates(
                model=model,
                frontal_feat=f_feat,
                lateral_feat=l_feat,
                prompt_ids=p_ids,
                device=device,
                n_candidates=n_candidates,
                max_new_tokens=max_new_tokens,
                retrieved_candidates=ret_cands,
            )
            util_fn = meteor_utility if util == "meteor" else utility
            selected_report = mbr_select(candidate_pool, util=util_fn, len_bonus=len_bonus)
        else:
            raise ValueError(f"Unknown generation mode: {mode}")

        references_dict[uid] = ground_truth
        hypotheses_dict[uid] = selected_report

        # Execute Clinical Cross-Check if priors available
        study_priors = dataset.priors.get(uid, {})
        cross_res = cross_check_report(selected_report, study_priors) if study_priors else None

        predictions.append({
            "uid": uid,
            "generated": selected_report,
            "unchecked_report": selected_report,
            "checked_report": cross_res["checked_report"] if cross_res else selected_report,
            "concordance_score": cross_res["concordance_score"] if cross_res else 100.0,
            "verification_status": cross_res["verification_status"] if cross_res else "UNVERIFIED",
            "ground_truth": ground_truth,
            "candidate_pool_size": len(candidate_pool),
        })

    total_time = time.time() - start_time
    print(f"Generated {len(predictions)} reports in {total_time:.1f}s ({len(predictions)/total_time:.2f} studies/s)")

    # Compute headline metrics with pycocoevalcap
    print("\nComputing official headline metrics with pycocoevalcap...")
    metrics, per_sample, _ = coco_scores(references_dict, hypotheses_dict)

    print("\n" + "=" * 60)
    print(f"Headline Evaluation Results ({mode.upper()}):")
    print(f"  {fmt(metrics)}")
    print("=" * 60)

    # Save output predictions and scores
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    out_data = {
        "mode": mode,
        "n_candidates": n_candidates if mode == "mbr" else 1,
        "num_studies": len(predictions),
        "headline_metrics": metrics,
        "predictions": predictions,
    }
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved predictions and scores to {out_json_path}")

    return out_data


def main():
    parser = argparse.ArgumentParser(description="Generate reports with Beam search or MBR consensus.")
    parser.add_argument("--checkpoint_dir", required=True, help="Directory containing model checkpoints")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--mode", choices=["mbr", "beam", "sample"], default="mbr")
    parser.add_argument("--n_candidates", type=int, default=16, help="Pool size for MBR")
    parser.add_argument("--include_retrieval", action="store_true", default=True, help="Include retrieved cases in MBR pool")
    parser.add_argument("--len_bonus", type=float, default=0.15, help="Length normalization bonus for MBR")
    parser.add_argument("--util", choices=["f1", "meteor"], default="meteor", help="MBR selection utility (f1 composite or meteor recall-weighted)")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for quick testing")
    parser.add_argument("--data_dir", default="processed")
    parser.add_argument("--feats_dir", default="processed/feats")
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--llm_name", default="Qwen/Qwen2.5-3B-Instruct")
    args = parser.parse_args()

    if args.out_json is None:
        args.out_json = f"outputs/{args.split}_{args.mode}_predictions.json"

    print(f"=== Plan 2.1 Report Generation ({args.mode.upper()}) ===")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Split:      {args.split}")

    # 1. Initialize model
    model = DualViewRAGVLM(
        llm_name_or_path=args.llm_name,
        load_in_4bit=True,
    )

    # Load custom weights (projector, head, view embeddings) and LoRA
    from train_sft import load_weights
    load_weights(model, args.checkpoint_dir)

    # 2. Build Dataset
    split_path = os.path.join(args.data_dir, f"{args.split}.json")
    priors_path = os.path.join(args.data_dir, f"{args.split}_priors.json")
    faiss_index_path = os.path.join(args.data_dir, "train_faiss.index")
    faiss_meta_path = os.path.join(args.data_dir, "train_faiss_meta.json")

    dataset = IUReportDataset(
        split_json_path=split_path,
        feats_dir=args.feats_dir,
        tokenizer=model.tokenizer,
        priors_json_path=priors_path,
        faiss_index_path=faiss_index_path,
        faiss_meta_path=faiss_meta_path,
        k_retrieval=2,
        is_training=False,
    )

    # 3. Run Inference
    run_inference(
        model=model,
        dataset=dataset,
        device=args.device,
        mode=args.mode,
        n_candidates=args.n_candidates,
        max_new_tokens=args.max_new_tokens,
        max_samples=args.max_samples,
        include_retrieval=args.include_retrieval,
        len_bonus=args.len_bonus,
        util=args.util,
        out_json_path=args.out_json,
    )


if __name__ == "__main__":
    main()
