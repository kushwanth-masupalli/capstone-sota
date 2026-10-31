"""
generate_consensus_fusion.py - Sentence-Level Clinical Consensus Fusion.

Fuses high-confidence non-redundant clinical anatomical statements across
diverse candidate pools (METEOR MBR, Enhanced MBR, Stochastic Samples) to maximize
both clinical completeness and natural language recall (METEOR > 0.20 - 0.22+).
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import coco_scores, truncate


def fuse_study_sentences(sources: list, jaccard_thresh: float = 0.50, max_sents: int = 6, max_words: int = 65) -> str:
    sents = []
    for t in sources:
        if not t:
            continue
        for s in re.split(r'[.!?]', str(t)):
            sc = s.strip().lower()
            if len(sc.split()) >= 3:
                sw = set(sc.split())
                is_redundant = False
                for existing in sents:
                    ew = set(existing.split())
                    jaccard = len(sw & ew) / max(len(sw | ew), 1)
                    if jaccard > jaccard_thresh:
                        is_redundant = True
                        break
                if not is_redundant:
                    sents.append(sc)
                    if len(sents) >= max_sents:
                        break
        if len(sents) >= max_sents:
            break

    fused_text = ". ".join(sents[:max_sents]).strip()
    if fused_text and not fused_text.endswith("."):
        fused_text += "."
    return truncate(fused_text, max_words)


def main():
    parser = argparse.ArgumentParser(description="Clinical Consensus Fusion across candidate pools.")
    parser.add_argument("--ref", default="processed/test.json")
    parser.add_argument("--meteor_pred", default="outputs/test_meteor_mbr_predictions.json")
    parser.add_argument("--enh_pred", default="outputs/test_enhanced_mbr_predictions.json")
    parser.add_argument("--sample_pred", default="outputs/test_sample_predictions.json")
    parser.add_argument("--out_json", default="outputs/test_consensus_fusion_predictions.json")
    parser.add_argument("--jaccard", type=float, default=0.50)
    parser.add_argument("--max_sents", type=int, default=6)
    parser.add_argument("--max_words", type=int, default=65)
    args = parser.parse_args()

    print("=== Running Sentence-Level Consensus Fusion ===")
    test_studies = json.load(open(args.ref))
    refs = {str(x["uid"]): x.get("findings", "") for x in test_studies}

    def load_p(path):
        if not os.path.exists(path):
            return {}
        d = json.load(open(path))
        if isinstance(d, dict) and "predictions" in d:
            return {str(x["uid"]): x.get("generated", "") for x in d["predictions"]}
        return {str(k): v for k, v in d.items()}

    p_meteor = load_p(args.meteor_pred)
    p_enh = load_p(args.enh_pred)
    p_sample = load_p(args.sample_pred)

    fused_dict = {}
    records = []
    for uid in refs.keys():
        sources = [
            p_meteor.get(uid, ""),
            p_enh.get(uid, ""),
            p_sample.get(uid, ""),
        ]
        fused = fuse_study_sentences(sources, jaccard_thresh=args.jaccard, max_sents=args.max_sents, max_words=args.max_words)
        fused_dict[uid] = fused
        records.append({
            "uid": uid,
            "generated": fused,
            "ground_truth": refs[uid],
        })

    print(f"Evaluating official headline metrics on {len(fused_dict)} studies...")
    scores, _, _ = coco_scores(refs, fused_dict)
    print("\n" + "=" * 60)
    print("CONSENUS FUSION HEADLINE EVALUATION RESULTS:")
    for k, v in scores.items():
        print(f"  {k:8s} = {v:.4f}")
    print("=" * 60)

    out_data = {
        "mode": "consensus_fusion",
        "num_studies": len(records),
        "headline_metrics": scores,
        "predictions": records,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved fused predictions to: {args.out_json}")


if __name__ == "__main__":
    main()

