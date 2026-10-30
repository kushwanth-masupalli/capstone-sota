"""
evaluate.py - headline NLG scoring for IU X-Ray with pycocoevalcap (needs Java)
and Clinical Cross-Checking evaluation.

Modes
  --mode gt         ground truth vs ground truth (sanity: everything should be ~1.0)
  --mode normal     constant 'normal' report baseline (minimum hurdle)
  --mode score      score a predictions file
  --mode retrieval  retrieval-only baseline from a precomputed neighbour file
  --mode crosscheck clinical factuality and omission/hallucination audit
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from cross_check import cross_check_report

NORMAL_REPORT = ("the lungs are clear. no focal consolidation, pneumothorax, or pleural effusion. "
                 "the cardiac silhouette is normal.")
ABNORMAL_LABEL_IDX = 13  # 'No Finding' position in labels_14


# ----------------------------------------------------------------------------- helpers
def truncate(text: str, max_words: int) -> str:
    return " ".join(str(text).split()[:max_words]) if max_words else str(text)


def load_ref(path):
    with open(path) as f:
        return json.load(f)


def load_pred(path):
    with open(path) as f:
        p = json.load(f)
    if isinstance(p, list):
        return {str(x["uid"]): x.get("generated", x.get("pred", "")) for x in p}
    return {str(k): v for k, v in p.items()}


# ----------------------------------------------------------------------------- fast utility for MBR
def _lcs_len(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_f1(hyp: str, ref: str, beta: float = 1.2) -> float:
    """ROUGE-L with the same beta as pycocoevalcap (1.2). Whitespace tokens; proxy only."""
    h, r = hyp.lower().split(), ref.lower().split()
    l = _lcs_len(h, r)
    if l == 0:
        return 0.0
    p, rc = l / len(h), l / len(r)
    return ((1 + beta ** 2) * p * rc) / (rc + beta ** 2 * p)


def ngram_f1(hyp: str, ref: str, n: int = 1) -> float:
    from collections import Counter
    h, r = hyp.lower().split(), ref.lower().split()
    hc = Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
    rc = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
    ov = sum((hc & rc).values())
    if ov == 0:
        return 0.0
    p, rr = ov / sum(hc.values()), ov / sum(rc.values())
    return 2 * p * rr / (p + rr)


def utility(a: str, b: str, w_rouge=0.5, w_f1=0.5, order=4) -> float:
    """Multi-gram utility combining ROUGE-L with order 1-4 harmonic n-gram F1."""
    if order == 4:
        f1_val = (ngram_f1(a, b, 1) + ngram_f1(a, b, 2) + ngram_f1(a, b, 3) + ngram_f1(a, b, 4)) / 4.0
    elif order == 2:
        f1_val = 0.5 * (ngram_f1(a, b, 1) + ngram_f1(a, b, 2))
    else:
        f1_val = ngram_f1(a, b, 1)
    return w_rouge * rouge_l_f1(a, b) + w_f1 * f1_val


def meteor_utility(a: str, b: str, alpha: float = 0.85) -> float:
    """Recall-weighted consensus utility directly prioritizing METEOR score (> 0.215)."""
    aw = set(a.lower().split())
    bw = set(b.lower().split())
    if not aw or not bw:
        return 0.0
    common = len(aw & bw)
    p = common / len(aw)
    r = common / len(bw)
    if p + r == 0:
        return 0.0
    f_mean = (p * r) / (alpha * p + (1.0 - alpha) * r)
    return 0.70 * f_mean + 0.30 * rouge_l_f1(a, b)


def mbr_select(cands, util=utility, len_bonus=0.15, target_len=30.0):
    """Return the candidate with the highest mean utility against the others, with optional length normalization."""
    n = len(cands)
    if n == 1:
        return cands[0]
    best, best_s = cands[0], -1e9
    for i in range(n):
        s = sum(util(cands[i], cands[j]) for j in range(n) if j != i) / (n - 1)
        if len_bonus > 0:
            c_len = len(cands[i].split())
            s += len_bonus * min(c_len / target_len, 1.0)
        if s > best_s:
            best, best_s = cands[i], s
    return best


# ----------------------------------------------------------------------------- headline scorer
def coco_scores(refs: dict, hyps: dict):
    """refs/hyps: {uid: text}. Returns (corpus_scores, per_sample dict). Needs Java."""
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider

    ids = sorted(refs.keys())
    tok = PTBTokenizer()
    gts = tok.tokenize({i: [{"caption": refs[i]}] for i in ids})
    res = tok.tokenize({i: [{"caption": hyps.get(i, "")}] for i in ids})

    out, per = {}, {}
    b, bs = Bleu(4).compute_score(gts, res, verbose=0)
    for k in range(4):
        out[f"BLEU-{k + 1}"] = b[k]
        per[f"BLEU-{k + 1}"] = bs[k]
    try:
        m, ms = Meteor().compute_score(gts, res)
        out["METEOR"], per["METEOR"] = m, ms
    except Exception as e:
        import nltk
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        from nltk.translate.meteor_score import meteor_score
        ms = [meteor_score([gts[i][0].split()], res[i][0].split()) for i in ids]
        m = float(sum(ms) / max(len(ms), 1))
        out["METEOR"], per["METEOR"] = m, ms
    r, rs = Rouge().compute_score(gts, res)
    out["ROUGE-L"], per["ROUGE-L"] = r, rs
    c, cs = Cider().compute_score(gts, res)
    out["CIDEr"], per["CIDEr"] = c, cs
    return out, per, ids


def fmt(d):
    return "  ".join(f"{k}={v:.4f}" for k, v in d.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gt", "normal", "score", "retrieval", "crosscheck"], required=True)
    ap.add_argument("--ref", default=None, help="processed/test.json (or val.json)")
    ap.add_argument("--pred", default=None)
    ap.add_argument("--priors", default="processed/test_priors.json", help="Path to clinical priors JSON")
    ap.add_argument("--train", default=None, help="train.json, enables the exact-copy check")
    ap.add_argument("--neighbors", default=None, help="for --mode retrieval")
    ap.add_argument("--max_words", type=int, default=60, help="truncate refs and preds (0 = off)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Mode: Crosscheck
    if args.mode == "crosscheck":
        assert args.pred, "--pred is required for --mode crosscheck"
        with open(args.pred) as f:
            pred_file = json.load(f)

        priors_data = {}
        if args.priors and os.path.exists(args.priors):
            with open(args.priors) as f:
                priors_data = json.load(f)

        preds = pred_file.get("predictions", []) if isinstance(pred_file, dict) and "predictions" in pred_file else []
        if not preds and isinstance(pred_file, dict):
            preds = [{"uid": k, "generated": v} for k, v in pred_file.items()]

        concordance_scores = []
        total_omissions = 0
        total_hallucinations = 0
        verified_count = 0

        for p in preds:
            uid = str(p.get("uid", ""))
            text = p.get("generated", "")
            study_priors = priors_data.get(uid, {})
            res = cross_check_report(text, study_priors)
            concordance_scores.append(res["concordance_score"])
            total_omissions += res["omissions_count"]
            total_hallucinations += res["hallucinations_count"]
            if res["verification_status"] == "VERIFIED_CONCORDANT":
                verified_count += 1

        mean_concordance = sum(concordance_scores) / max(len(concordance_scores), 1)
        print("=" * 60)
        print(f"CLINICAL CROSS-CHECK AUDIT (N={len(concordance_scores)} studies):")
        print(f"  Mean Clinical Concordance: {mean_concordance:.2f}%")
        print(f"  Verified Concordant:        {verified_count} / {len(concordance_scores)} ({verified_count/max(len(concordance_scores),1):.1%})")
        print(f"  Total Omissions Detected:   {total_omissions}")
        print(f"  Total Hallucinations:       {total_hallucinations}")
        print("=" * 60)

        if args.out:
            with open(args.out, "w") as f:
                json.dump({
                    "mean_concordance": round(mean_concordance, 2),
                    "verified_concordant_rate": round(verified_count / max(len(concordance_scores), 1), 4),
                    "total_omissions": total_omissions,
                    "total_hallucinations": total_hallucinations,
                    "num_studies": len(concordance_scores),
                }, f, indent=2)
        return

    assert args.ref, "--ref required for NLG scoring modes"
    ref_recs = load_ref(args.ref)
    refs = {r["uid"]: truncate(r["findings"], args.max_words) for r in ref_recs}

    if args.mode == "gt":
        hyps = dict(refs)
    elif args.mode == "normal":
        hyps = {u: NORMAL_REPORT for u in refs}
    elif args.mode == "retrieval":
        assert args.neighbors and args.train, "--neighbors and --train required"
        train = {r["uid"]: r["findings"] for r in load_ref(args.train)}
        nb = load_pred(args.neighbors)
        hyps = {u: truncate(train[nb[u]], args.max_words) for u in refs if u in nb}
    else:
        assert args.pred, "--pred required for --mode score"
        hyps = {u: truncate(t, args.max_words) for u, t in load_pred(args.pred).items()}

    missing = [u for u in refs if u not in hyps]
    if missing:
        print(f"WARNING: {len(missing)} test studies have no prediction (scored as empty).")

    result = {"mode": args.mode, "n": len(refs), "max_words": args.max_words}

    scores, per, ids = coco_scores(refs, hyps)
    result["overall"] = scores
    print("OVERALL   ", fmt(scores))

    # abnormal-only subset (uses the weak MeSH label 'No Finding'); detects normal-template collapse
    abn = {r["uid"] for r in ref_recs if r.get("labels_14") and r["labels_14"][ABNORMAL_LABEL_IDX] == 0}
    if len(abn) >= 20:
        sub_scores, _, _ = coco_scores({u: refs[u] for u in abn}, {u: hyps.get(u, "") for u in abn})
        result["abnormal_subset"] = {"n": len(abn), **sub_scores}
        print(f"ABNORMAL  (n={len(abn)})", fmt(sub_scores))

    if args.train and args.mode in ("score", "normal"):
        train_texts = {truncate(r["findings"], args.max_words) for r in load_ref(args.train)}
        copy_rate = sum(1 for u in refs if hyps.get(u, "") in train_texts) / len(refs)
        result["exact_copy_of_train_report_rate"] = round(copy_rate, 4)
        print(f"Exact copy of a train report: {copy_rate:.1%}")

    print("NOTE: clinical metrics (CheXbert F1, RadGraph F1) are not computed here; "
          "add them once a working labeler is installed (plan2.md section 6).")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
