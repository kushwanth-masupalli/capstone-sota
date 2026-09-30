"""
prepare_data.py - build locked train/val/test JSON files for IU X-Ray.

Primary protocol (R2Gen-compatible, headline numbers):
    * study needs a Frontal image, a Lateral image and non-empty `findings`
    * target text = cleaned findings, truncated to 60 words
Secondary variant (report separately, never as the headline):
    --include_single_view   keep studies without a lateral image (frontal duplicated)
    --impression_fallback   use `impression` when `findings` is empty

Usage:
    python src/prepare_data.py --data_dir data --out_dir processed
    python src/prepare_data.py --data_dir data --out_dir processed_v2 --impression_fallback
    python src/prepare_data.py --data_dir data --out_dir processed --r2gen_annotation annotation.json
"""
import argparse
import json
import os
import random
import re
from collections import Counter

import pandas as pd

MAX_WORDS = 60

LABELS_14 = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding",
]

# Weak labels from the dataset's own MeSH / Problems columns (positive findings only).
# This is a FALLBACK. Replace with CheXbert labels when available (see plan2.md 2.4).
WEAK_RULES = {
    "Cardiomegaly": r"cardiomegaly",
    "Lung Opacity": r"opacit|infiltrat",
    "Lung Lesion": r"nodule|\bmass\b|lesion|granuloma",
    "Edema": r"edema",
    "Consolidation": r"consolidat",
    "Pneumonia": r"pneumonia",
    "Atelectasis": r"atelecta",
    "Pneumothorax": r"pneumothorax",
    "Pleural Effusion": r"effusion",
    "Fracture": r"fracture",
    "Support Devices": r"catheter|pacemaker|\btube\b|device|stent|sternotomy|prosthe|implant|shunt|pacing|wire",
}


def clean_report(text) -> str:
    """Lowercase, remove ONLY anonymization placeholders (XXXX), normalise spacing, cap length."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    t = str(text).lower()
    t = re.sub(r"\bx{2,}\b", " ", t)            # placeholders only; never touches 'pneumothorax'
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s([.,;:])", r"\1", t)
    words = t.strip().split()
    return " ".join(words[:MAX_WORDS])


def weak_labels(mesh, problems):
    """14-dim weak label vector from MeSH/Problems strings."""
    s = " ".join(str(x).lower() for x in (mesh, problems) if isinstance(x, str))
    vec = [0] * 14
    for name, pat in WEAK_RULES.items():
        if re.search(pat, s):
            vec[LABELS_14.index(name)] = 1
    if re.search(r"mediastin", s) and re.search(r"widen|enlarg", s):
        vec[LABELS_14.index("Enlarged Cardiomediastinum")] = 1
    if re.search(r"pleura", s) and re.search(r"thicken|calcif|scar", s):
        vec[LABELS_14.index("Pleural Other")] = 1
    if isinstance(mesh, str) and mesh.strip().lower() == "normal":
        vec[LABELS_14.index("No Finding")] = 1
    return vec


def load_r2gen_ids(path):
    """Best-effort parse of an R2Gen-style annotation.json: {'train':[{'id':'CXR123_IM-...'}], ...}.
    [VERIFY] the id format against your copy of the file."""
    with open(path) as f:
        ann = json.load(f)
    out = {}
    for split in ("train", "val", "test"):
        uids = set()
        for e in ann[split]:
            m = re.match(r"CXR(\d+)_", e["id"])
            if m:
                uids.add(m.group(1))
        out[split] = uids
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="processed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include_single_view", action="store_true")
    ap.add_argument("--impression_fallback", action="store_true")
    ap.add_argument("--r2gen_annotation", default=None,
                    help="optional annotation.json to reuse R2Gen split ids exactly")
    args = ap.parse_args()

    rep = pd.read_csv(os.path.join(args.data_dir, "indiana_reports.csv"))
    proj = pd.read_csv(os.path.join(args.data_dir, "indiana_projections.csv"))
    img_dir = os.path.join(args.data_dir, "images")

    # ---- resolve views per uid ------------------------------------------------
    views = {}
    for uid, g in proj.groupby("uid", sort=False):
        fr = g[g["projection"].str.lower() == "frontal"]["filename"].tolist()
        la = g[g["projection"].str.lower() == "lateral"]["filename"].tolist()
        views[int(uid)] = (fr[0] if fr else None, la[0] if la else None)

    drops = Counter()
    records = []
    for _, r in rep.iterrows():
        uid = int(r["uid"])
        fr, la = views.get(uid, (None, None))
        if fr is None:
            drops["no_frontal"] += 1
            continue
        has_lat = la is not None
        if not has_lat:
            if not args.include_single_view:
                drops["no_lateral"] += 1
                continue
            la = fr
        text = clean_report(r["findings"])
        if not text and args.impression_fallback:
            text = clean_report(r["impression"])
        if not text:
            drops["empty_text"] += 1
            continue
        fp, lp = os.path.join(img_dir, fr), os.path.join(img_dir, la)
        if not (os.path.exists(fp) and os.path.exists(lp)):
            drops["missing_image_file"] += 1
            continue
        records.append({
            "uid": str(uid),
            "frontal_path": fp,
            "lateral_path": lp,
            "has_lateral": has_lat,
            "findings": text,
            "labels_14": weak_labels(r.get("MeSH"), r.get("Problems")),
            "label_source": "mesh_weak",
        })

    # ---- split (uid level; the CSVs contain no patient id) --------------------
    if args.r2gen_annotation:
        ids = load_r2gen_ids(args.r2gen_annotation)
        splits = {k: [x for x in records if x["uid"] in ids[k]] for k in ("train", "val", "test")}
        covered = sum(len(v) for v in splits.values())
        print(f"R2Gen ids matched {covered}/{len(records)} records")
    else:
        rng = random.Random(args.seed)
        order = list(range(len(records)))
        rng.shuffle(order)
        n = len(records)
        n_train, n_val = int(round(0.70 * n)), int(round(0.10 * n))
        pick = lambda idx: [records[i] for i in idx]
        splits = {
            "train": pick(order[:n_train]),
            "val": pick(order[n_train:n_train + n_val]),
            "test": pick(order[n_train + n_val:]),
        }

    # ---- safety checks --------------------------------------------------------
    sets = {k: {x["uid"] for x in v} for k, v in splits.items()}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]), \
        "uid overlap between splits"

    os.makedirs(args.out_dir, exist_ok=True)
    for k, v in splits.items():
        with open(os.path.join(args.out_dir, f"{k}.json"), "w") as f:
            json.dump(v, f, indent=1)
    with open(os.path.join(args.out_dir, "splits.json"), "w") as f:
        json.dump({k: sorted(s, key=int) for k, s in sets.items()}, f)
    with open(os.path.join(args.out_dir, "train_labels.json"), "w") as f:
        json.dump({x["uid"]: x["labels_14"] for x in splits["train"]}, f)

    # ---- stats ----------------------------------------------------------------
    lens = [len(x["findings"].split()) for x in records]
    train_texts = Counter(x["findings"] for x in splits["train"])
    prev = {LABELS_14[i]: sum(x["labels_14"][i] for x in records) for i in range(14)}
    stats = {
        "n_records": len(records),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "dropped": dict(drops),
        "mean_words": round(sum(lens) / max(1, len(lens)), 2),
        "frac_exact_duplicate_train_reports":
            round(sum(c for c in train_texts.values() if c > 1) / max(1, len(splits["train"])), 3),
        "most_common_train_report": train_texts.most_common(1)[0] if train_texts else None,
        "label_prevalence_weak": prev,
        "protocol": {"include_single_view": args.include_single_view,
                     "impression_fallback": args.impression_fallback,
                     "r2gen_annotation": args.r2gen_annotation, "seed": args.seed},
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)

    print(json.dumps({k: stats[k] for k in ("n_records", "split_sizes", "dropped", "mean_words",
                                             "frac_exact_duplicate_train_reports")}, indent=1))
    if not args.r2gen_annotation and not args.include_single_view and not args.impression_fallback:
        print("NOTE: R2Gen reports ~2,069/296/590. If your counts differ a lot, "
              "your split is NOT identical to R2Gen's; state that when comparing to papers.")


if __name__ == "__main__":
    main()
