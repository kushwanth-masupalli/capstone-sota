"""
cache_features.py - Pre-extract and cache dual-view visual features, build FAISS retrieval index,
run baseline retrieval (E1), and train the 14-finding linear probe for prompt conditioning.

Key specifications (Plan 2.1):
  * Frozen encoder: default microsoft/rad-dino (518x518, D=768)
  * Spatial pooling: AdaptiveAvgPool2d((12, 12)) -> exactly 144 tokens per view (288 dual-view)
  * Dual-view global representation: concat(frontal_mean, lateral_mean) -> 1536-dim, L2-normalized
  * Retrieval index: faiss.IndexFlatIP over train studies only
  * E1 hurdle: top-1 train retrieval evaluated on test set
  * 14-finding probe: 5-fold OOF probabilities on train (avoids train/test prompt mismatch)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel

# Import evaluate functions from evaluate.py for E1 baseline
sys.path.insert(0, os.path.dirname(__file__))
from evaluate import coco_scores, fmt, truncate

LABELS_14 = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding",
]


class DualViewDataset(Dataset):
    """Dataset reading paired frontal and lateral chest radiographs for a study."""

    def __init__(self, studies: List[Dict], img_base_dir: str = "", transform=None):
        self.studies = studies
        self.img_base_dir = img_base_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.studies)

    def _load_image(self, path: str) -> Image.Image:
        full_path = os.path.join(self.img_base_dir, path) if self.img_base_dir else path
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found at {full_path}")
        im = Image.open(full_path).convert("RGB")
        return im

    def __getitem__(self, idx: int) -> Dict:
        study = self.studies[idx]
        f_im = self._load_image(study["frontal_path"])
        if study.get("has_lateral", True) and study.get("lateral_path"):
            l_im = self._load_image(study["lateral_path"])
        else:
            l_im = f_im.copy()

        if self.transform is not None:
            f_tensor = self.transform(f_im)
            l_tensor = self.transform(l_im)
        else:
            f_tensor = T.ToTensor()(f_im)
            l_tensor = T.ToTensor()(l_im)

        labels = torch.tensor(study.get("labels_14", [0] * 14), dtype=torch.float32)
        return {
            "uid": str(study["uid"]),
            "frontal": f_tensor,
            "lateral": l_tensor,
            "labels": labels,
            "findings": study.get("findings", ""),
        }


def build_transform(encoder_name: str):
    """Build fast torchvision preprocessing transform matching the encoder configuration."""
    print(f"Configuring image transforms for {encoder_name}...")
    try:
        proc = AutoImageProcessor.from_pretrained(encoder_name)
        # Determine image size
        if hasattr(proc, "crop_size") and isinstance(proc.crop_size, dict):
            size = proc.crop_size.get("height", 518)
        elif hasattr(proc, "size") and isinstance(proc.size, dict):
            size = proc.size.get("height", proc.size.get("shortest_edge", 518))
        else:
            size = 518

        mean = getattr(proc, "image_mean", [0.5307, 0.5307, 0.5307])
        std = getattr(proc, "image_std", [0.2583, 0.2583, 0.2583])
    except Exception as e:
        print(f"Warning: Could not load AutoImageProcessor ({e}), using default 518x518")
        size = 518
        mean = [0.5307, 0.5307, 0.5307]
        std = [0.2583, 0.2583, 0.2583]

    print(f"  Target size: ({size}, {size}), mean: {mean}, std: {std}")
    transform = T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return transform, size


def extract_batch_features(
    model: nn.Module,
    images: torch.Tensor,
    pool: nn.AdaptiveAvgPool2d,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pass a batch of images through the frozen encoder, reshape patch grid, and pool to 12x12.
    Returns:
        pooled: [B, 144, D] fp16
        global_rep: [B, D] float32
    """
    images = images.to(device, dtype=torch.float16 if device == "cuda" else torch.float32)
    with torch.no_grad():
        out = model(images)
        hidden = out.last_hidden_state  # [B, S, D]
        B, S, D = hidden.shape

        # Identify CLS token and spatial patch grid
        # For DINOv2: S = 1370 = 1 + 37*37
        # For SigLIP: S = 576 = 24*24
        grid_dim = int(round((S - 1) ** 0.5))
        if grid_dim * grid_dim == S - 1:
            # Has CLS token at index 0
            cls_token = hidden[:, 0, :]
            patches = hidden[:, 1:, :]
            H_grid = W_grid = grid_dim
        else:
            grid_dim = int(round(S ** 0.5))
            if grid_dim * grid_dim == S:
                cls_token = None
                patches = hidden
                H_grid = W_grid = grid_dim
            else:
                raise ValueError(f"Cannot reshape sequence of length {S} into a square grid!")

        # Reshape to [B, D, H, W] for spatial pooling
        grid = patches.permute(0, 2, 1).reshape(B, D, H_grid, W_grid).float()
        pooled = pool(grid).flatten(2).permute(0, 2, 1)  # [B, 144, D]

        # Global vector for retrieval
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            global_rep = out.pooler_output.float()
        elif cls_token is not None:
            global_rep = cls_token.float()
        else:
            global_rep = pooled.mean(dim=1).float()

    return pooled.half(), global_rep


def cache_split_features(
    model: nn.Module,
    split_name: str,
    studies: List[Dict],
    out_dir: str,
    transform,
    device: str,
    batch_size: int = 16,
    num_workers: int = 0,
    skip_existing: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Process all studies in a split, save per-study .pt tensors, and collect global vectors.
    """
    os.makedirs(out_dir, exist_ok=True)
    dataset = DualViewDataset(studies, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    pool = nn.AdaptiveAvgPool2d((12, 12)).to(device)

    global_vectors = []
    labels_list = []
    uids_list = []

    print(f"\n--- Caching features for {split_name} ({len(studies)} studies) ---")
    start_time = time.time()
    processed_count = 0

    for batch in loader:
        b_uids = batch["uid"]
        b_frontal = batch["frontal"]
        b_lateral = batch["lateral"]
        b_labels = batch["labels"]

        # Check if all files in this batch already exist
        all_exist = skip_existing and all(
            os.path.exists(os.path.join(out_dir, f"{u}.pt")) for u in b_uids
        )

        if all_exist:
            for u, l in zip(b_uids, b_labels):
                data = torch.load(os.path.join(out_dir, f"{u}.pt"), map_location="cpu")
                global_vectors.append(data["global"].numpy())
                labels_list.append(l.numpy())
                uids_list.append(u)
            processed_count += len(b_uids)
            continue

        f_pooled, f_global = extract_batch_features(model, b_frontal, pool, device)
        l_pooled, l_global = extract_batch_features(model, b_lateral, pool, device)

        # Dual-view concatenated global vector: [B, 2*D]
        dual_global = torch.cat([f_global, l_global], dim=-1)
        dual_global = F.normalize(dual_global, p=2, dim=-1)  # L2 normalize

        for i, u in enumerate(b_uids):
            out_file = os.path.join(out_dir, f"{u}.pt")
            feat_dict = {
                "uid": u,
                "frontal": f_pooled[i].cpu(),  # [144, D] fp16
                "lateral": l_pooled[i].cpu(),  # [144, D] fp16
                "global": dual_global[i].cpu(),  # [2*D] float32
                "labels": b_labels[i].cpu(),  # [14]
            }
            torch.save(feat_dict, out_file)

            global_vectors.append(dual_global[i].cpu().numpy())
            labels_list.append(b_labels[i].numpy())
            uids_list.append(u)

        processed_count += len(b_uids)
        if processed_count % (batch_size * 5) == 0 or processed_count == len(studies):
            elapsed = time.time() - start_time
            rate = processed_count / max(elapsed, 0.001)
            print(f"  Processed {processed_count}/{len(studies)} studies ({rate:.1f} studies/s)")

    total_time = time.time() - start_time
    print(f"Finished {split_name}: {len(studies)} studies in {total_time:.1f}s ({len(studies)/max(total_time,0.001):.1f} studies/s)")

    return np.array(global_vectors, dtype=np.float32), np.array(labels_list, dtype=np.float32), uids_list


def build_faiss_retrieval_index(
    train_vectors: np.ndarray,
    train_studies: List[Dict],
    index_path: str,
    meta_path: str,
) -> faiss.Index:
    """Build and save FAISS flat inner-product (cosine similarity) index on train vectors."""
    dim = train_vectors.shape[1]
    print(f"\nBuilding FAISS IndexFlatIP (dim={dim}, N={len(train_vectors)})...")
    index = faiss.IndexFlatIP(dim)
    index.add(train_vectors)

    faiss.write_index(index, index_path)
    print(f"Saved FAISS index to {index_path}")

    # Save metadata aligned with index rows
    meta = [
        {
            "idx": i,
            "uid": str(s["uid"]),
            "findings": s["findings"],
            "labels_14": s.get("labels_14", []),
        }
        for i, s in enumerate(train_studies)
    ]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved FAISS metadata ({len(meta)} records) to {meta_path}")
    return index


def query_retrieval(
    query_uid: str,
    query_emb: np.ndarray,
    index: faiss.Index,
    train_meta: List[Dict],
    k: int = 2,
    training: bool = False,
) -> List[Dict]:
    """
    Retrieve top-k neighbors from the train index.
    Leakage guard: Exclude query study itself when training is True.
    """
    # Search extra candidates in case the query uid needs to be skipped
    _, indices = index.search(query_emb.reshape(1, -1), k + 10)
    out = []
    for idx in indices[0]:
        c = train_meta[idx]
        if training and c["uid"] == query_uid:
            continue
        out.append(c)
        if len(out) == k:
            break
    return out


def evaluate_e1_retrieval_baseline(
    test_vectors: np.ndarray,
    test_studies: List[Dict],
    index: faiss.Index,
    train_meta: List[Dict],
    out_json_path: str,
) -> Dict:
    """
    E1 Sanity Hurdle: Predict each test study report by copying top-1 train neighbor findings.
    """
    print("\n========================================================")
    print("Evaluating Baseline E1: Pure Retrieval (Top-1 Train Copy)")
    print("========================================================")

    hypotheses = {}
    references = {}
    records = []

    for i, study in enumerate(test_studies):
        uid = str(study["uid"])
        query_vec = test_vectors[i]
        top1 = query_retrieval(uid, query_vec, index, train_meta, k=1, training=False)[0]

        pred_text = truncate(top1["findings"], 60)
        ref_text = truncate(study["findings"], 60)

        hypotheses[uid] = pred_text
        references[uid] = ref_text

        records.append({
            "uid": uid,
            "prediction": pred_text,
            "ground_truth": ref_text,
            "retrieved_uid": top1["uid"],
        })

    metrics, per_sample, _ = coco_scores(references, hypotheses)

    print(f"\nE1 Retrieval Baseline Results on Test Set (N={len(test_studies)}):")
    print(f"  {fmt(metrics)}")

    out_data = {
        "baseline": "E1_retrieval_top1",
        "num_test_studies": len(test_studies),
        "headline_metrics": metrics,
        "sample_records": records[:5],
    }
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved E1 evaluation results to {out_json_path}")
    return metrics


def train_and_eval_linear_probe(
    train_vectors: np.ndarray,
    train_labels: np.ndarray,
    train_uids: List[str],
    val_vectors: np.ndarray,
    val_labels: np.ndarray,
    val_uids: List[str],
    test_vectors: np.ndarray,
    test_labels: np.ndarray,
    test_uids: List[str],
    out_dir: str,
) -> Dict:
    """
    Train 14-finding linear probe classifiers on train features.
    1. Generates 5-fold Out-Of-Fold (OOF) predicted probabilities on train (avoids train/test prompt leak).
    2. Generates test and val predicted probabilities.
    3. Evaluates Macro-AUC across findings to benchmark encoder visual representation.
    """
    print("\n========================================================")
    print("Training 14-Finding Linear Probe & 5-Fold OOF Predictions")
    print("========================================================")

    n_train = len(train_vectors)
    n_val = len(val_vectors)
    n_test = len(test_vectors)
    n_labels = len(LABELS_14)

    train_oof_probs = np.zeros((n_train, n_labels), dtype=np.float32)
    val_probs = np.zeros((n_val, n_labels), dtype=np.float32)
    test_probs = np.zeros((n_test, n_labels), dtype=np.float32)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    auc_per_label = {}

    for l_idx, label_name in enumerate(LABELS_14):
        y_train = train_labels[:, l_idx]
        pos_count = int(y_train.sum())

        if pos_count < 5 or pos_count > n_train - 5:
            # Degenerate label frequency, use prior
            prior = pos_count / max(n_train, 1)
            train_oof_probs[:, l_idx] = prior
            val_probs[:, l_idx] = prior
            test_probs[:, l_idx] = prior
            auc_per_label[label_name] = 0.5
            continue

        # 5-fold OOF predictions on train
        for tr_idx, val_fold_idx in kf.split(train_vectors):
            X_tr, y_tr = train_vectors[tr_idx], y_train[tr_idx]
            X_vf = train_vectors[val_fold_idx]

            clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
            if len(np.unique(y_tr)) > 1:
                clf.fit(X_tr, y_tr)
                train_oof_probs[val_fold_idx, l_idx] = clf.predict_proba(X_vf)[:, 1]
            else:
                train_oof_probs[val_fold_idx, l_idx] = float(y_tr[0])

        # Train on full train set for val and test inference
        full_clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
        full_clf.fit(train_vectors, y_train)
        val_probs[:, l_idx] = full_clf.predict_proba(val_vectors)[:, 1]
        test_probs[:, l_idx] = full_clf.predict_proba(test_vectors)[:, 1]

        # Calculate AUC on test set if positive examples exist
        y_test = test_labels[:, l_idx]
        try:
            if len(np.unique(y_test)) > 1 and int(y_test.sum()) >= 2:
                auc = roc_auc_score(y_test, test_probs[:, l_idx])
            else:
                auc = roc_auc_score(y_train, train_oof_probs[:, l_idx])
        except Exception:
            auc = 0.5
        auc_per_label[label_name] = round(float(auc), 4)

    macro_auc = float(np.mean(list(auc_per_label.values())))
    print(f"Linear Probe Macro-AUC: {macro_auc:.4f}")
    for k, v in auc_per_label.items():
        print(f"  {k:26s}: AUC = {v:.4f}")

    # Save probability mappings: {uid: {label: prob, ...}}
    def make_prob_dict(uids, probs):
        result = {}
        for i, u in enumerate(uids):
            result[u] = {LABELS_14[j]: round(float(probs[i, j]), 4) for j in range(n_labels)}
        return result

    train_oof_dict = make_prob_dict(train_uids, train_oof_probs)
    val_pred_dict = make_prob_dict(val_uids, val_probs)
    test_pred_dict = make_prob_dict(test_uids, test_probs)

    with open(os.path.join(out_dir, "train_oof_priors.json"), "w", encoding="utf-8") as f:
        json.dump(train_oof_dict, f, indent=2)
    with open(os.path.join(out_dir, "val_priors.json"), "w", encoding="utf-8") as f:
        json.dump(val_pred_dict, f, indent=2)
    with open(os.path.join(out_dir, "test_priors.json"), "w", encoding="utf-8") as f:
        json.dump(test_pred_dict, f, indent=2)

    auc_summary = {
        "macro_auc": round(macro_auc, 4),
        "per_label_auc": auc_per_label,
    }
    with open(os.path.join("outputs", "linear_probe_auc.json"), "w", encoding="utf-8") as f:
        json.dump(auc_summary, f, indent=2)

    print(f"Saved prior probability files to {out_dir} and AUC report to outputs/linear_probe_auc.json")
    return auc_summary


def main():
    parser = argparse.ArgumentParser(description="Cache visual features and build retrieval index.")
    parser.add_argument("--data_dir", default="processed", help="Path to processed JSON files")
    parser.add_argument("--img_base_dir", default="", help="Base directory for image paths")
    parser.add_argument("--out_dir", default="processed/feats", help="Directory to save cached feature tensors")
    parser.add_argument("--encoder", default="microsoft/rad-dino", help="Hugging Face model ID for vision encoder")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for feature extraction")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 recommended on Windows)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_existing", action="store_true", help="Skip feature extraction if .pt files already exist")
    args = parser.parse_args()

    print(f"=== Plan 2.1 Feature Caching & Setup ===")
    print(f"Encoder: {args.encoder}")
    print(f"Device:  {args.device}")

    # Load dataset splits
    train_path = os.path.join(args.data_dir, "train.json")
    val_path = os.path.join(args.data_dir, "val.json")
    test_path = os.path.join(args.data_dir, "test.json")

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing split file: {p}. Run prepare_data.py first.")

    with open(train_path, "r", encoding="utf-8") as f:
        train_studies = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_studies = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_studies = json.load(f)

    print(f"Loaded studies: train={len(train_studies)}, val={len(val_studies)}, test={len(test_studies)}")

    # Load encoder
    print(f"\nLoading vision encoder: {args.encoder}...")
    model = AutoModel.from_pretrained(args.encoder).to(args.device)
    if args.device == "cuda":
        model = model.half()
    model.eval()

    transform, img_size = build_transform(args.encoder)

    # 1. Cache features for all splits
    train_vecs, train_labels, train_uids = cache_split_features(
        model, "train", train_studies, args.out_dir, transform,
        args.device, args.batch_size, args.num_workers, args.skip_existing
    )
    val_vecs, val_labels, val_uids = cache_split_features(
        model, "val", val_studies, args.out_dir, transform,
        args.device, args.batch_size, args.num_workers, args.skip_existing
    )
    test_vecs, test_labels, test_uids = cache_split_features(
        model, "test", test_studies, args.out_dir, transform,
        args.device, args.batch_size, args.num_workers, args.skip_existing
    )

    # 2. Build FAISS index over train studies only
    index_path = os.path.join(args.data_dir, "train_faiss.index")
    meta_path = os.path.join(args.data_dir, "train_faiss_meta.json")
    index = build_faiss_retrieval_index(train_vecs, train_studies, index_path, meta_path)

    with open(meta_path, "r", encoding="utf-8") as f:
        train_meta = json.load(f)

    # 3. Evaluate E1 Sanity Hurdle (Baseline Retrieval)
    os.makedirs("outputs", exist_ok=True)
    e1_path = os.path.join("outputs", "e1_retrieval_baseline.json")
    evaluate_e1_retrieval_baseline(test_vecs, test_studies, index, train_meta, e1_path)

    # 4. Train linear probe & compute 5-fold OOF priors
    train_and_eval_linear_probe(
        train_vecs, train_labels, train_uids,
        val_vecs, val_labels, val_uids,
        test_vecs, test_labels, test_uids,
        args.data_dir,
    )

    print("\nFeature caching, index generation, E1 baseline, and prior generation complete!")


if __name__ == "__main__":
    main()
