"""
dataset.py - PyTorch Dataset and collator for Dual-View RAG-VLM training and evaluation.

Features:
  * Loads pre-cached dual-view features ([144, 768] frontal + [144, 768] lateral)
  * Dynamic train-safe retrieval: queries FAISS index and excludes query UID during training
  * OOF prior probabilities during training (val/test priors at evaluation)
  * Standard Qwen ChatML prompt format with 288 <|image_pad|> tokens
  * Assistant-only cross-entropy loss masking (labels = -100 for system & user prompt)
  * Returns 14-finding ground truth multi-labels for auxiliary classification head
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

LABELS_14 = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding",
]

IMAGE_PAD_TOKEN = "<|image_pad|>"
NUM_VISUAL_TOKENS_PER_VIEW = 144
TOTAL_VISUAL_TOKENS = 288


class IUReportDataset(Dataset):
    """
    Multimodal Dataset for IU X-Ray Report Generation.
    """

    def __init__(
        self,
        split_json_path: str,
        feats_dir: str,
        tokenizer: PreTrainedTokenizer,
        priors_json_path: Optional[str] = None,
        faiss_index_path: Optional[str] = None,
        faiss_meta_path: Optional[str] = None,
        k_retrieval: int = 2,
        is_training: bool = True,
        max_seq_len: int = 800,
        use_retrieval: bool = True,
        use_priors: bool = True,
        use_lateral: bool = True,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.feats_dir = feats_dir
        self.is_training = is_training
        self.max_seq_len = max_seq_len
        self.k_retrieval = k_retrieval
        self.use_retrieval = use_retrieval
        self.use_priors = use_priors
        self.use_lateral = use_lateral

        # 1. Load study metadata
        with open(split_json_path, "r", encoding="utf-8") as f:
            self.studies = json.load(f)

        # 2. Load prior probabilities (OOF for train, direct predictions for val/test)
        self.priors = {}
        if self.use_priors and priors_json_path and os.path.exists(priors_json_path):
            with open(priors_json_path, "r", encoding="utf-8") as f:
                self.priors = json.load(f)

        # 3. Load retrieval index and metadata
        self.index = None
        self.train_meta = None
        if self.use_retrieval and faiss_index_path and os.path.exists(faiss_index_path):
            self.index = faiss.read_index(faiss_index_path)
            if faiss_meta_path and os.path.exists(faiss_meta_path):
                with open(faiss_meta_path, "r", encoding="utf-8") as f:
                    self.train_meta = json.load(f)

        # Ensure image pad token is set
        if IMAGE_PAD_TOKEN not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_PAD_TOKEN]})
        self.image_pad_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)

    def __len__(self) -> int:
        return len(self.studies)

    def _format_priors_text(self, uid: str) -> str:
        """Format 14-finding prior probabilities into a concise clinical text."""
        if not self.use_priors or uid not in self.priors:
            return ""

        study_priors = self.priors[uid]
        # Sort by highest probability
        sorted_findings = sorted(study_priors.items(), key=lambda x: x[1], reverse=True)
        # Highlight top non-zero findings, plus No Finding
        items = []
        for name, prob in sorted_findings:
            if prob >= 0.05 or name == "No Finding":
                items.append(f"{name.lower()} {prob:.2f}")

        if not items:
            items = [f"{k.lower()} {v:.2f}" for k, v in sorted_findings[:5]]

        return "Predicted indicators: " + ", ".join(items) + ".\n"

    def _retrieve_similar_cases(self, uid: str, global_vec: torch.Tensor) -> List[str]:
        """Retrieve top-k similar training study findings, strictly excluding query UID when training."""
        if not self.use_retrieval or self.index is None or self.train_meta is None:
            return []

        emb = global_vec.numpy().reshape(1, -1)
        # Search k + 5 candidates to allow safe exclusion of own UID
        _, indices = self.index.search(emb, self.k_retrieval + 5)
        cases = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self.train_meta):
                continue
            cand = self.train_meta[idx]
            if self.is_training and cand["uid"] == uid:
                continue  # Train-time retrieval leakage guard
            cases.append(cand["findings"])
            if len(cases) == self.k_retrieval:
                break
        return cases

    def __getitem__(self, idx: int) -> Dict:
        study = self.studies[idx]
        uid = str(study["uid"])
        target_findings = study.get("findings", "")

        # 1. Load cached visual features
        feat_path = os.path.join(self.feats_dir, f"{uid}.pt")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Cached feature missing for UID {uid}: {feat_path}")

        cached = torch.load(feat_path, map_location="cpu")
        frontal_feat = cached["frontal"].float()  # [144, 768]
        if self.use_lateral:
            lateral_feat = cached["lateral"].float()  # [144, 768]
        else:
            lateral_feat = frontal_feat.clone()

        global_vec = cached["global"].float()  # [1536]
        labels_14 = cached.get("labels", torch.zeros(14, dtype=torch.float32)).float()

        # 2. Retrieve similar cases
        similar_cases = self._retrieve_similar_cases(uid, global_vec)
        rag_text = ""
        for c_idx, case_text in enumerate(similar_cases, 1):
            rag_text += f'Similar case {c_idx}: "{case_text}"\n'

        # 3. Clinical prior text
        prior_text = self._format_priors_text(uid)

        # 4. Construct ChatML prompt
        # Visual tokens placeholder: 288 tokens total (or 144 if single view)
        num_vis = TOTAL_VISUAL_TOKENS if self.use_lateral else NUM_VISUAL_TOKENS_PER_VIEW
        vis_tokens_str = IMAGE_PAD_TOKEN * num_vis

        system_prompt = (
            "<|im_start|>system\n"
            "You are an expert radiologist. Write the findings section of a chest X-ray report. "
            "Use only what the images and indicators support.<|im_end|>\n"
        )

        user_content = f"{vis_tokens_str}\n{prior_text}{rag_text}Write the findings.<|im_end|>\n"
        user_prompt = f"<|im_start|>user\n{user_content}"
        assistant_prefix = "<|im_start|>assistant\n"

        prompt_str = system_prompt + user_prompt + assistant_prefix

        # Tokenize prompt and target
        prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)

        if self.is_training:
            target_str = f"{target_findings}<|im_end|>"
            target_ids = self.tokenizer.encode(target_str, add_special_tokens=False)

            # Combined sequence
            input_ids = prompt_ids + target_ids
            # Assistant-only loss masking: prompt tokens are -100
            labels = [-100] * len(prompt_ids) + target_ids

            # Truncate if exceeds max length
            if len(input_ids) > self.max_seq_len:
                input_ids = input_ids[: self.max_seq_len]
                labels = labels[: self.max_seq_len]
        else:
            input_ids = prompt_ids
            labels = [-100] * len(prompt_ids)

        return {
            "uid": uid,
            "frontal_feat": frontal_feat,
            "lateral_feat": lateral_feat,
            "labels_14": labels_14,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "target_findings": target_findings,
            "prompt_length": len(prompt_ids),
        }


def collate_fn(batch: List[Dict], pad_token_id: int = 151643) -> Dict:
    """Collate batch with dynamic padding."""
    uids = [item["uid"] for item in batch]
    frontal_feats = torch.stack([item["frontal_feat"] for item in batch])
    lateral_feats = torch.stack([item["lateral_feat"] for item in batch])
    labels_14 = torch.stack([item["labels_14"] for item in batch])
    targets = [item["target_findings"] for item in batch]
    prompt_lengths = [item["prompt_length"] for item in batch]

    lengths = [len(item["input_ids"]) for item in batch]
    max_len = max(lengths)

    padded_input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i, item in enumerate(batch):
        l = len(item["input_ids"])
        padded_input_ids[i, :l] = item["input_ids"]
        padded_labels[i, :l] = item["labels"]
        attention_mask[i, :l] = 1

    return {
        "uids": uids,
        "frontal_feats": frontal_feats,      # [B, 144, 768]
        "lateral_feats": lateral_feats,      # [B, 144, 768]
        "labels_14": labels_14,              # [B, 14]
        "input_ids": padded_input_ids,       # [B, max_len]
        "attention_mask": attention_mask,    # [B, max_len]
        "labels": padded_labels,             # [B, max_len]
        "target_findings": targets,
        "prompt_lengths": prompt_lengths,
    }
