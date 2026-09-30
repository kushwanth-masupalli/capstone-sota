# Plan: Beat SOTA on IU X-Ray Report Generation

## 0. Goal and success criteria

Beat the current table on **at least one** metric, using a fixed, reproducible protocol.

| Metric | Target to beat | Difficulty |
|---|---|---|
| ROUGE-L | > 0.411 | Easiest |
| METEOR | > 0.316 | Easy |
| BLEU-4 | > 0.235 | Hard |
| BLEU-1 | > 0.531 | Hard |

**Primary targets: ROUGE-L and METEOR.** BLEU-1/4 are stretch goals.
Also report clinical metrics (CheXbert F1, RadGraph F1) so the win is credible.

**Known reference points from the literature:**
- LLaMA-XR (DenseNet-121 + LLaMA 3.1 + QLoRA): ROUGE-L 0.433, METEOR 0.336
- MLLM-RRG: BLEU-4 0.235 (0.240 on cleaned data)
- BoxMed-RL: CoT + RL, ~7% average gain in METEOR / ROUGE-L

---

## 1. Assumptions

- Only the IU X-ray dataset is available locally (images + reports).
- One GPU with 24 GB VRAM or more (A100/4090/3090 class). Adjust batch size and
  quantization if less.
- MIMIC-CXR pretraining is **optional** (needs PhysioNet credentialed access). The
  plan works without it.
- Python 3.10+, PyTorch 2.x, Hugging Face stack.

---

## 2. Project structure

```
project/
├── data/                 # your dataset folder (raw, read-only)
├── processed/            # cleaned csv/json, splits
├── src/
│   ├── prepare_data.py   # parse, clean, split
│   ├── dataset.py        # torch Dataset (2 views per study)
│   ├── model.py          # encoder + projector + LLM
│   ├── train_sft.py      # supervised fine-tuning
│   ├── train_rl.py       # optional RL stage
│   ├── generate.py       # beam search + sampling + MBR
│   └── evaluate.py       # NLG + clinical metrics
├── configs/
├── outputs/              # checkpoints, generated reports, score logs
└── plan.md
```

---

## 3. Phase 1: Data preparation (Day 1)

1. **Inspect the dataset folder.** Identify the layout: XML reports + PNGs (original
   Open-i release), or CSVs (Kaggle version). Confirm image-to-report mapping via
   the study/UID.
2. **Build one record per study:**
   `{uid, frontal_img, lateral_img, findings, impression, full_report}`
3. **Clean the text** (apply the same rules to train and test):
   - lowercase, strip de-identification tokens (`XXXX`), normalize whitespace
   - target text = `findings` (standard for IU X-ray); fall back to impression if
     findings is empty
   - drop studies with no images or empty reports
4. **Split.** Use the widely used R2Gen-style split (~70/10/20, split by patient/study,
   test ≈ 590 studies in that convention). Save split ids to
   `processed/splits.json` and **never change them**.
5. **Handle missing lateral views** by duplicating the frontal image, or by
   masking the second view.
6. **Extract weak labels** with CheXbert or CheXpert-labeler (14 findings) from the
   training reports. These are used for conditioning and clinical evaluation.

**Deliverable:** `processed/{train,val,test}.json`, label file, data stats notebook.

---

## 4. Phase 2: Evaluation harness first (Day 1-2)

Build the scorer **before** the model so all comparisons are fair.

- NLG: BLEU-1..4, METEOR, ROUGE-L, CIDEr via `pycocoevalcap`
- Clinical: CheXbert F1 (micro/macro), RadGraph F1
- Report both **cased** and **lowercased** scores, and document the tokenization
- Sanity checks:
  - copy of ground truth scores ~1.0
  - a constant "normal" report (e.g. "the lungs are clear. no pneumothorax or pleural
    effusion.") gives a strong baseline on IU. Record this number. Anything that
    doesn't clearly beat it is not learning much.

---

## 5. Phase 3: Baseline (Day 2-3)

Get a working end-to-end pipeline before adding tricks.

- **Encoder:** a medical vision encoder, frozen at first
  (try BiomedCLIP, MedSigLIP, or RAD-DINO; pick by quick linear-probe on the weak labels)
- **Projector:** 2-layer MLP mapping visual tokens to the LLM embedding space
- **Decoder:** Qwen3-4B (or LLaMA 3.1-8B if VRAM allows) with LoRA (r=16-64), 4-bit
  quantization if needed
- **Input:** both views' tokens concatenated, then a short instruction prompt
- **Training:** cross-entropy on report tokens, AdamW, lr 1e-4 (LoRA/projector),
  cosine schedule, 10-20 epochs, early stop on val ROUGE-L
- **Decode:** beam=3-5, `no_repeat_ngram_size=3`

**Exit criterion:** test ROUGE-L in the 0.36-0.40 range. If lower, debug the data
and prompt before moving on.

---

## 6. Phase 4: Improvements, ordered by expected gain (Week 1-2)

Add one at a time and log the ablation delta for each.

| # | Change | Expected effect |
|---|---|---|
| 1 | Unfreeze the top vision layers / LoRA on the encoder | +ROUGE-L |
| 2 | **Label conditioning:** predict 14 findings with a classifier head, then feed the labels as text in the prompt | + all metrics, clinical F1 |
| 3 | **Retrieval augmentation:** retrieve top-k similar training reports (image embedding similarity), add them to the prompt | + BLEU, ROUGE-L |
| 4 | **Draft, verify, refine** (two-pass generation) | + clinical accuracy |
| 5 | Generation-length and repetition-penalty tuning on val | + ROUGE-L, METEOR |
| 6 | Optional: **MIMIC-CXR pretraining**, then fine-tune on IU | + everything (largest lever if access exists) |

**Leakage rule for retrieval:** retrieve only from the training split, and never
from val/test.

---

## 7. Phase 5: Metric-targeted boosts (Week 2)

These directly attack ROUGE-L / METEOR.

1. **MBR decoding (no training needed).**
   Sample N=16-32 candidates (temperature 0.7 plus beam outputs), score every pair with
   ROUGE-L/METEOR, and output the candidate with the highest average agreement.
2. **RL fine-tuning (GRPO or PPO-style).**
   Reward = `0.5*ROUGE-L + 0.3*METEOR + 0.2*RadGraph-F1`, plus a KL penalty to the SFT
   model and a length penalty. Start from the best SFT checkpoint.
3. **Ensembling:** merge candidates from 3+ checkpoints or seeds into the MBR pool.

---

## 8. Phase 6: Final evaluation (Week 3)

- Train the final config with **3 seeds**; report mean +- std on the test set.
- Compare against the table **and** against your own re-run of one published baseline
  under the identical split and scorer.
- Run a paired bootstrap significance test (1000 resamples) against the best baseline
  for each metric.
- Error analysis: 30-50 sampled reports, checking hallucinated findings, omitted
  abnormalities, and laterality errors.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Scores not comparable to papers (split / lowercasing differences) | Fixed split and scorer; report cased and uncased; re-run a baseline |
| Model collapses to the "normal" template | Track clinical F1 and abnormal-only subset scores; use label conditioning |
| Overfitting (only ~3.9k reports) | LoRA, early stopping, augmentation (mild), pretraining if possible |
| Data leakage via retrieval or duplicate patients | Split by patient; retrieve from train only |
| RL reward hacking (length inflation) | Length penalty, KL term, clinical reward component |
| VRAM limits | 4-bit QLoRA, gradient checkpointing, smaller LLM (Qwen3-1.7B) |

---

## 10. Timeline

| Week | Milestone |
|---|---|
| 1, days 1-3 | Data prep, eval harness, baseline running |
| 1, days 4-7 | Encoder tuning, label conditioning, retrieval |
| 2 | Decoding tuning, MBR, RL stage, ablations |
| 3 | Multi-seed runs, significance tests, write-up |

---

## 11. Immediate next steps

1. Share the dataset folder structure (`ls` output) so `prepare_data.py` can be written to match.
2. Confirm GPU model and VRAM.
3. Confirm whether MIMIC-CXR access is available.
4. Start with Phase 1 and Phase 2 (data + evaluation harness).
