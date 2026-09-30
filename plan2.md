# Plan 2.1 (Corrected): Dual-View RAG-VLM for IU X-Ray Report Generation

Status of this document: revised after review. Items marked **[VERIFY]** are claims or
settings that must be checked when building. Expected results are **hypotheses**, not measurements.

---

## 0. Objective and success criteria

Beat the current benchmark table on **at least one headline metric**, under a fixed,
documented protocol, without degrading clinical accuracy.

| Paper | Year | BLEU-1 | BLEU-4 | METEOR | ROUGE-L |
|---|:---:|:---:|:---:|:---:|:---:|
| DART / DATR | 2025 | 0.486 | 0.208 | 0.205 | **0.411** |
| LePaX | 2026 | **0.531** | **0.235** | **0.316** | 0.402 |
| MPDRL | 2026 | 0.508 | 0.185 | 0.231 | 0.383 |
| DAMPER | 2025 | 0.520 | 0.225 | 0.284 | 0.397 |
| KiUT | 2023 | 0.525 | 0.185 | 0.242 | 0.409 |

Numbers above are taken from the benchmark listing provided at project start.
Competitor clinical-efficacy scores (CheXbert F1 etc.) are **not** included because
they could not be verified. Fill them in only from the papers themselves.

**Targets**
- Primary 1: ROUGE-L > 0.411
- Primary 2: METEOR > 0.316
- Stretch: BLEU-4 > 0.235, BLEU-1 > 0.531
- Clinical: report CheXbert F1 and RadGraph F1 for our model **and** for our own
  re-run baselines (constant normal report, frontal-only model). Do not claim a
  clinical win over papers unless their numbers are verified under the same tooling.

**Reference points from the literature** (unverified against our protocol):
LLaMA-XR reports ROUGE-L 0.433 / METEOR 0.336 on IU X-ray with DenseNet-121 + LLaMA 3.1
+ QLoRA, so a well-built LLM pipeline is plausible to clear the ROUGE-L / METEOR targets.

---

## 1. Fixed evaluation protocol (decide once, never change)

1. **Headline scorer:** `pycocoevalcap` (BLEU, METEOR 1.5 via Java, ROUGE-L, CIDEr) with
   its PTB tokenizer. Requires Java (`apt install default-jre` on Colab). **[VERIFY]**
   each competitor paper uses the same toolkit.
2. **Split:** R2Gen-style, target counts about 2,069 train / 296 val / 590 test.
   - Best case: obtain R2Gen's `annotation.json` (split ids) and use it verbatim.
   - Otherwise: build our own seeded split at `uid` level and **state clearly** it is not
     identical to R2Gen's. The local CSVs contain **no patient id**, so patient-level
     isolation is impossible; isolation is by `uid` only. Document this limitation.
3. **Study inclusion (R2Gen-compatible primary protocol):** keep studies with a frontal
   image, a lateral image, and a non-empty `findings` field. Single-view and
   empty-findings studies are dropped in the primary protocol.
   - Secondary variant (reported separately, not headline): include single-view studies
     with `has_lateral=False` and a duplicated frontal image; and/or impression fallback.
4. **Target text:** `findings` only (primary). **[VERIFY]** what each baseline used.
5. **Length handling:** references and generations are both truncated to 60 words in the
   primary protocol (R2Gen convention). Also log untruncated scores as a sensitivity check.
6. **Casing:** lowercase in the primary protocol; also log cased scores.
7. **Sanity baselines, measured on our split with the same scorer:**
   - ground truth vs ground truth (must be about 1.0)
   - constant "normal" report (record as the minimum hurdle)
   - nearest-neighbor retrieval only (copy the top-1 retrieved train report), which
     shows how much score comes from retrieval alone.

---

## 2. Data preparation (`src/prepare_data.py`)

### 2.1 Local files
- `data/images/`: 7,470 PNGs
- `data/indiana_projections.csv`: `uid`, `filename`, `projection` (Frontal / Lateral)
- `data/indiana_reports.csv`: `uid, MeSH, Problems, image, indication, comparison, findings, impression`

### 2.2 View resolution
- Frontal: first frontal image for the `uid` (prefer PA if the file distinguishes it).
- Lateral: first lateral image. If none exists, the study is dropped (primary) or gets
  `has_lateral=False` with a duplicated frontal image (secondary variant).

### 2.3 Text cleaning (corrected)
The earlier version used `re.sub(r'x+', '', ...)`, which deletes every "x" in normal words
("pneumothorax" becomes "pneumothora"). Only the anonymization placeholder must be removed.

```python
import re, pandas as pd

def clean_report(text) -> str:
    if pd.isna(text):
        return ""
    t = str(text).lower()
    t = re.sub(r"\bx{2,}\b", " ", t)          # anonymization placeholders only
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s([.,;:])", r"\1", t)
    words = t.strip().split()
    return " ".join(words[:60])               # 60-word cap (primary protocol)
```

Unit test: assert that `clean_report("No pneumothorax. Heart XXXX normal")` keeps
"pneumothorax" intact and drops only "xxxx". Print 20 random cleaned reports and eyeball them.

### 2.4 Labels for the auxiliary classifier
- Primary: run a CheXbert-based labeler on the cleaned training reports to get 14 labels
  (Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema,
  Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other,
  Fracture, Support Devices, No Finding). **[VERIFY]** which package provides a working,
  downloadable CheXbert checkpoint.
- Fallback: the rule-based CheXpert labeler, or weak labels mapped from the dataset's own
  `MeSH` / `Problems` columns.
- Uncertain labels (-1) are mapped to 0 for BCE (document this choice).

### 2.5 Record schema
```json
{"uid": "1",
 "frontal_path": "data/images/....png",
 "lateral_path": "data/images/....png",
 "has_lateral": true,
 "findings": "cleaned target text",
 "labels_14": [0,0,0,0,0,0,0,0,0,0,0,0,0,1]}
```
Outputs: `processed/{train,val,test}.json`, `processed/splits.json` (locked),
`processed/train_labels.json`, and a stats notebook (report length histogram, label
prevalence, fraction of exact-duplicate reports).

---

## 3. Model: Dual-View RAG-VLM

```
Frontal img ─┐                                    ┌─ 14-finding head (OOF probs in prompt)
             ├─ Vision encoder (frozen) ─ pool ───┤
Lateral img ─┘                                    ├─ Retriever (train pool, FAISS) ─ top-2 reports
                                                  └─ MLP projector ─ visual tokens
                                                                       │
                       Prompt (visual tokens + finding probs + retrieved reports)
                                                                       │
                                     Qwen2.5-3B-Instruct + QLoRA ── N candidates ── MBR ── report
```

### 3.1 Vision encoder (choose one, then set shapes from the real output)
Options: MedSigLIP (Google, gated access, 448 px), BiomedCLIP (native 224 px; higher
resolution needs position-embedding interpolation), or `google/siglip-base-patch16-384`
as a general-domain fallback. **[VERIFY]** the model id loads and record
`(num_tokens, hidden_dim)` from a test forward pass. Do not hardcode 576 / 768 / 1152.

- Reshape the patch grid to `[B, D, H, W]` and apply `AdaptiveAvgPool2d((12, 12))`, giving
  **144 tokens per view** regardless of the encoder's native grid.
- Add learned view embeddings (frontal / lateral), then concatenate: **288 visual tokens**.
- Encoder stays frozen, so **pre-compute and cache pooled features** to disk (fp16). This
  removes the encoder from the training loop and is the main speed lever. Augmentation is
  therefore limited to what is applied before caching (none by default).

### 3.2 Projector
2-layer MLP: `Linear(D_enc -> D_llm) -> GELU -> Linear(D_llm -> D_llm)`, with `D_llm`
read from the LLM config (2048 for Qwen2.5-3B **[VERIFY]**).

### 3.3 Auxiliary classifier and train/test mismatch fix
- Linear head on the globally pooled representation (both views concatenated), BCE loss
  against the 14 labels.
- The LLM prompt must contain probabilities that look the same at train and test time:
  - **Training prompts use out-of-fold (5-fold) predictions** from a linear probe on the
    cached features, never ground-truth labels.
  - Val/test prompts use predictions from the probe trained on the full train set.
  - Optional light noise on top; OOF predictions are the primary mechanism.

### 3.4 Retrieval (RAG) with correct leakage handling
- Index: `faiss.IndexFlatIP` over L2-normalized global embeddings of **train studies only**.
- Query embedding: concatenated frontal + lateral pooled global vectors.
- **Train time:** exclude the query's own `uid`. There is no patient id in the data, so
  same-patient exclusion cannot be done; note this limitation.
- **Val/test time:** search the full train index. Val and test are never indexed.
- Do not exclude candidates just because their text equals the target: exact-duplicate
  "normal" reports are legitimately common and occur at test time too.

```python
def retrieve(query_uid, query_emb, index, train_meta, k=2, training=True):
    _, idx = index.search(query_emb[None, :], k + 5)
    out = []
    for i in idx[0]:
        c = train_meta[i]
        if training and c["uid"] == query_uid:
            continue
        out.append(c["findings"])
        if len(out) == k:
            break
    return out
```

### 3.5 LLM and adaptation
- `Qwen/Qwen2.5-3B-Instruct`, 4-bit NF4 + double quantization, LoRA r=32, alpha=64,
  dropout 0.05 on all attention and MLP projections.
- Precision: bf16 on L4/A100. On a T4 (no bf16) use fp16 with fp32 LoRA/norm params;
  watch for NaN losses. If they persist, use Qwen2.5-1.5B or another small model.

### 3.6 Prompt template
```
<|im_start|>system
You are an expert radiologist. Write the findings section of a chest X-ray report.
Use only what the images and indicators support.<|im_end|>
<|im_start|>user
[VISUAL_TOKENS]
Predicted indicators: cardiomegaly 0.12, pleural effusion 0.05, lung opacity 0.08, ...
Similar case 1: "{retrieved_1}"
Similar case 2: "{retrieved_2}"
Write the findings.<|im_end|>
<|im_start|>assistant
{target_findings}
```
Loss is computed only on the assistant tokens (prompt tokens masked with -100).

---

## 4. Training protocol

**Loss:** `L = L_CE(report tokens) + 0.2 * L_BCE(14 findings)`

| Stage | What trains | Frozen | Epochs | LR |
|---|---|---|---|---|
| 1. Warmup | MLP projector + classifier head | encoder, LLM | 3 | 1e-3 |
| 2. SFT | LoRA + projector + head | encoder | 10-12, early stop on val ROUGE-L (patience 3) | LoRA 1e-4, projector 2e-5 |

Common: AdamW (weight decay 0.01), linear warmup then cosine, effective batch 16,
gradient checkpointing, seeds 42 / 1337 / 2026.

**Compute:** the earlier "3.5 min per epoch, zero OOM risk" figure was a guess. With
about 2,000 training studies and roughly 750-token sequences through a 3B QLoRA model, a
T4 epoch may take on the order of 10-20 minutes. **Time one epoch first** and set the
budget from the measurement. Peak VRAM must also be measured, not assumed.

Sequence budget: 288 visual tokens + prompt + two retrieved reports (about 80-100 tokens
each) + target, expected roughly 700-800 tokens. **[VERIFY]** by tokenizing real samples.

---

## 5. Decoding

1. **Baseline decoding:** beam search (beams 3-5), `no_repeat_ngram_size=3`,
   `max_new_tokens=128` (60 words can exceed 60 tokens, so a 60-token cap would cut
   sentences off). Tune length on val.
2. **Repetition penalty:** avoid 1.2; normal reports repeat phrases legitimately. Use
   `no_repeat_ngram_size` or a penalty near 1.05.
3. **MBR decoding:**
   - Pool: N=16 candidates (mix of beam outputs and samples at T=0.7, top-p=0.9).
   - Utility: a **fast** proxy such as `0.6*ROUGE-L + 0.4*unigram/bigram F1` computed with
     a local LCS implementation. Java METEOR over about 140,000 pairs is too slow.
   - Choose `argmax_i mean_{j != i} U(c_i, c_j)`.
   - Tune the utility weights, N, and temperature on **val**, never on test.
   - Run `pycocoevalcap` only on the final selected reports.

---

## 6. Evaluation and rigor

- NLG: BLEU-1..4, METEOR, ROUGE-L, CIDEr (`pycocoevalcap`).
- Clinical: CheXbert F1 (micro/macro) and RadGraph F1 **[VERIFY]** install paths.
  Also report scores on the **abnormal-only subset** to detect collapse to the normal template.
- 3 seeds, mean +- std.
- Paired bootstrap (1,000 resamples) versus our own baselines.
- Error analysis on 30-50 sampled reports: hallucinated findings, missed abnormalities,
  laterality mistakes, copying from retrieved text.
- Copy check: measure how often the output is identical to a retrieved report.

---

## 7. Ablation matrix (fill with measured values)

| ID | View(s) | Prior | RAG | Decoding | ROUGE-L | METEOR | B-4 | B-1 | CIDEr |
|---|---|---|---|---|---|---|---|---|---|
| E0 | constant normal report | - | - | - | 0.2809 | 0.1455 | 0.0603 | 0.2390 | 0.1855 |
| E1 | retrieval only (top-1 copy) | - | yes | - | 0.2513 | 0.1514 | 0.0871 | 0.3449 | 0.1970 |
| E2 | frontal | none | none | beam | | | | |
| E3 | frontal + lateral | none | none | beam | | | | |
| E4 | frontal + lateral | 14-finding | none | beam | | | | |
| E5 | frontal + lateral | 14-finding | top-2 | beam | | | | |
| E6 | frontal + lateral | 14-finding | none | MBR | | | | |
| E7 | frontal + lateral | 14-finding | top-2 | MBR (full) | | | | |

E6 versus E5 and E7 separates the contribution of RAG from that of MBR.

---

## 8. If the first build falls short (Phase 2 options)

1. **Encoder fine-tuning:** unfreeze the top 2 encoder blocks (lr 1e-5). This means
   giving up feature caching.
2. **RL fine-tuning (GRPO/PPO-style):** reward `0.5*ROUGE-L + 0.3*METEOR + 0.2*RadGraph F1`,
   with a KL penalty and a length penalty to prevent reward hacking.
3. **MIMIC-CXR pretraining** before IU fine-tuning (needs PhysioNet credentialed
   access). Likely the largest gain since IU is small.
4. **Checkpoint/seed ensembling** into the MBR candidate pool.
5. **Draft-verify-refine** second pass for clinical accuracy.

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Scores not comparable with papers | Fixed split/scorer; R2Gen ids if obtainable; report our own re-run baselines |
| Retrieval leakage | Train index only; exclude own uid at train time; val/test never indexed |
| Collapse to "normal" template | Classifier prior, abnormal-subset metrics, copy check |
| Generation truncated mid-sentence | `max_new_tokens=128`, tune length on val |
| Slow MBR | Fast utility proxy; headline scoring only on final outputs |
| fp16 instability on T4 | fp32 LoRA/norms, monitor NaN, smaller LLM fallback |
| Encoder id or shapes wrong | Verify load and print shapes before anything else |
| Overfitting on about 2,000 studies | LoRA, early stopping, OOF priors, optional pretraining |

---

## 10. Repository layout

```
project/
├── data/                       # read-only raw dataset
├── processed/
│   ├── splits.json             # locked split ids
│   ├── train.json  val.json  test.json
│   ├── train_labels.json
│   ├── feats/                  # cached pooled encoder features (fp16)
│   └── train_faiss.index
├── src/
│   ├── prepare_data.py         # aggregate, clean, split, label
│   ├── evaluate.py             # pycocoevalcap + clinical metrics + baselines
│   ├── cache_features.py       # encoder forward pass + pooling
│   ├── dataset.py              # dual-view features, RAG lookup, prompt building
│   ├── model.py                # projector + classifier + Qwen QLoRA
│   ├── train_sft.py            # two-stage training
│   └── generate_mbr.py         # beam, sampling, MBR
├── notebooks/colab_runner.ipynb
├── outputs/                    # checkpoints, generations, metric logs
├── plan.md
└── plan2.md
```

---

## 11. Build order

1. `prepare_data.py`: cleaning unit test, split, labels, stats.
2. `evaluate.py`: scorer, sanity baselines E0 and E1 on our split.
3. Encoder check: load, print shapes, `cache_features.py`.
4. Time one training epoch and measure peak VRAM.
5. Frontal-only baseline (E2); confirm it beats the constant normal report.
6. Add one component at a time (E3 to E7), logging each result.
7. Multi-seed runs, significance tests, error analysis, write-up.
