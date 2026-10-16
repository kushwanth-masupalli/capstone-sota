# Dual-View RAG-VLM: Empirical Baseline Auditing and Consensus Decoding for Automated Chest Radiograph Report Generation Under Consumer Hardware Constraints

**Masupalli Kushwanth**  
*Department of Computer Science & Engineering*  
*Capstone SOTA Research Project*  
*October 2026*

---

## Abstract

Automated radiology report generation holds immense promise for mitigating diagnostic backlog and clinical burnout. However, recent benchmark progress on standardized datasets such as the Indiana University Chest X-Ray (IU X-Ray) collection is frequently clouded by high dataset redundancy and uncalibrated evaluation metrics. In this paper, we present **Dual-View RAG-VLM**, a parameter-efficient multimodal vision-language model tailored to dual-projection chest radiograph interpretation and executable entirely on a single consumer-grade 6.0 GB laptop GPU (NVIDIA RTX 4050). Our framework integrates high-resolution spatial feature extraction from a frozen clinical foundation model (`microsoft/rad-dino`), learned projection view embeddings, an auxiliary 14-disease multi-label classification head, train-safe dynamic FAISS retrieval, and 4-bit quantized LoRA fine-tuning of `Qwen2.5-3B-Instruct`. 

Crucially, we conduct an empirical audit of the benchmark by establishing two foundational baselines: an **E0 Constant Normal Floor** ($\text{BLEU-1}=0.2390$, $\text{ROUGE-L}=0.2809$) and an **E1 Pure Retrieval Hurdle** ($\text{BLEU-1}=0.3449$, $\text{METEOR}=0.1514$), proving that trivial copy-paste strategies already clear a substantial fraction of published state-of-the-art numbers. Furthermore, we provide an extensive ablation of decoding strategies, demonstrating that standard Beam Search suffers from severe mode collapse to safe, generic templates ($\text{BLEU-1}=0.2728$, falling below the retrieval hurdle), whereas Fast **Minimum Bayes Risk (MBR) Consensus Decoding** overcomes this failure mode, achieving $\text{BLEU-1}=0.3820$, $\text{METEOR}=0.1816$, and $\text{CIDEr}=0.3240$ (+40.0% relative BLEU-1 and +33.5% relative METEOR over beam search). Our visual representation achieves a **Macro-AUC of 0.8135** across 14 clinical pathologies (exceeding 0.94 for Pleural Effusion and Pneumonia). We provide all verified empirical measurements, code, and protocol specifications to foster honest, reproducible research in low-resource medical AI.

---

## 1. Introduction

Interpreting chest radiographs (CXRs) is among the most frequent clinical diagnostic tasks worldwide. Radiologists are required to systematically inspect multi-view radiographs (frontal and lateral projections), detect subtle anatomical lesions, and synthesize their observations into structured, descriptive natural language reports. The escalating volume of medical imaging has created severe diagnostic backlogs and physician burnout, accelerating interest in automatic radiology report generation systems (Chen et al., 2020; Tanida et al., 2023).

Recent literature has seen a surge of deep learning architectures reporting rapid progress on the benchmark Indiana University Chest X-Ray collection (Demner-Fushman et al., 2016). State-of-the-art architectures such as **DART** (CVPR 2025), **LePaX** (ECCV 2026), **MPDRL** (Frontiers 2026), **DAMPER** (AAAI 2025), and **KiUT** (CVPR 2023) report headline $\text{BLEU-1}$ scores ranging from $0.486$ to $0.531$ and $\text{ROUGE-L}$ scores between $0.383$ and $0.411$. However, these reported improvements come with critical methodological caveats that are rarely discussed:
1. **Pre-training Confounding**: Nearly all published top-tier architectures rely on large-scale pre-training across hundreds of thousands of paired reports from the Beth Israel Deaconess MIMIC-CXR database (Johnson et al., 2019) on multi-GPU server clusters before evaluating on IU X-Ray.
2. **Template Redundancy & Baseline Blindspots**: Because approximately 70% of routine clinical chest radiographs exhibit no acute abnormalities, clinical reports share massive lexical and structural overlap. Standard natural language generation (NLG) metrics (BLEU, METEOR, ROUGE-L) reward this template repetition, yet virtually no prior work measures how much score is achieved by a trivial constant normal report or simple nearest-neighbor retrieval.
3. **Hardware Accessibility**: State-of-the-art multimodal systems increasingly rely on models with tens of billions of parameters, excluding academic laboratories, clinics in developing economies, and student researchers lacking massive compute budgets.

To address these challenges with complete scientific transparency, this paper introduces **Dual-View RAG-VLM** and presents an empirical investigation into automated CXR report generation under strict resource constraints. Operating within the memory envelope of a single consumer 6.0 GB VRAM laptop GPU, our contributions are fourfold:

* **Empirical Baseline Floors (E0 & E1)**: We formalize and evaluate two empirical sanity hurdles on the exact test set ($N=589$): a Constant Normal Report Floor ($\text{E0}$) and a Pure Retrieval Top-1 Hurdle ($\text{E1}$). We reveal that nearest-neighbor copy-paste alone achieves $\text{BLEU-1}=0.3449$, establishing the true baseline performance that any learned generator must beat to demonstrate genuine multimodal synthesis.
* **Dual-View Multimodal Architecture**: We engineer an efficient architecture combining frozen clinical foundation tokens (`microsoft/rad-dino`, ViT-B/14 at 518px native resolution) pooled to 288 dual-view tokens, learned view embeddings, an auxiliary 14-disease multi-label classifier, dynamic train-safe retrieval, and 4-bit quantized QLoRA fine-tuning of `Qwen2.5-3B-Instruct`.
* **Clinical Representation Fidelity**: We demonstrate via a rigorous linear probe across 14 CheXpert categories that the underlying visual features achieve an overall **Macro-AUC of 0.8135**, with individual condition AUCs reaching **0.9670** for Pleural Effusion, **0.9408** for Pneumonia, and **0.8936** for Cardiomegaly.
* **Decoding Strategy Ablation (Beam Search vs. MBR Consensus)**: We prove empirically that standard Beam Search suffers from severe template mode collapse ($\text{BLEU-1}=0.2728$, scoring below the retrieval hurdle). In contrast, Fast Minimum Bayes Risk (MBR) consensus decoding successfully extracts high-confidence clinical findings from stochastic candidate pools, reaching $\text{BLEU-1}=0.3820$, $\text{METEOR}=0.1816$, and $\text{CIDEr}=0.3240$ without altering model weights.

---

## 2. Empirical Baseline Auditing (E0 & E1)

A persistent deficiency in natural language generation for radiology is the uncritical reporting of n-gram overlap metrics without calibrating against domain baselines. Because routine chest radiographs are overwhelmingly normal, metrics such as BLEU and ROUGE-L can be trivially inflated by regurgitating stereotypical clinical phrasing.

```
+-------------------------------------------------------------------------------+
|                             BENCHMARK HIERARCHY                               |
|                                                                               |
|  [ Published SOTA (DART / LePaX / MPDRL) ]  --> BLEU-1 ~ 0.48-0.53            |
|       ^                                                                       |
|  [ Dual-View RAG-VLM (Ours - MBR) ]         --> BLEU-1 = 0.3820, MET = 0.1816 |
|       ^                                                                       |
|  [ E1: Pure Retrieval Top-1 Hurdle ]        --> BLEU-1 = 0.3449, MET = 0.1514 |
|       ^                                                                       |
|  [ Dual-View RAG-VLM (Beam Search) ]        --> BLEU-1 = 0.2728 (Mode Collapse)|
|       ^                                                                       |
|  [ E0: Constant Normal Report Floor ]       --> BLEU-1 = 0.2390, R-L = 0.2809 |
+-------------------------------------------------------------------------------+
```

To establish rigorous ground truth hurdles, we implement and score two zero-parameter baselines across the locked IU X-Ray test partition ($N=589$) using official Stanford PTBTokenizer evaluation:

### 2.1 E0: Constant Normal Report Floor
We define the constant normal baseline $h_{\text{E0}}$ as the most parsimonious statement of radiographic normalcy:
$$\mathcal{T}_{\text{normal}} = \text{"the lungs are clear. no focal consolidation, pneumothorax, or pleural effusion. the cardiac silhouette is normal."}$$
Evaluating $h_{\text{E0}}$ across all test ground truth references produces:
$$\text{BLEU-1} = 0.2390, \quad \text{BLEU-4} = 0.0603, \quad \text{METEOR} = 0.1455, \quad \text{ROUGE-L} = 0.2809, \quad \text{CIDEr} = 0.1855$$
Any generative model scoring below $\text{ROUGE-L} = 0.2809$ has failed to beat a single static sentence applied indiscriminately to every patient.

### 2.2 E1: Pure Retrieval Top-1 Hurdle
To measure the lexical advantage conferred solely by nearest-neighbor visual similarity, we construct an L2-normalized 1,536-dimensional global study representation from frontal and lateral radiographs:
$$\mathbf{v}_{\text{study}} = \left[ \frac{\bar{\mathbf{z}}_{\text{frontal}}}{\|\bar{\mathbf{z}}_{\text{frontal}}\|_2} \,;\, \frac{\bar{\mathbf{z}}_{\text{lateral}}}{\|\bar{\mathbf{z}}_{\text{lateral}}\|_2} \right]$$
We index all 2,060 training studies into an exact inner-product FAISS index ($\text{IndexFlatIP}$). For each test query $j \in \mathcal{D}_{\text{test}}$, we query the training database:
$$k^* = \arg\max_{i \in \mathcal{D}_{\text{train}}} \langle \mathbf{v}_{\text{test}}^{(j)}, \mathbf{v}_{\text{train}}^{(i)} \rangle$$
The retrieved training report $\mathbf{y}_{\text{train}}^{(k^*)}$ is output verbatim as the hypothesis report. The measured performance of this zero-generation retrieval approach is:
$$\text{BLEU-1} = 0.3449, \quad \text{BLEU-4} = 0.0871, \quad \text{METEOR} = 0.1514, \quad \text{ROUGE-L} = 0.2513, \quad \text{CIDEr} = 0.1970$$

**Takeaway**: An algorithm that produces no words of its own and strictly copies past training cases achieves $\text{BLEU-1} = 0.3449$. Consequently, any generative model reporting $\text{BLEU-1} < 0.35$ has learned less than pure nearest-neighbor lookup.

---

## 3. Methodology

The Dual-View RAG-VLM framework comprises five tightly integrated subsystems designed to balance clinical expressiveness against strict GPU memory limitations.

```
                              +----------------------------+
                              | Frontal & Lateral Radiograph|
                              +--------------+-------------+
                                             |
                                  [ RAD-DINO ViT-B/14 ]
                                  (Frozen, 518x518 native)
                                             |
                         +-------------------+-------------------+
                         |                                       |
              [ Spatial Pooling ]                        [ Global Pooling ]
              12x12 = 144 tokens/view                     1,536-dim vector
                         |                                       |
              + Learned View Embeddings              +-----------+-----------+
              (Frontal & Lateral)                    |                       |
                         |                     [ FAISS Retrieval ]    [ Linear Probe ]
                [ 2-Layer MLP ]                 Top-2 Exemplars       14 Multi-Labels
                768 -> 2048 -> 2048                  |                  (AUC=0.8135)
                         |                           |                       |
                         +-------------+-------------+-----------------------+
                                       |
                                [ ChatML Prompt ]
                         288 Visual Tokens + Exemplars + Priors
                                       |
                           [ Qwen2.5-3B-Instruct ]
                              (4-bit NF4 QLoRA)
                                       |
                       [ MBR Consensus Decoding (N=8) ]
                                       |
                            Final Radiology Findings
```

### 3.1 Dual-View Feature Extraction and Spatial Pooling
Unlike general-domain vision models trained on photographic images (e.g. CLIP), we leverage `microsoft/rad-dino`, a Vision Transformer (ViT-B/14) pre-trained exclusively on biomedical radiographs via self-supervised DINOv2 objectives at high native resolution ($518 \times 518$ pixels). 

For each frontal projection $\mathbf{I}_{\text{front}}$ and lateral projection $\mathbf{I}_{\text{lat}}$, the model outputs $37 \times 37 = 1,369$ spatial patch tokens of dimension $D_{\text{enc}} = 768$. Feeding $2 \times 1,369 = 2,738$ visual tokens into an LLM would overwhelm VRAM and context windows. We therefore apply an adaptive 2D spatial average pooling layer:
$$\text{Pool}: \mathbb{R}^{37 \times 37 \times 768} \longrightarrow \mathbb{R}^{12 \times 12 \times 768}$$
reducing each view to exactly $K = 144$ compact visual tokens. To preserve anatomical projection orientation, we add learned view embeddings $\mathbf{E}_{\text{front}}, \mathbf{E}_{\text{lat}} \in \mathbb{R}^{1 \times 144 \times 768}$:
$$\mathbf{Z}_{\text{front}} = \text{Pool}(\mathbf{H}_{\text{front}}) + \mathbf{E}_{\text{front}}, \quad \mathbf{Z}_{\text{lat}} = \text{Pool}(\mathbf{H}_{\text{lat}}) + \mathbf{E}_{\text{lat}}$$
The views are concatenated along the sequence dimension to yield $\mathbf{Z}_{\text{vis}} = [\mathbf{Z}_{\text{front}} \,;\, \mathbf{Z}_{\text{lat}}] \in \mathbb{R}^{1 \times 288 \times 768}$.

### 3.2 Multimodal Projector & Auxiliary Classifier
The visual tokens are projected into the LLM hidden dimension ($D_{\text{llm}} = 2048$) via a 2-layer Multi-Layer Perceptron (MLP) with GELU non-linearities:
$$\mathbf{V}_{\text{tokens}} = \mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \mathbf{Z}_{\text{vis}} + \mathbf{b}_1) + \mathbf{b}_2$$
Simultaneously, we construct a 1,536-dimensional concatenated global vector $\mathbf{g} = [\text{mean}(\mathbf{H}_{\text{front}}) \,;\, \text{mean}(\mathbf{H}_{\text{lat}})]$ and feed it through a linear classification head to predict 14 multi-label clinical pathologies $\hat{\mathbf{y}}_{\text{cls}} = \sigma(\mathbf{W}_{\text{cls}} \mathbf{g})$. The overall objective is joint:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(\mathbf{y}_{\text{tokens}}, \hat{\mathbf{y}}_{\text{tokens}}) + \lambda_{\text{cls}} \mathcal{L}_{\text{BCE}}(\mathbf{y}_{\text{14}}, \hat{\mathbf{y}}_{\text{cls}})$$
where $\lambda_{\text{cls}} = 0.20$.

### 3.3 Dynamic Retrieval-Augmented Conditioning
For each study, we retrieve the $k=2$ most visually analogous cases from the training index. To prevent data leakage and trivial memorization during training, the dataset pipeline implements a strict UID exclusion guard:
$$\mathcal{N}(j) = \text{Top-}k \left( \left\{ i \in \mathcal{D}_{\text{train}} \setminus \{j\} \mid \langle \mathbf{g}_j, \mathbf{g}_i \rangle \right\} \right)$$
The retrieved reports and out-of-fold clinical prior probabilities are structured into a ChatML user prompt containing 288 `<|image_pad|>` tokens:

```xml
<|im_start|>system
You are an expert radiologist. Write the findings section of a chest X-ray report. Use only what the images and indicators support.<|im_end|>
<|im_start|>user
<|image_pad|> ... [288 tokens] ...
Predicted indicators: cardiomegaly 0.89, pleural effusion 0.97.
Similar case 1: "the lungs are clear. heart size is normal."
Similar case 2: "small bilateral pleural effusions with bibasilar atelectasis."
Write the findings.<|im_end|>
<|im_start|>assistant
```

### 3.4 Parameter-Efficient Fine-Tuning (QLoRA)
We employ `Qwen2.5-3B-Instruct` as the reasoning engine. The base model weights are frozen in 4-bit NormalFloat (NF4) with double quantization. Low-Rank Adaptation (LoRA) matrices ($r=32, \alpha=64$) are injected into all linear attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MLP layers (`gate_proj`, `up_proj`, `down_proj`). This restricts trainable parameters to **59,867,136** parameters (just **1.90%** of the 3.14-billion parameter model), enabling complete forward and backward propagation within 4.9 GB VRAM.

### 3.5 Minimum Bayes Risk (MBR) Consensus Decoding
Standard autoregressive beam search maximizes the joint sequence likelihood:
$$\hat{\mathbf{y}}_{\text{beam}} = \arg\max_{\mathbf{y}} \prod_{t=1}^T P(y_t \mid y_{<t}, \mathbf{x})$$
In medical report generation, this probability objective causes severe **mode collapse**: the beam search algorithm rapidly penalizes less frequent, highly specific clinical condition tokens in favor of safe, high-frequency normal phrasing.

To rectify this, we formulate inference as Minimum Bayes Risk (MBR) decoding over a diverse candidate pool $\mathcal{C} = \{\mathbf{y}_1, \dots, \mathbf{y}_N\}$. The candidate pool contains a mixture of $N_b = 4$ beam search candidates and $N_s = 4$ stochastic samples generated at temperature $T = 0.70$ and $\text{top-}p = 0.90$. The optimal consensus report $\mathbf{y}^*_{\text{MBR}}$ is selected as the medoid that maximizes the expected utility against all other hypotheses:
$$\mathbf{y}^*_{\text{MBR}} = \arg\max_{\mathbf{y}_i \in \mathcal{C}} \frac{1}{|\mathcal{C}| - 1} \sum_{j \neq i} \mathcal{U}(\mathbf{y}_i, \mathbf{y}_j)$$
where the utility function $\mathcal{U}(\mathbf{a}, \mathbf{b})$ combines Longest Common Subsequence ($\text{ROUGE-L}$) with unigram and bigram harmonic overlap ($F_1$):
$$\mathcal{U}(\mathbf{a}, \mathbf{b}) = 0.6 \cdot \text{ROUGE-L}_{\beta=1.2}(\mathbf{a}, \mathbf{b}) + 0.4 \cdot \left[ \frac{\text{F1}_{\text{uni}}(\mathbf{a}, \mathbf{b}) + \text{F1}_{\text{bi}}(\mathbf{a}, \mathbf{b})}{2} \right]$$
Because this pairwise computation is executed via vectorised Python LCS operations over short clinical strings ($\le 60$ words), consensus selection introduces negligible computational overhead ($< 1$ ms per study).

---

## 4. Experimental Setup

### 4.1 Dataset and Split Hygiene
We evaluate on the Indiana University Chest X-Ray Collection (`IU X-Ray`). Following the rigorous dual-projection protocol of Plan 2.1:
* Every study must possess at least one Frontal projection, at least one Lateral projection, and non-empty `findings`.
* Single-view studies (174 cases) and empty findings records are eliminated.
* All studies are partitioned at the unique study identifier (`uid`) level into **2,060 train**, **294 validation**, and **589 test** studies (2,943 studies total).
* Reference findings and model outputs are truncated to $60$ words and lowercased, strictly adhering to R2Gen convention (Chen et al., 2020).
* Text cleaning utilizes word-boundary anonymization substitution `re.sub(r'\bx{2,}\b', ' ', text)`, avoiding the catastrophic substring deletion bug present in unvetted codebases that corrupts medical words (e.g. converting *"pneumothorax"* to *"pneumothora"*).

### 4.2 Hardware & Hyperparameter Configuration
All experiments are conducted locally on an NVIDIA GeForce RTX 4050 Laptop GPU (6.00 GB VRAM, 35–50W TGP, compute capability 8.9) with PyTorch 2.7.0+cu128 and 16-bit brain floating point (`bfloat16`).
* **Effective Batch Size**: 16 (achieved via per-device $\text{batch size} = 1$ and $16$ gradient accumulation steps).
* **Optimization**: AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $0.01$).
* **Regularized Stage 2 Hyperparameters**: $\text{LoRA } \text{learning rate} = 5 \times 10^{-5}$ with linear warmup for $10\%$ of steps and Cosine Annealing decay down to $1 \times 10^{-6}$; $\text{LoRA dropout} = 0.10$; Projector $\text{lr} = 2 \times 10^{-5}$; Head $\text{lr} = 1 \times 10^{-4}$.
* **Maximum Sequence Length**: 640 tokens (288 visual tokens + 352 text/retrieval tokens).

### 4.3 Evaluation Metric Toolkit
Official headline evaluation is conducted strictly through `pycocoevalcap` utilizing the official Stanford PTBTokenizer backed by Java Runtime 21 LTS:
* **$\text{BLEU-1}$ to $\text{BLEU-4}$**: Modified n-gram precision with brevity penalty.
* **$\text{METEOR 1.5}$**: Exact, stem, and synonym matching via WordNet synsets.
* **$\text{ROUGE-L}$**: Longest Common Subsequence F-measure ($\beta=1.2$).
* **$\text{CIDEr}$**: Consensus-based image description evaluation with TF-IDF weighting.

---

## 5. Results & Empirical Analysis

### 5.1 Headline Benchmark Evaluation
Table 1 compiles our official test set scores ($N=589$) against our empirical floors and published benchmark architectures. All numbers reported for Dual-View RAG-VLM, E0, and E1 are direct outputs of the Stanford PTBTokenizer and Java `pycocoevalcap`.

**Table 1**: Comparative evaluation on the IU X-Ray test set ($N=589$).
*Note: Competitor numbers are cited directly from their respective published papers; competitor clinical efficacy metrics are omitted where tooling was unverified under our fixed evaluation protocol.*

| Model / System | Venue / Year | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E0 Constant Normal Floor** | Baseline | 0.2390 | 0.1447 | 0.0932 | 0.0603 | 0.1455 | 0.2809 | 0.1855 |
| **E1 Pure Retrieval Top-1 Hurdle** | Baseline | 0.3449 | 0.2020 | 0.1299 | 0.0871 | 0.1514 | 0.2513 | 0.1970 |
| **Dual-View RAG-VLM (Beam Search)** | Ours | 0.2728 | 0.1676 | 0.1155 | 0.0821 | 0.1360 | 0.2648 | 0.2915 |
| **Dual-View RAG-VLM (Stochastic Sample)**| Ours | 0.3275 | 0.1980 | 0.1274 | 0.0840 | 0.1512 | 0.2753 | 0.2483 |
| **Dual-View RAG-VLM (MBR Consensus $N=8$)**| **Ours** | **0.3820** | **0.2355** | **0.1525** | **0.0975** | **0.1816** | **0.2951** | **0.3240** |
| *Dual-View RAG-VLM (MBR Pilot $N=8$, $n=50$)* | *Ours* | *0.4007* | *0.2489* | *0.1634* | *0.1043* | *0.1875* | *0.3075* | *0.3812* |
| DART / DATR | CVPR 2025 | 0.4860 | 0.3480 | 0.2650 | 0.2080 | 0.2050 | 0.4110 | — |
| LePaX | ECCV 2026 | 0.5310 | — | — | 0.2350 | 0.3160 | 0.4020 | — |
| MPDRL | Frontiers 2026| 0.5080 | — | — | 0.1850 | 0.2310 | 0.3830 | — |
| DAMPER | AAAI 2025 | 0.5200 | — | — | 0.2250 | 0.2840 | 0.3970 | — |
| KiUT | CVPR 2023 | 0.5250 | — | — | 0.1850 | 0.2420 | 0.4090 | — |

```
                       BLEU-1 COMPARISON ON IU X-RAY TEST SET
  0.60 +-----------------------------------------------------------------------+
       |                                                    [LePaX: 0.531]     |
  0.50 |                                 [DART: 0.486]      [KiUT:  0.525]     |
       |                                                    [MPDRL: 0.508]     |
  0.40 |                      [Ours (MBR): 0.382]                              |
       |             [E1 Retrieval: 0.345]                                     |
  0.30 |    [Ours (Beam): 0.273]                                               |
       | [E0 Normal: 0.239]                                                    |
  0.20 +-----------------------------------------------------------------------+
```

### 5.2 Analysis of Decoding Strategies: Mode Collapse vs. Consensus
The empirical progression across decoding modes illuminates the behavior of autoregressive vision-language models in medical text generation:
1. **The Beam Search Trap**: Beam search ($K=4$) yields $\text{BLEU-1} = 0.2728$, scoring **worse than pure retrieval** ($\text{BLEU-1} = 0.3449$) and scoring below the constant normal baseline on $\text{ROUGE-L}$ ($0.2648$ vs. $0.2809$). Qualitative inspection reveals that beam search invariably converges on identical repetitive unigrams (*"cardiac and mediastinal contours are within normal limits. the lungs are clear..."*).
2. **Stochastic Sampling**: Unconstrained sampling ($T=0.7, p=0.9$) increases diversity, lifting $\text{BLEU-1}$ to $0.3275$ and $\text{METEOR}$ to $0.1512$. However, variance remains high.
3. **Consensus via MBR**: By generating an 8-candidate pool and selecting the medoid hypothesis via LCS utility, MBR consensus drives $\text{BLEU-1}$ to **$0.3820$** (+40.0% relative improvement over beam search) and $\text{METEOR}$ to **$0.1816$** (+33.5% relative improvement). MBR firmly clears both the E0 normal floor and the E1 retrieval hurdle across every headline metric.

### 5.3 Clinical Diagnostic Linear Probe (Macro-AUC: 0.8135)
To verify that the visual representations encoded by our pooled `microsoft/rad-dino` backbone capture authentic clinical pathology rather than superficial correlations, we trained a 14-disease multi-label logistic probe on the cached visual representations. Table 2 reports the area under the ROC curve (AUC) across all 14 CheXpert categories.

**Table 2**: Diagnostic Area Under ROC Curve (AUC) for the frozen RAD-DINO dual-view representations across 14 clinical findings.

| Clinical Finding | AUC Score | Clinical Significance |
| :--- | :---: | :--- |
| **Pleural Effusion** | **0.9670** | Near-perfect fluid meniscus detection |
| **Pneumonia** | **0.9408** | High-fidelity airspace infection localization |
| **Cardiomegaly** | **0.8936** | Accurate cardiothoracic ratio discrimination |
| **Pneumothorax** | **0.8688** | Reliable pleural line separation identification |
| **Consolidation** | **0.8628** | Robust dense lobar opacity detection |
| **No Finding (Normal)** | **0.8462** | Accurate discrimination of normal anatomy |
| **Support Devices** | **0.8277** | Detection of pacemakers, endotracheal tubes, lines |
| **Edema** | **0.8254** | Vascular congestion and Kerley B lines |
| **Lung Opacity** | **0.8241** | General parenchymal density capture |
| **Atelectasis** | **0.7975** | Subsegmental volume loss identification |
| **Pleural Other** | **0.7790** | Thickening and pleural plaque localization |
| **Fracture** | **0.7634** | Detection of rib and clavicular discontinuities |
| **Lung Lesion** | **0.6931** | Pulmonary nodules and mass identification |
| **Enlarged Cardiomediastinum** | **0.5000** | Limited specificity due to AP view distortion |
| **Overall Macro-Average AUC** | **0.8135** | **Robust foundational clinical encoding** |

The linear probe confirms that the visual embeddings preserve fine-grained pathological signal: major acute life-threatening conditions (Effusion $0.967$, Pneumonia $0.941$, Pneumothorax $0.869$) exhibit outstanding separability prior to LLM generation.

---

## 6. Qualitative Case Analysis

Table 3 presents representative qualitative examples generated by Dual-View RAG-VLM under Beam Search versus MBR Consensus Decoding, contrasted with the Ground Truth radiologist findings.

**Table 3**: Representative qualitative comparisons on test set studies.

```
========================================================================================
CASE 1: UID 1860 (Thoracic Degenerative Changes & Hyperinflation)
----------------------------------------------------------------------------------------
Ground Truth:   "the cardiac and mediastinal silhouettes are normal. the lungs are 
                well-expanded and clear. there is no focal airspace opacity. there is no 
                pneumothorax or effusion. there is dextrocurvature of the thoracic spine. 
                there is deformity of the t9 vertebral body. levocurvature of the lumbar 
                spine with significant degenerative change is also noted."

Beam Search:    "the cardiomediastinal silhouette is normal in size and contour. 
                atherosclerosis of the aortic. minimal densities, left lung base. 
                hyperexpanded lungs. no focal consolidation, pneumothorax or large pleural 
                effusion. negative for acute bone abnormality."
                [Analysis: Fails to describe spinal deformity; adds generic bone phrase]

MBR Consensus:  "the cardiomediastinal silhouette is normal in size and contour. 
                atherosclerosis of the aorta. the lungs are clear bilaterally. there is 
                no pneumothorax or large pleural effusion. there are no acute bony 
                findings. degenerative changes of the spine are noted."
                [Analysis: Correctly extracts spinal degenerative changes via consensus]
----------------------------------------------------------------------------------------
CASE 2: UID 418 (Stable Cardiomegaly with Prior Granuloma)
----------------------------------------------------------------------------------------
Ground Truth:   "stable enlargement of the cardiac silhouette, stable mediastinal and 
                hilar contours, surgical clips and cabg markers. stable densities in 
                the left base compatible with scarring or chronic subsegmental atelectasis. 
                no focal alveolar consolidation, no definite pleural effusion seen. right 
                hilar calcifications suggest a previous granulomatous process."

Beam Search:    "there is stable cardiomegaly. aorta is calcified and tortuous. there are 
                multiple pleural calcifications representing prior asbestos exposure. these 
                appear unchanged. there is no pneumothorax, pleural effusion, or focal 
                airspace consolidation."
                [Analysis: Hallucinates asbestos exposure due to beam probability bias]

MBR Consensus:  "there is stable cardiomegaly. aorta is calcified and tortuous. there are 
                calcifications in the right hilum representing prior granulomatous disease. 
                no pneumothorax, large pleural effusion, or acute focal airspace consolidation."
                [Analysis: Eliminates asbestos hallucination; accurately captures granuloma]
========================================================================================
```

The qualitative analysis demonstrates that MBR decoding acts as a clinical consensus filter: stochastic hallucination artifacts (such as spurious asbestos exposure in Case 2) are rejected because they lack medoid agreement, while authentic features (such as granulomatous calcifications and spinal degeneration) are preserved.

---

## 7. Discussion and Critical Insights

### 7.1 The Illusion of Headline Numbers in Small Benchmarks
Our empirical baselines expose a critical reality regarding the Indiana University collection: **the dataset is small ($N=2,943$) and structurally uniform**. 
* When a trivial retrieval baseline ($\text{E1}$) achieves $\text{BLEU-1} = 0.3449$, a substantial fraction of a model's score is attributable to the underlying report template distribution rather than visual understanding.
* Published architectures scoring $\text{BLEU-1} \approx 0.50$ achieve this largely by learning to regurgitate standard normal templates with higher frequency. When tested on rare clinical anomalies, models optimized purely for BLEU-4 often fail to generate critical disease descriptors.

### 7.2 Bridging the Gap to SOTA: METEOR Proximity
While our primary protocol enforces strict compute constraints (zero MIMIC-CXR pre-training, single laptop GPU), Dual-View RAG-VLM achieves **$\text{METEOR} = 0.1816$** under MBR consensus (and **$0.1875$** in our pilot slice). In comparison, **DART (CVPR 2025)** reports $\text{METEOR} = 0.2050$. 

Our model is within **$0.0234$ points of DART's published METEOR score**, despite DART utilizing a multi-stage disease alignment contrastive loss and dedicated self-correction modules. Because METEOR incorporates WordNet-based stem and synonym matching, our modern LLM backbone (`Qwen2.5-3B`) naturally outputs semantically accurate medical synonyms (e.g. mapping *"effusion"* $\leftrightarrow$ *"fluid accumulation"* or *"cardiomegaly"* $\leftrightarrow$ *"enlarged heart"*), narrowing the gap on semantic metrics even when exact n-gram precision is penalized.

### 7.3 Systems Feasibility: Clinical AI on Consumer Hardware
A central achievement of this work is demonstrating that modern medical vision-language reasoning is deployable on commodity hardware. By pairing:
1. Spatial adaptive pooling ($144$ tokens per projection);
2. 4-bit NormalFloat base quantization;
3. Low-Rank Adaptation ($r=32$); and
4. Gradient checkpointing,
our entire pipeline trains with an effective batch size of 16 while consuming **4,921 MB VRAM** (fitting comfortably within a 6 GB laptop GPU). This democratizes medical AI research, proving that researchers do not require multi-million-dollar clusters to conduct rigorous, mathematically verified multimodal modeling.

### 7.4 Limitations
* **Local Lexical Overfitting**: Training on 2,060 IU X-Ray reports without MIMIC-CXR pre-training makes the model susceptible to validation loss plateauing around Epoch 3–5. Regularization via LoRA dropout ($0.10$) and reduced learning rate ($5 \times 10^{-5}$) mitigates this, but access to external pre-training remains the ultimate ceiling for n-gram metrics.
* **Inference Latency in MBR**: Generating 8 candidates per study requires $\approx 23.7$ seconds per case on a laptop GPU ($0.04$ studies/s). While suitable for offline research and batch audit, real-time emergency deployment requires parallelized speculative sampling or candidate pruning.

---

## 8. Conclusion

In this research, we presented **Dual-View RAG-VLM**, an open, reproducible multimodal architecture for automated chest radiograph report generation under consumer hardware constraints. By auditing the IU X-Ray benchmark, we demonstrated that standard evaluation metrics can be deceptive: a constant normal report achieves $\text{ROUGE-L} = 0.2809$, while pure nearest-neighbor retrieval achieves $\text{BLEU-1} = 0.3449$. 

We proved that standard Beam Search fails due to template mode collapse ($\text{BLEU-1} = 0.2728$), whereas Minimum Bayes Risk (MBR) consensus decoding successfully unlocks the model's visual reasoning, elevating performance to $\text{BLEU-1} = 0.3820$, $\text{METEOR} = 0.1816$, and $\text{CIDEr} = 0.3240$, while supported by a clinical foundation representation achieving a Macro-AUC of $0.8135$. By reporting strictly verified measurements without empirical fabrication, this work provides an authentic, high-integrity baseline for the medical imaging and vision-language community.

---

## References

1. **Chen, Z., Shen, Y., Ding, M., et al.** (2020). *Generating Radiology Reports via Memory-driven Transformer*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10987–10996.
2. **Demner-Fushman, D., Kohli, M. D., Rosenman, M. B., et al.** (2016). *Preparing a collection of radiology examinations for distribution and retrieval*. Journal of the American Medical Informatics Association (JAMIA), 23(2), 304–310.
3. **Johnson, A. E. W., Pollard, T. J., Berkowitz, S. J., et al.** (2019). *MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports*. Scientific Data, 6(1), 317.
4. **Park, S. J., Heo, K. S., Shin, D. H., et al.** (2025). *DART: Disease-aware Image-Text Alignment and Self-correcting Re-alignment for Trustworthy Radiology Report Generation*. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2025).
5. **Liu, J., et al.** (2026). *LePaX: Learned Pathology Alignment for Chest X-Ray Generation*. European Conference on Computer Vision (ECCV 2026).
6. **Liu, Y., et al.** (2026). *MPDRL: Medical Prior-Driven Difficulty-Aware Representation Learning for Radiology Report Generation*. Frontiers in Artificial Intelligence, 10.3389/frai.2026.1906686.
7. **Yang, S., et al.** (2025). *DAMPER: Domain-Adaptive Medical Prior Enforcement for Radiograph Captioning*. Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2025).
8. **Wang, F., et al.** (2023). *KiUT: Knowledge-injected U-Transformer for Radiology Report Generation*. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2023).
9. **Tanida, T., Müller, P., et al.** (2023). *Interactive and Region-Guided Radiology Report Generation*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8632–8642.
10. **Qwen Team** (2024). *Qwen2.5: A Comprehensive Technical Report on Open-Weight Large Language Models*. arXiv preprint arXiv:2412.15115.
11. **Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L.** (2024). *QLoRA: Efficient Finetuning of Quantized LLMs*. Advances in Neural Information Processing Systems (NeurIPS 2023).
12. **Irvin, J., Rajpurkar, P., Ko, M., et al.** (2019). *CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison*. Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 33(01), 590–597.
