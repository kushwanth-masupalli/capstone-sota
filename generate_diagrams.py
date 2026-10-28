"""
generate_diagrams.py - Generates ultra-crisp, publication-quality architecture
and benchmark diagrams for the Dual-View RAG-VLM Capstone presentation.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def create_system_architecture_diagram():
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Color Palette (Clinical Dark Theme)
    c_card = '#161b22'
    c_border = '#30363d'
    c_blue = '#58a6ff'
    c_cyan = '#39c5cf'
    c_green = '#3fb950'
    c_purple = '#bc8cff'
    c_orange = '#d29922'
    c_red = '#f85149'
    c_text = '#f0f6fc'
    c_sub = '#8b949e'

    # Title Banner
    ax.text(8.0, 9.6, "DUAL-VIEW RAG-VLM: SYSTEM ARCHITECTURE", 
            fontsize=20, weight='bold', color=c_text, ha='center', va='center')
    ax.text(8.0, 9.2, "Automated Chest Radiograph Report Generation | Consumer GPU Envelope (6.0 GB VRAM)", 
            fontsize=11, color=c_blue, ha='center', va='center')

    # -------------------------------------------------------------
    # 1. INPUT & VISION ENCODING STAGE (Left Column: x=0.6 to 4.2)
    # -------------------------------------------------------------
    rect1 = patches.FancyBboxPatch((0.5, 0.8), 3.8, 8.0, boxstyle="round,pad=0.2,rounding_size=0.3",
                                  facecolor=c_card, edgecolor=c_blue, linewidth=1.8)
    ax.add_patch(rect1)
    ax.text(2.4, 8.4, "1. DUAL-VIEW VISION BACKBONE", fontsize=12, weight='bold', color=c_blue, ha='center')
    ax.text(2.4, 8.1, "microsoft/rad-dino (ViT-B/14, 518x518 native)", fontsize=8.5, color=c_sub, ha='center')

    # Frontal Box
    bf = patches.FancyBboxPatch((0.8, 6.2), 3.2, 1.6, boxstyle="round,pad=0.1,rounding_size=0.2",
                                facecolor='#1f242c', edgecolor='#388bfd', linewidth=1.2)
    ax.add_patch(bf)
    ax.text(2.4, 7.3, "Frontal Radiograph (PA/AP)", fontsize=10, weight='bold', color=c_text, ha='center')
    ax.text(2.4, 6.9, "Resolution: 518 x 518 x 3", fontsize=8.5, color=c_sub, ha='center')
    ax.text(2.4, 6.5, "Patch Grid: 37 x 37 = 1,369 tokens", fontsize=8, color=c_cyan, ha='center')

    # Lateral Box
    bl = patches.FancyBboxPatch((0.8, 4.3), 3.2, 1.6, boxstyle="round,pad=0.1,rounding_size=0.2",
                                facecolor='#1f242c', edgecolor='#388bfd', linewidth=1.2)
    ax.add_patch(bl)
    ax.text(2.4, 5.4, "Lateral Radiograph (Lat)", fontsize=10, weight='bold', color=c_text, ha='center')
    ax.text(2.4, 5.0, "Resolution: 518 x 518 x 3", fontsize=8.5, color=c_sub, ha='center')
    ax.text(2.4, 4.6, "Patch Grid: 37 x 37 = 1,369 tokens", fontsize=8, color=c_cyan, ha='center')

    # Spatial Pooling Block
    b_pool = patches.FancyBboxPatch((0.8, 1.2), 3.2, 2.7, boxstyle="round,pad=0.1,rounding_size=0.2",
                                   facecolor='#1f242c', edgecolor=c_border, linewidth=1.2)
    ax.add_patch(b_pool)
    ax.text(2.4, 3.5, "Adaptive Spatial Pooling", fontsize=10, weight='bold', color=c_text, ha='center')
    ax.text(2.4, 3.1, "AdaptiveAvgPool2d((12, 12))", fontsize=8.5, color=c_cyan, ha='center')
    ax.text(2.4, 2.6, "• 144 Frontal Tokens (768-dim)\n• 144 Lateral Tokens (768-dim)\n+ Learned View Embeddings (768-dim)\n= 288 Total Dual Tokens", 
            fontsize=8, color=c_sub, ha='center', va='center')
    ax.text(2.4, 1.5, "Cached to disk: 100x Training Speedup", fontsize=7.5, weight='bold', color=c_green, ha='center')

    # Arrows from Inputs to Pooling
    ax.annotate("", xy=(2.4, 4.0), xytext=(2.4, 4.3), arrowprops=dict(arrowstyle="->", color=c_blue, lw=1.5))

    # -------------------------------------------------------------
    # 2. RAG & CLINICAL PROBING STAGE (Middle-Top: x=5.0 to 10.2, y=5.0 to 8.8)
    # -------------------------------------------------------------
    rect2 = patches.FancyBboxPatch((4.8, 5.0), 5.6, 3.8, boxstyle="round,pad=0.2,rounding_size=0.3",
                                  facecolor=c_card, edgecolor=c_green, linewidth=1.8)
    ax.add_patch(rect2)
    ax.text(7.6, 8.4, "2. LEAKAGE-FREE RAG & CLINICAL PROBES", fontsize=12, weight='bold', color=c_green, ha='center')
    ax.text(7.6, 8.1, "Dual Global Vector: concat(mean(F), mean(L)) = 1,536-dim (L2-norm)", fontsize=8.5, color=c_sub, ha='center')

    # FAISS Sub-box
    b_faiss = patches.FancyBboxPatch((5.1, 5.3), 2.4, 2.5, boxstyle="round,pad=0.1,rounding_size=0.2",
                                    facecolor='#1b2b22', edgecolor=c_green, linewidth=1.2)
    ax.add_patch(b_faiss)
    ax.text(6.3, 7.4, "FAISS IndexFlatIP", fontsize=9.5, weight='bold', color=c_text, ha='center')
    ax.text(6.3, 7.0, "Cosine Similarity", fontsize=8, color=c_sub, ha='center')
    ax.text(6.3, 6.4, "Train Index Only\n(N=2,060 studies)\nOwn UID Excluded\nTop-2 Similar Cases", 
            fontsize=7.8, color=c_sub, ha='center')
    ax.text(6.3, 5.5, "Leakage-Free Guard", fontsize=7.5, weight='bold', color=c_green, ha='center')

    # Linear Probe Sub-box
    b_probe = patches.FancyBboxPatch((7.7, 5.3), 2.4, 2.5, boxstyle="round,pad=0.1,rounding_size=0.2",
                                    facecolor='#1b2b22', edgecolor=c_green, linewidth=1.2)
    ax.add_patch(b_probe)
    ax.text(8.9, 7.4, "14-Disease Probe", fontsize=9.5, weight='bold', color=c_text, ha='center')
    ax.text(8.9, 7.0, "Macro-AUC: 0.8135", fontsize=8.5, weight='bold', color=c_cyan, ha='center')
    ax.text(8.9, 6.4, "Effusion: 0.967\nPneumonia: 0.941\nCardiomegaly: 0.894\n5-Fold OOF Priors", 
            fontsize=7.8, color=c_sub, ha='center')
    ax.text(8.9, 5.5, "Aux BCE Loss (wt=0.2)", fontsize=7.5, weight='bold', color=c_orange, ha='center')

    # -------------------------------------------------------------
    # 3. MULTIMODAL PROJECTION & LLM (Middle-Bottom: x=5.0 to 10.2, y=0.8 to 4.6)
    # -------------------------------------------------------------
    rect3 = patches.FancyBboxPatch((4.8, 0.8), 5.6, 3.8, boxstyle="round,pad=0.2,rounding_size=0.3",
                                  facecolor=c_card, edgecolor=c_purple, linewidth=1.8)
    ax.add_patch(rect3)
    ax.text(7.6, 4.2, "3. MULTIMODAL FUSION & QLoRA LLM", fontsize=12, weight='bold', color=c_purple, ha='center')
    
    # 2-Layer MLP
    b_mlp = patches.FancyBboxPatch((5.1, 2.4), 5.0, 1.4, boxstyle="round,pad=0.1,rounding_size=0.2",
                                  facecolor='#261f33', edgecolor=c_purple, linewidth=1.2)
    ax.add_patch(b_mlp)
    ax.text(7.6, 3.4, "2-Layer MLP Projector (fp32)", fontsize=9.5, weight='bold', color=c_text, ha='center')
    ax.text(7.6, 2.9, "Linear(768 -> 2048)  -->  GELU  -->  Linear(2048 -> 2048)", fontsize=8, color=c_cyan, ha='center')
    ax.text(7.6, 2.5, "Replaces 288 <|image_pad|> tokens in ChatML prompt with projected visual embeddings", 
            fontsize=7.5, color=c_sub, ha='center')

    # Qwen2.5-3B Box
    b_qwen = patches.FancyBboxPatch((5.1, 1.0), 5.0, 1.2, boxstyle="round,pad=0.1,rounding_size=0.2",
                                   facecolor='#261f33', edgecolor=c_purple, linewidth=1.2)
    ax.add_patch(b_qwen)
    ax.text(7.6, 1.8, "Qwen2.5-3B-Instruct (4-Bit NF4 Quantized)", fontsize=9.5, weight='bold', color=c_text, ha='center')
    ax.text(7.6, 1.3, "QLoRA: r=32, alpha=64, dropout 0.1 | 59.8M Trainable Parameters (1.9%) | 4.9 GB VRAM Peak", 
            fontsize=7.8, color=c_sub, ha='center')

    # Arrow from pooling to MLP
    ax.annotate("", xy=(4.8, 2.7), xytext=(3.7, 2.7), arrowprops=dict(arrowstyle="->", color=c_blue, lw=2.0))
    # Arrow from pooling to RAG stage
    ax.annotate("", xy=(4.8, 6.8), xytext=(3.7, 3.2), arrowprops=dict(arrowstyle="->", color=c_cyan, lw=2.0))
    # Arrow from RAG stage to MLP/Prompt
    ax.annotate("", xy=(7.6, 3.9), xytext=(7.6, 4.9), arrowprops=dict(arrowstyle="->", color=c_green, lw=2.0))

    # -------------------------------------------------------------
    # 4. DECODING & CLINICAL CROSS-CHECKING (Right Column: x=10.8 to 15.6)
    # -------------------------------------------------------------
    rect4 = patches.FancyBboxPatch((10.8, 0.8), 4.8, 8.0, boxstyle="round,pad=0.2,rounding_size=0.3",
                                  facecolor=c_card, edgecolor=c_orange, linewidth=1.8)
    ax.add_patch(rect4)
    ax.text(13.2, 8.45, "4. CONSENSUS & CROSS-CHECKING", fontsize=12, weight='bold', color=c_orange, ha='center')
    ax.text(13.2, 8.15, "MBR Consensus Medoid + Factuality Audit", fontsize=8.5, color=c_sub, ha='center')

    # Candidate Pool & MBR Selection Box (Unchecked Draft)
    b_cand = patches.FancyBboxPatch((11.1, 5.95), 4.2, 1.9, boxstyle="round,pad=0.1,rounding_size=0.2",
                                   facecolor='#2a2216', edgecolor=c_orange, linewidth=1.2)
    ax.add_patch(b_cand)
    ax.text(13.2, 7.5, "MBR Consensus (Unchecked Report)", fontsize=9.5, weight='bold', color=c_text, ha='center')
    ax.text(13.2, 6.95, "• 16 Diverse Candidates (4 Beams + 12 Samples)\n• Medoid argmax sum U(c_i, c_j) (ROUGE+F1)", 
            fontsize=8, color=c_sub, ha='center')
    ax.text(13.2, 6.25, "BLEU-1: 0.382 - 0.405 (+40% over Beam)", fontsize=8, weight='bold', color=c_cyan, ha='center')

    # Clinical Cross-Checking Engine Box
    b_cross = patches.FancyBboxPatch((11.1, 3.4), 4.2, 2.25, boxstyle="round,pad=0.1,rounding_size=0.2",
                                    facecolor='#1b2b22', edgecolor=c_green, linewidth=1.4)
    ax.add_patch(b_cross)
    ax.text(13.2, 5.3, "Clinical Cross-Checking Engine", fontsize=9.5, weight='bold', color=c_green, ha='center')
    ax.text(13.2, 4.5, "Cross-Checks 14 Classifier Priors vs Text\n• Negation-Aware Parsing (Absent vs Present)\n• Omission (Risk >= 40%) & Hallucination Guard\n• Computes Clinical Concordance Score", 
            fontsize=7.8, color=c_text, ha='center', va='center')
    ax.text(13.2, 3.65, "Factuality & Safety Verification Layer", fontsize=7.8, weight='bold', color=c_cyan, ha='center')

    # Final Verified Checked Report Output Box
    b_out = patches.FancyBboxPatch((11.1, 1.0), 4.2, 2.1, boxstyle="round,pad=0.1,rounding_size=0.2",
                                  facecolor='#16251b', edgecolor='#3fb950', linewidth=1.6)
    ax.add_patch(b_out)
    ax.text(13.2, 2.75, "Final Verified Checked Report", fontsize=10, weight='bold', color=c_green, ha='center')
    ax.text(13.2, 2.1, "\"the heart size and pulmonary vascularity\nare normal. lungs are clear without focal\nconsolidation, pneumothorax, or effusion.\"", 
            fontsize=7.5, style='italic', color=c_text, ha='center')
    ax.text(13.2, 1.3, "[VERIFIED CONCORDANT: 100% CLINICAL ALIGNMENT]", fontsize=7.5, weight='bold', color='#3fb950', ha='center')

    # Arrows in Column 4
    ax.annotate("", xy=(13.2, 5.65), xytext=(13.2, 5.95), arrowprops=dict(arrowstyle="->", color=c_orange, lw=1.5))
    ax.annotate("", xy=(13.2, 3.1), xytext=(13.2, 3.4), arrowprops=dict(arrowstyle="->", color=c_green, lw=1.8))
    # Arrow from LLM to MBR Candidate Pool
    ax.annotate("", xy=(11.1, 6.8), xytext=(10.2, 1.6), arrowprops=dict(arrowstyle="->", color=c_orange, lw=2.0))
    # Arrow from 14-Disease Probe directly to Cross-Checking Engine
    ax.annotate("", xy=(11.1, 4.5), xytext=(10.1, 6.4), 
                arrowprops=dict(arrowstyle="->", color=c_cyan, lw=2.0, connectionstyle="arc3,rad=-0.15"))
    ax.text(10.6, 5.8, "Probabilities", fontsize=7.5, color=c_cyan, weight='bold', ha='center')

    # Bottom Footer
    ax.text(8.0, 0.3, "Dual-View RAG-VLM Architecture | IU X-Ray Locked Split: 2,060 Train / 294 Val / 589 Test", 
            fontsize=9, color=c_sub, ha='center')

    out_path = "architecture_pipeline.png"
    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=300)
    plt.close()
    print(f"Saved {out_path}")

def create_benchmark_and_mbr_diagram():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), dpi=300)
    fig.patch.set_facecolor('#0d1117')
    for ax in (ax1, ax2):
        ax.set_facecolor('#0d1117')

    # Color Palette
    c_card = '#161b22'
    c_blue = '#58a6ff'
    c_cyan = '#39c5cf'
    c_green = '#3fb950'
    c_purple = '#bc8cff'
    c_orange = '#d29922'
    c_red = '#f85149'
    c_text = '#f0f6fc'
    c_sub = '#8b949e'

    # -------------------------------------------------------------
    # LEFT PANEL: BENCHMARK HIERARCHY & THE EMPIRICAL BASELINE AUDIT
    # -------------------------------------------------------------
    ax1.set_title("THE EMPIRICAL BASELINE AUDIT ON IU X-RAY", fontsize=13, weight='bold', color=c_text, pad=15)
    
    models = [
        "E0 Constant Normal Floor",
        "Dual-View Beam Search (Ours)",
        "E1 Pure Retrieval Hurdle",
        "Dual-View Stochastic (Ours)",
        "Dual-View MBR N=8 (Ours)",
        "Dual-View MBR N=16 Pilot (Ours)",
        "DART / DATR (CVPR 2025)",
        "LePaX (ECCV 2026)"
    ]
    bleu1_scores = [0.2390, 0.2728, 0.3449, 0.3275, 0.3820, 0.4053, 0.4860, 0.5310]
    colors = [c_red, '#d29922', '#e3b341', c_blue, c_green, '#2ea043', c_purple, '#a371f7']

    y_pos = range(len(models))
    bars = ax1.barh(y_pos, bleu1_scores, color=colors, height=0.6, edgecolor=c_card, linewidth=1.5)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models, fontsize=9.5, color=c_text)
    ax1.set_xlabel("BLEU-1 Score (Stanford PTBTokenizer, 60 words)", fontsize=9.5, color=c_sub)
    ax1.set_xlim(0, 0.6)
    ax1.grid(axis='x', linestyle='--', alpha=0.25, color=c_sub)

    # Annotations on bars
    for bar, score in zip(bars, bleu1_scores):
        ax1.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height()/2, 
                 f"{score:.4f}", va='center', fontsize=8.5, weight='bold', color=c_text)

    # Highlight lines for E0 and E1
    ax1.axvline(x=0.2390, color=c_red, linestyle=':', alpha=0.7, label='E0 Normal Floor (0.2390)')
    ax1.axvline(x=0.3449, color='#e3b341', linestyle='--', alpha=0.8, label='E1 Retrieval Hurdle (0.3449)')
    ax1.legend(loc='lower right', facecolor=c_card, edgecolor='#30363d', labelcolor=c_text, fontsize=8)

    # Key callout box on Left Panel
    ax1.text(0.02, 0.05, 
             "• Pure nearest-neighbor copy clears 0.3449 BLEU-1 without any LLM!\n"
             "• Standard Beam Search collapses (0.2728), underperforming retrieval.\n"
             "• MBR Consensus decisively clears the retrieval hurdle (0.3820 - 0.4053).\n"
             "• Competitors (DART/LePaX) pre-trained on 400k MIMIC-CXR multi-GPU clusters.",
             transform=ax1.transAxes, fontsize=8, color=c_sub,
             bbox=dict(boxstyle="round,pad=0.4", facecolor=c_card, edgecolor='#30363d'))

    # -------------------------------------------------------------
    # RIGHT PANEL: BEAM SEARCH MODE COLLAPSE VS MBR CONSENSUS
    # -------------------------------------------------------------
    ax2.set_title("BEAM SEARCH COLLAPSE VS. MBR CONSENSUS", fontsize=13, weight='bold', color=c_text, pad=15)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')

    # Beam Search Box
    box_beam = patches.FancyBboxPatch((0.5, 5.3), 9.0, 4.0, boxstyle="round,pad=0.2,rounding_size=0.2",
                                     facecolor='#261517', edgecolor=c_red, linewidth=1.5)
    ax2.add_patch(box_beam)
    ax2.text(5.0, 8.8, "FAILURE MODE: Standard Beam Search (num_beams=4)", fontsize=10, weight='bold', color=c_red, ha='center')
    ax2.text(5.0, 8.3, "Greedy joint probability optimization heavily penalizes rare disease tokens.", fontsize=8, color=c_sub, ha='center')
    ax2.text(5.0, 7.5, "Generates generic safe template: 'heart and lungs normal...'\nDrops subtle focal opacities, pleural thickening, or calcified granulomas.", fontsize=8, color=c_text, ha='center')
    ax2.text(5.0, 6.4, "Measured Score: BLEU-1 = 0.2728 | METEOR = 0.1360 | CIDEr = 0.2915", fontsize=8.5, weight='bold', color=c_red, ha='center')
    ax2.text(5.0, 5.7, "CRITICAL FLAW: Scores BELOW trivial nearest-neighbor retrieval (0.3449)!", fontsize=8, style='italic', color='#ff7b72', ha='center')

    # MBR Consensus Box
    box_mbr = patches.FancyBboxPatch((0.5, 0.5), 9.0, 4.4, boxstyle="round,pad=0.2,rounding_size=0.2",
                                    facecolor='#14291f', edgecolor=c_green, linewidth=1.5)
    ax2.add_patch(box_mbr)
    ax2.text(5.0, 4.4, "BREAKTHROUGH: Minimum Bayes Risk (MBR) Consensus", fontsize=10, weight='bold', color=c_green, ha='center')
    ax2.text(5.0, 3.9, "1. Samples 16 diverse candidates (4 beams + 12 stochastic at T=0.7).", fontsize=8, color=c_sub, ha='center')
    ax2.text(5.0, 3.4, "2. Pairwise clinical agreement utility: U = 0.6*ROUGE-L + 0.4*ngram_F1.", fontsize=8, color=c_sub, ha='center')
    ax2.text(5.0, 2.9, "3. Selects the medoid hypothesis maximizing expected clinical utility.", fontsize=8, color=c_text, ha='center')
    ax2.text(5.0, 2.1, "Measured Score: BLEU-1 = 0.3820 - 0.4053 | METEOR = 0.1816 - 0.1926", fontsize=8.5, weight='bold', color=c_green, ha='center')
    ax2.text(5.0, 1.4, "+40.0% Relative BLEU-1 Jump over Beam Search Without Modifying Weights!", fontsize=8.5, weight='bold', color=c_cyan, ha='center')

    out_path2 = "decoding_and_baselines.png"
    plt.tight_layout()
    plt.savefig(out_path2, facecolor=fig.get_facecolor(), edgecolor='none', dpi=300)
    plt.close()
    print(f"Saved {out_path2}")

if __name__ == "__main__":
    create_system_architecture_diagram()
    create_benchmark_and_mbr_diagram()
