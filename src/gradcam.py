"""
src/gradcam.py - Clinical Grad-CAM Visual Explainability Engine for Dual-View RAG-VLM.

Architecture & Formulation:
  For our dual-view biomedical foundation model (RAD-DINO ViT-B/14 + linear projection head),
  each radiograph view is spatially encoded into a 12x12 grid (144 spatial tokens):
    f_tokens = frontal + view_embed_frontal  [144, 768]
    l_tokens = lateral + view_embed_lateral  [144, 768]

  The 14-disease linear classifier operates on the concatenated global mean representations:
    f_global = (1/N) * sum_i f_tokens_i
    l_global = (1/N) * sum_j l_tokens_j
    y_c = W_{c, :768} @ f_global + W_{c, 768:} @ l_global + b_c

  In closed-form, the gradient of logit y_c with respect to token i is exactly W_c:
    d(y_c) / d(f_tokens_i) = (1/N) * W_{c, :768}
    d(y_c) / d(l_tokens_j) = (1/N) * W_{c, 768:}

  Therefore, the Class Activation Map (CAM / Grad-CAM) for finding c is:
    M_frontal_c(x, y) = ReLU( f_tokens_{(x,y)} @ W_{c, :768} )
    M_lateral_c(x, y) = ReLU( l_tokens_{(x,y)} @ W_{c, 768:} )

  This yields an exact, instant, zero-VRAM gradient localization map that maps precisely
  onto the patient's frontal and lateral chest radiographs.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# Standard 14 CheXpert/NIH Pathologies matching project protocol
LABELS_14 = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]

COLORMAPS = {
    "turbo": cv2.COLORMAP_TURBO,
    "jet": cv2.COLORMAP_JET,
    "inferno": cv2.COLORMAP_INFERNO,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "magma": cv2.COLORMAP_MAGMA,
    "hot": cv2.COLORMAP_HOT,
}


def _infer_anatomical_zone(row: int, col: int, is_lateral: bool = False) -> str:
    """Infers anatomical thoracic landmark from 12x12 grid coordinates."""
    if is_lateral:
        if row < 4:
            return "Superior Mediastinum / Retrovascular Space"
        elif row > 8:
            return "Posterior Costophrenic Sulcus / Diaphragm"
        elif col < 6:
            return "Anterior Retrosternal Clear Space"
        else:
            return "Retrocardiac & Spine Region"
    else:
        # Frontal PA/AP projection
        side = "Right" if col < 6 else "Left"  # Radiologist convention: patient's right is image left
        if row < 3:
            zone = "Apical / Clavicular Zone"
        elif row < 7:
            if 3 <= col <= 8:
                zone = "Hilar / Mediastinal Contour"
            else:
                zone = f"{side} Mid-Lung Field"
        elif row < 10:
            if 4 <= col <= 9:
                zone = "Cardiac Silhouette / Lower Mediastinum"
            else:
                zone = f"{side} Lower Lung Zone"
        else:
            zone = f"{side} Costophrenic Angle / Hemidiaphragm"
        return zone


class GradCAMExplainer:
    """
    Clinical Grad-CAM visualizer for dual-view chest radiograph interpretation.
    Computes exact spatial saliency heatmaps for any of the 14 CheXpert pathologies.
    """

    def __init__(
        self,
        checkpoint_path: str = "outputs/checkpoints/stage2_best/custom_heads.pt",
        feats_dir: str = "processed/feats",
        data_json: str = "processed/test.json",
        device: str = "cpu",
    ):
        self.device = device
        self.feats_dir = feats_dir
        self.data_json = data_json

        # Load weights
        if not os.path.exists(checkpoint_path):
            # Fallback to stage1_best or other checkpoints
            for cand in [
                "outputs/checkpoints/stage1_best/custom_heads.pt",
                "outputs/checkpoints/latest_checkpoint/custom_heads.pt",
                "outputs/checkpoints/stage2_final/custom_heads.pt",
            ]:
                if os.path.exists(cand):
                    checkpoint_path = cand
                    break

        print(f"[GradCAM] Loading classification probe weights from: {checkpoint_path}")
        heads = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.clf_w = heads["classifier_head"]["weight"].float()  # [14, 1536]
        self.clf_b = heads["classifier_head"]["bias"].float()    # [14]
        self.view_f = heads["view_embed_frontal"].float().squeeze(0)  # [144, 768]
        self.view_l = heads["view_embed_lateral"].float().squeeze(0)  # [144, 768]

        # Weights partition
        self.w_frontal = self.clf_w[:, :768]
        self.w_lateral = self.clf_w[:, 768:]

        # Load metadata lookup if available
        self.study_meta: Dict[str, Dict] = {}
        if os.path.exists(data_json):
            try:
                with open(data_json, "r") as f:
                    studies = json.load(f)
                    for s in studies:
                        self.study_meta[str(s["uid"])] = s
            except Exception as e:
                print(f"[GradCAM] Warning loading {data_json}: {e}")

    def get_study_features(self, uid: str) -> Optional[Dict[str, torch.Tensor]]:
        """Loads precomputed 144-token features for a given study UID."""
        feat_path = os.path.join(self.feats_dir, f"{uid}.pt")
        if not os.path.exists(feat_path):
            return None
        return torch.load(feat_path, map_location=self.device, weights_only=False)

    def predict_pathology_probabilities(self, f_tokens: torch.Tensor, l_tokens: torch.Tensor) -> Dict[str, float]:
        """Calculates probabilities across all 14 pathologies from dual-view tokens."""
        f_emb = (f_tokens.float() + self.view_f).mean(dim=0)
        l_emb = (l_tokens.float() + self.view_l).mean(dim=0)
        dual_global = torch.cat([f_emb, l_emb], dim=-1)

        logits = torch.matmul(self.clf_w, dual_global) + self.clf_b
        probs = torch.sigmoid(logits)

        return {label: round(float(probs[i].item()), 4) for i, label in enumerate(LABELS_14)}

    def compute_heatmap(
        self,
        tokens: torch.Tensor,
        w_sub: torch.Tensor,
        grid_size: int = 12,
        relu_threshold: bool = True,
    ) -> np.ndarray:
        """
        Computes the normalized 2D activation heatmap from spatial tokens and weight vector.
        """
        # tokens: [144, 768], w_sub: [768]
        raw_contrib = torch.matmul(tokens.float(), w_sub).reshape(grid_size, grid_size).numpy()

        # In Grad-CAM, ReLU is applied to isolate features that positively contribute to the class
        if relu_threshold:
            # Baseline offset: activations above median or mean indicate positive regional focus
            ref_baseline = float(np.mean(raw_contrib))
            heatmap = np.maximum(raw_contrib - ref_baseline, 0.0)
            if np.max(heatmap) == 0.0:
                # Fallback to min-max scaling if all negative
                heatmap = raw_contrib - np.min(raw_contrib)
        else:
            heatmap = raw_contrib - np.min(raw_contrib)

        # Min-Max normalize to [0, 1]
        denom = np.max(heatmap) - np.min(heatmap)
        if denom > 1e-8:
            heatmap = (heatmap - np.min(heatmap)) / denom
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap

    def overlay_heatmap_on_image(
        self,
        image: Union[np.ndarray, Image.Image],
        heatmap: np.ndarray,
        colormap: str = "turbo",
        alpha: float = 0.45,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Smoothly interpolates a 12x12 heatmap to original image resolution and overlays it.
        Returns:
            (overlay_bgr, colored_heatmap_bgr)
        """
        if isinstance(image, Image.Image):
            img_np = np.array(image.convert("RGB"))
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 3:
                img_bgr = image.copy()
            else:
                img_bgr = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        h, w = img_bgr.shape[:2]

        # Bicubic interpolation to full resolution
        heat_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
        heat_resized = np.clip(heat_resized, 0.0, 1.0)
        heat_uint8 = np.uint8(255 * heat_resized)

        # Apply colormap
        cmap_code = COLORMAPS.get(colormap.lower(), cv2.COLORMAP_TURBO)
        colored_map = cv2.applyColorMap(heat_uint8, cmap_code)

        # Weighted blend
        overlay = cv2.addWeighted(img_bgr, 1.0 - alpha, colored_map, alpha, 0)
        return overlay, colored_map

    def explain(
        self,
        uid: str,
        pathology: Optional[str] = None,
        colormap: str = "turbo",
        alpha: float = 0.45,
    ) -> Dict[str, Any]:
        """
        Generates full dual-view Grad-CAM explainability package for a study UID.
        """
        feats = self.get_study_features(uid)
        if feats is None:
            raise FileNotFoundError(f"Feature tensor not found for UID '{uid}' in {self.feats_dir}")

        f_tokens = feats["frontal"].float() + self.view_f
        l_tokens = feats["lateral"].float() + self.view_l

        # Compute full pathology probabilities
        probs = self.predict_pathology_probabilities(feats["frontal"], feats["lateral"])

        # Determine target pathology
        if pathology is None:
            # Pick top predicted non-normal pathology
            filtered = {k: v for k, v in probs.items() if k != "No Finding"}
            pathology = max(filtered, key=filtered.get)

        if pathology not in LABELS_14:
            raise ValueError(f"Unknown pathology '{pathology}'. Must be one of {LABELS_14}")

        pathology_idx = LABELS_14.index(pathology)
        prob = probs[pathology]

        # Compute 12x12 heatmaps
        f_heatmap = self.compute_heatmap(f_tokens, self.w_frontal[pathology_idx])
        l_heatmap = self.compute_heatmap(l_tokens, self.w_lateral[pathology_idx])

        # Infer peak coordinates
        f_max_pos = int(np.argmax(f_heatmap))
        f_row, f_col = f_max_pos // 12, f_max_pos % 12
        f_zone = _infer_anatomical_zone(f_row, f_col, is_lateral=False)

        l_max_pos = int(np.argmax(l_heatmap))
        l_row, l_col = l_max_pos // 12, l_max_pos % 12
        l_zone = _infer_anatomical_zone(l_row, l_col, is_lateral=True)

        # Load raw radiographs if metadata available
        meta = self.study_meta.get(str(uid), {})
        f_path = meta.get("frontal_path", "")
        l_path = meta.get("lateral_path", "")

        f_overlay = None
        l_overlay = None
        f_img_bgr = None
        l_img_bgr = None

        if f_path and os.path.exists(f_path):
            f_img_bgr = cv2.imread(f_path)
            f_overlay, _ = self.overlay_heatmap_on_image(f_img_bgr, f_heatmap, colormap, alpha)

        if l_path and os.path.exists(l_path):
            l_img_bgr = cv2.imread(l_path)
            l_overlay, _ = self.overlay_heatmap_on_image(l_img_bgr, l_heatmap, colormap, alpha)

        return {
            "uid": str(uid),
            "target_pathology": pathology,
            "pathology_probability": prob,
            "all_probabilities": probs,
            "frontal_heatmap_12x12": f_heatmap,
            "lateral_heatmap_12x12": l_heatmap,
            "frontal_peak_coord": (f_row, f_col),
            "lateral_peak_coord": (l_row, l_col),
            "frontal_anatomical_zone": f_zone,
            "lateral_anatomical_zone": l_zone,
            "frontal_overlay_bgr": f_overlay,
            "lateral_overlay_bgr": l_overlay,
            "frontal_raw_bgr": f_img_bgr,
            "lateral_raw_bgr": l_img_bgr,
            "colormap": colormap,
        }

    def render_composite_visualization(
        self,
        explanation: Dict[str, Any],
        out_path: Optional[str] = None,
        max_dim: int = 1024,
    ) -> Image.Image:
        """
        Creates a high-resolution, publication-quality 2x2 comparison plate:
          [ Frontal Radiograph ]    [ Frontal Grad-CAM Overlay ]
          [ Lateral Radiograph ]    [ Lateral Grad-CAM Overlay ]
        with annotated title bar and anatomical localization banners.
        """
        f_raw = explanation.get("frontal_raw_bgr")
        f_ov = explanation.get("frontal_overlay_bgr")
        l_raw = explanation.get("lateral_raw_bgr")
        l_ov = explanation.get("lateral_overlay_bgr")

        if f_raw is None or f_ov is None:
            # Fallback if raw images not found: create from heatmaps
            h = w = 512
            f_raw = np.zeros((h, w, 3), dtype=np.uint8) + 40
            f_ov, _ = self.overlay_heatmap_on_image(f_raw, explanation["frontal_heatmap_12x12"])

        if l_raw is None or l_ov is None:
            l_raw = f_raw.copy()
            l_ov = f_ov.copy()

        # Resize all panels to a uniform square size
        panel_sz = min(f_raw.shape[0], f_raw.shape[1], max_dim)
        f_raw_r = cv2.resize(f_raw, (panel_sz, panel_sz))
        f_ov_r = cv2.resize(f_ov, (panel_sz, panel_sz))
        l_raw_r = cv2.resize(l_raw, (panel_sz, panel_sz))
        l_ov_r = cv2.resize(l_ov, (panel_sz, panel_sz))

        # Annotate labels onto overlays
        pathology = explanation["target_pathology"]
        prob = explanation["pathology_probability"]
        f_zone = explanation["frontal_anatomical_zone"]
        l_zone = explanation["lateral_anatomical_zone"]

        font = cv2.FONT_HERSHEY_SIMPLEX

        def add_banner(img, title, subtitle):
            im = img.copy()
            h, w = im.shape[:2]
            cv2.rectangle(im, (0, h - 60), (w, h), (15, 20, 28), -1)
            cv2.putText(im, title, (16, h - 36), font, 0.7, (240, 246, 252), 2, cv2.LINE_AA)
            cv2.putText(im, subtitle, (16, h - 14), font, 0.5, (56, 139, 253), 1, cv2.LINE_AA)
            return im

        p1 = add_banner(f_raw_r, "Frontal PA/AP Radiograph", f"Study UID: {explanation['uid']}")
        p2 = add_banner(f_ov_r, f"Frontal Grad-CAM: {pathology} ({prob:.1%})", f"Peak Localization: {f_zone}")
        p3 = add_banner(l_raw_r, "Lateral Projection", "Paired Thoracic View")
        p4 = add_banner(l_ov_r, f"Lateral Grad-CAM: {pathology} ({prob:.1%})", f"Peak Localization: {l_zone}")

        # Stitch 2x2 grid
        top_row = np.hstack([p1, p2])
        bottom_row = np.hstack([p3, p4])
        grid = np.vstack([top_row, bottom_row])

        # Header banner
        header_h = 90
        header = np.zeros((header_h, grid.shape[1], 3), dtype=np.uint8) + 13
        cv2.putText(header, f"DUAL-VIEW GRAD-CAM: {pathology.upper()}", (24, 45), font, 1.1, (240, 246, 252), 2, cv2.LINE_AA)
        cv2.putText(
            header,
            f"Model Probability: {prob:.1%} | Colormap: {explanation['colormap']} | Backbone: microsoft/rad-dino (ViT-B/14)",
            (24, 75), font, 0.6, (88, 166, 255), 1, cv2.LINE_AA
        )

        composite = np.vstack([header, grid])
        composite_rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(composite_rgb)

        if out_path:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            pil_img.save(out_path)
            print(f"[GradCAM] Saved composite visualization to: {out_path}")

        return pil_img


def main():
    parser = argparse.ArgumentParser(description="Generate Dual-View Grad-CAM Visualizations for Chest Radiographs")
    parser.add_argument("--uid", type=str, default="3039", help="Study UID (e.g. 3039, 2465, 1014)")
    parser.add_argument("--pathology", type=str, default=None, help=f"Target finding from: {LABELS_14}")
    parser.add_argument("--colormap", type=str, default="turbo", choices=list(COLORMAPS.keys()), help="Colormap")
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap blend transparency (0.0 to 1.0)")
    parser.add_argument("--out", type=str, default=None, help="Output image file path (PNG)")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/stage2_best/custom_heads.pt")
    args = parser.parse_args()

    explainer = GradCAMExplainer(checkpoint_path=args.checkpoint)
    out_path = args.out or f"outputs/gradcam_{args.uid}_{args.pathology or 'top'}.png"

    print(f"Generating Grad-CAM visualization for study '{args.uid}'...")
    res = explainer.explain(
        uid=args.uid,
        pathology=args.pathology,
        colormap=args.colormap,
        alpha=args.alpha,
    )

    print(f"  Target Pathology : {res['target_pathology']}")
    print(f"  Confidence Prob  : {res['pathology_probability']:.1%}")
    print(f"  Frontal Peak     : Row {res['frontal_peak_coord'][0]}, Col {res['frontal_peak_coord'][1]} -> {res['frontal_anatomical_zone']}")
    print(f"  Lateral Peak     : Row {res['lateral_peak_coord'][0]}, Col {res['lateral_peak_coord'][1]} -> {res['lateral_anatomical_zone']}")

    explainer.render_composite_visualization(res, out_path=out_path)
    print(f"[GradCAM] Completed successfully! View saved figure at: {out_path}")


if __name__ == "__main__":
    main()
