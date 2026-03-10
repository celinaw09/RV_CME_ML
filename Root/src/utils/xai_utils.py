import os
import pandas as pd
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import glob

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image


# ----------------------------
# Helpers
# ----------------------------
def _denormalize_grayscale(img_tensor: torch.Tensor, mean=0.289, std=0.146):
    """
    img_tensor: shape (1, H, W) or (H, W) torch tensor, normalized.
    returns float32 numpy (H, W) in [0,1] approx (clipped).
    """
    if img_tensor.dim() == 3:
        img_tensor = img_tensor[0]
    x = img_tensor.detach().cpu().float()
    x = x * std + mean
    x = torch.clamp(x, 0.0, 1.0)
    return x.numpy()


def _to_3ch_uint8(gray_01: np.ndarray):
    """
    gray_01: (H,W) float in [0,1]
    returns (H,W,3) uint8
    """
    gray_uint8 = (gray_01 * 255.0).astype(np.uint8)
    rgb = np.stack([gray_uint8, gray_uint8, gray_uint8], axis=-1)
    return rgb


def _find_resnet_target_layer(model: torch.nn.Module):
    """
    Best-effort: choose the last block in layer4 for torchvision-style ResNet.
    If your build_resnet_for_grayscale wraps the resnet, adapt here.
    """
    # Common case: model has attribute 'layer4'
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    # Sometimes model.model or model.backbone contains the resnet
    for attr in ["model", "backbone", "net", "resnet"]:
        if hasattr(model, attr):
            sub = getattr(model, attr)
            if hasattr(sub, "layer4"):
                return sub.layer4[-1]
    raise ValueError("Could not find a ResNet-like layer4[-1] target layer. "
                     "Pass target_layers explicitly.")


def _predict_class_and_probs(model, input_tensor, device):
    """
    Returns predicted class index and probs (softmax) for binary/multiclass.
    """
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor.to(device))
        probs = torch.softmax(logits, dim=1)
        pred = int(torch.argmax(probs, dim=1).item())
    return pred, probs.squeeze(0).detach().cpu().numpy()


# ----------------------------
# Main API
# ----------------------------
def run_gradcam_single(
    model,
    device,
    sample_path,
    transform,
    out_dir,                 # <-- changed: pass folder, not full file path
    sample_id,               # <-- used in filenames
    true_label=None,         # optional, for filenames/metadata
    class_idx=None,          # None => predicted class
    target_layers=None,
    mean=0.289,
    std=0.146,
    alpha=0.45,
    save_pdf=True,
):
    """
    Saves two figures for one sample:
      1) overlay-only:  <out_dir>/<sample_id>_overlay.png
      2) comparison:    <out_dir>/<sample_id>_comparison.(png|pdf)
                        left: raw cam, right: overlay

    Returns dict with prediction + file paths.
    """
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)

    # ---- Load + preprocess exactly like training ----
    img = Image.open(sample_path)
    input_tensor = transform(img)              # (1,H,W)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)  # (1,1,H,W)
    input_tensor = input_tensor.to(device)

    # ---- Determine target layers ----
    if target_layers is None:
        layer = _find_resnet_target_layer(model)
        target_layers = [layer]

    # ---- Prediction ----
    pred_idx, probs = _predict_class_and_probs(model, input_tensor, device)
    if class_idx is None:
        class_idx = pred_idx

    # ---- Run Grad-CAM ----
    model.eval()
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(class_idx)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)  # (1,H,W)
    cam_map = grayscale_cam[0]                                       # (H,W) in [0,1]

    # ---- Convert original input back to visible image ----
    input_cpu = input_tensor[0].detach().cpu()   # (1,H,W)
    gray_01 = _denormalize_grayscale(input_cpu, mean=mean, std=std)  # (H,W)
    rgb_uint8 = _to_3ch_uint8(gray_01)                                # (H,W,3) uint8

    # ---- Create overlay ----
    heat_uint8 = (cam_map * 255.0).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    overlay = (rgb_uint8 * (1 - alpha) + heat_color * alpha).astype(np.uint8)

    # ---- Filenames (self-documenting) ----
    # Note: label mapping in your code: non_CME=0, CME=1
    def _lbl_name(v):
        if v is None:
            return "unk"
        return "CME" if int(v) == 1 else "nonCME"

    true_name = _lbl_name(true_label)
    pred_name = _lbl_name(pred_idx)

    probs_str = f"p0_{probs[0]:.3f}_p1_{probs[1]:.3f}"
    base = f"{sample_id}_true_{true_name}_pred_{pred_name}_{probs_str}_target_{_lbl_name(class_idx)}"

    overlay_path = os.path.join(out_dir, base + "_overlay.png")
    comparison_ext = "pdf" if save_pdf else "png"
    comparison_path = os.path.join(out_dir, base + f"_comparison.{comparison_ext}")

    # -----------------------------
    # Figure 1: Overlay ONLY
    # -----------------------------
    plt.figure(figsize=(7, 7))
    plt.imshow(overlay)
    plt.title(f"Grad-CAM Overlay | true={true_name} pred={pred_name} target={_lbl_name(class_idx)}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=300, bbox_inches="tight")
    plt.close()

    # -----------------------------
    # Figure 2: Raw CAM + Overlay Side-by-Side
    # -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(cam_map, cmap="jet")
    axes[0].set_title("Raw Grad-CAM (Normalized)")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Overlay on Image")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "sample_path": sample_path,
        "sample_id": sample_id,
        "true_label": None if true_label is None else int(true_label),
        "pred_idx": int(pred_idx),
        "target_idx": int(class_idx),
        "probs": probs,
        "overlay_path": overlay_path,
        "comparison_path": comparison_path,
        "target_layer": str(target_layers[0]),
    }


def run_gradcam_on_balanced_samples(df, model, device,transform, xai_root_dir, n_per_class=5,seed=42):
    """
    Picks n_per_class samples from label 0 and 1, runs prediction-based Grad-CAM.
    Saves results under xai_root_dir/xai_plots/.
    Returns a dataframe report.
    """
    out_dir = os.path.join(xai_root_dir, "xai_plots")
    os.makedirs(out_dir, exist_ok=True)

    df0 = df[df["label"] == 0].sample(n=min(n_per_class, (df["label"] == 0).sum()), random_state=seed)
    df1 = df[df["label"] == 1].sample(n=min(n_per_class, (df["label"] == 1).sum()), random_state=seed)
    df_sel = pd.concat([df0, df1]).reset_index(drop=False)  # keep original index as an id

    records = []
    for i, row in df_sel.iterrows():
        sample_path = row["image_path"]
        true_label = int(row["label"])
        # Use a stable ID: original df index + filename stem
        orig_idx = row["index"]  # from reset_index(drop=False)
        fname = os.path.splitext(os.path.basename(sample_path))[0]
        sample_id = f"idx{orig_idx}_{fname}"

        # Grad-CAM on predicted class (class_idx=None)
        info = run_gradcam_single(
            model=model,
            device=device,
            sample_path=sample_path,
            transform=transform,
            out_dir=out_dir,
            sample_id=sample_id,
            true_label=true_label,
            class_idx=None,      # prediction-based CAM
            target_layers=None,  # auto-detect layer4[-1]
            save_pdf=True
        )

        # Flatten probs for report
        probs = info["probs"]
        info["prob_nonCME"] = float(probs[0])
        info["prob_CME"] = float(probs[1])
        del info["probs"]

        records.append(info)

    report_df = pd.DataFrame(records)
    report_path = os.path.join(out_dir, "xai_report.csv")
    report_df.to_csv(report_path, index=False)

    print(f"[XAI] Saved {len(report_df)} samples to: {out_dir}")
    print(f"[XAI] Report: {report_path}")
    return report_df



def run_gradcam_on_true_positives(
    df,
    model,
    device,
    transform,
    xai_root_dir,
    n_tp=10,
    seed=42,
    threshold=0.5,
):
    """
    1) Predict on ALL df rows.
    2) Filter TRUE POSITIVES (true=1, pred=1) using threshold on prob_CME.
    3) Run Grad-CAM (target class = CME) on those samples.
    4) Save:
        - PNG: 2-panel (original sample | heatmap overlay)
        - PDF: 3-panel (raw cam | original sample | heatmap overlay)
        - CSV report

    Returns:
        report_df (pd.DataFrame)
    """
    out_dir = os.path.join(xai_root_dir, "xai_plots", "TP")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Step A: run predictions for the entire df ----
    from PIL import Image

    model.eval()
    preds = []
    prob_cmes = []

    for idx, row in df.reset_index(drop=False).iterrows():
        sample_path = row["image_path"]
        img = Image.open(sample_path)
        x = transform(img)  # (1,H,W) for grayscale
        if x.dim() == 3:
            x = x.unsqueeze(0)  # (1,1,H,W)

        pred_idx, probs = _predict_class_and_probs(model, x.to(device), device)
        prob_cme = float(probs[1])

        preds.append(pred_idx)
        prob_cmes.append(prob_cme)

    df_pred = df.reset_index(drop=False).copy()
    df_pred["pred_idx"] = preds
    df_pred["prob_CME"] = prob_cmes
    df_pred["pred_thresh"] = (df_pred["prob_CME"] >= threshold).astype(int)

    # ---- Step B: TRUE POSITIVES: true label=1 and predicted=1 (thresholded) ----
    df_tp = df_pred[(df_pred["label"] == 1) & (df_pred["pred_thresh"] == 1)].copy()

    if len(df_tp) == 0:
        print("[XAI] No TRUE POSITIVES found at this threshold.")
        return df_tp

    df_tp = df_tp.sample(n=min(n_tp, len(df_tp)), random_state=seed).reset_index(drop=True)

    # ---- Step C: run Grad-CAM on CME class explicitly (class_idx=1) ----
    records = []
    for i, row in df_tp.iterrows():
        sample_path = row["image_path"]
        true_label = int(row["label"])
        pred_idx = int(row["pred_thresh"])

        orig_idx = row["index"]
        fname = os.path.splitext(os.path.basename(sample_path))[0]
        sample_id = f"TP_idx{orig_idx}_{fname}"

        # -----------------------------
        # Build CAM + original + overlay (same logic as run_gradcam_single)
        # but change ONLY what we save.
        # -----------------------------
        img = Image.open(sample_path)
        input_tensor = transform(img)  # (1,H,W)
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)  # (1,1,H,W)
        input_tensor = input_tensor.to(device)

        # Target layer auto-detect (same behavior as your existing pipeline)
        layer = _find_resnet_target_layer(model)
        target_layers = [layer]

        # Predict (for metadata)
        pred_raw_idx, probs = _predict_class_and_probs(model, input_tensor, device)

        # Force CME CAM
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(1)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)  # (1,H,W)
        cam_map = grayscale_cam[0]  # (H,W) in [0,1]

        # Denormalize + render original (grayscale -> 3ch)
        input_cpu = input_tensor[0].detach().cpu()  # (1,H,W)
        gray_01 = _denormalize_grayscale(input_cpu, mean=0.289, std=0.146)  # (H,W) in [0,1]
        original_rgb_uint8 = _to_3ch_uint8(gray_01)  # (H,W,3) uint8

        # Build overlay
        alpha = 0.45
        heat_uint8 = (cam_map * 255.0).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
        overlay_uint8 = (original_rgb_uint8 * (1 - alpha) + heat_color * alpha).astype(np.uint8)

        # Label helper (for filenames)
        def _lbl_name(v):
            if v is None:
                return "unk"
            return "CME" if int(v) == 1 else "nonCME"

        true_name = _lbl_name(true_label)
        pred_name = _lbl_name(pred_idx)  # thresholded pred for TP is 1
        probs_str = f"p0_{probs[0]:.3f}_p1_{probs[1]:.3f}"
        base = f"{sample_id}_true_{true_name}_pred_{pred_name}_{probs_str}_target_CME"

        # Paths
        pair_png_path = os.path.join(out_dir, base + "_pair.png")
        triple_pdf_path = os.path.join(out_dir, base + "_triple.pdf")

        # -----------------------------
        # SAVE PNG: (original | overlay)
        # -----------------------------
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(original_rgb_uint8)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(overlay_uint8)
        axes[1].set_title("Grad-CAM Overlay")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(pair_png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # -----------------------------
        # SAVE PDF: (raw cam | original | overlay)
        # -----------------------------
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(cam_map, cmap="jet")
        axes[0].set_title("Raw Grad-CAM (Normalized)")
        axes[0].axis("off")

        axes[1].imshow(original_rgb_uint8)
        axes[1].set_title("Original")
        axes[1].axis("off")

        axes[2].imshow(overlay_uint8)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(triple_pdf_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Record row
        info = {
            "sample_path": sample_path,
            "sample_id": sample_id,
            "true_label": int(true_label),
            "pred_idx": int(pred_raw_idx),
            "target_idx": 1,
            "prob_nonCME": float(probs[0]),
            "prob_CME": float(probs[1]),
            "pred_idx_thresh": int(pred_idx),
            "pair_png_path": pair_png_path,
            "triple_pdf_path": triple_pdf_path,
            "target_layer": str(target_layers[0]),
        }

        records.append(info)

    report_df = pd.DataFrame(records)
    report_path = os.path.join(out_dir, "xai_report_TP.csv")
    report_df.to_csv(report_path, index=False)

    print(f"[XAI] TRUE POSITIVES saved to: {out_dir}")
    print(f"[XAI] Report: {report_path}")
    return report_df


def _first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def make_category_grid_figure(
    report_csv,
    out_pdf,
    n_rows=5,
    seed=42,
    category_name="TP",
    dpi=300,
    fontsize_row=9,
):
    """
    Builds a grid PDF for one category:
      - left: original image
      - right: Grad-CAM overlay

    Expected CSV columns:
      - sample_path or image_path
      - pair_png_path
      - sample_id
      - prob_CME
      - pred_idx_thresh (optional)

    Saves:
      out_pdf
    """
    df = pd.read_csv(report_csv)

    if len(df) == 0:
        raise ValueError(f"{report_csv} is empty.")

    orig_col = _first_existing_col(df, ["sample_path", "image_path"])
    if orig_col is None:
        raise KeyError("Could not find original image path column.")

    pair_col = _first_existing_col(df, ["pair_png_path", "pair_path"])
    if pair_col is None:
        raise KeyError("Could not find pair PNG path column.")

    sid_col = _first_existing_col(df, ["sample_id", "id"])
    prob_col = _first_existing_col(df, ["prob_CME", "prob_cme", "p_cme"])
    pred_col = _first_existing_col(df, ["pred_idx_thresh", "pred_thresh", "pred_idx"])

    df = df.sample(n=min(n_rows, len(df)), random_state=seed).reset_index(drop=True)
    n = len(df)

    fig_h = max(6, 1.55 * n * 2)
    fig = plt.figure(figsize=(10.5, fig_h))

    gs = fig.add_gridspec(
        nrows=2 * n,
        ncols=2,
        height_ratios=[0.18, 1.0] * n,
        wspace=0.02,
        hspace=0.08,
    )

    fig.text(0.27, 0.995, "Original", ha="center", va="top", fontsize=12, fontweight="bold")
    fig.text(0.75, 0.995, "Grad-CAM Overlay", ha="center", va="top", fontsize=12, fontweight="bold")
    fig.suptitle(f"{category_name} Samples: Original vs Grad-CAM Overlay", fontsize=13, y=0.999)

    for i in range(n):
        row = df.iloc[i]

        orig_path = row[orig_col]
        pair_path = row[pair_col]

        if not os.path.exists(orig_path):
            raise FileNotFoundError(f"Original image not found: {orig_path}")
        if not os.path.exists(pair_path):
            raise FileNotFoundError(f"Pair PNG not found: {pair_path}")

        # load original
        orig = Image.open(orig_path).convert("RGB")

        # load pair image and crop only the overlay half
        pair_img = Image.open(pair_path).convert("RGB")
        w, h = pair_img.size
        overlay = pair_img.crop((w // 2, 0, w, h))

        ax_text = fig.add_subplot(gs[2 * i, :])
        ax_text.axis("off")

        bits = []
        if sid_col is not None:
            bits.append(str(row[sid_col]))
        if prob_col is not None:
            bits.append(f"p(CME)={float(row[prob_col]):.3f}")
        if pred_col is not None:
            bits.append(f"pred={int(row[pred_col])}")

        meta = " | ".join(bits) if bits else f"{category_name} {i+1}"

        ax_text.text(
            0.0, 0.2, meta,
            ha="left", va="bottom",
            fontsize=fontsize_row
        )

        axL = fig.add_subplot(gs[2 * i + 1, 0])
        axR = fig.add_subplot(gs[2 * i + 1, 1])

        axL.imshow(orig)
        axL.axis("off")

        axR.imshow(overlay)
        axR.axis("off")

    fig.subplots_adjust(top=0.985, left=0.02, right=0.98, bottom=0.02)
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"[GRID] Saved {category_name} grid to: {out_pdf}")
    return out_pdf




def run_gradcam_on_category(
    df,
    model,
    device,
    transform,
    xai_root_dir,
    category="TP",
    n_samples=10,
    seed=42,
    threshold=0.5,
):
    """
    category in {"TP", "TN", "FP", "FN"}

    Saves:
      - PNG: original + overlay
      - PDF: raw cam + original + overlay
      - CSV report

    Returns:
        report_df (pd.DataFrame)
    """
    out_dir = os.path.join(xai_root_dir, "xai_plots", category)
    os.makedirs(out_dir, exist_ok=True)


    model.eval()
    preds = []
    prob_cmes = []

    # -----------------------------
    # Step A: Predict on all rows
    # -----------------------------
    for _, row in df.reset_index(drop=False).iterrows():
        sample_path = row["image_path"]
        img = Image.open(sample_path)
        x = transform(img)
        if x.dim() == 3:
            x = x.unsqueeze(0)

        pred_idx, probs = _predict_class_and_probs(model, x.to(device), device)
        preds.append(pred_idx)
        prob_cmes.append(float(probs[1]))

    df_pred = df.reset_index(drop=False).copy()
    df_pred["pred_idx"] = preds
    df_pred["prob_CME"] = prob_cmes
    df_pred["pred_thresh"] = (df_pred["prob_CME"] >= threshold).astype(int)

    # -----------------------------
    # Step B: Filter category
    # -----------------------------
    if category == "TP":
        df_cat = df_pred[(df_pred["label"] == 1) & (df_pred["pred_thresh"] == 1)].copy()
    elif category == "TN":
        df_cat = df_pred[(df_pred["label"] == 0) & (df_pred["pred_thresh"] == 0)].copy()
    elif category == "FP":
        df_cat = df_pred[(df_pred["label"] == 0) & (df_pred["pred_thresh"] == 1)].copy()
    elif category == "FN":
        df_cat = df_pred[(df_pred["label"] == 1) & (df_pred["pred_thresh"] == 0)].copy()
    else:
        raise ValueError("category must be one of: TP, TN, FP, FN")

    if len(df_cat) == 0:
        print(f"[XAI] No samples found for category {category}.")
        return df_cat

    df_cat = df_cat.sample(n=min(n_samples, len(df_cat)), random_state=seed).reset_index(drop=True)

    # -----------------------------
    # Step C: Generate CAMs
    # -----------------------------
    records = []
    for _, row in df_cat.iterrows():
        sample_path = row["image_path"]
        true_label = int(row["label"])
        pred_idx_thresh = int(row["pred_thresh"])

        orig_idx = row["index"]
        fname = os.path.splitext(os.path.basename(sample_path))[0]
        sample_id = f"{category}_idx{orig_idx}_{fname}"

        img = Image.open(sample_path)
        input_tensor = transform(img)
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(device)

        layer = _find_resnet_target_layer(model)
        target_layers = [layer]

        pred_raw_idx, probs = _predict_class_and_probs(model, input_tensor, device)

        # Force CME CAM for all categories so comparison is meaningful
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(1)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        cam_map = grayscale_cam[0]

        input_cpu = input_tensor[0].detach().cpu()
        gray_01 = _denormalize_grayscale(input_cpu, mean=0.289, std=0.146)
        original_rgb_uint8 = _to_3ch_uint8(gray_01)

        alpha = 0.45
        heat_uint8 = (cam_map * 255.0).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
        overlay_uint8 = (original_rgb_uint8 * (1 - alpha) + heat_color * alpha).astype(np.uint8)

        def _lbl_name(v):
            return "CME" if int(v) == 1 else "nonCME"

        true_name = _lbl_name(true_label)
        pred_name = _lbl_name(pred_idx_thresh)
        probs_str = f"p0_{probs[0]:.3f}_p1_{probs[1]:.3f}"
        base = f"{sample_id}_true_{true_name}_pred_{pred_name}_{probs_str}_target_CME"

        pair_png_path = os.path.join(out_dir, base + "_pair.png")
        triple_pdf_path = os.path.join(out_dir, base + "_triple.pdf")
        raw_cam_npy_path = os.path.join(out_dir, base + "_cam.npy")

        # Save raw CAM for later similarity analysis
        np.save(raw_cam_npy_path, cam_map)

        # PNG: original + overlay
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(original_rgb_uint8)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(overlay_uint8)
        axes[1].set_title("Grad-CAM Overlay")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(pair_png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # PDF: raw cam + original + overlay
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(cam_map, cmap="jet")
        axes[0].set_title("Raw Grad-CAM")
        axes[0].axis("off")

        axes[1].imshow(original_rgb_uint8)
        axes[1].set_title("Original")
        axes[1].axis("off")

        axes[2].imshow(overlay_uint8)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(triple_pdf_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        info = {
            "sample_path": sample_path,
            "sample_id": sample_id,
            "category": category,
            "true_label": int(true_label),
            "pred_idx": int(pred_raw_idx),
            "pred_idx_thresh": int(pred_idx_thresh),
            "target_idx": 1,
            "prob_nonCME": float(probs[0]),
            "prob_CME": float(probs[1]),
            "pair_png_path": pair_png_path,
            "triple_pdf_path": triple_pdf_path,
            "raw_cam_npy_path": raw_cam_npy_path,
            "target_layer": str(target_layers[0]),
        }
        records.append(info)

    report_df = pd.DataFrame(records)
    report_path = os.path.join(out_dir, f"xai_report_{category}.csv")
    report_df.to_csv(report_path, index=False)

    print(f"[XAI] {category} saved to: {out_dir}")
    print(f"[XAI] Report: {report_path}")
    return report_df


import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _load_cam_arrays_from_category_dir(category_dir):
    """
    Loads all *_cam.npy files from a category directory.
    Returns:
        cams: list of np.ndarray
        names: list of filenames
    """
    cam_paths = sorted(glob.glob(os.path.join(category_dir, "*_cam.npy")))
    cams = []
    names = []

    for p in cam_paths:
        cam = np.load(p)
        cams.append(cam.astype(np.float32))
        names.append(os.path.basename(p))

    return cams, names


def _pairwise_cam_correlations(cams, names, category):
    """
    Compute pairwise Pearson correlation between all CAM maps in a category.
    Returns a list of dict records.
    """
    records = []

    if len(cams) < 2:
        return records

    # flatten each CAM once
    flat_cams = [cam.flatten() for cam in cams]

    for i in range(len(flat_cams)):
        for j in range(i + 1, len(flat_cams)):
            a = flat_cams[i]
            b = flat_cams[j]

            # guard against constant arrays
            if np.std(a) < 1e-8 or np.std(b) < 1e-8:
                corr = np.nan
            else:
                corr = float(np.corrcoef(a, b)[0, 1])

            records.append({
                "category": category,
                "sample_a": names[i],
                "sample_b": names[j],
                "pearson_corr": corr,
            })

    return records


def analyze_cam_similarity(
    xai_root_dir,
    categories=("TP", "TN", "FP", "FN"),
    out_filename_prefix="cam_similarity",
    make_boxplot=True,
):
    """
    Analyze Grad-CAM similarity across TP/TN/FP/FN groups.

    Expected folder structure:
        xai_root_dir/
            xai_plots/
                TP/
                    *_cam.npy
                TN/
                    *_cam.npy
                FP/
                    *_cam.npy
                FN/
                    *_cam.npy

    Saves:
        - cam_similarity_scores.csv
        - cam_similarity_summary.csv
        - cam_similarity_boxplot.pdf

    Returns:
        scores_df, summary_df
    """
    base_dir = os.path.join(xai_root_dir, "xai_plots")
    os.makedirs(base_dir, exist_ok=True)

    all_records = []

    for category in categories:
        cat_dir = os.path.join(base_dir, category)
        if not os.path.isdir(cat_dir):
            print(f"[SIM] Skipping missing category dir: {cat_dir}")
            continue

        cams, names = _load_cam_arrays_from_category_dir(cat_dir)
        print(f"[SIM] {category}: loaded {len(cams)} CAM maps")

        records = _pairwise_cam_correlations(cams, names, category)
        all_records.extend(records)

    if len(all_records) == 0:
        print("[SIM] No pairwise CAM similarity records generated.")
        return pd.DataFrame(), pd.DataFrame()

    scores_df = pd.DataFrame(all_records)

    # summary stats per category
    summary_df = (
        scores_df.groupby("category")["pearson_corr"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
        .rename(columns={"count": "n_pairs"})
    )

    # save CSVs
    scores_csv = os.path.join(base_dir, f"{out_filename_prefix}_scores.csv")
    summary_csv = os.path.join(base_dir, f"{out_filename_prefix}_summary.csv")
    scores_df.to_csv(scores_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print(f"[SIM] Saved pairwise scores to: {scores_csv}")
    print(f"[SIM] Saved summary stats to: {summary_csv}")

    # boxplot
    if make_boxplot:
        plot_path = os.path.join(base_dir, f"{out_filename_prefix}_boxplot.pdf")

        ordered_categories = [c for c in categories if c in scores_df["category"].unique()]
        data = [
            scores_df.loc[scores_df["category"] == c, "pearson_corr"].dropna().values
            for c in ordered_categories
        ]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.boxplot(data, labels=ordered_categories, showmeans=True)
        ax.set_ylabel("Pairwise CAM Pearson Correlation")
        ax.set_title("Grad-CAM Similarity Across Prediction Groups")
        ax.grid(True, axis="y", alpha=0.3)

        fig.tight_layout(pad=0.5)
        fig.savefig(plot_path, format="pdf", bbox_inches="tight", pad_inches=0.02, dpi=300)
        plt.close(fig)

        print(f"[SIM] Saved boxplot to: {plot_path}")

    return scores_df, summary_df


def _load_cam_arrays_from_category_dir(category_dir):
    cam_paths = sorted(glob.glob(os.path.join(category_dir, "*_cam.npy")))
    cams = []
    names = []

    for p in cam_paths:
        cam = np.load(p).astype(np.float32)
        cams.append(cam)
        names.append(os.path.basename(p))

    return cams, names


def _normalize_cam(cam):
    cam = cam.astype(np.float32)
    cam = cam - cam.min()
    denom = cam.max()
    if denom > 1e-8:
        cam = cam / denom
    return cam


def _compute_center_of_mass(cam):
    """
    cam: 2D numpy array, assumed nonnegative
    returns: (y_cm, x_cm)
    """
    cam = np.maximum(cam, 0)
    total_mass = cam.sum()

    if total_mass < 1e-8:
        return np.nan, np.nan

    h, w = cam.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    y_cm = (yy * cam).sum() / total_mass
    x_cm = (xx * cam).sum() / total_mass

    return float(y_cm), float(x_cm)


def analyze_mean_cam_centers(
    xai_root_dir,
    categories=("TP", "TN", "FP", "FN"),
    out_filename_prefix="mean_cam_center",
    normalize_each_cam=True,
    save_figures=True,
):
    """
    For each category:
      1) Load *_cam.npy
      2) Compute mean CAM
      3) Compute center of mass of mean CAM
      4) Save summary CSV
      5) Optionally save figure with center marked

    Returns:
      summary_df
    """
    base_dir = os.path.join(xai_root_dir, "xai_plots")
    os.makedirs(base_dir, exist_ok=True)

    summary_records = []

    for category in categories:
        cat_dir = os.path.join(base_dir, category)
        if not os.path.isdir(cat_dir):
            print(f"[COM] Skipping missing category dir: {cat_dir}")
            continue

        cams, names = _load_cam_arrays_from_category_dir(cat_dir)
        if len(cams) == 0:
            print(f"[COM] No CAM files found in {cat_dir}")
            continue

        if normalize_each_cam:
            cams = [_normalize_cam(cam) for cam in cams]

        mean_cam = np.mean(np.stack(cams, axis=0), axis=0)
        mean_cam = _normalize_cam(mean_cam)

        y_cm, x_cm = _compute_center_of_mass(mean_cam)

        summary_records.append({
            "category": category,
            "n_cams": len(cams),
            "y_cm": y_cm,
            "x_cm": x_cm,
            "height": mean_cam.shape[0],
            "width": mean_cam.shape[1],
            "y_cm_norm": y_cm / mean_cam.shape[0] if np.isfinite(y_cm) else np.nan,
            "x_cm_norm": x_cm / mean_cam.shape[1] if np.isfinite(x_cm) else np.nan,
        })

        # save mean CAM array
        mean_cam_npy = os.path.join(base_dir, f"{out_filename_prefix}_{category}.npy")
        np.save(mean_cam_npy, mean_cam)

        if save_figures:
            fig_path = os.path.join(base_dir, f"{out_filename_prefix}_{category}.pdf")

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(mean_cam, cmap="jet")
            ax.scatter([x_cm], [y_cm], c="white", s=80, marker="x", linewidths=2)
            ax.set_title(f"{category} Mean CAM\nCenter=({x_cm:.1f}, {y_cm:.1f})")
            ax.axis("off")

            fig.tight_layout(pad=0.5)
            fig.savefig(fig_path, format="pdf", bbox_inches="tight", pad_inches=0.02, dpi=300)
            plt.close(fig)

            print(f"[COM] Saved mean CAM figure for {category}: {fig_path}")

    summary_df = pd.DataFrame(summary_records)

    summary_csv = os.path.join(base_dir, f"{out_filename_prefix}_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print(f"[COM] Saved summary to: {summary_csv}")
    return summary_df