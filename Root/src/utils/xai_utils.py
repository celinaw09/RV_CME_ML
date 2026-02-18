import os
import pandas as pd
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


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