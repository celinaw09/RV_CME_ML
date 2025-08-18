import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

def load_checkpoint(checkpoint_path, model, optimizer=None, map_location="cuda"):
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    # If checkpoint contains 'model_state_dict'
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        # assume checkpoint is raw state_dict
        try:
            model.load_state_dict(ckpt)
        except Exception as e:
            raise RuntimeError(f"Unable to load model state: {e}")
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def get_probs_and_preds_from_outputs(outputs):
    """
    Accepts: torch.Tensor outputs from model (logits or probs)
    Returns:
      - probs_or_scores: np.ndarray, either shape (B,) for binary or (B,C) for multiclass
      - preds: np.ndarray shape (B,) of predicted class indices (0..C-1) or {0,1} for binary
    Handles:
      - outputs shape (B,)  -> single-logit? treat as binary logit
      - outputs shape (B,1) -> binary logit -> squeeze to (B,)
      - outputs shape (B,C) -> multiclass logits or probs -> softmax if logits
    """
    # Convert to tensor on cpu
    if isinstance(outputs, dict):
        # common pattern if model returns dict
        # try common keys
        for k in ('logits','out','output'):
            if k in outputs:
                outputs = outputs[k]
                break
        else:
            raise ValueError("Model returned dict but no `logits`/`out`/`output` key found.")
    if isinstance(outputs, torch.Tensor):
        out = outputs.detach().cpu()
    else:
        # assume numpy-like
        out = torch.tensor(outputs).detach().cpu()

    # deal with shapes
    if out.ndim == 1:
        # treat as binary logits -> probability
        probs = torch.sigmoid(out).numpy()             # (B,)
        preds = (probs >= 0.5).astype(int)
        return probs, preds

    if out.ndim == 2 and out.size(1) == 1:
        out1 = out.squeeze(1)                         # (B,)
        probs = torch.sigmoid(out1).numpy()
        preds = (probs >= 0.5).astype(int)
        return probs, preds

    if out.ndim == 2 and out.size(1) >= 2:
        # multiclass: determine if logits or probs
        arr = out.numpy()
        # heuristic: if values outside [0,1] -> logits
        if arr.min() < 0 or arr.max() > 1:
            probs = torch.softmax(out, dim=1).numpy()
        else:
            probs = arr
        preds = np.argmax(probs, axis=1)
        return probs, preds

    raise ValueError(f"Unexpected model output shape: {out.shape}")
        


# ---- robust evaluate_loader ----
def evaluate_loader(model, loader, criterion=None, device=None, debug=False):
    """
    Evaluate model on loader. Handles binary (N,) or (N,1) and multiclass (N,C).
    Returns dict with metrics and ROC/PR curves.
    Set debug=True to print shapes and unique labels.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_y_true = []
    all_y_pred = []
    all_scores = []
    losses = []
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs, labels = batch[0], batch[1]
            else:
                raise RuntimeError("Loader must return (inputs, labels) pairs.")
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            # compute loss if possible
            if criterion is not None:
                try:
                    loss_val = criterion(outputs, labels).item()
                    losses.append(loss_val * inputs.size(0))
                except Exception:
                    # skip if shapes mismatch
                    pass

            probs_or_scores, preds = get_probs_and_preds_from_outputs(outputs)

            all_scores.append(probs_or_scores)
            all_y_pred.append(preds)
            all_y_true.append(labels.detach().cpu().numpy())
            total_samples += labels.size(0)

    # concat
    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)
    y_scores = np.concatenate(all_scores, axis=0)

    # DEBUG prints so we can see what shapes are actually flowing
    if debug:
        print("DEBUG evaluate_loader:")
        print("  y_true.shape:", y_true.shape, "unique labels:", np.unique(y_true))
        print("  y_pred.shape:", y_pred.shape, "unique preds:", np.unique(y_pred))
        print("  y_scores.shape:", y_scores.shape, "dtype:", y_scores.dtype)
        # show a few score values
        print("  y_scores sample:", y_scores.ravel()[:8])

    # Normalize single-column case (N,1) -> (N,)
    if y_scores.ndim == 2 and y_scores.shape[1] == 1:
        y_scores = y_scores.squeeze(1)

    avg_loss = (sum(losses) / total_samples) if (len(losses) > 0 and total_samples > 0) else None
    acc = accuracy_score(y_true, y_pred)

    # Binary branch
    if y_scores.ndim == 1:
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        precs, recs, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recs, precs)
        aps = average_precision_score(y_true, y_scores)

        roc_curves = (fpr, tpr)
        pr_curves = (recs, precs)

    # Multiclass branch
    else:
        n_classes = y_scores.shape[1]
        # if labels are not 0..n_classes-1, remap
        unique_labels = np.unique(y_true)
        if not np.array_equal(unique_labels, np.arange(len(unique_labels))) or len(unique_labels) != n_classes:
            mapping = {v: i for i, v in enumerate(unique_labels)}
            y_true_mapped = np.array([mapping[v] for v in y_true])
            y_true_bin = label_binarize(y_true_mapped, classes=list(range(len(unique_labels))))
        else:
            y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

        try:
            roc_auc = roc_auc_score(y_true_bin, y_scores, average="macro", multi_class="ovr")
        except Exception:
            roc_auc = None

        try:
            pr_auc = average_precision_score(y_true_bin, y_scores, average="macro")
        except Exception:
            pr_auc = None

        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        roc_curves = {}
        pr_curves = {}
        for c in range(y_scores.shape[1]):
            try:
                fpr, tpr, _ = roc_curve(y_true_bin[:, c], y_scores[:, c])
                roc_curves[c] = (fpr, tpr)
            except Exception:
                roc_curves[c] = (None, None)
            try:
                precs, recs, _ = precision_recall_curve(y_true_bin[:, c], y_scores[:, c])
                pr_curves[c] = (recs, precs)
            except Exception:
                pr_curves[c] = (None, None)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_scores": y_scores,
        "loss": avg_loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "roc_curves": roc_curves,
        "pr_curves": pr_curves,
    }


def plot_epoch_curves(history, out_dir="plots"):
    """
    history should be a dict-like with keys:
      train_losses, val_losses, train_accs, val_accs  (lists)
    Optionals: val_roc_auc, val_pr_auc, train_roc_auc, train_pr_auc
    """
    os.makedirs(out_dir, exist_ok=True)
    epochs = np.arange(1, len(history["train_losses"]) + 1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, history["train_losses"], label="train loss")
    plt.plot(epochs, history["val_losses"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(epochs, history["train_accs"], label="train acc")
    plt.plot(epochs, history["val_accs"], label="val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "acc_curve.png"))
    plt.close()

    # optional AUC plots if present
    if "val_roc_auc" in history:
        plt.figure(figsize=(6,4))
        plt.plot(epochs, history["val_roc_auc"], label="val ROC-AUC")
        if "train_roc_auc" in history:
            plt.plot(epochs, history["train_roc_auc"], label="train ROC-AUC")
        plt.xlabel("Epoch")
        plt.ylabel("ROC-AUC")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "roc_auc_curve.png"))
        plt.close()

    if "val_pr_auc" in history:
        plt.figure(figsize=(6,4))
        plt.plot(epochs, history["val_pr_auc"], label="val PR-AUC")
        if "train_pr_auc" in history:
            plt.plot(epochs, history["train_pr_auc"], label="train PR-AUC")
        plt.xlabel("Epoch")
        plt.ylabel("PR-AUC")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, "pr_auc_curve.png"))
        plt.close()

def plot_bar_metrics(train_metrics, val_metrics, out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)
    metrics = ["accuracy", "loss", "precision", "recall", "f1"]
    train_vals = [train_metrics.get(k, np.nan) for k in metrics]
    val_vals = [val_metrics.get(k, np.nan) for k in metrics]

    x = np.arange(len(metrics))
    width = 0.35
    plt.figure(figsize=(10,4))
    plt.bar(x - width/2, train_vals, width, label="train")
    plt.bar(x + width/2, val_vals, width, label="val")
    plt.xticks(x, metrics)
    plt.ylabel("Value")
    plt.title("Train vs Val metrics")
    plt.legend()
    plt.grid(axis="y")
    plt.savefig(os.path.join(out_dir, "metrics_bar.png"))
    plt.close()

def plot_roc_pr(train_metrics, val_metrics, out_dir="plots"):
    """
    Robust plotting of ROC and Precision-Recall curves.
    - train_metrics["roc_curves"] / ["pr_curves"] may be:
        * tuple: (fpr, tpr) or (recall, precision) for binary
        * dict: {class_id: (fpr, tpr) or (rec, prec)} for multiclass
        * some entries may be (None, None) when the curve couldn't be computed
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---------- ROC ----------
    plt.figure(figsize=(6,6))
    skipped_train = []
    skipped_val = []

    # multiclass case if roc_curves is dict
    if isinstance(train_metrics["roc_curves"], dict) or isinstance(val_metrics["roc_curves"], dict):
        # plot per-class; iterate over union of class keys
        train_rc = train_metrics["roc_curves"]
        val_rc = val_metrics["roc_curves"]
        all_classes = sorted(set(list(train_rc.keys()) + list(val_rc.keys())))
        for c in all_classes:
            # train
            tr = train_rc.get(c, (None, None))
            if tr is None or tr[0] is None or tr[1] is None:
                skipped_train.append(c)
            else:
                fpr, tpr = tr
                plt.plot(fpr, tpr, linestyle='--', alpha=0.6, label=f"train class {c}")

            # val
            vr = val_rc.get(c, (None, None))
            if vr is None or vr[0] is None or vr[1] is None:
                skipped_val.append(c)
            else:
                fpr, tpr = vr
                auc_label = ""
                try:
                    auc_val = val_metrics.get("roc_auc")
                    if isinstance(auc_val, dict):
                        auc_label = f" (AUC={auc_val.get(c):.3f})"
                    elif isinstance(auc_val, (float, int)):
                        auc_label = f" (AUC={auc_val:.3f})"
                except Exception:
                    auc_label = ""
                plt.plot(fpr, tpr, label=f"val class {c}{auc_label}")
    else:
        # binary case: tuples expected
        try:
            fpr_t, tpr_t = train_metrics["roc_curves"]
            if fpr_t is not None and tpr_t is not None:
                plt.plot(fpr_t, tpr_t, linestyle='--', label=f"train (AUC={train_metrics.get('roc_auc', float('nan')):.3f})")
            else:
                skipped_train.append("binary")
        except Exception:
            skipped_train.append("binary")

        try:
            fpr_v, tpr_v = val_metrics["roc_curves"]
            if fpr_v is not None and tpr_v is not None:
                plt.plot(fpr_v, tpr_v, label=f"val (AUC={val_metrics.get('roc_auc', float('nan')):.3f})")
            else:
                skipped_val.append("binary")
        except Exception:
            skipped_val.append("binary")

    plt.plot([0,1],[0,1],'k:', alpha=0.3)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"))
    plt.close()

    if skipped_train or skipped_val:
        print(f"plot_roc_pr: skipped ROC plotting for train classes: {skipped_train}, val classes: {skipped_val}")

    # ---------- Precision-Recall ----------
    plt.figure(figsize=(6,6))
    skipped_train_pr = []
    skipped_val_pr = []

    if isinstance(train_metrics["pr_curves"], dict) or isinstance(val_metrics["pr_curves"], dict):
        train_pc = train_metrics["pr_curves"]
        val_pc = val_metrics["pr_curves"]
        all_classes = sorted(set(list(train_pc.keys()) + list(val_pc.keys())))
        for c in all_classes:
            tr = train_pc.get(c, (None, None))
            if tr is None or tr[0] is None or tr[1] is None:
                skipped_train_pr.append(c)
            else:
                rec, prec = tr
                plt.plot(rec, prec, linestyle='--', alpha=0.6, label=f"train class {c}")
            vr = val_pc.get(c, (None, None))
            if vr is None or vr[0] is None or vr[1] is None:
                skipped_val_pr.append(c)
            else:
                rec, prec = vr
                ap_label = ""
                try:
                    ap_val = val_metrics.get("pr_auc")
                    if isinstance(ap_val, dict):
                        ap_label = f" (AP={ap_val.get(c):.3f})"
                    elif isinstance(ap_val, (float, int)):
                        ap_label = f" (AP={ap_val:.3f})"
                except Exception:
                    ap_label = ""
                plt.plot(rec, prec, label=f"val class {c}{ap_label}")
    else:
        # binary case: tuples expected (rec, prec)
        try:
            rec_t, prec_t = train_metrics["pr_curves"]
            if rec_t is not None and prec_t is not None:
                plt.plot(rec_t, prec_t, linestyle='--', label=f"train (AP={train_metrics.get('pr_auc', float('nan')):.3f})")
            else:
                skipped_train_pr.append("binary")
        except Exception:
            skipped_train_pr.append("binary")

        try:
            rec_v, prec_v = val_metrics["pr_curves"]
            if rec_v is not None and prec_v is not None:
                plt.plot(rec_v, prec_v, label=f"val (AP={val_metrics.get('pr_auc', float('nan')):.3f})")
            else:
                skipped_val_pr.append("binary")
        except Exception:
            skipped_val_pr.append("binary")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pr_curve.png"))
    plt.close()

    if skipped_train_pr or skipped_val_pr:
        print(f"plot_roc_pr: skipped PR plotting for train classes: {skipped_train_pr}, val classes: {skipped_val_pr}")
