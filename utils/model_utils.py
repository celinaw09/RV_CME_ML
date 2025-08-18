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
    Convert model outputs to:
      - y_scores: probability scores suitable for ROC/PR (shape (N, n_classes) or (N,) for binary)
      - y_pred: predicted class indices (N,)
    Accepts:
      - outputs raw logits (shape (N, C)) or (N,) for binary logit
      - outputs already softmaxed (probabilities)
    """
    with torch.no_grad():
        out = outputs.detach().cpu()
        if out.ndim == 1 or (out.ndim == 2 and out.size(1) == 1):
            # binary single-logit case
            # if shape (N,1) squeeze it
            if out.ndim == 2:
                out = out.squeeze(1)
            probs = torch.sigmoid(out).numpy()  # shape (N,)
            preds = (probs >= 0.5).astype(int)
            return probs, preds
        elif out.ndim == 2 and out.size(1) >= 2:
            # multi-class logits/probs
            # convert to probabilities with softmax if logits
            # check range to see if likely probs or logits
            if (out.min().item() < 0) or (out.max().item() > 1):
                # treat as logits -> softmax
                probs = torch.softmax(out, dim=1).numpy()
            else:
                probs = out.numpy()
            preds = np.argmax(probs, axis=1)
            return probs, preds
        else:
            raise ValueError(f"Unexpected output shape: {out.shape}")
        

def evaluate_loader(model, loader, criterion=None, device=DEVICE, multilabel=False):
    """
    Runs model over loader and returns aggregated metrics & raw arrays.
    Returns dict:
        { 'y_true', 'y_pred', 'y_scores', 'loss', 'accuracy', 'precision', 'recall', 'f1', 'roc' , 'pr' , 'auc_roc', 'auc_pr' }
    """
    model.eval()
    ys = []
    y_preds = []
    y_scores_list = []  # either shape (N,) for binary or (N, C) for multiclass
    losses = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs, labels = batch[0], batch[1]
            else:
                raise RuntimeError("Loader must return (inputs, labels).")
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            # compute loss if possible
            if criterion is not None:
                try:
                    loss_val = criterion(outputs, labels).item()
                    losses.append(loss_val * inputs.size(0))
                except Exception:
                    # can't compute loss (e.g., shapes mismatch); skip
                    pass

            probs_or_scores, preds = get_probs_and_preds_from_outputs(outputs)
            # attach
            if isinstance(probs_or_scores, np.ndarray):
                y_scores_list.append(probs_or_scores)
            else:
                y_scores_list.append(probs_or_scores.numpy())
            y_preds.append(preds)
            ys.append(labels.detach().cpu().numpy())

    # concat
    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(y_preds, axis=0)
    y_scores_arr = np.concatenate(y_scores_list, axis=0)

    # compute averaged loss
    total_examples = sum([s.size(0) for s in getattr(loader, "batch_sampler").sampler.__iter__()], 0) if False else None
    # simpler: average using counts from concatenation
    avg_loss = (sum(losses) / len(y_true)) if (len(losses) > 0) else None

    # basic metrics
    acc = accuracy_score(y_true, y_pred)

    # precision/recall/f1: choose average depending on classes
    # if binary (y_scores_arr.ndim==1 or y_pred in {0,1})
    if y_scores_arr.ndim == 1:
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        # ROC/PR curves
        fpr, tpr, _ = roc_curve(y_true, y_scores_arr)
        roc_auc = auc(fpr, tpr)
        precs, recs, _ = precision_recall_curve(y_true, y_scores_arr)
        pr_auc = auc(recs, precs)
        aps = average_precision_score(y_true, y_scores_arr)
        roc_curves = (fpr, tpr)
        pr_curves = (recs, precs)
    else:
        # multiclass case -> use one-vs-rest for ROC
        n_classes = y_scores_arr.shape[1]
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        try:
            roc_auc = roc_auc_score(y_true_bin, y_scores_arr, average="macro", multi_class="ovr")
        except Exception:
            roc_auc = None
        # compute micro-averaged PR-auc (average_precision_score can compute macro)
        try:
            pr_auc = average_precision_score(y_true_bin, y_scores_arr, average="macro")
        except Exception:
            pr_auc = None
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        # produce per-class ROC curves (return dict)
        roc_curves = {}
        pr_curves = {}
        for c in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, c], y_scores_arr[:, c])
            roc_curves[c] = (fpr, tpr)
            precs, recs, _ = precision_recall_curve(y_true_bin[:, c], y_scores_arr[:, c])
            pr_curves[c] = (recs, precs)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_scores": y_scores_arr,
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
    os.makedirs(out_dir, exist_ok=True)
    # ROC
    plt.figure(figsize=(6,6))
    if isinstance(train_metrics["roc_curves"], dict):
        # multiclass: plot per-class ROC for readability (train & val)
        for c, (fpr, tpr) in train_metrics["roc_curves"].items():
            plt.plot(fpr, tpr, linestyle='--', alpha=0.6, label=f"train class {c}")
        for c, (fpr, tpr) in val_metrics["roc_curves"].items():
            plt.plot(fpr, tpr, label=f"val class {c}")
    else:
        fpr_t, tpr_t = train_metrics["roc_curves"]
        fpr_v, tpr_v = val_metrics["roc_curves"]
        plt.plot(fpr_t, tpr_t, linestyle='--', label=f"train (AUC={train_metrics['roc_auc']:.3f})")
        plt.plot(fpr_v, tpr_v, label=f"val (AUC={val_metrics['roc_auc']:.3f})")
    plt.plot([0,1],[0,1],'k:', alpha=0.3)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "roc_curve.png"))
    plt.close()

    # Precision-Recall
    plt.figure(figsize=(6,6))
    if isinstance(train_metrics["pr_curves"], dict):
        for c, (rec, prec) in train_metrics["pr_curves"].items():
            plt.plot(rec, prec, linestyle='--', alpha=0.6, label=f"train class {c}")
        for c, (rec, prec) in val_metrics["pr_curves"].items():
            plt.plot(rec, prec, label=f"val class {c}")
    else:
        rec_t, prec_t = train_metrics["pr_curves"]
        rec_v, prec_v = val_metrics["pr_curves"]
        plt.plot(rec_t, prec_t, linestyle='--', label=f"train (AP={train_metrics['pr_auc']:.3f})")
        plt.plot(rec_v, prec_v, label=f"val (AP={val_metrics['pr_auc']:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "pr_curve.png"))
    plt.close()
