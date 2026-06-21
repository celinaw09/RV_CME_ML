import torch
import torch.nn.functional as F

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
)

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

def find_best_threshold(y_true, y_prob):

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_prob
    )

    j_scores = tpr - fpr

    best_idx = np.argmax(j_scores)

    return thresholds[best_idx]


def train(
    model,
    device,
    train_loader,
    optimizer,
    criterion,
    epoch,
    log_interval=10,
    val_loader=None,
    validate_every_epochs=None,  # <-- validate every N epochs
    val_threshold=0.5,
    early_stopping_on_loss=False,  # <-- name kept for compatibility, now uses monitor_metric
    es_patience=3,
    es_min_delta=1e-4,
    checkpoint_path=None,        # where to save best model (weights only)
    monitor_metric="accuracy",   # metric to choose best model AND early stopping
):
    """
    Train for ONE epoch.

    Validation is performed every `validate_every_epochs` epochs (epoch % N == 0).

    Early stopping:
        • If `early_stopping_on_loss` is True, we actually early-stop based on
          the *validation `monitor_metric`* (default: accuracy), i.e., we stop
          when that metric stops improving for `es_patience` validation cycles.
        • State is persisted across epochs via attributes on `model`.

    Best-model saving:
        • if `checkpoint_path` is not None and validation runs,
          we save the model with the HIGHEST validation `monitor_metric`
          (by default, 'accuracy').

    Returns:
        avg_loss, accuracy on the training set for THIS epoch.
    """
    import torch.nn.functional as F

    model.train()
    total_loss = 0.0
    correct = 0
    num_samples = 0

    # ---- Best-model / early-stopping state (persistent across epochs) ----
    if checkpoint_path is not None or early_stopping_on_loss:
        # We maximize this metric (accuracy by default)
        if not hasattr(model, "_best_val_metric_for_es"):
            model._best_val_metric_for_es = float("-inf")
            model._es_no_improve = 0

    # Also keep a separate best metric for saving checkpoint (can share with ES)
    if checkpoint_path is not None:
        if not hasattr(model, "_best_val_metric"):
            model._best_val_metric = float("-inf")

    # ---- TRAIN OVER THIS EPOCH ----
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # print(data.min().item(), data.max().item(), data.mean().item(), data.std().item())

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # accumulate statistics
        batch_size = data.size(0)
        total_loss += loss.item() * batch_size
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        num_samples += batch_size

        # log
        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * batch_size}/{len(train_loader.dataset)} "
                f"({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
            )

    # ---- END-OF-EPOCH training stats ----
    avg_loss = total_loss / num_samples
    accuracy = 100.0 * correct / num_samples

    print(
        f"Train Epoch {epoch} Average loss: {avg_loss:.4f}, "
        f"Accuracy: {correct}/{num_samples} ({accuracy:.2f}%)"
    )

    # ---- CONDITIONAL VALIDATION (every N epochs) ----
    should_validate = (
        val_loader is not None
        and validate_every_epochs is not None
        and (epoch % validate_every_epochs == 0)
    )

    if should_validate:
        print(f"\n=== Validation at epoch {epoch} (every {validate_every_epochs} epochs) ===")
        val_metrics = validate(
            model,
            device,
            criterion,
            val_loader,
            threshold=0.5,
            optimize_threshold=False
        )

        # Pretty-print validation metrics, including accuracy:
        print("Validation results:")
        for k in ["loss", "accuracy", "auroc", "auprc", "sensitivity", "specificity", "f1"]:
            if k in val_metrics and val_metrics[k] is not None:
                print(f"  {k:12s}: {val_metrics[k]:.4f}")
        print("=====================================================================\n")

        # ---- Metric used for both best-model & early stopping ----
        current_val_metric = val_metrics.get(monitor_metric, None)

        # ---- Save best model based on VALIDATION monitor_metric (accuracy by default) ----
        if checkpoint_path is not None and current_val_metric is not None:
            if current_val_metric > model._best_val_metric + 1e-8:
                print(
                    f"[Best Model Update] {monitor_metric} improved "
                    f"from {model._best_val_metric:.4f} to {current_val_metric:.4f}. "
                    f"Saving checkpoint to {checkpoint_path}"
                )
                model._best_val_metric = current_val_metric
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "threshold": float(val_metrics["threshold"]),
                        "best_metric": float(current_val_metric),
                        "epoch": int(epoch),
                    },
                    checkpoint_path
                )

        # ---- Early stopping based on VALIDATION monitor_metric (maximize) ----
        if early_stopping_on_loss and current_val_metric is not None:
            if current_val_metric > model._best_val_metric_for_es + es_min_delta:
                # improvement
                model._best_val_metric_for_es = current_val_metric
                model._es_no_improve = 0
            else:
                # no improvement
                model._es_no_improve += 1
                print(
                    f"[ES] No improvement in {monitor_metric}: "
                    f"{model._es_no_improve}/{es_patience} "
                    f"(best={model._best_val_metric_for_es:.4f}, current={current_val_metric:.4f})"
                )
                if model._es_no_improve >= es_patience:
                    print(
                        f"\n[Early stopping triggered at epoch {epoch}] "
                        f"Validation {monitor_metric} has not improved for {es_patience} periods.\n"
                    )
                    model._stop_training = True  # main() should check this

    return avg_loss, accuracy

def validate(model, device,criterion, val_loader, threshold=0.5,optimize_threshold=False):
    model.eval()
    val_loss = 0.0
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item() * data.size(0)

            # --- Predictions ---
            probs = F.softmax(output, dim=1)  # (batch, num_classes)
            preds = probs.argmax(dim=1)

            all_targets.append(target.cpu())
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

    # --- Concatenate ---
    all_targets = torch.cat(all_targets).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()

    num_classes = all_probs.shape[1]
    avg_loss = val_loss / len(val_loader.dataset)

    print(f"Validation Loss: {avg_loss:.4f}")

    # -------------------------------
    #       MULTI-CLASS SUPPORT
    # -------------------------------
    if num_classes > 2:
        acc = accuracy_score(all_targets, all_preds)

        # Use macro for fairness across classes
        f1 = f1_score(all_targets, all_preds, average="macro")

        print(f"Accuracy: {acc:.4f}")
        print(f"F1-score (macro): {f1:.4f}")
        print("AUROC: Not applicable (multi-class without one-vs-rest).")
        print("AUPRC: Not applicable (multi-class).")
        print("Sensitivity/Specificity: Binary only.\n")

        return {
            "loss": avg_loss,
            "accuracy": acc,
            "f1": f1,
        }

    # -------------------------------
    #       BINARY CLASSIFICATION
    # -------------------------------
    pos_probs = all_probs[:, 1]

    if optimize_threshold:
        threshold = find_best_threshold(
            all_targets,
            pos_probs
        )

        print(
            f"Best threshold from validation: "
            f"{threshold:.4f}"
        )

    preds_thresh = (
        pos_probs >= threshold
    ).astype(int)

    # Accuracy
    acc = accuracy_score(all_targets, preds_thresh)

    # AUROC / AUPRC
    try:
        auroc = roc_auc_score(all_targets, pos_probs)
    except:
        auroc = float("nan")
    try:
        auprc = average_precision_score(all_targets, pos_probs)
    except:
        auprc = float("nan")

    # F1
    f1 = f1_score(all_targets, preds_thresh)

    # Confusion matrix → TN, FP, FN, TP
    cm = confusion_matrix(
        all_targets,
        preds_thresh,
        labels=[0,1]
    )

    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    # -------------------------------
    #           PRINT METRICS
    # -------------------------------
    print(f"Accuracy:          {acc:.4f}")
    print(f"AUROC:             {auroc:.4f}")
    print(f"AUPRC:             {auprc:.4f}")
    print(f"F1-score:          {f1:.4f}")
    print(f"Sensitivity:       {sensitivity:.4f}")
    print(f"Specificity:       {specificity:.4f}")
    print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}\n")

    return {
    "loss": avg_loss,
    "accuracy": acc,
    "auroc": auroc,
    "auprc": auprc,
    "f1": f1,
    "sensitivity": sensitivity,
    "specificity": specificity,
    "threshold": threshold,   # NEW
    "tp": tp,
    "fp": fp,
    "tn": tn,
    "fn": fn,
}


def get_probs_and_targets(model, device, loader):
    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            probs = F.softmax(output, dim=1)  # (batch, num_classes)

            all_targets.append(target.cpu())
            all_probs.append(probs.cpu())

    all_targets = torch.cat(all_targets).numpy()
    all_probs = torch.cat(all_probs).numpy()
    return all_targets, all_probs


def plot_roc_curve(model, device, loader, save_path="roc_curve.png"):
    # binary case: class 1 = positive
    y_true, probs = get_probs_and_targets(model, device, loader)
    y_scores = probs[:, 1]  # P(y=1)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved ROC curve to: {save_path}  (AUC = {roc_auc:.3f})")
    return roc_auc



def plot_roc_curve(targets, probs, save_path):

    from sklearn.metrics import roc_curve, roc_auc_score

    fpr, tpr, _ = roc_curve(targets, probs)

    auc = roc_auc_score(targets, probs)

    fig, ax = plt.subplots(figsize=(6,5))

    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"AUC = {auc:.3f}"
    )

    ax.plot(
        [0,1],
        [0,1],
        linestyle="--"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")

    ax.legend()

    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        save_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
        dpi=300
    )

    plt.close(fig)

    return auc