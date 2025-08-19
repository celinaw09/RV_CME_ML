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
        

def train_and_validate(
    model, train_loader, val_loader, criterion, optimizer, device,
    num_epochs=50, eval_interval=10, out_dir="checkpoints"
):
    os.makedirs(out_dir, exist_ok=True)

    # History dictionary
    history = {
        "train_losses": [],
        "val_losses": [],
        "train_accs": [],
        "val_accs": [],
        "train_roc_auc": [],
        "val_roc_auc": [],
        "train_pr_auc": [],
        "val_pr_auc": [],
    }

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        avg_train_loss = running_loss / len(train_loader)
        history["train_losses"].append(avg_train_loss)
        history["train_accs"].append(train_acc)

        print(f"[Epoch {epoch}/{num_epochs}] Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # Evaluate every eval_interval epochs
        if epoch % eval_interval == 0 or epoch == num_epochs:
            model.eval()
            with torch.no_grad():
                train_loss, train_acc, train_roc_auc, train_pr_auc = evaluate_loader(model, train_loader, criterion=criterion, device=device)
                val_loss, val_acc, val_roc_auc, val_pr_auc = evaluate_loader(model, val_loader, criterion=criterion, device=device)

            # Record metrics
            history["val_losses"].append(val_loss)
            history["train_losses"].append(train_loss)
            history["train_accs"].append(train_acc)
            history["val_accs"].append(val_acc)
            history["train_roc_auc"].append(train_roc_auc)
            history["val_roc_auc"].append(val_roc_auc)
            history["train_pr_auc"].append(train_pr_auc)
            history["val_pr_auc"].append(val_pr_auc)

            print(f"  ▶ Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                  f"ROC AUC: {val_roc_auc:.4f}, PR AUC: {val_pr_auc:.4f}")

            # Save checkpoint
            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                **history
            }
            torch.save(ckpt, os.path.join(out_dir, f"checkpoint_epoch_{epoch}.pt"))


def evaluate_loader(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_targets = []
    total = 0
    correct = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets.long())

            _, predicted = torch.max(outputs.data, 1)

            # 🔧 Convert one-hot or multi-label targets to class indices if needed
            if targets.ndim > 1:
                targets = torch.argmax(targets, dim=1)

            total_loss += loss.item()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())  # fixed

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_targets, all_preds)

    try:
        roc_auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        roc_auc = float('nan')  # Cannot compute AUC if only one class present

    try:
        precision, recall, _ = precision_recall_curve(all_targets, all_probs)
        pr_auc = auc(recall, precision)
    except ValueError:
        pr_auc = float('nan')  # Cannot compute PR AUC if only one class present

    return avg_loss, acc * 100.0, roc_auc, pr_auc


def plot_epoch_curves(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    def save_plot(x_vals, y_vals_list, labels, ylabel, title, filename):
        plt.figure()
        for y_vals, label in zip(y_vals_list, labels):
            plt.plot(x_vals[:len(y_vals)], y_vals, label=label, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename))
        plt.close()

    # Epochs
    eval_epochs = list(range(len(history['val_losses'])))
    train_epochs = list(range(len(history['train_losses'])))

    # Loss
    save_plot(
        train_epochs,
        [history["train_losses"]],
        ["Train Loss"],
        "Loss",
        "Training Loss",
        "train_loss.png"
    )
    save_plot(
        eval_epochs,
        [history["val_losses"]],
        ["Val Loss"],
        "Loss",
        "Validation Loss",
        "val_loss.png"
    )

    # Accuracy
    save_plot(
        train_epochs,
        [history["train_accs"]],
        ["Train Accuracy"],
        "Accuracy (%)",
        "Training Accuracy",
        "train_acc.png"
    )
    save_plot(
        eval_epochs,
        [history["val_accs"]],
        ["Val Accuracy"],
        "Accuracy (%)",
        "Validation Accuracy",
        "val_acc.png"
    )

    # ROC AUC
    save_plot(
        eval_epochs,
        [history["train_roc_auc"], history["val_roc_auc"]],
        ["Train ROC AUC", "Val ROC AUC"],
        "ROC AUC",
        "ROC AUC over Epochs",
        "roc_auc.png"
    )

    # PR AUC
    save_plot(
        eval_epochs,
        [history["train_pr_auc"], history["val_pr_auc"]],
        ["Train PR AUC", "Val PR AUC"],
        "PR AUC",
        "PR AUC over Epochs",
        "pr_auc.png"
    )
