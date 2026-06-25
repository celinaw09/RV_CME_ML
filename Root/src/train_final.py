import os
import sys
import json
import time
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.train import train
from utils.misc_utils import build_classification_dataset, build_resnet_for_grayscale
from dataset.dataset import EyeFFEDataset


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


EXPERIMENT_ROOT = "experiment_logs"
CHECKPOINT_DIR = "checkpoint_dir"


def reset_random_seed(seed: int = 0):
    g = torch.Generator()
    g.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return g


def make_transforms(image_size: int):
    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.289], std=[0.146]),
    ])

    return train_transforms


def save_hyperparameters(out_dir: str, hyperparameters: dict):
    os.makedirs(out_dir, exist_ok=True)
    hparam_path = os.path.join(out_dir, "hyperparameters.json")
    with open(hparam_path, "w") as f:
        json.dump(hyperparameters, f, indent=4)


def setup_experiment_folders():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(EXPERIMENT_ROOT, f"final_training_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return run_dir


def freeze_model_backbone(model):
    # Freeze everything first, then unfreeze layer4 and fc only.
    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True


def build_dataloader(df, image_size: int, batch_size: int):
    train_transform = make_transforms(image_size)
    dataset = EyeFFEDataset(df, transform=train_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader


def configure_logger(run_dir: str):
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "console_output.txt")
    log_file = open(log_path, "w")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    return log_file


def main():
    reset_random_seed(0)

    print(torch.__version__)
    gpu = 0
    if torch.cuda.is_available():
        try:
            device = torch.device(f"cuda:{gpu}")
            _ = torch.cuda.get_device_name(device)
        except AssertionError:
            print(f"GPU {gpu} is not available. Falling back to GPU 0.")
            device = torch.device("cuda:0")
    else:
        print("CUDA is not available. Using CPU.")
        device = torch.device("cpu")

    run_dir = setup_experiment_folders()
    log_file = configure_logger(run_dir)

    hyperparameters = {
        "learning_rate": 1e-4,
        "image_size": 224,
        "augmentation": "basic",
        "batch_size": 16,
        "weight_decay": 1e-3,
        "epochs": 30,
    }
    save_hyperparameters(run_dir, hyperparameters)

    print("=" * 60)
    print(f"Experiment folder: {run_dir}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")
    print("=" * 60)

    root_dir = "/data2/users/koushani/chbmit/data/allpatients_resized"
    df_final = build_classification_dataset(root_dir)

    label_mapping = {"non_CME": 0, "CME": 1}
    df_final["label"] = df_final["label"].map(label_mapping)

    print("\nFinal dataset summary")
    print(f"Images   : {len(df_final)}")
    print(f"Patients : {df_final['patient_id'].nunique()}")
    print("Label distribution:")
    print(df_final["label"].value_counts())

    train_loader = build_dataloader(df_final, hyperparameters["image_size"], hyperparameters["batch_size"])

    model = build_resnet_for_grayscale(num_classes=2)
    freeze_model_backbone(model)
    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("\n=== TRAINABLE PARAMETERS ===")
    for name, param in model.named_parameters():
        print(f"{name:50s} {param.requires_grad}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Total params: {total_params:,}")
    print(f"Percent trainable: {100 * trainable_params / total_params:.2f}%")

    train_counts = df_final["label"].value_counts()
    num_non_cme = int(train_counts[0])
    num_cme = int(train_counts[1])

    class_weights = torch.tensor([
        len(df_final) / (2 * num_non_cme),
        len(df_final) / (2 * num_cme),
    ], dtype=torch.float32).to(device)
    print("Class weights:", class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=hyperparameters["learning_rate"],
        weight_decay=hyperparameters["weight_decay"],
    )

    num_epochs = hyperparameters["epochs"]
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train(
            model,
            device,
            train_loader,
            optimizer,
            criterion,
            epoch,
            log_interval=10,
            val_loader=None,
            validate_every_epochs=None,
            val_threshold=0.5,
            early_stopping_on_loss=False,
            es_patience=10,
            es_min_delta=1e-4,
            checkpoint_path=None,
            monitor_metric="auroc",
        )

    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "learning_rate": hyperparameters["learning_rate"],
            "image_size": hyperparameters["image_size"],
            "augmentation": hyperparameters["augmentation"],
            "epochs": hyperparameters["epochs"],
        },
        checkpoint_path,
    )

    copy_path = os.path.join(run_dir, "best_model.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "learning_rate": hyperparameters["learning_rate"],
            "image_size": hyperparameters["image_size"],
            "augmentation": hyperparameters["augmentation"],
            "epochs": hyperparameters["epochs"],
        },
        copy_path,
    )

    print("\nTraining complete.")
    print(f"Final model saved to: {checkpoint_path}")
    print(f"Duplicate model saved to: {copy_path}")

    log_file.close()


if __name__ == "__main__":
    main()
