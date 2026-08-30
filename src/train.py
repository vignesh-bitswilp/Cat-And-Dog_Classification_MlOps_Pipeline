"""
train.py
--------
Trains the SimpleCNN baseline on the Cats vs Dogs dataset and logs the
full run (params, metrics, loss curves, confusion matrix, model artifact)
to MLflow (Assignment M1, Task 3).

Usage:
    python src/data_preprocessing.py     # build train/val/test split
    python src/train.py                  # train + log to MLflow + save model
    mlflow ui --port 5000                # inspect runs in the browser
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.data_preprocessing import get_dataloaders
from src.model import SimpleCNN

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy, all_preds, all_labels


def train(
    epochs: int = 5,
    lr: float = 1e-3,
    batch_size: int = 16,
    experiment_name: str = "cats-vs-dogs-baseline",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODELS_DIR.mkdir(exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params(
            {
                "epochs": epochs,
                "learning_rate": lr,
                "batch_size": batch_size,
                "optimizer": "Adam",
                "model": "SimpleCNN",
                "img_size": 224,
            }
        )

        train_losses, val_losses, val_accs = [], [], []

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss, seen = 0.0, 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                seen += images.size(0)

            train_loss = running_loss / max(seen, 1)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            print(f"Epoch {epoch}/{epochs} - train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # Final test evaluation
        test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)
        print(f"Test: loss={test_loss:.4f} acc={test_acc:.4f}")

        # --- Artifact: loss curve plot ---
        fig, ax = plt.subplots()
        ax.plot(range(1, epochs + 1), train_losses, label="train_loss")
        ax.plot(range(1, epochs + 1), val_losses, label="val_loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training / Validation Loss")
        ax.legend()
        loss_curve_path = MODELS_DIR / "loss_curve.png"
        fig.savefig(loss_curve_path)
        plt.close(fig)
        mlflow.log_artifact(str(loss_curve_path))

        # --- Artifact: confusion matrix ---
        cm = confusion_matrix(test_labels, test_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["cat", "dog"])
        fig2, ax2 = plt.subplots()
        disp.plot(ax=ax2)
        cm_path = MODELS_DIR / "confusion_matrix.png"
        fig2.savefig(cm_path)
        plt.close(fig2)
        mlflow.log_artifact(str(cm_path))

        # --- Save serialized model (.pt) ---
        model_path = MODELS_DIR / "model.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(str(model_path))

        # Log an MLflow-native model too, with an input example so MLflow
        # can capture the model signature (required by newer MLflow versions).
        example_input = torch.randn(1, 3, 224, 224)
        mlflow.pytorch.log_model(model, name="pytorch-model", input_example=example_input.numpy())

        print(f"Model saved to {model_path}")
        print(f"MLflow run logged under experiment '{experiment_name}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    train(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
