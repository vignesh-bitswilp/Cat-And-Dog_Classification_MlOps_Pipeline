"""
monitoring/track_model_performance.py
--------------------------------------
Post-deployment model performance tracking (Assignment M5, Task 2).

Collects a small batch of requests with known true labels and sends them
to the live inference service, then computes accuracy, a per-class
report, and a confusion matrix -- so a team can track how the deployed
model performs on fresh traffic over time and catch drift early.

WHERE THE LABELED BATCH COMES FROM
------------------------------------
This script samples real, already-labeled images from the held-out test
split (data/processed/test/{cats,dogs}), which is a reasonable stand-in
for "a small batch of real or simulated requests with true labels" per
the assignment. In a production deployment you would instead:
  - Sample from logged production requests once ground truth becomes
    available (e.g. an adopter/staff member confirms the actual species), or
  - Periodically re-score a fixed, curated labeled holdout set.

Usage:
    python monitoring/track_model_performance.py --url http://localhost:8000 --n 20
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TEST_DIR = ROOT / "data" / "processed" / "test"
REPORTS_DIR = ROOT / "monitoring" / "performance_reports"
CLASSES = ["cats", "dogs"]
LABEL_MAP = {"cats": "cat", "dogs": "dog"}  # folder name -> model output label


def collect_batch(n_per_class: int, seed: int = 42):
    """
    Sample `n_per_class` real, labeled images per class from the held-out
    test split. Returns a list of (image_path, true_label) tuples.
    """
    if not TEST_DIR.exists():
        sys.exit(
            f"Test split not found at {TEST_DIR}. Run the data pipeline first:\n"
            "  python scripts/download_kaggle_data.py\n"
            "  dvc add data/raw/cats data/raw/dogs\n"
            "  python -m src.data_preprocessing"
        )

    random.seed(seed)
    batch = []
    for cls in CLASSES:
        files = sorted((TEST_DIR / cls).glob("*.jpg"))
        if not files:
            sys.exit(f"No test images found under {TEST_DIR / cls}.")
        sample = random.sample(files, min(n_per_class, len(files)))
        batch.extend((f, LABEL_MAP[cls]) for f in sample)
    return batch


def score_batch(base_url: str, batch):
    y_true, y_pred = [], []
    for image_path, true_label in batch:
        with open(image_path, "rb") as f:
            response = requests.post(
                f"{base_url}/predict",
                files={"file": (image_path.name, f, "image/jpeg")},
                timeout=10,
            )
        response.raise_for_status()
        pred_label = response.json()["label"]
        y_true.append(true_label)
        y_pred.append(pred_label)
    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=10, help="images per class")
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    batch = collect_batch(args.n)
    y_true, y_pred = score_batch(args.url, batch)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=["cat", "dog"]).tolist()

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(y_true),
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "labels_order": ["cat", "dog"],
        "source": "held-out test split (data/processed/test)",
    }

    out_path = REPORTS_DIR / f"perf_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(f"Post-deployment accuracy on {len(y_true)} samples: {acc:.3f}")
    print(f"Report written to {out_path}")

    # Simple alerting threshold example: flag if accuracy drops below 0.6
    if acc < 0.6:
        print("WARNING: accuracy below 0.6 threshold -- possible model drift.")


if __name__ == "__main__":
    main()
