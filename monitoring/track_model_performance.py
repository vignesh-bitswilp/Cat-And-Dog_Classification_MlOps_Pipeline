"""
monitoring/track_model_performance.py
--------------------------------------
Post-deployment model performance tracking (Assignment M5, Task 2).

Simulates/collects a small batch of requests with known true labels,
sends them to the live inference service, and computes drift-relevant
metrics (accuracy, per-class breakdown, confusion matrix) so a team can
track how the deployed model performs on fresh traffic over time.

In a real deployment, `collect_batch()` would be replaced with a query
against logged production requests + a source of ground truth (e.g.
labels supplied later by adopters/staff at the pet-adoption platform).
Here we simulate that batch using freshly generated synthetic images
with known labels, and call the same synthetic generators used in the
training pipeline for consistency.

Usage:
    python monitoring/track_model_performance.py --url http://localhost:8000 --n 20
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.generate_synthetic_data import make_cat_image, make_dog_image  # noqa: E402

REPORTS_DIR = ROOT / "monitoring" / "performance_reports"


def collect_batch(n_per_class: int = 10):
    """Simulate a batch of real/labeled traffic: (image_bytes, true_label)."""
    batch = []
    for _ in range(n_per_class):
        buf = io.BytesIO()
        make_cat_image().save(buf, format="JPEG")
        batch.append((buf.getvalue(), "cat"))

        buf = io.BytesIO()
        make_dog_image().save(buf, format="JPEG")
        batch.append((buf.getvalue(), "dog"))
    return batch


def score_batch(base_url: str, batch):
    y_true, y_pred = [], []
    for image_bytes, true_label in batch:
        response = requests.post(
            f"{base_url}/predict",
            files={"file": ("sample.jpg", image_bytes, "image/jpeg")},
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
