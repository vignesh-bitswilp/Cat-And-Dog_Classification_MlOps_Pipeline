#!/usr/bin/env bash
# smoke_test.sh
# --------------
# Post-deploy smoke test (Assignment M4, Task 3).
# Calls the health endpoint and one prediction call; exits non-zero
# (failing the CD pipeline) if either check fails.
#
# Usage: bash monitoring/smoke_test.sh [BASE_URL]
#   BASE_URL defaults to http://localhost:8000

set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLACEHOLDER_IMAGE="$(mktemp /tmp/smoke_test_image_XXXX.jpg)"
TMP_IMAGE="$PLACEHOLDER_IMAGE"

cleanup() {
  # Only remove the placeholder we created ourselves -- never delete a
  # real sample image from data/processed/test if that's what got used.
  rm -f "$PLACEHOLDER_IMAGE"
}
trap cleanup EXIT

echo "== Smoke test: $BASE_URL =="

# 1. Health check
# (no -f on curl: we want to inspect non-2xx responses ourselves below
# rather than have `set -e` kill the script before printing a clear message)
echo "-> Checking /health ..."
HEALTH_RESPONSE=$(curl -s -w "\n%{http_code}" "$BASE_URL/health")
HTTP_CODE=$(echo "$HEALTH_RESPONSE" | tail -n1)
BODY=$(echo "$HEALTH_RESPONSE" | sed '$d')

if [ "$HTTP_CODE" != "200" ]; then
  echo "FAIL: /health returned HTTP $HTTP_CODE"
  exit 1
fi
echo "   /health OK: $BODY"

# 2. Prediction call
# Prefer a real image from the held-out test split if this checkout has
# one (e.g. running in CI right after the data pipeline); otherwise fall
# back to a plain solid-colour placeholder. Either way this step only
# checks that /predict responds successfully with a well-formed payload
# -- it is a connectivity/contract check, not a classification-accuracy
# check (see monitoring/track_model_performance.py for that).
echo "-> Locating a test image and checking /predict ..."
REAL_SAMPLE=$(find "$SCRIPT_DIR/../data/processed/test" -type f \( -iname "*.jpg" -o -iname "*.jpeg" \) 2>/dev/null | head -n 1 || true)

if [ -n "${REAL_SAMPLE:-}" ]; then
  echo "   Using real sample image: $REAL_SAMPLE"
  TMP_IMAGE="$REAL_SAMPLE"
else
  echo "   No processed test images found -- using a placeholder image instead."
  python3 - "$TMP_IMAGE" <<'PY'
import sys
from PIL import Image
img = Image.new("RGB", (224, 224), color=(120, 80, 40))
img.save(sys.argv[1])
PY
fi

PREDICT_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST -F "file=@${TMP_IMAGE}" "$BASE_URL/predict")
HTTP_CODE=$(echo "$PREDICT_RESPONSE" | tail -n1)
BODY=$(echo "$PREDICT_RESPONSE" | sed '$d')

if [ "$HTTP_CODE" != "200" ]; then
  echo "FAIL: /predict returned HTTP $HTTP_CODE"
  exit 1
fi
echo "   /predict OK: $BODY"

echo "== All smoke tests passed =="
