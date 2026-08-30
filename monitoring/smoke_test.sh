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
TMP_IMAGE="$(mktemp /tmp/smoke_test_image_XXXX.jpg)"

cleanup() {
  rm -f "$TMP_IMAGE"
}
trap cleanup EXIT

echo "== Smoke test: $BASE_URL =="

# 1. Health check
echo "-> Checking /health ..."
HEALTH_RESPONSE=$(curl -sf -w "\n%{http_code}" "$BASE_URL/health")
HTTP_CODE=$(echo "$HEALTH_RESPONSE" | tail -n1)
BODY=$(echo "$HEALTH_RESPONSE" | sed '$d')

if [ "$HTTP_CODE" != "200" ]; then
  echo "FAIL: /health returned HTTP $HTTP_CODE"
  exit 1
fi
echo "   /health OK: $BODY"

# 2. Prediction call using a tiny generated test image
echo "-> Generating test image and checking /predict ..."
python3 - "$TMP_IMAGE" <<'PY'
import sys
from PIL import Image
img = Image.new("RGB", (224, 224), color=(120, 80, 40))
img.save(sys.argv[1])
PY

PREDICT_RESPONSE=$(curl -sf -w "\n%{http_code}" -X POST -F "file=@${TMP_IMAGE}" "$BASE_URL/predict")
HTTP_CODE=$(echo "$PREDICT_RESPONSE" | tail -n1)
BODY=$(echo "$PREDICT_RESPONSE" | sed '$d')

if [ "$HTTP_CODE" != "200" ]; then
  echo "FAIL: /predict returned HTTP $HTTP_CODE"
  exit 1
fi
echo "   /predict OK: $BODY"

echo "== All smoke tests passed =="
