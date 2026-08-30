# MLOps Pipeline — Cats vs Dogs Classifier (Pet Adoption Platform)

**Course:** MLOps (S1-25_AIMLCZG523) — Assignment 2
**Use case:** Binary image classification (cat vs dog) for a pet-adoption platform, deployed as an end-to-end MLOps pipeline: model training -> experiment tracking -> packaging -> containerization -> CI -> CD -> monitoring.

This repository implements every module (M1-M5) in the assignment brief against the **real Kaggle Cats and Dogs dataset**. The application code, tests, Docker image, and CI/CD workflows have all been executed and verified in development; the note in the "Verification status" section below is important context on which parts required an actual Kaggle download and which didn't.

---

## 0. Dataset setup (real Kaggle data — no synthetic fallback)

### 0.1 Get Kaggle API credentials
1. Log into Kaggle -> **Account -> Settings -> API -> Create New Token**. This downloads `kaggle.json`.
2. Make it available to the `kaggle` package, either:
   - Place it at `~/.kaggle/kaggle.json` and `chmod 600 ~/.kaggle/kaggle.json`, **or**
   - Export environment variables:
     ```bash
     export KAGGLE_USERNAME=<your-username>
     export KAGGLE_KEY=<your-key>
     ```

### 0.2 Download and organize the dataset
```bash
pip install kaggle
python scripts/download_kaggle_data.py
# By default this pulls the Kaggle dataset "salader/dogs-vs-cats".
# If your assignment links to a different Kaggle dataset/competition, pass its slug:
python scripts/download_kaggle_data.py --dataset <owner>/<dataset-slug>
```
This downloads, unzips, and reorganizes the images into the layout the rest of the pipeline expects:
```
data/raw/cats/*.jpg
data/raw/dogs/*.jpg
```
If your chosen dataset has an unusual folder layout, the `scripts/download_kaggle_data.py` docstring explains how `organize_into_class_folders()` matches folders/filenames and how to adjust it.

### 0.3 Version the raw data with DVC
```bash
dvc add data/raw/cats data/raw/dogs
git add data/raw/cats.dvc data/raw/dogs.dvc data/raw/.gitignore
git commit -m "Track raw Kaggle dataset with DVC"
```

### 0.4 Run the rest of the pipeline
```bash
dvc repro          # preprocess (80/10/10 split) -> train (MLflow-tracked)
dvc push           # push versioned data + trained model to your DVC remote (see 0.5)
```

### 0.5 (For CI/CD) Configure a DVC remote
CI/CD does **not** download from Kaggle on every run (see `.github/workflows/ci.yml`) — it pulls the already-versioned, already-trained model artifact via `dvc pull`. Configure a remote once:
```bash
dvc remote add -d storage s3://your-bucket/cats-dogs-dvc   # or gs://, azure://, or a local path
dvc push
```
Then add the remote's credentials as GitHub Actions repo secrets (e.g. `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` for an S3 remote) so `ci.yml`'s `dvc pull` step can authenticate.

---

## Verification status (what has and hasn't been run end-to-end)

- **Verified in this environment:** all 9 unit tests, the FastAPI service (`/health`, `/predict`, `/metrics`) with a locally trained model, the DVC pipeline mechanics (`dvc repro`/`dvc.yaml` structure), the smoke test script, and the model-performance-tracking script — using a small dataset produced for local testing during development.
- **Not verified in this environment:** the actual Kaggle download and a full training run on the real dataset. The sandbox this repo was built in cannot reach `kaggle.com`. You will need to run Section 0 above yourself (locally or in a training environment with internet access) before `models/model.pt` exists and before `dvc repro`/the FastAPI service/the Docker image have a real trained model to work with.
- Everything downstream of "a `model.pt` file exists" (the API, Docker, CI/CD YAML, monitoring scripts) is dataset-agnostic and does not need to change once you've completed Section 0.

---

## Repository layout

```
mlops-cats-dogs/
├── data/                        # raw + processed images (DVC-tracked, not Git-tracked)
├── models/                      # model.pt, loss_curve.png, confusion_matrix.png (DVC-tracked)
├── mlruns/ , mlflow.db          # MLflow experiment tracking store (local; gitignored)
├── scripts/
│   └── download_kaggle_data.py  # M1.1: fetches + organizes the real Kaggle dataset
├── src/
│   ├── data_preprocessing.py    # M1.1 preprocessing + train/val/test split
│   ├── model.py                 # M1.2 SimpleCNN + shared inference utility
│   └── train.py                 # M1.3 training loop + MLflow logging
├── tests/
│   ├── test_data_preprocessing.py   # M3.1 unit tests
│   └── test_model_utils.py          # M3.1 unit tests
├── app/
│   └── main.py                  # M2.1 FastAPI service (/health, /predict, /metrics)
├── monitoring/
│   ├── smoke_test.sh                 # M4.3 post-deploy smoke test
│   ├── track_model_performance.py    # M5.2 post-deployment performance tracking
│   └── prometheus.yml                # M5.1 Prometheus scrape config (compose stack)
├── k8s/
│   ├── deployment.yaml          # M4.1 Kubernetes Deployment
│   └── service.yaml             # M4.1 Kubernetes Service
├── .github/workflows/
│   ├── ci.yml                   # M3.2/M3.3 CI: test, dvc pull model, build, push image
│   └── cd.yml                   # M4.2/M4.3 CD: deploy to kind cluster, smoke test
├── Dockerfile                   # M2.3
├── docker-compose.yml           # M4.1 alternative deployment target
├── dvc.yaml / dvc.lock          # M1.1 DVC pipeline (data & model versioning)
├── requirements.txt             # M2.2 pinned dependencies
└── .gitignore
```

---

## M1 — Model Development & Experiment Tracking

### 1. Data & code versioning
- **Git** tracks all source code (`src/`, `app/`, `tests/`, configs).
- **DVC** tracks the Kaggle dataset and model artifacts, with a pipeline defined in `dvc.yaml`:

```bash
dvc repro     # runs preprocess -> train (assumes data/raw/cats, data/raw/dogs already exist; see Section 0)
dvc dag       # visualize the pipeline graph
```

There is deliberately **no** `dvc.yaml` stage that downloads the data automatically — Kaggle downloads require per-user credentials, so that step is manual/one-time (Section 0), and only its *output* (`data/raw/cats`, `data/raw/dogs`) is a tracked DVC dependency for the `preprocess` stage. `dvc.lock` then pins the exact command, dependency hashes, and output hashes for `preprocess` and `train`, so `dvc repro` only re-runs a stage when its inputs actually changed.

### 2. Model building
`src/model.py` implements `SimpleCNN`: 3 conv blocks (16->32->64 channels) + adaptive pooling + a small classifier head — a lightweight baseline per the assignment's "at least one baseline model" requirement. Weights are serialized with `torch.save(model.state_dict(), "models/model.pt")` (standard `.pt` format).

### 3. Experiment tracking (MLflow)
`src/train.py` logs, for every run:
- **Params:** epochs, learning rate, batch size, optimizer, model name, image size
- **Metrics per epoch:** train_loss, val_loss, val_accuracy; plus final test_loss/test_accuracy
- **Artifacts:** `loss_curve.png`, `confusion_matrix.png`, the serialized `model.pt`, and an MLflow-native logged model

```bash
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
python -m src.train --epochs 15 --batch-size 32

mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# open http://localhost:5000 to browse runs, metrics, and artifacts
```
On the real ~25,000-image Kaggle dataset, expect this SimpleCNN baseline to reach roughly 80-90% test accuracy after ~15 epochs on CPU/GPU (adjust `--epochs`/`--batch-size` for your hardware and time budget); exact numbers will depend on your run.

---

## M2 — Model Packaging & Containerization

### 1. Inference service
`app/main.py` (FastAPI) exposes:
- `GET /health` -> `{"status": "ok", "model_ready": true}`
- `POST /predict` -> accepts a multipart image file, returns `{"label": "cat"|"dog", "class_index": 0|1, "probabilities": {"cat": ..., "dog": ...}}`
- `GET /metrics` -> Prometheus-format metrics (see M5)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
curl -X POST -F "file=@/path/to/image.jpg" http://localhost:8000/predict
```

### 2. Environment specification
`requirements.txt` pins exact versions of torch, torchvision, mlflow, fastapi, uvicorn, scikit-learn, pillow, matplotlib, prometheus_client, dvc, pytest, httpx, kaggle, and requests.

### 3. Containerization
```bash
docker build -t cats-dogs-inference:latest .
docker run -p 8000:8000 cats-dogs-inference:latest
curl http://localhost:8000/health
curl -X POST -F "file=@sample.jpg" http://localhost:8000/predict
```
The `Dockerfile` uses a slim Python base, installs pinned deps, copies `src/`, `app/`, and the trained `models/` artifact (which must exist locally first — see Section 0), runs as a non-root user, and defines a container `HEALTHCHECK` against `/health`.

---

## M3 — CI Pipeline (`.github/workflows/ci.yml`)

On every push/PR to `main`/`develop`, GitHub Actions:
1. Checks out the repo
2. Installs pinned dependencies
3. Runs **`pytest tests/ -v`** — 9 unit tests covering:
   - preprocessing: output shape, value range/normalization, grayscale/RGBA handling, custom sizes (`tests/test_data_preprocessing.py`)
   - inference utility: output schema, probability normalization, input validation, determinism (`tests/test_model_utils.py`)

   These tests intentionally don't need the real Kaggle images or a trained model: preprocessing tests use tiny in-memory PIL images, and the inference-utility tests validate an untrained model's I/O plumbing (shapes/schema/determinism), not classification accuracy.
4. Pulls the already-trained model artifact via `dvc pull` (requires a DVC remote — see Section 0.5); training on the full dataset is a deliberately separate, on-demand step rather than something every push re-runs
5. Builds the Docker image with Buildx (layer-cached via GitHub Actions cache)
6. On push events, logs into **GitHub Container Registry (GHCR)** and pushes the image tagged by commit SHA, branch, and `latest`

Run the test suite locally:
```bash
pytest tests/ -v
# ============================== 9 passed ==============================
```

---

## M4 — CD Pipeline & Deployment

### 1. Deployment target
Two supported targets are provided:
- **Kubernetes** (`k8s/deployment.yaml`, `k8s/service.yaml`) — a `Deployment` with 2 replicas, resource requests/limits, readiness/liveness probes against `/health`, and rolling-update strategy; exposed via a `ClusterIP` `Service`.
- **Docker Compose** (`docker-compose.yml`) — the inference service plus an optional Prometheus container for local monitoring.

### 2. CD / GitOps flow
`.github/workflows/cd.yml` triggers after CI succeeds on `main` (or manually via `workflow_dispatch`):
1. Spins up a local **kind** Kubernetes cluster inside the runner
2. Pulls the freshly-published image from GHCR
3. Loads it into the kind cluster and applies the Deployment/Service manifests
4. Waits for the rollout to complete

For a Docker Compose target instead:
```bash
docker compose pull      # or: docker compose build
docker compose up -d
```

### 3. Smoke tests / health check
`monitoring/smoke_test.sh` calls `/health`, then calls `/predict` — using a real image from `data/processed/test` if one is present in the checkout, or a plain placeholder image otherwise (this step is a connectivity/contract check, not an accuracy check). It exits non-zero if either call fails, and the CD workflow fails the pipeline on smoke-test failure.

```bash
bash monitoring/smoke_test.sh http://localhost:8000
```

---

## M5 — Monitoring, Logs & Final Submission

### 1. Basic monitoring & logging
`app/main.py`:
- Logs every request's **metadata only** (filename, content-type, predicted label, latency) — never raw image bytes or full payloads, per the "excluding sensitive data" requirement.
- Exposes Prometheus counters (`inference_requests_total{endpoint,status}`) and a latency histogram (`inference_request_latency_seconds{endpoint}`) at `GET /metrics`.
- `monitoring/prometheus.yml` configures a Prometheus instance (wired up in `docker-compose.yml`) to scrape these metrics every 15s.

### 2. Model performance tracking (post-deployment)
`monitoring/track_model_performance.py` samples a small batch of **real, labeled images from the held-out test split** (`data/processed/test/{cats,dogs}`), sends each to the live `/predict` endpoint, and computes accuracy, a full classification report, and a confusion matrix, writing a timestamped JSON report to `monitoring/performance_reports/`. It flags a simple drift-alert if accuracy drops below a configurable threshold. In a production deployment you'd instead sample from logged production requests once ground truth becomes available.

```bash
python monitoring/track_model_performance.py --url http://localhost:8000 --n 20
```

---

## Quickstart — full pipeline

```bash
# 0. One-time: get Kaggle data (see Section 0 for credentials setup)
pip install -r requirements.txt
python scripts/download_kaggle_data.py
dvc add data/raw/cats data/raw/dogs

# 1. Data -> preprocessing -> training (via DVC, reproducible)
dvc repro

# 2. Run unit tests
pytest tests/ -v

# 3. Serve the model
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
sleep 3
curl http://localhost:8000/health

# 4. Smoke test
bash monitoring/smoke_test.sh http://localhost:8000

# 5. Post-deploy performance tracking
python monitoring/track_model_performance.py --url http://localhost:8000 --n 10

# 6. Containerize
docker build -t cats-dogs-inference:latest .
docker run -d -p 8001:8000 --name cats-dogs-inference cats-dogs-inference:latest
curl http://localhost:8001/health

# 7. (Optional) docker-compose stack with Prometheus
docker compose up -d

# 8. Push versioned data + model so CI/CD can `dvc pull` them
dvc push
```

---

## Notes on the screen-recording deliverable

A <5 minute recording should demonstrate, in order:
1. A code change (e.g. tweak `SimpleCNN` or a hyperparameter in `src/train.py`)
2. `git commit` + `git push` triggering **CI** (GitHub Actions tab: tests passing, model pulled via DVC, image built & pushed to GHCR)
3. **CD** workflow deploying to the kind cluster (or `docker compose up -d` for the Compose target) and the smoke test step going green
4. A live `curl`/Postman call to the deployed `/predict` endpoint showing a real prediction
5. (Optional) `mlflow ui` showing the new run, `/metrics` showing incremented request counters, and `python monitoring/track_model_performance.py` showing a fresh accuracy report

---

## Design notes & assumptions

- **Framework choice:** PyTorch (over TensorFlow/Keras) for the CNN — used consistently across `model.py`, `train.py`, and the FastAPI service.
- **Experiment tracker:** MLflow with a SQLite backend (`sqlite:///mlflow.db`) rather than the legacy file store, since MLflow 3.x deprecated `./mlruns` file-store writes in favor of a DB backend.
- **Registry:** GitHub Container Registry (GHCR) is used in `ci.yml`/`cd.yml` since it requires no extra secrets beyond the built-in `GITHUB_TOKEN`; swapping to Docker Hub only requires changing the `REGISTRY`/login step.
- **CD deployment target:** a local **kind** cluster inside the CD workflow keeps the pipeline fully self-contained and runnable without a real cloud cluster; `k8s/*.yaml` manifests work unmodified against any real Kubernetes cluster by pointing `kubectl` at it instead.
- **Kaggle dataset slug:** `scripts/download_kaggle_data.py` defaults to `salader/dogs-vs-cats` (a mirror of the classic Dogs vs Cats dataset, pre-split into per-class folders). If your assignment links to a different Kaggle dataset, pass `--dataset <owner>/<slug>` — no other file needs to change since `data_preprocessing.py`, `train.py`, and the FastAPI service only assume the canonical `data/raw/<class>/*.jpg` layout that the download script produces.
