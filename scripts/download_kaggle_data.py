"""
download_kaggle_data.py
------------------------
Downloads the real "Cats and Dogs" dataset from Kaggle and organizes it
into the folder layout expected by the rest of the pipeline:

    data/raw/cats/*.jpg
    data/raw/dogs/*.jpg

PREREQUISITES
-------------
1. Install the Kaggle CLI/SDK:
       pip install kaggle

2. Get a Kaggle API token:
       Kaggle account -> Settings -> API -> "Create New Token"
       This downloads a `kaggle.json` file.

3. Make the credentials available, either:
       a) Place the file at ~/.kaggle/kaggle.json  (chmod 600), or
       b) Export environment variables:
              export KAGGLE_USERNAME=<your-username>
              export KAGGLE_KEY=<your-key>

USAGE
-----
    python scripts/download_kaggle_data.py
    python scripts/download_kaggle_data.py --dataset salader/dogs-vs-cats

By default this pulls the "salader/dogs-vs-cats" Kaggle dataset (a mirror
of the classic Dogs vs Cats dataset, already split into train/cat and
train/dog folders with ~12,500 images per class). If your course points
to a different Kaggle dataset/competition, pass its slug via --dataset
(competition slugs use --competition instead -- see the Kaggle CLI docs).

WHAT THIS SCRIPT DOES
----------------------
1. Downloads and unzips the dataset into a temporary folder under data/raw/_kaggle_download
2. Walks the extracted contents looking for cat/dog images (by folder name
   containing "cat"/"dog", case-insensitively -- this covers the common
   Kaggle layouts such as `train/cats`, `train/Cat`, `PetImages/Cat`, etc.)
3. Copies (not moves) them into the canonical data/raw/cats and
   data/raw/dogs folders that src/data_preprocessing.py expects
4. Leaves the raw download in data/raw/_kaggle_download in case you want
   to inspect it; delete it manually once you're happy with the result

After running this script, version the raw data with DVC:
    dvc add data/raw/cats data/raw/dogs
    git add data/raw/cats.dvc data/raw/dogs.dvc data/raw/.gitignore
    git commit -m "Track raw Kaggle dataset with DVC"
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
DOWNLOAD_DIR = RAW_DIR / "_kaggle_download"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def download_dataset(dataset_slug: str) -> Path:
    """Download and unzip a Kaggle dataset via the Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise SystemExit(
            "The 'kaggle' package is not installed. Run: pip install kaggle"
        ) from exc

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()  # reads ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY env vars

    print(f"Downloading Kaggle dataset '{dataset_slug}' ...")
    api.dataset_download_files(dataset_slug, path=str(DOWNLOAD_DIR), unzip=False, quiet=False)

    # Unzip whatever archive(s) were downloaded
    for zip_path in DOWNLOAD_DIR.glob("*.zip"):
        print(f"Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DOWNLOAD_DIR)
        zip_path.unlink()

    return DOWNLOAD_DIR


def organize_into_class_folders(source_dir: Path, raw_dir: Path = RAW_DIR) -> None:
    """
    Walk `source_dir` and copy every image found under a folder whose name
    contains "cat" or "dog" into data/raw/cats or data/raw/dogs respectively.
    """
    cats_dir = raw_dir / "cats"
    dogs_dir = raw_dir / "dogs"
    cats_dir.mkdir(parents=True, exist_ok=True)
    dogs_dir.mkdir(parents=True, exist_ok=True)

    n_cats, n_dogs, n_skipped = 0, 0, 0

    for path in source_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        parent_chain = " ".join(p.name.lower() for p in path.parents)
        filename_lower = path.name.lower()

        if "cat" in parent_chain or filename_lower.startswith("cat"):
            dest = cats_dir / f"cat_{n_cats:05d}{path.suffix.lower()}"
            shutil.copy(path, dest)
            n_cats += 1
        elif "dog" in parent_chain or filename_lower.startswith("dog"):
            dest = dogs_dir / f"dog_{n_dogs:05d}{path.suffix.lower()}"
            shutil.copy(path, dest)
            n_dogs += 1
        else:
            n_skipped += 1

    print(f"Organized {n_cats} cat images and {n_dogs} dog images into {raw_dir}")
    if n_skipped:
        print(
            f"Skipped {n_skipped} files that didn't match a cat/dog folder or "
            f"filename pattern -- inspect {source_dir} if this looks wrong."
        )
    if n_cats == 0 or n_dogs == 0:
        print(
            "WARNING: one of the classes has zero images. The Kaggle dataset's "
            "folder layout may differ from what this script expects -- open "
            f"{source_dir} and adjust organize_into_class_folders() accordingly."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="salader/dogs-vs-cats",
        help="Kaggle dataset slug, e.g. 'salader/dogs-vs-cats' (default) or 'shaunthesheep/microsoft-catsvsdogs-dataset'",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the download step and only (re-)organize an already-extracted folder",
    )
    args = parser.parse_args()

    if not args.skip_download:
        download_dataset(args.dataset)
    else:
        if not DOWNLOAD_DIR.exists():
            sys.exit(f"--skip-download was set but {DOWNLOAD_DIR} does not exist.")

    organize_into_class_folders(DOWNLOAD_DIR)


if __name__ == "__main__":
    main()
