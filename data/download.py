"""Download the Kaggle apparel dataset (paramaggarwal/fashion-product-images-dataset).

Run with: uv run python -m data.download

Only `styles.csv` (a few MB) is fetched — the ~600 MB image archive is not
needed for this demo (every attribute we use lives in the CSV / in
`productDisplayName`).

Kaggle CLI 2.2.3 quirks (verified on this machine):
  - Auth file is ~/.kaggle/access_token (a single-line KGAT_... token),
    NOT the legacy ~/.kaggle/kaggle.json. We do not write kaggle.json.
  - `kaggle datasets download --unzip` does NOT actually unzip in 2.2.3,
    so we call zipfile.ZipFile().extractall() ourselves and delete the zip.
  - The file path inside the dataset is `fashion-dataset/styles.csv`
    (passed as `-f fashion-dataset/styles.csv`).
  - If data/styles.csv already exists, the download step is a no-op.

License: MIT (confirmed via Kaggle dataset metadata). Attribution:
paramaggarwal on Kaggle, sourced from myntra.com.

The full styles.csv is gitignored; a ~500-row deterministic slice filtered
to Apparel + Footwear is committed as data/sample.parquet.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DATASET = "paramaggarwal/fashion-product-images-dataset"
REMOTE_FILE = "fashion-dataset/styles.csv"
HERE = Path(__file__).resolve().parent
DEST = HERE / "styles.csv"
TOKEN = Path.home() / ".kaggle" / "access_token"

# TODO(fallback): amazon-reviews-2023 loader — if Aryan switches the data
# source, implement a loader for McAuley-Lab/Amazon-Reviews-2023 subset
# raw_meta_Clothing_Shoes_and_Jewelry here and adapt the column mapping.


def _data_dir() -> Path:
    """Resolve the data dir, honoring SUBSTITUTES_AGENT_DATA_DIR if set."""
    env = os.environ.get("SUBSTITUTES_AGENT_DATA_DIR")
    return Path(env).resolve() if env else HERE


def _kaggle_available() -> bool:
    """True if the kaggle CLI is on PATH and an access_token is present."""
    if not shutil.which("kaggle"):
        return False
    return TOKEN.exists()


def download(dest: Path | None = None) -> Path:
    """Ensure styles.csv is on disk. No-op if already present.

    Degrades gracefully: if the kaggle CLI or token is missing, prints a
    clear, actionable error and exits non-zero (the deterministic pipeline
    refuses to run without its input data).
    """
    target = Path(dest) if dest else _data_dir() / "styles.csv"
    if target.exists() and target.stat().st_size > 0:
        print(f"found cached styles.csv, skipping ({target.stat().st_size} bytes)")
        return target

    if not _kaggle_available():
        print(
            "error: styles.csv not found and kaggle CLI is not usable.\n"
            "  Install the Kaggle CLI (pip install kaggle) and place a valid\n"
            "  access token at ~/.kaggle/access_token (CLI 2.x format, a\n"
            "  single KGAT_... line), then re-run `substitutes-agent download`.\n"
            "  Alternatively, set SUBSTITUTES_AGENT_DATA_DIR to a directory\n"
            "  that already contains styles.csv.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    target.parent.mkdir(parents=True, exist_ok=True)
    zip_path = target.with_suffix(".zip")
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET,
        "-f",
        REMOTE_FILE,
        "-p",
        str(target.parent),
    ]
    print(f"downloading {REMOTE_FILE} from {DATASET} -> {target}")
    subprocess.run(cmd, check=True)

    if not zip_path.exists():
        # Some CLI versions name the file after the remote basename + .zip.
        cand = target.parent / f"{Path(REMOTE_FILE).name}.zip"
        if cand.exists():
            zip_path = cand
    if not zip_path.exists():
        raise SystemExit(f"download command succeeded but no zip found at {zip_path}")

    # CLI 2.2.3's --unzip is a no-op, so extract manually.
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target.parent)
    zip_path.unlink()

    # The extracted member is named styles.csv; rename if needed.
    extracted = target.parent / Path(REMOTE_FILE).name
    if extracted != target and extracted.exists():
        shutil.move(str(extracted), target)

    if not target.exists():
        raise SystemExit(f"extraction finished but {target} is missing")
    print(f"done: {target} ({target.stat().st_size} bytes)")
    return target


def main() -> None:
    """CLI entry: download styles.csv."""
    arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    download(arg)


if __name__ == "__main__":
    main()
