"""Download the Open Beauty Facts beauty.parquet dataset (resume-friendly).

Run with: uv run python -m data.download

The full file (~52 MB) is gitignored. A deterministic ~500-row slice
filtered to face-cream categories is committed as data/sample.parquet.
"""

from __future__ import annotations

import hashlib
import sys
import time
import urllib.request
from pathlib import Path

URL = (
    "https://huggingface.co/datasets/openfoodfacts/"
    "product-database/resolve/main/beauty.parquet"
)
HERE = Path(__file__).resolve().parent
DEST = HERE / "beauty.parquet"
CHUNK = 1 << 16  # 64 KiB
USER_AGENT = "substitutes-agent-demo/0.1 (portfolio; contact: see repo)"


def _remote_size() -> int | None:
    req = urllib.request.Request(URL, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            cl = resp.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not determine remote size ({exc})", file=sys.stderr)
        return None


def download(dest: Path = DEST) -> Path:
    """Download `URL` to `dest`, resuming if a partial file exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    remote = _remote_size()
    existing = dest.stat().st_size if dest.exists() else 0

    if remote and existing == remote:
        print(f"already complete: {dest} ({existing} bytes)")
        return dest

    headers = {"User-Agent": USER_AGENT}
    mode = "ab" if existing else "wb"
    if existing:
        headers["Range"] = f"bytes={existing}-"
        print(f"resuming at byte {existing} / {remote or '?'}")
    else:
        print(f"downloading {URL} -> {dest}")

    req = urllib.request.Request(URL, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, mode) as fh:  # noqa: S310
        start = existing
        t0 = time.monotonic()
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            fh.write(chunk)
            start += len(chunk)
            if remote:
                pct = 100.0 * start / remote
                rate = start / (time.monotonic() - t0 + 1e-9) / (1 << 20)
                print(f"  {start}/{remote} ({pct:.1f}%) @ {rate:.2f} MiB/s", end="\r")
        print()

    final = dest.stat().st_size
    if remote and final != remote:
        raise SystemExit(f"size mismatch after download: {final} != {remote}")
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    print(f"done: {dest} ({final} bytes, sha256={digest[:12]})")
    return dest


def main() -> None:
    """CLI entry: download beauty.parquet."""
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEST
    download(target)


if __name__ == "__main__":
    main()
