from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.scifact.dataset import (  # noqa: E402
    SciFactDataError,
    build_manifest,
    validate_manifest,
)

DATA_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
EXPECTED_ARCHIVE_SHA256 = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"
ALLENAI_REPOSITORY_REVISION = "68b98a56d93e0f9da0d2aab4e6c3294699a0f72e"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and validate a pinned SciFact snapshot")
    parser.add_argument("--root", default="evals/data/scifact")
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    data_dir = root / "data"
    archive = root / "data.tar.gz"
    manifest_path = root / "manifest.json"
    root.mkdir(parents=True, exist_ok=True)
    if manifest_path.is_file():
        try:
            validate_manifest(manifest_path)
        except SciFactDataError as exc:
            print(f"DATASET_MANIFEST=FAIL reason={exc}")
            return 1
        print(f"DATASET_MANIFEST=PASS path={manifest_path}")
        return 0
    if args.download:
        if archive.is_file():
            print(f"DATASET_ARCHIVE=EXISTS path={archive}")
        else:
            with urlopen(DATA_URL, timeout=30) as response:
                archive.write_bytes(response.read())
            print(f"DATASET_ARCHIVE=DOWNLOADED path={archive}")
        from evals.scifact.dataset import sha256_file

        archive_hash = sha256_file(archive)
        if archive_hash != EXPECTED_ARCHIVE_SHA256:
            print("DATASET_ARCHIVE=FAIL reason=immutable archive hash mismatch")
            return 1
        if not data_dir.is_dir():
            with tarfile.open(archive, "r:gz") as handle:
                handle.extractall(root, filter="data")
        if not data_dir.is_dir():
            print("DATASET_MANIFEST=FAIL reason=archive did not produce data directory")
            return 1
    if not data_dir.is_dir():
        print("DATASET_MANIFEST=FAIL reason=missing data directory; use --download")
        return 1
    manifest = build_manifest(
        data_dir=data_dir,
        archive_path=archive if archive.is_file() else None,
        source="https://github.com/allenai/scifact plus official release data.tar.gz",
        revision=(
            f"allenai/scifact@{ALLENAI_REPOSITORY_REVISION}; "
            f"release archive sha256={EXPECTED_ARCHIVE_SHA256}"
        ),
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_manifest(manifest_path)
    print(f"DATASET_MANIFEST=PASS path={manifest_path}")
    print(f"DATASET_REVISION={manifest['revision']}")
    print(f"SPLIT_ROWS={json.dumps(manifest['split_rows'], sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
