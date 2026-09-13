"""Fail early if the private, git-ignored inference artifacts are incomplete."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
SHARED = (
    "track_catalog.csv.gz",
    "artist_catalog.parquet",
    "content_features.npy",
    "artist_content_features.npy",
    "artist_metadata_index.npz",
    "track_artist_index.npz",
    "metadata_tokens.json",
)
EXPECTED = {"lightgcn", "sign", "residual_sign"}


def main():
    manifest_path = MODELS / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing {manifest_path}. Mount the complete models folder.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {item["id"]: item["file"] for item in manifest["models"]}
    missing = [name for name in SHARED if not (MODELS / name).is_file()]
    missing.extend(entries[model_id] for model_id in sorted(EXPECTED & entries.keys()) if not (MODELS / entries[model_id]).is_file())
    missing.extend(f"manifest entry: {model_id}" for model_id in sorted(EXPECTED - entries.keys()))
    if missing:
        raise SystemExit("Deployment model files are incomplete:\n- " + "\n- ".join(missing))
    print("Deployment artifacts ready: three models and shared catalogs/indexes.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (KeyError, ValueError, TypeError) as error:
        sys.exit(f"Invalid model manifest: {error}")

