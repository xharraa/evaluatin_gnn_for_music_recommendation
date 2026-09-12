"""Download both sources and normalize Kaggle's sometimes-zipped CSV responses."""

from pathlib import Path
import argparse
import csv
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATASET_HANDLE = "yamaerenay/spotify-dataset-19212020-600k-tracks/versions/1"


def normalize_csv(path):
    if zipfile.is_zipfile(path):
        temporary = path.with_suffix(".unpacked")
        with zipfile.ZipFile(path) as archive:
            member = next(
                (n for n in archive.namelist() if Path(n).name == path.name), None
            )
            if member is None:
                raise ValueError(f"{path.name} is missing from its download archive")
            with archive.open(member) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
        temporary.replace(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream))
    if not {"id", "name"}.issubset(header):
        raise ValueError(f"Unexpected CSV schema: {path}")


def download_catalog(force=False):
    import kagglehub

    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in ["tracks.csv", "artists.csv"]:
        destination = raw / name
        if force or not destination.exists() or not destination.stat().st_size:
            result = Path(
                kagglehub.dataset_download(
                    DATASET_HANDLE, path=name, output_dir=str(raw), force_download=force
                )
            )
            if not destination.exists() and result.is_file():
                shutil.copyfile(result, destination)
        normalize_csv(destination)
        print(f"Ready: {name} ({destination.stat().st_size:,} bytes)", flush=True)


def download_playlists(force=False):
    root = ROOT / "data/raw/playlists/spud"
    root.mkdir(parents=True, exist_ok=True)
    db = root / "spud.sqlite"
    archive = root / "spud.zip"
    if db.exists() and not force:
        return
    if force or not archive.exists() or not zipfile.is_zipfile(archive):
        command = [
            "curl.exe" if __import__("os").name == "nt" else "curl",
            "-fL",
            "--retry",
            "6",
            "--retry-all-errors",
            "--retry-delay",
            "3",
        ]
        if archive.exists() and not force:
            command += ["-C", "-"]
        subprocess.run(
            command
            + ["-o", str(archive), "https://www.dcs.gla.ac.uk/~daniel/spud/spud.zip"],
            check=True,
        )
    with zipfile.ZipFile(archive) as z:
        member = next(n for n in z.namelist() if Path(n).name == "spud.sqlite")
        temporary = db.with_suffix(".partial")
        with z.open(member) as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        temporary.replace(db)
    print(f"Ready: {db}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--source", choices=["all", "catalog", "playlists"], default="all"
    )
    args = parser.parse_args()
    if args.source in ["all", "catalog"]:
        download_catalog(args.force)
    if args.source in ["all", "playlists"]:
        download_playlists(args.force)
