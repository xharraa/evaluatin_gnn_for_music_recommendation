"""Create clear, executable notebooks; keep original Spark work as an archive."""

from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
BOOT = """from pathlib import Path
import sys
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'scripts/data_pipeline.py').exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from IPython.display import display, Image
import pandas as pd
from scripts.data_pipeline import OUT, REPORTS
"""


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}


def code(s):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": s.splitlines(True),
        "execution_count": None,
        "outputs": [],
    }


def notebook(name, title, cells):
    path = ROOT / "notebooks" / name
    archive = ROOT / "notebooks/legacy" / name
    if path.exists() and not archive.exists() and name != "00_environment_check.ipynb":
        archive.parent.mkdir(parents=True, exist_ok=True)
        old = json.loads(path.read_text(encoding="utf-8-sig"))
        old["cells"].insert(
            0,
            md(
                "# Archived Spark implementation\nHistorical source from the previous laptop. Run the notebooks one directory above for the current portable pipeline. Outputs have been cleared to avoid presenting old results as current."
            ),
        )
        for cell in old["cells"]:
            if cell["cell_type"] == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
        archive.write_text(
            json.dumps(old, indent=1, ensure_ascii=False), encoding="utf-8"
        )
    obj = {
        "cells": [md(title), code(BOOT), *cells],
        "metadata": {
            "kernelspec": {
                "display_name": "Python (Music GNN)",
                "language": "python",
                "name": "music-gnn",
            },
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for i, c in enumerate(obj["cells"]):
        c["id"] = f"cell-{i:03d}"
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


notebook(
    "00_environment_check.ipynb",
    "# 00 · This laptop, from scratch\nInspect Python, RAM, CPU, NVIDIA driver and PyTorch before building any data. The portable workflow uses CPU by default and does not need Java or Spark.",
    [
        code(
            'from scripts.project_analysis import hardware\nhardware_report = hardware()\ndisplay(pd.Series(hardware_report, name="Current machine"))'
        ),
        md(
            "The GTX 1650 has 4 GB VRAM. The original CUDA 13 configuration cannot run with driver 457.49. A CPU wheel works immediately; GPU use requires a compatible driver and PyTorch installation. Data stages run in separate kernels to release memory."
        ),
    ],
)
notebook(
    "01_preprocessing.ipynb",
    "# 01 · Catalog tracks and every credited artist\nThe catalog is the base discovery collection. Keep valid Spotify track IDs even when audio fields are missing; represent invalid audio values explicitly. Parse artist lists safely and retain all credits, including collaborations.",
    [
        code("from scripts.data_pipeline import catalog\nquality = catalog()"),
        code(
            'tracks = pd.read_parquet(OUT / "catalog_tracks.parquet")\ndisplay(tracks[["id", "name", "artists", "id_artists"]].head(10))\ndisplay(pd.read_parquet(OUT / "catalog_track_artists.parquet").head(10))'
        ),
        md(
            "Invalid IDs and duplicate track IDs are removed. Numeric audio validity is audited column by column; missing audio never removes a searchable song. Artist ID is the join key, so artists with the same name are not merged."
        ),
        code(
            'display(pd.read_csv(REPORTS / "catalog_audio_correlations.csv", index_col=0).round(3))'
        ),
    ],
)
notebook(
    "01b_playlist_preprocessing.ipynb",
    "# 01b · SPUD playlist relationships\nSPUD supplies Last.fm playlists with Spotify IDs. Read its SQLite database in read-only mode, audit its schema, deduplicate edges, remove orphan endpoints and recompute playlist sizes.",
    [
        code("from scripts.data_pipeline import playlists\nquality = playlists()"),
        code(
            'import json\nschema = json.loads((REPORTS / "spud_schema.json").read_text())\ndisplay(pd.DataFrame([{"table": table, **column} for table, columns in schema.items() for column in columns]))'
        ),
        code(
            'display(pd.read_parquet(OUT / "playlists.parquet").head(10))\ndisplay(pd.read_parquet(OUT / "playlist_edges.parquet").head(10))'
        ),
        md(
            "Playlist absence means no observed membership. We do not create fake playlists for catalog artists. Playlists with at least five distinct tracks support train/validation/test evaluation; smaller playlists supply training context only."
        ),
    ],
)
notebook(
    "02_joining_and_segmentation.ipynb",
    "# 02 · One catalog-inclusive music graph\nUnion catalog and playlist tracks by Spotify ID. Retain every track–artist credit. Add artist–genre edges and optional sourced artist–country edges. Shared track credits provide collaboration paths. Split playlist edges before any graph propagation.",
    [
        code("from scripts.data_pipeline import integrate\ncoverage = integrate()"),
        md(
            "For each eligible playlist, hide 15% of edges for validation and 15% for test (with at least one in each). All other edges are training data. Negative sampling excludes every known positive. Static artist/audio metadata is available for the full catalog, but validation/test playlist composition never enters graph features."
        ),
        code(
            'display(pd.read_parquet(OUT / "tracks.parquet").head(10))\ndisplay(pd.read_parquet(OUT / "track_artist_edges.parquet").head(10))\ndisplay(pd.read_parquet(OUT / "split.parquet").groupby("split").size())'
        ),
        md(
            "Compute normalized sparse propagation once, writing each hop to a memory-mapped file. This is algebraically equivalent to linear LightGCN propagation of projected features and supports the SIGN encoders without retaining full-graph gradients in RAM."
        ),
        code("from scripts.data_pipeline import propagate\npropagate(max_hops=4)"),
    ],
)
notebook(
    "02b_data_analysis.ipynb",
    "# 02b · Coverage, audio and graph relationships\nRegenerate observations from the current data. Full graph degrees here are descriptive only; training features use the training split.",
    [
        code(
            "from scripts.project_analysis import analysis\nobservations = analysis()"
        ),
        code(
            'display(Image(filename=str(REPORTS / "dataset_overview.png")))\ndisplay(Image(filename=str(REPORTS / "audio_correlations.png")))'
        ),
    ],
)
notebook(
    "02c_further_data_analysis.ipynb",
    "# 02c · Missing playlist history and integrity\nExplicitly test the Elvana Gjata case, unique identities, valid graph endpoints, split disjointness and finite feature vectors.",
    [
        code(
            "from scripts.project_analysis import diagnostics\nchecks = diagnostics()"
        ),
        code(
            'elvana = pd.read_csv(REPORTS / "elvana_gjata_coverage.csv")\ndisplay(elvana[["spotify_track_id", "track_title", "artist_name", "in_catalog", "in_playlists"]])'
        ),
        md(
            "Country is not inferred from an artist name or genre. Optional country metadata must be supplied as `data/raw/artist_countries.csv` with `spotify_artist_id,country,source`. Catalog-only artists can receive scores from metadata and graph paths, but absent playlist labels cannot prove their recommendation quality."
        ),
    ],
)
notebook(
    "03_model_creation.ipynb",
    "# 03 · Three GNNs, one controlled experiment\nTrain LightGCN, SIGN and Residual SIGN on the same graph, seeds, negatives, optimizer, learning rate, batch size and epoch budget. Only model capacity and architecture differ.",
    [
        md(
            "| Model | Graph hops | Hidden width | Output dimensions | Nonlinear trunk layers |\n|---|---:|---:|---:|---:|\n| LightGCN | 2 | 32 | 32 | 0 |\n| SIGN | 2 | 128 | 64 | 2 |\n| Residual SIGN | 4 | 256 | 128 | 4 |\n\n**LightGCN variant:** initial embeddings are `XW`, not free node-ID parameters. Averaging `A^h X` before applying `W` is exactly equivalent to linear propagation of `XW`. This intentional attribute-based extension makes metadata available to catalog-only tracks. SIGN uses hop-specific nonlinear projections; the larger model adds depth, width and residual layers."
        ),
        code(
            'from scripts.train_models import run\n# One common configuration for all three models.\nCONFIG = dict(epochs=10, batch_size=2048, lr=0.001, weight_decay=0.0001, max_eval_playlists=1024, device="cpu")\nmanifest = run(**CONFIG, resume=True)'
        ),
        md(
            "Select checkpoints and the metadata mixture using validation only. Calibrate relative fit scores using validation only. Evaluate each frozen model once on held-out test playlists. Metrics use all held-out positives plus up to 100 sampled negatives per playlist, on a fixed sample of up to 1,024 playlists; they are not full-catalog ranking metrics. Random and training-popularity baselines use the same candidate protocol."
        ),
        code(
            'from scripts.project_analysis import training_figures\ndisplay(training_figures())\ndisplay(Image(filename=str(REPORTS / "model_comparison.png")))'
        ),
        code(
            'from scripts.recommender_inference import load_default_recommender\nimport gc\nartists = pd.read_parquet(OUT / "artists.parquet")\nelvana = artists.loc[artists.artist_name.str.casefold().eq("elvana gjata")]\nif len(elvana):\n    artist_id = str(elvana.iloc[0].spotify_artist_id)\n    print("Artist metadata:")\n    display(elvana[["spotify_artist_id", "artist_name", "genres", "track_count"]])\n    for model_id in ["lightgcn", "sign", "residual_sign"]:\n        engine = load_default_recommender(ROOT, model_id)\n        recommendations, metadata = engine.recommend([], k=8, artist_ids=[artist_id])\n        print(model_id, metadata)\n        display(recommendations[["track_title", "artist_name", "match_percent", "in_playlists"]])\n        recommendations.to_csv(REPORTS / f"elvana_{model_id}_recommendations.csv", index=False)\n        del engine\n        gc.collect()\nelse:\n    print("Elvana Gjata is absent from the artist source.")\n'
        ),
        md(
            "The three `.pt` bundles and their measured scores are listed in `models/manifest.json`. Search uses the shared union catalog; switching models preserves the user playlist. `scripts/start_web_demo.ps1` starts the local API and web interface."
        ),
    ],
)
print(
    "Created seven executable notebooks; original Spark source preserved in notebooks/legacy."
)
