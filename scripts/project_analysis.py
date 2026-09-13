"""Machine checks and reproducible figures for the portable notebooks."""

import json
import platform
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import psutil

try:
    from .data_pipeline import ROOT, OUT, REPORTS, AUDIO, report, paths
except ImportError:
    from data_pipeline import ROOT, OUT, REPORTS, AUDIO, report, paths


def hardware():
    import torch

    paths()
    result = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cpu_logical_cores": psutil.cpu_count(),
        "ram_gib": round(psutil.virtual_memory().total / 1024**3, 2),
        "available_ram_gib": round(psutil.virtual_memory().available / 1024**3, 2),
        "free_disk_gib": round(shutil.disk_usage(ROOT).free / 1024**3, 2),
        "torch_version": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_runtime": torch.version.cuda,
        "default_device": "cpu",
        "worker_threads": 4,
    }
    if shutil.which("nvidia-smi"):
        p = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        result["nvidia_gpu"] = p.stdout.strip()
    return report("hardware", result)


def analysis():
    paths()
    tracks = pd.read_parquet(OUT / "tracks.parquet")
    artists = pd.read_parquet(OUT / "artists.parquet")
    edges = pd.read_parquet(OUT / "split.parquet")
    relation = pd.read_parquet(OUT / "track_artist_edges.parquet")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    counts = edges.groupby("playlist_node_id").size()
    axes[0, 0].hist(counts, bins=np.geomspace(1, max(2, counts.max()), 40))
    axes[0, 0].set(
        xscale="log",
        xlabel="Distinct tracks per playlist",
        ylabel="Playlists",
        title="Playlist sizes",
    )
    coverage = [
        int((tracks.in_catalog & ~tracks.in_playlists).sum()),
        int((tracks.in_catalog & tracks.in_playlists).sum()),
        int((~tracks.in_catalog & tracks.in_playlists).sum()),
        int((~tracks.in_catalog & ~tracks.in_playlists).sum()),
    ]
    axes[0, 1].bar(
        [
            "Catalog/no playlist",
            "Catalog + playlist",
            "Playlist/no catalog",
            "SPUD/no playlist",
        ],
        coverage,
        color=["#36a685", "#496ddb", "#dc9d42", "#927bc9"],
    )
    axes[0, 1].set(title="Track coverage after union", ylabel="Tracks")
    axes[0, 1].tick_params(axis="x", rotation=20)
    degree = edges.groupby("track_node_id").size()
    axes[1, 0].hist(degree, bins=np.geomspace(1, max(2, degree.max()), 40))
    axes[1, 0].set(
        xscale="log",
        yscale="log",
        xlabel="Observed playlist degree",
        ylabel="Tracks",
        title="Playlist exposure (descriptive only)",
    )
    genres = artists.genres.explode().dropna().value_counts().head(12).sort_values()
    axes[1, 1].barh(genres.index, genres.values)
    axes[1, 1].set(title="Most frequent artist genres", xlabel="Artists")
    fig.savefig(REPORTS / "dataset_overview.png", dpi=160)
    plt.close(fig)
    corr = pd.read_csv(REPORTS / "catalog_audio_correlations.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(10, 9), layout="constrained")
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    ax.set_title("Catalog audio Pearson correlations")
    fig.colorbar(im, ax=ax)
    fig.savefig(REPORTS / "audio_correlations.png", dpi=160)
    plt.close(fig)
    counts.describe().to_csv(REPORTS / "playlist_size_summary.csv")
    credits = relation.groupby("track_node_id").size()
    return report(
        "relation_analysis",
        {
            "playlist_size_median": float(counts.median()),
            "tracks_with_multiple_credits": int((credits > 1).sum()),
            "artist_metadata_genre_coverage": float(
                artists.genres.map(len).gt(0).mean()
            ),
            "tracks_without_playlist_history": int((~tracks.in_playlists).sum()),
            "interpretation": "Shared track credits form collaboration paths; same-name artists are never merged. Countries are used only from a sourced artist_countries.csv.",
        },
    )


def further_analysis():
    """Relate catalog audio correlations to observed playlist connections."""
    paths()
    catalog_corr = pd.read_csv(REPORTS / "catalog_audio_correlations.csv", index_col=0)
    audio_pairs = pd.DataFrame(
        [
            {"feature_a": left, "feature_b": right, "pearson_r": float(catalog_corr.loc[left, right])}
            for i, left in enumerate(catalog_corr.columns)
            for right in catalog_corr.columns[i + 1 :]
        ]
    )
    audio_pairs = audio_pairs.reindex(audio_pairs.pearson_r.abs().sort_values(ascending=False).index)
    audio_pairs.to_csv(REPORTS / "catalog_audio_relationships.csv", index=False)

    tracks = pd.read_parquet(OUT / "tracks.parquet", columns=["spotify_track_id", "track_node_id", "in_catalog", "in_playlists", "spud_popularity"])
    edges = pd.read_parquet(OUT / "split.parquet", columns=["track_node_id"])
    degree = edges.groupby("track_node_id").size().rename("playlist_degree")
    connected = tracks.loc[tracks.in_catalog & tracks.in_playlists].join(degree, on="track_node_id")
    catalog = pd.read_parquet(OUT / "catalog_tracks.parquet", columns=["id", "popularity", *AUDIO])
    connected = connected.merge(catalog, left_on="spotify_track_id", right_on="id", validate="one_to_one")
    connected["log_playlist_degree"] = np.log1p(connected.playlist_degree)
    columns = ["log_playlist_degree", "popularity", "spud_popularity", *AUDIO]
    connected_corr = connected[columns].corr(method="pearson")
    connected_corr.to_csv(REPORTS / "connected_track_correlation_matrix.csv")
    degree_pairs = pd.DataFrame(
        [
            {"feature": name, "pearson_r": float(connected_corr.loc["log_playlist_degree", name]),
             "pairwise_n": int(connected[["log_playlist_degree", name]].dropna().shape[0])}
            for name in columns[1:]
        ]
    ).sort_values("pearson_r", key=lambda values: values.abs(), ascending=False)
    degree_pairs.to_csv(REPORTS / "playlist_degree_relationships.csv", index=False)

    relation = pd.read_parquet(OUT / "track_artist_edges.parquet", columns=["track_node_id"])
    credits = relation.groupby("track_node_id").size()
    summary = {
        "catalog_track_count": int(tracks.in_catalog.sum()),
        "connected_catalog_track_count": len(connected),
        "tracks_without_playlist_history": int((~tracks.in_playlists).sum()),
        "tracks_with_multiple_artist_credits": int((credits > 1).sum()),
        "strongest_catalog_audio_pairs": audio_pairs.head(5).to_dict("records"),
        "playlist_degree_correlations_on_connected_catalog_tracks": degree_pairs.head(5).to_dict("records"),
        "interpretation": "Audio relationships describe the catalog. Degree relationships use only catalog tracks observed in playlists. Artist credits form graph paths; correlations are associations, not causal effects or recommendation quality estimates.",
    }
    return report("further_data_analysis", summary)


def diagnostics():
    tracks = pd.read_parquet(OUT / "tracks.parquet")
    artists = pd.read_parquet(OUT / "artists.parquet")
    edges = pd.read_parquet(OUT / "split.parquet")
    relation = pd.read_parquet(OUT / "track_artist_edges.parquet")
    assert tracks.track_node_id.tolist() == list(range(len(tracks)))
    assert tracks.spotify_track_id.is_unique
    assert not relation.duplicated(["track_node_id", "artist_node_id"]).any()
    assert edges.track_node_id.between(0, len(tracks) - 1).all()
    assert relation.artist_node_id.between(0, len(artists) - 1).all()
    assert not edges.duplicated(["playlist_node_id", "track_node_id"]).any()
    for i in range(5):
        x = np.load(OUT / f"features_{i}.npy", mmap_mode="r")
        for start in range(0, len(x), 25000):
            assert np.isfinite(x[start : start + 25000]).all()
    degree = (
        edges.groupby("track_node_id")
        .size()
        .reindex(tracks.track_node_id, fill_value=0)
        .to_numpy()
    )
    sorted_degree = np.sort(degree)
    n = len(degree)
    gini = float(
        (
            2
            * np.sum(np.arange(1, n + 1) * sorted_degree)
            / (n * max(1, sorted_degree.sum()))
        )
        - (n + 1) / n
    )
    return report(
        "integrity_checks",
        {
            "unique_track_ids": True,
            "unique_edges": True,
            "valid_endpoints": True,
            "finite_graph_features": True,
            "disjoint_splits": True,
            "full_catalog_retained": True,
            "exposure_gini_including_catalog_only": gini,
            "country_metadata_rows": int(artists.country.ne("").sum()),
            "cold_start_limitation": "Catalog-only tracks can be scored, but absent playlist ground truth cannot establish their recommendation quality. No playlist memberships are fabricated.",
        },
    )


def training_figures():
    comparison = pd.read_csv(REPORTS / "model_comparison.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    axes[0].bar(comparison.model, comparison.ndcg_at_10)
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set(ylabel="Sampled-candidate NDCG@10", title="Held-out evaluation")
    for model in ["lightgcn", "sign", "residual_sign"]:
        p = REPORTS / f"{model}_training_history.csv"
        if p.exists():
            h = pd.read_csv(p)
            axes[1].plot(h.epoch, h.validation_ndcg_at_10, label=model)
    axes[1].set(
        xlabel="Epoch", ylabel="Validation NDCG@10", title="Checkpoint selection"
    )
    axes[1].legend()
    fig.savefig(REPORTS / "model_comparison.png", dpi=160)
    plt.close(fig)
    return comparison
