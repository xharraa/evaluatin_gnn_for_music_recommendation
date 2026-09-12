"""Portable, catalog-inclusive preprocessing shared by the executable notebooks."""

from __future__ import annotations
import ast
import hashlib
import json
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/portable"
REPORTS = ROOT / "reports/current"
AUDIO = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
    "key",
    "mode",
]
SEED = 42
FEATURE_DIM = 64


def paths():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)


def report(name, value):
    paths()
    (REPORTS / f"{name}.json").write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(value, indent=2, ensure_ascii=False), flush=True)
    return value


def listed(value):
    if not isinstance(value, str):
        return []
    try:
        result = ast.literal_eval(value)
        return (
            [str(x).strip() for x in result if str(x).strip()]
            if isinstance(result, list)
            else []
        )
    except (ValueError, SyntaxError):
        return []


def valid_ids(series):
    return series.fillna("").astype(str).str.fullmatch(r"[A-Za-z0-9]{22}")


def catalog():
    paths()
    tracks = pd.read_csv(ROOT / "data/raw/tracks.csv")
    original = len(tracks)
    tracks = tracks.loc[valid_ids(tracks.id)].drop_duplicates("id").copy()
    tracks["id_artists"] = tracks.id_artists.map(listed)
    tracks["artists"] = tracks.artists.map(listed)
    tracks["name"] = tracks.name.fillna("Unknown track")
    invalid_audio = {}
    for col in AUDIO:
        tracks[col] = pd.to_numeric(tracks[col], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        valid = tracks[col].notna()
        if col in [
            "danceability",
            "energy",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "mode",
        ]:
            valid &= tracks[col].between(0, 1)
        elif col in ["duration_ms", "tempo"]:
            valid &= tracks[col] > 0
        elif col == "key":
            valid &= tracks[col].between(-1, 11)
        invalid_audio[col] = int((~valid).sum())
        tracks.loc[~valid, col] = np.nan
        tracks[col] = tracks[col].astype("float32")
    tracks.to_parquet(OUT / "catalog_tracks.parquet", index=False)
    artists = pd.read_csv(ROOT / "data/raw/artists.csv")
    artists = artists.loc[valid_ids(artists.id)].drop_duplicates("id").copy()
    artists["genres"] = artists.genres.map(listed)
    artists.to_parquet(OUT / "catalog_artists.parquet", index=False)
    relation = (
        tracks[["id", "id_artists"]]
        .explode("id_artists")
        .rename(columns={"id": "spotify_track_id", "id_artists": "spotify_artist_id"})
    )
    relation = relation.loc[valid_ids(relation.spotify_artist_id)].drop_duplicates()
    relation.to_parquet(OUT / "catalog_track_artists.parquet", index=False)
    tracks[AUDIO].corr().to_csv(REPORTS / "catalog_audio_correlations.csv")
    return report(
        "catalog_quality",
        {
            "source_tracks": original,
            "clean_tracks": len(tracks),
            "artist_metadata_rows": len(artists),
            "track_artist_edges": len(relation),
            "tracks_with_multiple_artists": int((tracks.id_artists.map(len) > 1).sum()),
            "invalid_audio_values_preserved_as_missing": invalid_audio,
            "country_column_in_source": "country" in artists.columns,
        },
    )


def playlists():
    paths()
    db = ROOT / "data/raw/playlists/spud/spud.sqlite"
    if not db.exists():
        raise FileNotFoundError(f"Download SPUD first: {db}")
    with sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        tracks = pd.read_sql_query(
            "SELECT t.trackid AS spud_track_id,t.spotifyid AS spotify_track_id,t.title AS track_title,a.spotifyid AS spotify_artist_id,a.name AS artist_name,t.popularity AS spud_popularity,t.duration AS duration_seconds FROM tracks t LEFT JOIN artists a ON t.artist=a.artistid ORDER BY t.trackid",
            conn,
        )
        edges = pd.read_sql_query(
            "SELECT playlist AS playlist_id,track AS spud_track_id FROM lastfmplayliststracks",
            conn,
        )
        pls = pd.read_sql_query(
            "SELECT playlistid AS playlist_id,title AS playlist_title FROM lastfmplaylists",
            conn,
        )
        schema = {
            row[0]: [
                dict(
                    zip(
                        [
                            "position",
                            "name",
                            "type",
                            "not_null",
                            "default",
                            "primary_key",
                        ],
                        c,
                    )
                )
                for c in conn.execute(
                    'PRAGMA table_info("' + row[0].replace('"', '""') + '")'
                )
            ]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    (REPORTS / "spud_schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8"
    )
    raw_edges = len(edges)
    raw_tracks = len(tracks)
    tracks = (
        tracks.loc[valid_ids(tracks.spotify_track_id)]
        .drop_duplicates("spud_track_id")
        .copy()
    )
    edges = edges.merge(
        tracks[["spud_track_id", "spotify_track_id"]],
        on="spud_track_id",
        validate="many_to_one",
    )
    edges = edges.loc[
        edges.playlist_id.isin(pls.playlist_id), ["playlist_id", "spotify_track_id"]
    ].drop_duplicates()
    tracks["duration_seconds"] = pd.to_numeric(tracks.duration_seconds, errors="coerce")
    bad = ~np.isfinite(tracks.duration_seconds) | (tracks.duration_seconds <= 0)
    tracks.loc[bad, "duration_seconds"] = tracks.loc[~bad, "duration_seconds"].median()
    tracks = tracks.drop_duplicates("spotify_track_id")
    tracks["track_title"] = tracks.track_title.fillna("Unknown track")
    tracks["artist_name"] = tracks.artist_name.fillna("Unknown artist")
    pls = pls.drop_duplicates("playlist_id").copy()
    pls["playlist_title"] = (
        pls.playlist_title.fillna("").str.strip().replace("", "Untitled playlist")
    )
    counts = edges.groupby("playlist_id").size()
    pls["track_count"] = pls.playlist_id.map(counts).fillna(0).astype("int32")
    pls["eligible_for_evaluation"] = pls.track_count >= 5
    for name, table in [
        ("spud_tracks", tracks),
        ("playlists", pls),
        ("playlist_edges", edges),
    ]:
        table.to_parquet(OUT / f"{name}.parquet", index=False)
    return report(
        "playlist_quality",
        {
            "source_tracks": raw_tracks,
            "clean_unique_spotify_tracks": len(tracks),
            "source_edges": raw_edges,
            "clean_edges": len(edges),
            "removed_duplicate_or_invalid_edges": raw_edges - len(edges),
            "playlists": len(pls),
            "duration_repairs": int(bad.sum()),
            "evaluation_playlists": int(pls.eligible_for_evaluation.sum()),
        },
    )


def split_edges(edges, seed=SEED):
    """All positives are partitioned before any graph aggregation or negative sampling."""
    rng = np.random.default_rng(seed)
    ordered = (
        edges.sort_values(["playlist_node_id", "track_node_id"])
        .reset_index(drop=True)
        .copy()
    )
    labels = np.full(len(ordered), "train", dtype="U5")
    for indices in ordered.groupby("playlist_node_id", sort=True).indices.values():
        if len(indices) < 5:
            continue
        perm = rng.permutation(indices)
        count = max(1, int(len(indices) * 0.15))
        labels[perm[:count]] = "val"
        labels[perm[count : 2 * count]] = "test"
    ordered["split"] = labels
    return ordered


def hash_features(tokens, width=48):
    result = np.zeros(width, dtype=np.float32)
    for token in tokens:
        digest = hashlib.blake2b(str(token).encode("utf-8"), digest_size=8).digest()
        result[int.from_bytes(digest[:4], "little") % width] += (
            1 if digest[4] % 2 else -1
        )
    return result / max(np.linalg.norm(result), 1.0)


def integrate():
    paths()
    cat = pd.read_parquet(OUT / "catalog_tracks.parquet")
    ar = pd.read_parquet(OUT / "catalog_artists.parquet").set_index("id")
    spud = pd.read_parquet(OUT / "spud_tracks.parquet")
    pls = pd.read_parquet(OUT / "playlists.parquet")
    edges = pd.read_parquet(OUT / "playlist_edges.parquet")
    # Union on stable Spotify IDs: neither artist names nor track titles are identity keys.
    # Keep even SPUD tracks with no observed playlist edge.
    spud = spud.copy()
    cat["artist_name"] = cat.artists.map(lambda values: " / ".join(values))
    cat_info = cat[["id", "name", "artist_name"]].rename(
        columns={"id": "spotify_track_id", "name": "track_title"}
    )
    tracks = (
        pd.concat(
            [cat_info, spud[["spotify_track_id", "track_title", "artist_name"]]],
            ignore_index=True,
        )
        .drop_duplicates("spotify_track_id")
        .sort_values("spotify_track_id")
        .reset_index(drop=True)
    )
    tracks["track_node_id"] = np.arange(len(tracks), dtype=np.int32)
    tracks["in_catalog"] = tracks.spotify_track_id.isin(cat.id)
    tracks["in_playlists"] = tracks.spotify_track_id.isin(edges.spotify_track_id)
    tracks["spud_popularity"] = (
        tracks.spotify_track_id.map(spud.set_index("spotify_track_id").spud_popularity)
        .fillna(0)
        .clip(0, 1)
    )
    relation = pd.concat(
        [
            pd.read_parquet(OUT / "catalog_track_artists.parquet"),
            spud[["spotify_track_id", "spotify_artist_id"]],
        ],
        ignore_index=True,
    ).drop_duplicates()
    relation = relation.loc[valid_ids(relation.spotify_artist_id)].copy()
    artists = pd.DataFrame(
        {"spotify_artist_id": sorted(set(relation.spotify_artist_id) | set(ar.index))}
    )
    artists["artist_node_id"] = np.arange(len(artists), dtype=np.int32)
    artists["artist_name"] = (
        artists.spotify_artist_id.map(ar["name"])
        .fillna(
            artists.spotify_artist_id.map(
                spud.drop_duplicates("spotify_artist_id")
                .set_index("spotify_artist_id")
                .artist_name
            )
        )
        .fillna("Unknown artist")
    )
    artists["genres"] = artists.spotify_artist_id.map(ar.genres).map(
        lambda v: list(v) if isinstance(v, (list, np.ndarray)) else []
    )
    artists["track_count"] = (
        artists.spotify_artist_id.map(relation.groupby("spotify_artist_id").size())
        .fillna(0)
        .astype("int32")
    )
    artists["country"] = ""
    country_file = ROOT / "data/raw/artist_countries.csv"
    if country_file.exists():
        countries = pd.read_csv(country_file).dropna(
            subset=["spotify_artist_id", "country", "source"]
        )
        if countries.spotify_artist_id.duplicated().any():
            raise ValueError(
                "Country metadata must contain one sourced row per artist ID."
            )
        countries = countries.loc[
            countries.source.str.strip().ne("") & countries.country.str.strip().ne("")
        ]
        artists["country"] = artists.spotify_artist_id.map(
            countries.set_index("spotify_artist_id").country
        ).fillna("")
    pls = pls.loc[pls.track_count > 0].sort_values("playlist_id").reset_index(drop=True)
    pls["playlist_node_id"] = np.arange(len(pls), dtype=np.int32)
    tid = tracks.set_index("spotify_track_id").track_node_id
    aid = artists.set_index("spotify_artist_id").artist_node_id
    pid = pls.set_index("playlist_id").playlist_node_id
    relation["track_node_id"] = relation.spotify_track_id.map(tid)
    relation["artist_node_id"] = relation.spotify_artist_id.map(aid)
    edges["playlist_node_id"] = edges.playlist_id.map(pid)
    edges["track_node_id"] = edges.spotify_track_id.map(tid)
    split = split_edges(edges[["playlist_node_id", "track_node_id"]])
    nt, na, np_ = len(tracks), len(artists), len(pls)
    train = split.loc[split.split.eq("train")]
    # Track -> every credited artist -> genres/countries; collaboration is shared track credit.
    tokens = sorted(
        {f"genre:{g}" for values in artists.genres for g in values}
        | {f"country:{c}" for c in artists.country if c}
    )
    token_id = {t: i + nt + na + np_ for i, t in enumerate(tokens)}
    rows = [train.track_node_id.to_numpy(), relation.track_node_id.to_numpy()]
    cols = [
        train.playlist_node_id.to_numpy() + nt + na,
        relation.artist_node_id.to_numpy() + nt,
    ]
    left, right = [], []
    for a, genres, country in artists[
        ["artist_node_id", "genres", "country"]
    ].itertuples(index=False, name=None):
        for token in [f"genre:{g}" for g in genres] + (
            [f"country:{country}"] if country else []
        ):
            left.append(nt + a)
            right.append(token_id[token])
    rows.append(np.asarray(left, dtype=np.int64))
    cols.append(np.asarray(right, dtype=np.int64))
    src = np.concatenate(rows).astype(np.int32)
    dst = np.concatenate(cols).astype(np.int32)
    total = nt + na + np_ + len(tokens)
    adj = sparse.coo_matrix(
        (np.ones(2 * len(src), dtype=np.float32), (np.r_[src, dst], np.r_[dst, src])),
        shape=(total, total),
    ).tocsr()
    adj.sum_duplicates()
    adj.data[:] = 1
    degree = np.asarray(adj.sum(axis=1)).ravel()
    inv = 1 / np.sqrt(np.maximum(degree, 1))
    adj = (sparse.diags(inv) @ adj @ sparse.diags(inv)).tocsr().astype(np.float32)
    sparse.save_npz(OUT / "train_graph.npz", adj)
    features = np.lib.format.open_memmap(
        OUT / "features_0.npy", mode="w+", dtype="float32", shape=(total, FEATURE_DIM)
    )
    features[:] = 0
    # Static metadata may cover the whole catalog; no held-out playlist composition is used.
    values = (
        cat.set_index("id")[AUDIO]
        .reindex(tracks.spotify_track_id)
        .to_numpy(dtype=np.float32)
    )
    mask = np.isfinite(values)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std = np.where(std > 1e-6, std, 1)
    features[:nt, :12] = np.nan_to_num(np.clip((values - mean) / std, -4, 4)) / 4
    features[:nt, 12] = mask.mean(axis=1)
    features[:nt, 13] = 1
    tracks["audio_features_available"] = mask.any(axis=1)
    artist_features = np.stack(
        [
            hash_features(
                [f"artist:{i}"]
                + [f"genre:{g}" for g in gs]
                + ([f"country:{c}"] if c else [])
            )
            for i, gs, c in artists[
                ["spotify_artist_id", "genres", "country"]
            ].itertuples(index=False, name=None)
        ]
    )
    features[nt : nt + na, 16:] = artist_features
    features[nt : nt + na, 14] = 1
    incidence = sparse.coo_matrix(
        (
            np.ones(len(relation), dtype=np.float32),
            (relation.track_node_id, relation.artist_node_id),
        ),
        shape=(nt, na),
    ).tocsr()
    denom = np.maximum(np.asarray(incidence.sum(axis=1)).ravel(), 1)
    features[:nt, 16:] = (incidence @ artist_features) / denom[:, None]
    features[nt + na : nt + na + np_, 15] = 1
    for i, title in pls[["playlist_node_id", "playlist_title"]].itertuples(
        index=False, name=None
    ):
        features[nt + na + i, 16:] = hash_features(
            ["title:" + t for t in title.casefold().split()]
        )
    for token, index in token_id.items():
        features[index, 16:] = hash_features([token])
    features.flush()
    tracks.to_parquet(OUT / "tracks.parquet", index=False)
    artists.to_parquet(OUT / "artists.parquet", index=False)
    pls.to_parquet(OUT / "playlist_nodes.parquet", index=False)
    relation.to_parquet(OUT / "track_artist_edges.parquet", index=False)
    split.to_parquet(OUT / "split.parquet", index=False)
    (OUT / "feature_scaling.json").write_text(
        json.dumps(
            {
                "names": AUDIO,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "hash_algorithm": "blake2b signed 48 bins",
                "feature_dimension": FEATURE_DIM,
            }
        ),
        encoding="utf-8",
    )
    assert relation[["track_node_id", "artist_node_id"]].notna().all().all()
    assert not split.duplicated(["playlist_node_id", "track_node_id"]).any()
    assert np.isfinite(features).all()
    elvana = tracks.loc[
        tracks.artist_name.str.contains("elvana gjata", case=False, regex=False)
    ]
    elvana.to_csv(REPORTS / "elvana_gjata_coverage.csv", index=False)
    return report(
        "graph_coverage",
        {
            "tracks": nt,
            "artists": na,
            "playlists": np_,
            "metadata_nodes": len(tokens),
            "total_nodes": total,
            "catalog_only_tracks": int(
                (tracks.in_catalog & ~tracks.in_playlists).sum()
            ),
            "playlist_only_tracks": int(
                (~tracks.in_catalog & tracks.in_playlists).sum()
            ),
            "spud_only_tracks_without_playlist_history": int(
                (~tracks.in_catalog & ~tracks.in_playlists).sum()
            ),
            "overlapping_tracks": int((tracks.in_catalog & tracks.in_playlists).sum()),
            "track_artist_edges": len(relation),
            "split_counts": {
                str(k): int(v) for k, v in split.split.value_counts().items()
            },
            "countries_available": int(artists.country.ne("").sum()),
            "elvana_gjata_artist_records": int(
                artists.artist_name.str.contains(
                    "elvana gjata", case=False, regex=False
                ).sum()
            ),
            "artists_without_track_records": int(artists.track_count.eq(0).sum()),
            "elvana_gjata_tracks": len(elvana),
            "elvana_gjata_playlist_tracks": int(elvana.in_playlists.sum()),
            "graph_uses_only_train_playlist_edges": True,
        },
    )


def propagate(max_hops=4):
    adj = sparse.load_npz(OUT / "train_graph.npz")
    previous = np.load(OUT / "features_0.npy", mmap_mode="r")
    for hop in range(1, max_hops + 1):
        current = np.lib.format.open_memmap(
            OUT / f"features_{hop}.npy",
            mode="w+",
            dtype="float32",
            shape=previous.shape,
        )
        for start in range(0, adj.shape[0], 25000):
            current[start : start + 25000] = adj[start : start + 25000] @ previous
        current.flush()
        previous = current
        print(f"Graph propagation {hop}/{max_hops} saved", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["catalog", "playlists", "integrate", "propagate"]
    )
    globals()[parser.parse_args().stage]()
