"""Exact artist/genre/country evidence for artist-inspired discovery candidates.

The GNN ranks this candidate set. This policy applies to artist inspirations,
not the track-seeded benchmark, and does not fabricate source relationships.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse


def export_discovery_index(processed, destination):
    processed, destination = Path(processed), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    artists = pd.read_parquet(processed / "artists.parquet")
    tracks = pd.read_parquet(processed / "tracks.parquet", columns=["track_node_id"])
    relation = pd.read_parquet(
        processed / "track_artist_edges.parquet",
        columns=["track_node_id", "artist_node_id"],
    )
    tokens = sorted(
        {f"genre:{g}" for genres in artists.genres for g in genres}
        | {f"country:{c}" for c in artists.country if c}
    )
    mapping = {token: i for i, token in enumerate(tokens)}
    rows, cols = [], []
    for index, genres, country in artists[
        ["artist_node_id", "genres", "country"]
    ].itertuples(index=False, name=None):
        for token in [f"genre:{g}" for g in genres] + (
            [f"country:{country}"] if country else []
        ):
            rows.append(index)
            cols.append(mapping[token])
    metadata = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(artists), len(tokens)),
    ).tocsr()
    credits = sparse.coo_matrix(
        (
            np.ones(len(relation), dtype=np.float32),
            (relation.track_node_id, relation.artist_node_id),
        ),
        shape=(len(tracks), len(artists)),
    ).tocsr()
    sparse.save_npz(destination / "artist_metadata_index.npz", metadata)
    sparse.save_npz(destination / "track_artist_index.npz", credits)
    (destination / "metadata_tokens.json").write_text(
        json.dumps(tokens, ensure_ascii=False), encoding="utf-8"
    )


class DiscoveryIndex:
    def __init__(self, root):
        root = Path(root)
        self.metadata = sparse.load_npz(root / "artist_metadata_index.npz")
        self.credits = sparse.load_npz(root / "track_artist_index.npz")
        self.tokens = json.loads(
            (root / "metadata_tokens.json").read_text(encoding="utf-8")
        )

    def candidates(self, artist_indices):
        direct = np.asarray(self.credits[:, artist_indices].sum(axis=1)).ravel() > 0
        # All artists credited on one of the inspiration's tracks are collaborators.
        collaborators = (
            np.asarray(self.credits.T @ direct.astype(np.float32)).ravel() > 0
        )
        token_mask = np.asarray(self.metadata[artist_indices].sum(axis=0)).ravel() > 0
        related = np.asarray(self.metadata @ token_mask.astype(np.float32)).ravel() > 0
        related |= collaborators
        related[np.asarray(artist_indices, dtype=np.int64)] = True
        eligible = np.asarray(self.credits @ related.astype(np.float32)).ravel() > 0
        return eligible, [self.tokens[i] for i in np.flatnonzero(token_mask)]
