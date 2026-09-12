"""Versioned multi-model inference. Search includes the union of both datasets."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch


class PlaylistRecommender:
    def __init__(
        self,
        bundle_path,
        catalog_path=None,
        *,
        catalog=None,
        content=None,
        artists=None,
        artist_content=None,
        generation=None,
        discovery=None,
    ):
        self.bundle_path = Path(bundle_path)
        self.bundle = torch.load(
            self.bundle_path, map_location="cpu", weights_only=True
        )
        if self.bundle.get("schema_version") != 2:
            raise ValueError(
                "Legacy checkpoint: rerun notebook 03 to build the current three models."
            )
        if generation is not None and self.bundle["generation"] != generation:
            raise ValueError(
                "Model and catalog are from different runs. Rebuild all three models."
            )
        self.catalog = (
            catalog
            if catalog is not None
            else pd.read_csv(catalog_path)
            .sort_values("track_node_id")
            .reset_index(drop=True)
        )
        if not np.array_equal(
            self.catalog.track_node_id.to_numpy(), np.arange(len(self.catalog))
        ):
            raise ValueError("Catalog node IDs must be contiguous and ordered.")
        self.track_embeddings = (
            self.bundle.pop("final_track_embeddings").float().numpy()
        )
        self.content = (
            content
            if content is not None
            else np.load(
                self.bundle_path.parent / "content_features.npy", mmap_mode="r"
            )
        )
        if len(self.track_embeddings) != len(self.catalog) or len(self.content) != len(
            self.catalog
        ):
            raise ValueError("Checkpoint, catalog, and content dimensions disagree.")
        self.artists = (
            artists
            if artists is not None
            else pd.read_parquet(self.bundle_path.parent / "artist_catalog.parquet")
        )
        self.artist_embeddings = self.bundle.pop("final_artist_embeddings").numpy()
        self.artist_content = (
            artist_content
            if artist_content is not None
            else np.load(
                self.bundle_path.parent / "artist_content_features.npy", mmap_mode="r"
            )
        )
        self.artist_to_index = dict(
            zip(self.artists.spotify_artist_id, self.artists.artist_node_id.astype(int))
        )
        if len(self.artists) != len(self.artist_embeddings):
            raise ValueError("Artist catalog and model dimensions disagree.")
        try:
            from .discovery import DiscoveryIndex
        except ImportError:
            from discovery import DiscoveryIndex
        self.discovery = (
            discovery
            if discovery is not None
            else DiscoveryIndex(self.bundle_path.parent)
        )
        if self.discovery.credits.shape != (len(self.catalog), len(self.artists)):
            raise ValueError("Discovery index and catalogs disagree.")
        self.content_weight = float(self.bundle["config"]["selected_content_weight"])
        self.spotify_to_index = dict(
            zip(
                self.catalog.spotify_track_id.astype(str),
                self.catalog.track_node_id.astype(int),
            )
        )

    @staticmethod
    def _unit(x):
        return x / max(float(np.linalg.norm(x)), 1e-12)

    def recommend(self, spotify_track_ids, k=10, artist_ids=()):
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        requested = list(dict.fromkeys(str(x) for x in spotify_track_ids))
        recognized = [x for x in requested if x in self.spotify_to_index]
        unknown = [x for x in requested if x not in self.spotify_to_index]
        requested_artists = list(dict.fromkeys(str(x) for x in artist_ids))
        recognized_artists = [x for x in requested_artists if x in self.artist_to_index]
        unknown_artists = [
            x for x in requested_artists if x not in self.artist_to_index
        ]
        if not recognized and not recognized_artists:
            raise ValueError(
                "None of these tracks or artists are present in either source dataset."
            )
        seeds = np.array([self.spotify_to_index[x] for x in recognized], dtype=np.int64)
        artist_indices = np.array(
            [self.artist_to_index[x] for x in recognized_artists], dtype=np.int64
        )
        if not len(seeds) and all(
            int(self.artists.iloc[i].track_count) == 0
            and len(self.artists.iloc[i].genres) == 0
            and not str(self.artists.iloc[i].country).strip()
            for i in artist_indices
        ):
            raise ValueError(
                "These artist profiles have no tracks, genres, or country metadata. Add a track or an artist with metadata."
            )
        query = self._unit(
            np.concatenate(
                [
                    self.track_embeddings[seeds],
                    np.asarray(
                        self.artist_embeddings[artist_indices], dtype=np.float32
                    ),
                ]
            ).mean(axis=0)
        )
        content_query = self._unit(
            np.concatenate(
                [
                    np.asarray(self.content[seeds], dtype=np.float32),
                    np.asarray(self.artist_content[artist_indices], dtype=np.float32),
                ]
            ).mean(axis=0)
        )
        scores = np.empty(len(self.catalog), dtype=np.float32)
        for start in range(0, len(scores), 25000):
            end = start + 25000
            scores[start:end] = (1 - self.content_weight) * (
                self.track_embeddings[start:end] @ query
            ) + self.content_weight * (
                np.asarray(self.content[start:end], dtype=np.float32) @ content_query
            )
        evidence = []
        if len(artist_indices):
            eligible, evidence = self.discovery.candidates(artist_indices)
            if not eligible.any():
                raise ValueError(
                    "No tracks share this artist inspiration's credits, collaborators, genres or sourced country in these datasets."
                )
            scores[~eligible] = -np.inf
        scores[seeds] = -np.inf
        count = min(k, int(np.isfinite(scores).sum()))
        if count:
            # Stable tie order makes repeated recommendations deterministic.
            top = np.argpartition(-scores, count - 1)[:count]
            top = top[np.lexsort((top, -scores[top]))]
        else:
            top = np.array([], dtype=np.int64)
        output = self.catalog.iloc[top].copy()
        logits = float(self.bundle["fit_calibration_scale"]) * scores[top] + float(
            self.bundle["fit_calibration_bias"]
        )
        output["match_percent"] = np.round(
            100 / (1 + np.exp(-np.clip(logits, -30, 30))), 1
        )
        output["rank"] = np.arange(1, len(output) + 1)
        output["collaborative_similarity"] = self.track_embeddings[top] @ query
        output["audio_similarity"] = np.nan
        output["reason"] = np.where(
            output.in_playlists,
            "Artist, genre and training playlist graph",
            "Artist and audio metadata; no source playlist history",
        )
        if len(artist_indices):
            output["reason"] = (
                "Connected to your artist inspiration by a genre, country or shared artist credit"
            )
        metadata = {
            "artist_candidate_filter_used": bool(len(artist_indices)),
            "artist_metadata_evidence": evidence,
            "eligible_track_count": int(np.isfinite(scores).sum()),
            "recognized_track_ids": recognized,
            "unknown_track_ids": unknown,
            "recognized_artist_ids": recognized_artists,
            "unknown_artist_ids": unknown_artists,
            "model_type": self.bundle["model_type"],
            "model_id": self.bundle["model_id"],
            "selected_content_weight": self.content_weight,
            "catalog_only_seed_count": int(
                (~self.catalog.iloc[seeds].in_playlists).sum()
            ),
            "match_percent_definition": "Validation-calibrated relative fit, not a liking probability.",
        }
        return output.reset_index(drop=True), metadata


def load_default_recommender(project_root, model_id="lightgcn"):
    root = Path(project_root) / "models"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    entry = next((x for x in manifest["models"] if x["id"] == model_id), None)
    if entry is None:
        raise ValueError(f"Model {model_id!r} is not trained.")
    return PlaylistRecommender(
        root / entry["file"],
        root / "track_catalog.csv.gz",
        generation=manifest["generation"],
    )
