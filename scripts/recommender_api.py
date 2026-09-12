"""Local JSON API; loads only the selected model to fit an 8 GB laptop."""

from __future__ import annotations
import argparse
import gc
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import unicodedata
import numpy as np
import pandas as pd
import torch

try:
    from .recommender_inference import PlaylistRecommender
except ImportError:
    from recommender_inference import PlaylistRecommender
ROOT = Path(__file__).resolve().parents[1]


def normalized(value):
    return " ".join(
        "".join(
            c
            for c in unicodedata.normalize("NFKD", str(value).casefold())
            if not unicodedata.combining(c)
        ).split()
    )


class ModelStore:
    def __init__(self, root=ROOT):
        self.root = Path(root) / "models"
        self.lock = threading.Lock()
        self.engine = None
        self.selected = None
        self.manifest = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )
        self.entries = {
            x["id"]: x
            for x in self.manifest["models"]
            if (self.root / x["file"]).is_file()
        }
        self.catalog = (
            pd.read_csv(self.root / "track_catalog.csv.gz")
            .sort_values("track_node_id")
            .reset_index(drop=True)
        )
        self.catalog["track_title"] = self.catalog.track_title.fillna("Unknown track")
        self.catalog["artist_name"] = self.catalog.artist_name.fillna("Unknown artist")
        self.catalog["spud_popularity"] = self.catalog.spud_popularity.fillna(0)
        self.titles = self.catalog.track_title.map(normalized)
        self.artists = self.catalog.artist_name.map(normalized)
        try:
            from .discovery import DiscoveryIndex
        except ImportError:
            from discovery import DiscoveryIndex
        self.discovery = DiscoveryIndex(self.root)
        self.artist_catalog = pd.read_parquet(self.root / "artist_catalog.parquet")
        self.artist_names = self.artist_catalog.artist_name.map(normalized)
        self.artist_content = np.load(
            self.root / "artist_content_features.npy", mmap_mode="r"
        )
        self.content = np.load(self.root / "content_features.npy", mmap_mode="r")
        self.popular = np.argsort(
            -self.catalog.spud_popularity.to_numpy(), kind="stable"
        )

    def health(self):
        return {
            "status": "ready" if self.entries else "untrained",
            "trackCount": len(self.catalog),
            "catalogOnlyTrackCount": int((~self.catalog.in_playlists).sum()),
            "models": [
                {
                    "id": x["id"],
                    "name": x["name"],
                    "architecture": x["architecture"],
                    "metrics": x["metrics"],
                }
                for x in self.entries.values()
            ],
        }

    def search(self, query, limit=8):
        q = normalized(query)
        if not q:
            return self.catalog.iloc[self.popular[:limit]]
        matches = np.flatnonzero(
            (
                self.titles.str.contains(q, regex=False)
                | self.artists.str.contains(q, regex=False)
            ).to_numpy()
        )
        priority = (
            500 * self.titles.iloc[matches].eq(q).to_numpy()
            + 400 * self.artists.iloc[matches].eq(q).to_numpy()
            + 80 * self.titles.iloc[matches].str.startswith(q).to_numpy()
            + 60 * self.artists.iloc[matches].str.startswith(q).to_numpy()
            + self.catalog.iloc[matches].spud_popularity.to_numpy()
        )
        return self.catalog.iloc[matches[np.lexsort((matches, -priority))[:limit]]]

    def search_artists(self, query, limit=5):
        q = normalized(query)
        if not q:
            return []
        matches = np.flatnonzero(
            self.artist_names.str.contains(q, regex=False).to_numpy()
        )
        priority = (
            100 * self.artist_names.iloc[matches].eq(q).to_numpy()
            + 10 * self.artist_names.iloc[matches].str.startswith(q).to_numpy()
        )
        selected = matches[np.lexsort((matches, -priority))[:limit]]
        return [
            {
                "spotifyArtistId": str(r.spotify_artist_id),
                "name": str(r.artist_name),
                "genres": list(r.genres),
                "trackCount": int(r.track_count),
            }
            for r in self.artist_catalog.iloc[selected].itertuples()
        ]

    def recommend(self, model_id, track_ids, limit, artist_ids=()):
        if model_id not in self.entries:
            raise ValueError(f"Unknown or untrained model: {model_id}")
        with self.lock:
            if self.selected != model_id:
                self.engine = None
                gc.collect()
                self.engine = PlaylistRecommender(
                    self.root / self.entries[model_id]["file"],
                    catalog=self.catalog,
                    content=self.content,
                    artists=self.artist_catalog,
                    artist_content=self.artist_content,
                    discovery=self.discovery,
                    generation=self.manifest["generation"],
                )
                self.selected = model_id
            return self.engine.recommend(track_ids, limit, artist_ids=artist_ids)


def records(frame):
    result = []
    for row in frame.to_dict(orient="records"):
        item = {
            "trackNodeId": int(row["track_node_id"]),
            "spotifyTrackId": str(row["spotify_track_id"]),
            "title": str(row["track_title"]),
            "artist": str(row["artist_name"]),
            "popularity": float(row["spud_popularity"]),
            "audioAvailable": bool(row["audio_features_available"]),
            "inPlaylists": bool(row["in_playlists"]),
            "inCatalog": bool(row["in_catalog"]),
        }
        if "rank" in row:
            item.update(
                {
                    "rank": int(row["rank"]),
                    "matchPercent": float(row["match_percent"]),
                    "reason": row["reason"],
                }
            )
        result.append(item)
    return result


def positive_int(value, maximum):
    if isinstance(value, bool):
        raise ValueError("limit must be an integer")
    n = int(value)
    if str(n) != str(value) or not 1 <= n <= maximum:
        raise ValueError(f"limit must be between 1 and {maximum}")
    return n


class Handler(BaseHTTPRequestHandler):
    def _json(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        try:
            url = urlparse(self.path)
            if url.path == "/health":
                self._json(self.server.store.health())
                return
            if url.path == "/search":
                params = parse_qs(url.query)
                query = params.get("q", [""])[0]
                if len(query) > 200:
                    raise ValueError("Search query is too long.")
                limit = positive_int(params.get("limit", ["8"])[0], 50)
                self._json(
                    {
                        "results": records(self.server.store.search(query, limit)),
                        "artists": self.server.store.search_artists(query),
                        "query": query,
                    }
                )
                return
            self._json({"error": "Not found"}, 404)
        except (ValueError, TypeError) as e:
            self._json({"error": str(e)}, 400)

    def do_POST(self):
        if urlparse(self.path).path != "/recommend":
            self._json({"error": "Not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 64000:
                self._json({"error": "Invalid request size"}, 413)
                return
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Request must be a JSON object.")
            ids = payload.get("trackIds", [])
            artist_ids = payload.get("artistIds", [])
            if (
                not isinstance(ids, list)
                or not isinstance(artist_ids, list)
                or not (ids or artist_ids)
                or len(ids) + len(artist_ids) > 500
                or not all(isinstance(x, str) for x in ids + artist_ids)
            ):
                raise ValueError("Supply between 1 and 500 track or artist ID strings.")
            model = payload.get("modelId", "lightgcn")
            if not isinstance(model, str):
                raise ValueError("modelId must be a string.")
            limit = positive_int(payload.get("limit", 8), 50)
            frame, metadata = self.server.store.recommend(
                model, ids, limit, artist_ids=artist_ids
            )
            self._json({"results": records(frame), "metadata": metadata})
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            self._json({"error": str(e)}, 400)
        except Exception:
            import traceback

            traceback.print_exc()
            self._json({"error": "Recommendation failed; inspect the API log."}, 500)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    torch.set_num_threads(4)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.store = ModelStore()
    print(f"Playlist API ready at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
