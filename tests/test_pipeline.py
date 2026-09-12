import json
import sqlite3
import threading
from urllib.request import urlopen, Request
from urllib.error import HTTPError
import numpy as np
import pandas as pd
import pytest
import torch
from scipy import sparse
from scripts import data_pipeline as data
from scripts import train_models as training
from scripts.recommender_api import ModelStore, Handler
from scripts.recommender_inference import load_default_recommender
from http.server import ThreadingHTTPServer


def sid(prefix, index):
    return prefix + str(index).zfill(21)


@pytest.fixture()
def music(tmp_path, monkeypatch):
    out = tmp_path / "data/processed/portable"
    reports = tmp_path / "reports/current"
    for module in [data, training]:
        monkeypatch.setattr(module, "ROOT", tmp_path)
        monkeypatch.setattr(module, "OUT", out)
        monkeypatch.setattr(module, "REPORTS", reports)
    raw = tmp_path / "data/raw"
    raw.mkdir(parents=True)
    tracks = []
    for i in range(24):
        row = {
            "id": sid("T", i),
            "name": f"Song {i}",
            "artists": repr(["Elvana Gjata"] if i >= 18 else [f"Artist {i%3}"]),
            "id_artists": repr([sid("A", 3)] if i >= 18 else [sid("A", i % 3)]),
            "popularity": 20,
        }
        row.update({col: 0.5 for col in data.AUDIO})
        row.update(duration_ms=180000, tempo=120, key=2, mode=1)
        tracks.append(row)
    tracks[0]["id_artists"] = repr([sid("A", 0), sid("A", 1)])
    pd.DataFrame(tracks).to_csv(raw / "tracks.csv", index=False)
    pd.DataFrame(
        [
            {
                "id": sid("A", i),
                "name": "Elvana Gjata" if i == 3 else f"Artist {i}",
                "genres": repr(
                    [] if i == 5 else (["albanian pop"] if i == 3 else ["pop"])
                ),
                "followers": 20,
                "popularity": 20,
            }
            for i in range(6)
        ]
    ).to_csv(raw / "artists.csv", index=False)
    directory = raw / "playlists/spud"
    directory.mkdir(parents=True)
    with sqlite3.connect(directory / "spud.sqlite") as con:
        con.executescript(
            "CREATE TABLE tracks(trackid INTEGER,spotifyid TEXT,title TEXT,artist INTEGER,popularity REAL,duration REAL);CREATE TABLE artists(artistid INTEGER,spotifyid TEXT,name TEXT);CREATE TABLE lastfmplaylists(playlistid INTEGER,title TEXT);CREATE TABLE lastfmplayliststracks(playlist INTEGER,track INTEGER);"
        )
        con.executemany(
            "INSERT INTO artists VALUES(?,?,?)",
            [(i, sid("A", i), f"Artist {i}") for i in range(3)],
        )
        con.executemany(
            "INSERT INTO tracks VALUES(?,?,?,?,?,?)",
            [(i, sid("T", i), f"Song {i}", i % 3, 0.2, 180) for i in range(18)],
        )
        con.executemany(
            "INSERT INTO lastfmplaylists VALUES(?,?)",
            [(i, f"Playlist {i}") for i in range(3)],
        )
        con.executemany(
            "INSERT INTO lastfmplayliststracks VALUES(?,?)",
            [(i // 6, i) for i in range(18)] + [(0, 0), (0, 9999)],
        )
    data.catalog()
    data.playlists()
    data.integrate()
    data.propagate()
    return tmp_path


def test_graph_retains_cold_artists_and_all_credits_without_leakage(music):
    tracks = pd.read_parquet(data.OUT / "tracks.parquet")
    relation = pd.read_parquet(data.OUT / "track_artist_edges.parquet")
    split = pd.read_parquet(data.OUT / "split.parquet")
    assert len(tracks) == 24
    assert len(tracks.loc[~tracks.in_playlists]) == 6
    assert len(relation.loc[relation.track_node_id.eq(0)]) == 2
    artists = pd.read_parquet(data.OUT / "artists.parquet")
    assert artists.country.eq("").all()
    graph = sparse.load_npz(data.OUT / "train_graph.npz")
    offset = len(tracks) + len(artists)
    for row in split.itertuples():
        assert bool(graph[row.track_node_id, offset + row.playlist_node_id]) == (
            row.split == "train"
        )
    assert np.allclose(
        data.split_edges(split[["playlist_node_id", "track_node_id"]]).index,
        split.index,
    )
    pd.testing.assert_frame_equal(
        data.split_edges(split[["playlist_node_id", "track_node_id"]]), split
    )


def test_lightgcn_equivalence_and_capacity():
    torch.manual_seed(42)
    x = torch.randn(6, 64)
    a = torch.rand(6, 6)
    a = a / a.sum(1, keepdim=True)
    hops = torch.stack([x, a @ x, a @ a @ x], dim=1)
    model = training.GraphEncoder("lightgcn")
    expected = torch.nn.functional.normalize(
        (model.projection(x) + a @ model.projection(x) + a @ a @ model.projection(x))
        / 3,
        dim=1,
    )
    assert torch.allclose(model(hops), expected, atol=1e-6)
    counts = []
    for name in training.SPECS:
        model = training.GraphEncoder(name)
        input = torch.randn(8, model.hops + 1, 64)
        values = model(input)
        (values[:, 0].sum()).backward()
        assert torch.isfinite(values).all()
        assert all(p.grad is not None for p in model.parameters())
        counts.append(sum(p.numel() for p in model.parameters()))
    assert counts[0] < counts[1] < counts[2]


def test_three_bundle_roundtrip_search_selection_and_http(music):
    manifest = training.run(epochs=1, batch_size=8, max_eval_playlists=3)
    assert [x["id"] for x in manifest["models"]] == list(training.SPECS)
    store = ModelStore(music)
    results = store.search("ELVANA GJATA")
    assert len(results) == 6 and not results.in_playlists.any()
    seed = str(results.iloc[0].spotify_track_id)
    for name in training.SPECS:
        frame, metadata = store.recommend(name, [seed, seed], 8)
        assert len(frame) == 8 and seed not in set(frame.spotify_track_id)
        assert metadata["model_id"] == name and metadata["catalog_only_seed_count"] == 1
        assert frame.match_percent.between(0, 100).all()
        engine = load_default_recommender(music, name)
        assert engine.recommend(store.catalog.spotify_track_id.tolist(), 8)[0].empty
    for name in training.SPECS:
        frame, metadata = store.recommend(name, [], 8, artist_ids=[sid("A", 4)])
        assert len(frame) == 8 and metadata["recognized_artist_ids"] == [sid("A", 4)]
        assert metadata["artist_candidate_filter_used"]
        assert metadata["artist_metadata_evidence"] == ["genre:pop"]
        assert frame.track_node_id.lt(18).all()
    assert store.search_artists("Artist 4")[0]["trackCount"] == 0
    with pytest.raises(ValueError, match="no tracks, genres, or country"):
        store.recommend("lightgcn", [], 8, artist_ids=[sid("A", 5)])
    with pytest.raises(ValueError):
        store.recommend("not_a_model", [seed], 8)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.store = store
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        assert len(json.load(urlopen(base + "/health"))["models"]) == 3
        for path, body in [
            ("/search?limit=abc", None),
            ("/recommend", []),
            ("/recommend", {"trackIds": [seed], "modelId": []}),
            ("/recommend", {"trackIds": [seed], "limit": 0}),
        ]:
            request = Request(
                base + path,
                data=json.dumps(body).encode() if body is not None else None,
                headers={"Content-Type": "application/json"},
            )
            with pytest.raises(HTTPError) as error:
                urlopen(request)
            assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_safe_lists():
    assert data.listed("__import__('os').system('anything')") == []
    assert data.listed("['a', 'b']") == ["a", "b"]


def test_resume_keeps_completed_models_and_rejects_changed_config(music):
    first = training.run(epochs=1, batch_size=8, max_eval_playlists=3, model_ids=["lightgcn"])
    path = music / "models/lightgcn_playlist_recommender.pt"
    original = path.read_bytes()
    resumed = training.run(epochs=1, batch_size=8, max_eval_playlists=3, resume=True)
    assert [entry["id"] for entry in resumed["models"]] == list(training.SPECS)
    assert path.read_bytes() == original
    assert resumed["models"][0] == first["models"][0]
    with pytest.raises(ValueError, match="settings changed"):
        training.run(epochs=2, batch_size=8, max_eval_playlists=3, resume=True)


def test_interrupted_epoch_resume_matches_uninterrupted_training(music, monkeypatch):
    original_embed = training.embed
    calls = 0

    def interrupted_embed(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Simulated interruption")
        return original_embed(*args, **kwargs)

    config = dict(epochs=2, batch_size=8, max_eval_playlists=3, model_ids=["residual_sign"])
    monkeypatch.setattr(training, "embed", interrupted_embed)
    with pytest.raises(RuntimeError, match="Simulated interruption"):
        training.run(**config)
    monkeypatch.setattr(training, "embed", original_embed)
    training.run(**config, resume=True)
    path = music / "models/residual_sign_playlist_recommender.pt"
    resumed = torch.load(path, weights_only=False)
    training.run(**config)
    fresh = torch.load(path, weights_only=False)
    assert torch.equal(resumed["final_track_embeddings"], fresh["final_track_embeddings"])
    assert resumed["validation_metrics"] == fresh["validation_metrics"]


def test_csv_zip_response_is_normalized(tmp_path):
    import zipfile
    from scripts.download_data import normalize_csv

    path = tmp_path / "tracks.csv"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("tracks.csv", "id,name\nabc,Song\n")
    normalize_csv(path)
    assert path.read_text() == "id,name\nabc,Song\n"
    normalize_csv(path)
