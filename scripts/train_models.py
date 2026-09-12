"""Three scalable GNN encoders over the same training-only heterogeneous graph.

LightGCN uses E0=XW, E=mean(A^h X)W, h=0..2. This is an attribute-based
LightGCN variant, not the canonical free node-ID embedding baseline. SIGN
precomputes graph propagation, then learns nonlinear hop-specific encoders.
"""

from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression

try:
    from .data_pipeline import ROOT, OUT, REPORTS, SEED, propagate
except ImportError:
    from data_pipeline import ROOT, OUT, REPORTS, SEED, propagate

SPECS = {
    "lightgcn": {"name": "LightGCN", "hops": 2, "hidden": 32, "dim": 32, "depth": 0},
    "sign": {"name": "SIGN", "hops": 2, "hidden": 128, "dim": 64, "depth": 2},
    "residual_sign": {
        "name": "Residual SIGN",
        "hops": 4,
        "hidden": 256,
        "dim": 128,
        "depth": 4,
    },
}


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


class GraphEncoder(nn.Module):
    def __init__(self, model_id, input_dim=64):
        super().__init__()
        self.model_id = model_id
        s = SPECS[model_id]
        self.hops = s["hops"]
        if model_id == "lightgcn":
            self.projection = nn.Linear(input_dim, s["dim"], bias=False)
        else:
            self.branches = nn.ModuleList(
                nn.Linear(input_dim, s["hidden"]) for _ in range(self.hops + 1)
            )
            self.layers = nn.ModuleList(
                nn.Linear(s["hidden"], s["hidden"]) for _ in range(s["depth"])
            )
            self.output = nn.Linear(s["hidden"], s["dim"], bias=False)

    def forward(self, x):
        if self.model_id == "lightgcn":
            y = self.projection(x.mean(dim=1))
        else:
            y = sum(F.relu(layer(x[:, i])) for i, layer in enumerate(self.branches)) / (
                self.hops + 1
            )
            for layer in self.layers:
                z = F.relu(layer(y))
                y = (y + z) / 2 if self.model_id == "residual_sign" else z
            y = self.output(y)
        return F.normalize(y, dim=1, eps=1e-8)


def inputs(features, ids, hops, device):
    return torch.from_numpy(
        np.stack([f[ids] for f in features[: hops + 1]], axis=1)
    ).to(device)


def fingerprint(tracks, split):
    h = hashlib.sha256()
    h.update("\n".join(tracks.spotify_track_id).encode())
    h.update(pd.util.hash_pandas_object(split, index=False).to_numpy().tobytes())
    for path in [OUT / "features_0.npy", OUT / "train_graph.npz"]:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
    return h.hexdigest()


def candidates(split, which, pool, max_users, seed):
    rng = np.random.default_rng(seed)
    known = split.groupby("playlist_node_id").track_node_id.agg(list).to_dict()
    train = (
        split.loc[split.split.eq("train")]
        .groupby("playlist_node_id")
        .track_node_id.agg(list)
        .to_dict()
    )
    held = (
        split.loc[split.split.eq(which)]
        .groupby("playlist_node_id")
        .track_node_id.agg(list)
        .to_dict()
    )
    users = np.array(sorted(held))
    if max_users and len(users) > max_users:
        users = np.sort(rng.choice(users, max_users, replace=False))
    result = []
    for user in users:
        forbidden = set(known[user])
        n = min(100, len(pool) - len(forbidden))
        negatives = set()
        while len(negatives) < n:
            draws = rng.choice(pool, max(128, 2 * n))
            negatives.update(int(i) for i in draws if int(i) not in forbidden)
        neg = np.array(sorted(negatives), dtype=np.int64)
        if len(neg) > n:
            neg = rng.choice(neg, n, replace=False)
        positive = np.array(sorted(held[user]), dtype=np.int64)
        result.append(
            (
                int(user),
                np.array(train[user], dtype=np.int64),
                np.r_[positive, neg],
                len(positive),
            )
        )
    return result


def evaluate(
    embeddings, cases, content, weight=0, baseline=None, popularity=None, collect=False
):
    rows = []
    scores_all = []
    labels_all = []
    recommended = set()
    rng = np.random.default_rng(SEED)
    for user, seeds, candidate, n_positive in cases:
        if baseline == "random":
            scores = rng.random(len(candidate))
        elif baseline == "popularity":
            scores = popularity[candidate]
        else:
            query = unit(embeddings[seeds].mean(axis=0))
            scores = (1 - weight) * (embeddings[candidate] @ query) + weight * (
                content[candidate] @ unit(content[seeds].mean(axis=0))
            )
        order = np.argsort(-scores, kind="stable")[:10]
        hits = order < n_positive
        dcg = float(np.sum(hits / np.log2(np.arange(len(order)) + 2)))
        ideal = float(np.sum(1 / np.log2(np.arange(min(n_positive, 10)) + 2)))
        hit_indices = np.flatnonzero(hits)
        rows.append(
            {
                "playlist_node_id": user,
                "recall_at_10": float(hits.sum() / n_positive),
                "ndcg_at_10": dcg / max(ideal, 1e-12),
                "hit_rate_at_10": float(hits.any()),
                "mrr_at_10": float(1 / (hit_indices[0] + 1)) if len(hit_indices) else 0,
            }
        )
        recommended.update(candidate[order].tolist())
        if collect:
            scores_all.extend(scores.tolist())
            labels_all.extend([1] * n_positive + [0] * (len(candidate) - n_positive))
    frame = pd.DataFrame(rows)
    metrics = frame.drop(columns="playlist_node_id").mean().to_dict()
    metrics.update(
        {
            "evaluated_playlists": len(cases),
            "recommended_unique_tracks": len(recommended),
            "catalog_coverage_at_10": len(recommended) / len(content),
        }
    )
    return metrics, frame, (scores_all, labels_all)


@torch.inference_mode()
def embed(model, features, nt, device, batch_size=8192, offset=0):
    result = np.empty((nt, SPECS[model.model_id]["dim"]), dtype=np.float32)
    model.eval()
    for start in range(0, nt, batch_size):
        ids = np.arange(start, min(start + batch_size, nt))
        result[ids] = (
            model(inputs(features, ids + offset, model.hops, device)).cpu().numpy()
        )
    return result


def run(
    epochs=10,
    batch_size=2048,
    lr=0.001,
    weight_decay=0.0001,
    max_eval_playlists=1024,
    device="cpu",
    model_ids=None,
    resume=False,
):
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    torch.set_num_threads(4)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Use --device cpu or update the NVIDIA driver and install a compatible PyTorch build."
        )
    if not (OUT / "features_4.npy").exists():
        propagate()
    features = [np.load(OUT / f"features_{i}.npy", mmap_mode="r") for i in range(5)]
    tracks = pd.read_parquet(OUT / "tracks.parquet")
    split = pd.read_parquet(OUT / "split.parquet")
    artists = pd.read_parquet(OUT / "artists.parquet")
    nt = len(tracks)
    pool = np.sort(split.track_node_id.unique()).astype(np.int64)
    train = split.loc[split.split.eq("train")]
    # Website-style seed query is the mean of a training playlist's track embeddings.
    positives = train.track_node_id.to_numpy(dtype=np.int64)
    users = train.playlist_node_id.to_numpy(dtype=np.int64)
    train_by_user = train.groupby("playlist_node_id").track_node_id.agg(list).to_dict()
    known_keys = np.sort(
        (
            split.playlist_node_id.to_numpy(dtype=np.int64) * nt
            + split.track_node_id.to_numpy(dtype=np.int64)
        )
    )
    val = candidates(split, "val", pool, max_eval_playlists, SEED + 1)
    test = candidates(split, "test", pool, max_eval_playlists, SEED + 2)
    if not val or not test:
        raise ValueError(
            "No evaluable playlists: at least one playlist with five distinct tracks is required."
        )
    content = unit(np.asarray(features[0][:nt]).copy())
    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    generation = fingerprint(tracks, split)
    config = {
        "seed": SEED,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "optimizer": "Adam",
        "loss": "BPR",
        "device": device,
        "max_eval_playlists": max_eval_playlists,
        "negatives_per_evaluation_playlist": 100,
        "candidate_protocol": "all held-out positives plus up to 100 sampled non-positive playlist tracks; train seeds; same candidates for all models",
        "graph": "train-only playlist-track plus all catalog track-artist, artist-genre, sourced artist-country edges",
        "negative_pool": "playlist-observed tracks only; excludes every known positive for each playlist",
        "score": "mean seed track embeddings; validation-selected metadata mixture",
    }
    manifest = {
        "schema_version": 2,
        "generation": generation,
        "config": config,
        "models": [],
    }
    manifest_path = model_dir / "manifest.json"
    if resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("generation") != generation or previous.get("config") != config:
            raise ValueError("Cannot resume: dataset or training settings changed.")
        for entry in previous["models"]:
            path = model_dir / entry["file"]
            if path.is_file():
                bundle = torch.load(path, map_location="cpu", weights_only=False)
                if bundle.get("generation") != generation or bundle.get("model_id") != entry["id"]:
                    raise ValueError(f"Incompatible saved model: {path}")
                manifest["models"].append(entry)
                del bundle
                gc.collect()
    # Validate compatibility before replacing any shared inference artifacts.
    tracks.to_csv(model_dir / "track_catalog.csv.gz", index=False, compression="gzip")
    try:
        from .discovery import export_discovery_index
    except ImportError:
        from discovery import export_discovery_index
    export_discovery_index(OUT, model_dir)
    np.save(model_dir / "content_features.npy", content.astype(np.float16))
    artists.to_parquet(model_dir / "artist_catalog.parquet", index=False)
    np.save(
        model_dir / "artist_content_features.npy",
        unit(np.asarray(features[0][nt : nt + len(artists)]).copy()).astype(np.float16),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    popularity = np.bincount(positives, minlength=nt).astype(float)
    summary = []
    for baseline in ["random", "popularity"]:
        metrics, _, _ = evaluate(
            None, test, content, baseline=baseline, popularity=popularity
        )
        summary.append({"model": baseline, **metrics})
    summary.extend(entry["metrics"] for entry in manifest["models"])
    for model_id in model_ids or list(SPECS):
        if model_id not in SPECS:
            raise ValueError(f"Unknown model: {model_id}")
        if any(entry["id"] == model_id for entry in manifest["models"]):
            print(f"Keeping completed {model_id}", flush=True)
            continue
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model = GraphEncoder(model_id).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        best = -float("inf")
        history = []
        started = time.monotonic()
        best_state = None
        # Reset the sampler for every model: identical positive order, seeds and negatives.
        rng = np.random.default_rng(SEED)
        checkpoint_path = model_dir / f"{model_id}_training_resume.pt"
        elapsed_before = 0.0
        if resume and checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if checkpoint["generation"] != generation or checkpoint["config"] != config:
                raise ValueError("Cannot resume epoch: dataset or training settings changed.")
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            history = checkpoint["history"]
            best, best_epoch, best_state = checkpoint["best"], checkpoint["best_epoch"], checkpoint["best_state"]
            rng.bit_generator.state = checkpoint["rng_state"]
            elapsed_before = history[-1]["elapsed_seconds"]
            del checkpoint
            print(f"Resuming {model_id} after epoch {len(history)}", flush=True)
        for epoch in range(len(history), epochs):
            model.train()
            loss_sum = 0.0
            permutation = rng.permutation(len(train))
            for begin in range(0, len(train), batch_size):
                selected = permutation[begin : begin + batch_size]
                p = positives[selected]
                u = users[selected]
                # Exclude target from its seed playlist; singletons have no seed-completion example.
                keep = np.array([len(train_by_user[int(user)]) > 1 for user in u])
                p = p[keep]
                u = u[keep]
                if not len(p):
                    continue
                seed_ids = np.array(
                    [
                        rng.choice([t for t in train_by_user[int(user)] if t != target])
                        for user, target in zip(u, p)
                    ],
                    dtype=np.int64,
                )
                negatives = rng.choice(pool, len(p))
                while True:
                    keys = u * nt + negatives
                    pos = np.searchsorted(known_keys, keys)
                    bad = (pos < len(known_keys)) & (
                        known_keys[np.minimum(pos, len(known_keys) - 1)] == keys
                    )
                    if not bad.any():
                        break
                    negatives[bad] = rng.choice(pool, int(bad.sum()))
                ids, inverse = np.unique(
                    np.r_[seed_ids, p, negatives], return_inverse=True
                )
                encoded = model(inputs(features, ids, model.hops, device))
                q, pos_vec, neg_vec = encoded[
                    torch.from_numpy(inverse).to(device)
                ].chunk(3)
                margin = (q * pos_vec).sum(1) - (q * neg_vec).sum(1)
                loss = F.softplus(-margin * 5).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(p)
            vectors = embed(model, features, nt, device)
            metrics, _, _ = evaluate(vectors, val, content)
            row = {
                "epoch": epoch + 1,
                "bpr_loss": loss_sum / len(train),
                "validation_ndcg_at_10": metrics["ndcg_at_10"],
                "elapsed_seconds": elapsed_before + time.monotonic() - started,
            }
            history.append(row)
            print(model_id, json.dumps(row), flush=True)
            pd.DataFrame(history).to_csv(
                REPORTS / f"{model_id}_training_history.csv", index=False
            )
            if metrics["ndcg_at_10"] > best:
                best = metrics["ndcg_at_10"]
                best_state = copy.deepcopy(
                    {k: v.cpu() for k, v in model.state_dict().items()}
                )
                best_epoch = epoch + 1
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save({
                "generation": generation, "config": config,
                "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                "history": history, "best": best, "best_epoch": best_epoch,
                "best_state": best_state, "rng_state": rng.bit_generator.state,
            }, temporary)
            temporary.replace(checkpoint_path)
            del vectors
        model.load_state_dict(best_state)
        vectors = embed(model, features, nt, device)
        trials = []
        for weight in [0.0, 0.15, 0.35, 0.6]:
            metrics, _, _ = evaluate(vectors, val, content, weight=weight)
            trials.append((metrics["ndcg_at_10"], weight))
        _, weight = max(trials, key=lambda pair: (pair[0], -pair[1]))
        validation, _, (scores, labels) = evaluate(
            vectors, val, content, weight=weight, collect=True
        )
        calibration = LogisticRegression(
            class_weight="balanced", random_state=SEED
        ).fit(np.array(scores).reshape(-1, 1), labels)
        # Test is evaluated only after checkpoint, mixture and calibration are fixed.
        metrics, segments, _ = evaluate(vectors, test, content, weight=weight)
        metrics.update(
            {
                "model": model_id,
                "best_epoch": best_epoch,
                "parameters": sum(p.numel() for p in model.parameters()),
                "training_seconds": elapsed_before + time.monotonic() - started,
                "metadata_weight": weight,
            }
        )
        artist_vectors = embed(model, features, len(artists), device, offset=nt)
        filename = f"{model_id}_playlist_recommender.pt"
        bundle = {
            "schema_version": 2,
            "generation": generation,
            "model_id": model_id,
            "model_type": SPECS[model_id]["name"],
            "architecture": SPECS[model_id],
            "state_dict": best_state,
            "final_track_embeddings": torch.from_numpy(vectors.astype(np.float16)),
            "final_artist_embeddings": torch.from_numpy(
                artist_vectors.astype(np.float16)
            ),
            "fit_calibration_scale": float(calibration.coef_[0, 0]),
            "fit_calibration_bias": float(calibration.intercept_[0]),
            "config": {**config, "selected_content_weight": weight},
            "validation_metrics": validation,
            "test_metrics": metrics,
        }
        temporary = model_dir / (filename + ".tmp")
        torch.save(bundle, temporary)
        temporary.replace(model_dir / filename)
        pd.DataFrame(history).to_csv(
            REPORTS / f"{model_id}_training_history.csv", index=False
        )
        segments.to_csv(REPORTS / f"{model_id}_test_playlist_metrics.csv", index=False)
        summary.append(metrics)
        manifest["models"].append(
            {
                "id": model_id,
                "name": SPECS[model_id]["name"],
                "file": filename,
                "architecture": SPECS[model_id],
                "metrics": metrics,
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        pd.DataFrame(summary).to_csv(REPORTS / "model_comparison.csv", index=False)
        print(f"Saved {filename}", flush=True)
        del model, optimizer, vectors, artist_vectors, best_state, bundle
        gc.collect()
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--max-eval-playlists", type=int, default=1024)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--resume", action="store_true", help="Keep compatible completed models and resume saved epochs.")
    args = parser.parse_args()
    run(**vars(args))
