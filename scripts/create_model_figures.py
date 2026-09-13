"""Create dissertation-ready comparisons from the saved three-model experiment."""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports/current"
OUT = REPORTS / "model_analysis_figures"
ORDER = ["lightgcn", "sign", "residual_sign"]
COLORS = {"lightgcn": "#0072B2", "sign": "#D55E00", "residual_sign": "#009E73"}


def save(fig, stem):
    for extension in ("png", "svg"):
        fig.savefig(OUT / f"{stem}.{extension}", dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, x, y, width, height, label, color):
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.03", linewidth=1.3, edgecolor=color, facecolor="white"))
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=10, linespacing=1.35)


def architecture(models):
    fig, ax = plt.subplots(figsize=(14, 7.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    fig.suptitle("Three graph encoders: what changes", x=0.06, y=0.98, ha="left", fontsize=19, weight="bold")
    for index, model_id in enumerate(ORDER):
        item = models[model_id]
        spec = item["architecture"]
        metrics = item["metrics"]
        top = 5.75 - index * 2.02
        color = COLORS[model_id]
        ax.text(0.2, top + 0.91, item["name"], fontsize=15, weight="bold", color=color)
        stages = [
            (2.2, "64-D node\nfeatures"),
            (4.75, f"Graph hops\n0 to {spec['hops']}"),
            (7.3, "Mean + linear\nprojection" if model_id == "lightgcn" else f"Hop-specific branches\n{spec['hidden']}-D hidden"),
            (10.25, "No nonlinear\ntrunk" if model_id == "lightgcn" else f"{spec['depth']} {'residual ' if model_id == 'residual_sign' else ''}layers"),
            (12.55, f"{spec['dim']}-D\noutput"),
        ]
        widths = [2.15, 2.15, 2.55, 1.95, 1.1]
        for (x, label), width in zip(stages, widths):
            box(ax, x, top - 0.03, width, 0.77, label, color)
        for (x, _), width, (next_x, _) in zip(stages[:-1], widths[:-1], stages[1:]):
            ax.add_patch(FancyArrowPatch((x + width + 0.04, top + 0.35), (next_x - 0.06, top + 0.35), arrowstyle="-|>", mutation_scale=12, color="#64748B", linewidth=1.1))
        ax.text(2.23, top - 0.31, f"{metrics['parameters']:,} trainable parameters", fontsize=10, color="#334155")
        if index < 2:
            ax.axhline(top - 0.67, color="#E2E8F0", lw=1)
    fig.text(0.06, 0.035, "All use the same train-only graph, 64-D input features, BPR loss, and evaluation protocol. Hops are precomputed.", fontsize=10, color="#475569")
    save(fig, "01_model_architectures")


def training(models):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), layout="constrained")
    for model_id in ORDER:
        item = models[model_id]
        history = pd.read_csv(REPORTS / f"{model_id}_training_history.csv")
        color = COLORS[model_id]
        axes[0].plot(history.epoch, history.validation_ndcg_at_10, marker="o", ms=4, lw=2.2, color=color, label=item["name"])
        axes[1].plot(history.epoch, history.bpr_loss, marker="o", ms=4, lw=2.2, color=color, label=item["name"])
        best = history.loc[history.epoch.eq(item["metrics"]["best_epoch"])].iloc[0]
        axes[0].scatter([best.epoch], [best.validation_ndcg_at_10], color=color, s=95, marker="*", zorder=5)
    axes[0].set(title="Validation ranking quality", xlabel="Epoch", ylabel="Validation NDCG@10")
    axes[1].set(title="Training objective", xlabel="Epoch", ylabel="BPR loss (lower is better)")
    for ax in axes:
        ax.set_xticks(range(1, 11))
        ax.grid(axis="y", color="#E2E8F0")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Training loss kept falling; validation chose epoch 1", fontsize=18, weight="bold")
    fig.text(0.5, -0.025, "Stars mark the selected checkpoints. Validation rankings and training loss measure different objectives; test results are shown separately.", ha="center", fontsize=9.5, color="#475569")
    save(fig, "02_training_and_validation")


def outcomes(models, comparison):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7), gridspec_kw={"width_ratios": [1.45, 1]}, layout="constrained")
    metrics = [("recall_at_10", "Recall@10"), ("ndcg_at_10", "NDCG@10"), ("hit_rate_at_10", "HitRate@10"), ("mrr_at_10", "MRR@10")]
    positions = np.arange(len(metrics))
    width = 0.23
    for index, model_id in enumerate(ORDER):
        item = models[model_id]
        values = [item["metrics"][key] for key, _ in metrics]
        bars = axes[0].bar(positions + (index - 1) * width, values, width, label=item["name"], color=COLORS[model_id])
        axes[0].bar_label(bars, fmt="%.3f", padding=3, fontsize=7.8, rotation=90)
        seconds = item["metrics"]["training_seconds"]
        score = item["metrics"]["ndcg_at_10"]
        axes[1].scatter(seconds / 60, score, s=130, color=COLORS[model_id], zorder=3)
        axes[1].annotate(f"{item['name']}\n{item['metrics']['parameters']:,} parameters", (seconds / 60, score), xytext=(8, -5 if index != 1 else 8), textcoords="offset points", fontsize=9, color=COLORS[model_id])
    baseline = comparison.set_index("model").loc["popularity", "ndcg_at_10"]
    axes[1].axhline(baseline, color="#64748B", ls="--", lw=1.3, label=f"Popularity baseline: {baseline:.3f}")
    axes[0].set(xticks=positions, xticklabels=[name for _, name in metrics], ylim=(0, 1.02), ylabel="Held-out score", title="Ranking metrics on the same test sample")
    axes[1].set(xlabel="Full 10-epoch CPU training time (minutes)", ylabel="Held-out NDCG@10", title="Quality versus computation", xlim=(0, 24), ylim=(0.35, 0.72))
    axes[0].legend(frameon=False, loc="upper left", ncol=3, fontsize=9)
    axes[1].legend(frameon=False, loc="lower right", fontsize=8.5)
    for ax in axes:
        ax.grid(axis="y", color="#E2E8F0", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Held-out results and compute trade-off", fontsize=18, weight="bold")
    fig.text(0.5, -0.025, "1,024 sampled test playlists; all held-out positives plus up to 100 sampled negatives each. Scores are not full-catalog ranking metrics.", ha="center", fontsize=9.5, color="#475569")
    save(fig, "03_test_results_and_cost")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 13, "savefig.pad_inches": 0.18})
    manifest = json.loads((ROOT / "models/manifest.json").read_text(encoding="utf-8"))
    models = {item["id"]: item for item in manifest["models"]}
    comparison = pd.read_csv(REPORTS / "model_comparison.csv")
    assert all(model_id in models for model_id in ORDER)
    assert all(model_id in set(comparison.model) for model_id in ORDER)
    architecture(models)
    training(models)
    outcomes(models, comparison)
    light = models["lightgcn"]["metrics"]
    sign = models["sign"]["metrics"]
    residual = models["residual_sign"]["metrics"]
    captions = f"""# Three-model analysis figures

These figures come from `models/manifest.json`, `reports/current/model_comparison.csv`, and the three saved training-history CSV files. They describe the completed experiment; no models were retrained to create them. PNG files are ready for insertion into a dissertation, and matching SVG files allow lossless resizing.

1. **Model architectures (`01_model_architectures`)** — LightGCN uses two hops and a linear 32-dimensional projection ({light['parameters']:,} parameters). SIGN uses two hops, separate nonlinear projections, two trunk layers and 64-dimensional output ({sign['parameters']:,} parameters). Residual SIGN uses four hops, four residual trunk layers and 128-dimensional output ({residual['parameters']:,} parameters). The larger models differ in several ways at once, so the experiment cannot isolate which architectural change caused a score difference.

2. **Training and validation (`02_training_and_validation`)** — Training BPR loss continued to decrease through epoch 10, but validation NDCG@10 peaked at epoch 1 for all three models. The plotted stars identify the selected checkpoints. This gap is consistent with diminishing generalization after the first epoch under this protocol, but does not by itself establish a specific cause.

3. **Held-out results and cost (`03_test_results_and_cost`)** — NDCG@10 was {light['ndcg_at_10']:.3f} for LightGCN, {sign['ndcg_at_10']:.3f} for SIGN and {residual['ndcg_at_10']:.3f} for Residual SIGN. The gain from LightGCN to Residual SIGN was {residual['ndcg_at_10'] - light['ndcg_at_10']:.3f} absolute points, with {residual['parameters'] / light['parameters']:.1f} times as many parameters and {residual['training_seconds'] / light['training_seconds']:.1f} times the full-run CPU training time. The baseline is training-popularity ranking. These are sampled-candidate results on 1,024 held-out playlists, with all held-out positives and up to 100 sampled negatives per playlist; they do not measure full-catalog or artist-only recommendation quality.
"""
    (OUT / "figure_captions.md").write_text(captions, encoding="utf-8")
    print(f"Created three PNG/SVG figure pairs and captions in {OUT}")


if __name__ == "__main__":
    main()
