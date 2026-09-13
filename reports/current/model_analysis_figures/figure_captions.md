# Three-model analysis figures

These figures come from `models/manifest.json`, `reports/current/model_comparison.csv`, and the three saved training-history CSV files. They describe the completed experiment; no models were retrained to create them. PNG files are ready for insertion into a dissertation, and matching SVG files allow lossless resizing.

1. **Model architectures (`01_model_architectures`)** — LightGCN uses two hops and a linear 32-dimensional projection (2,048 parameters). SIGN uses two hops, separate nonlinear projections, two trunk layers and 64-dimensional output (66,176 parameters). Residual SIGN uses four hops, four residual trunk layers and 128-dimensional output (379,136 parameters). The larger models differ in several ways at once, so the experiment cannot isolate which architectural change caused a score difference.

2. **Training and validation (`02_training_and_validation`)** — Training BPR loss continued to decrease through epoch 10, but validation NDCG@10 peaked at epoch 1 for all three models. The plotted stars identify the selected checkpoints. This gap is consistent with diminishing generalization after the first epoch under this protocol, but does not by itself establish a specific cause.

3. **Held-out results and cost (`03_test_results_and_cost`)** — NDCG@10 was 0.623 for LightGCN, 0.642 for SIGN and 0.665 for Residual SIGN. The gain from LightGCN to Residual SIGN was 0.043 absolute points, with 185.1 times as many parameters and 4.4 times the full-run CPU training time. The baseline is training-popularity ranking. These are sampled-candidate results on 1,024 held-out playlists, with all held-out positives and up to 100 sampled negatives per playlist; they do not measure full-catalog or artist-only recommendation quality.
