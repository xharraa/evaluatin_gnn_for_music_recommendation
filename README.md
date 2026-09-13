# Music recommendation with three graph neural networks

A local playlist builder backed by LightGCN, SIGN and Residual SIGN. Search tracks or artists, add/remove songs, choose artist inspirations, and switch models without losing your playlist.

The public demo is [Playlist Lab](https://playlist-lab-ruddy.vercel.app). Vercel hosts the interface; the three models run in Docker on this laptop through a temporary tunnel, so recommendations require the laptop to stay awake and online. After restarting Docker or Windows, run `powershell -ExecutionPolicy Bypass -File scripts/start_vercel_demo.ps1` from this folder to reconnect and redeploy. See the [deployment guide](docs/DEPLOYMENT.md) for the Vercel setup, restart/stop commands, private model files, and an always-online VPS alternative. GitHub is not required for the current deployment.

## Run on this laptop

From the repository root in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_environment.ps1
$MusicPython = Join-Path $env:USERPROFILE '.venvs\music\Scripts\python.exe'
& $MusicPython scripts/download_data.py
& $MusicPython scripts/run_notebooks.py
powershell -ExecutionPolicy Bypass -File scripts/start_web_demo.ps1
```

Open **http://localhost:3000**. After the datasets and checkpoints have been built, only the last command is needed. The playlist is local to the browser session; this app does not upload playlists to Spotify. The external link on a selected track opens it on Spotify.

Setup uses installed Python 3.13, a short environment path to avoid Windows' path-length limit, a CPU PyTorch wheel, and a checksum-verified project-local Node.js runtime. No Java, Spark, Windows Hadoop helpers, old repository name, or previous laptop path is required. `-VenvPath` overrides the environment location; the launcher reads `.tools/environment.json`.

The inspected laptop has an AMD Ryzen 5 4600H, approximately 8 GB RAM, and a GTX 1650 with 4 GB VRAM. Its NVIDIA driver 457.49 cannot use the previous CUDA 13 configuration. CPU is the working default. Optional `requirements-cuda.txt` uses CUDA 12.6 and requires a compatible updated driver. See [NVIDIA's driver compatibility requirements](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html). The current machine audit is saved in `reports/current/hardware.json`.

For manual notebook use in VS Code, select **Python (Music GNN)**. Run the notebooks in order using a fresh kernel for each. The automated runner saves executed outputs and stops on an error. To resume training after preprocessing:

```powershell
& $MusicPython scripts/run_notebooks.py --from-notebook 03
```

Notebook 03 now resumes compatible completed models and saved training epochs.
It verifies the dataset fingerprint and training settings before reusing work.
An epoch checkpoint includes optimizer, sampler and best-validation weights, so
an interrupted model can continue without changing the experiment. For a fresh
training run use `scripts/train_models.py`; add `--resume` to continue it.

The app loads song covers and artist photos from Spotify's public oEmbed metadata
using the exact Spotify IDs. Artwork refreshes daily and requires internet access;
unavailable artwork keeps the colored placeholder. Search and recommendations
continue to use the local datasets and saved models.

## Two datasets, one discovery collection

- [Spotify 600k-track catalog](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks/data): track audio features, every credited artist, and a larger artist metadata table.
- [University of Glasgow SPUD](https://www.dcs.gla.ac.uk/~daniel/spud/): Last.fm playlists mapped to Spotify IDs, plus track and artist metadata. These are not the Spotify Million Playlist Dataset.

`download_data.py` retrieves both sources. It detects ZIP responses incorrectly named `.csv`, extracts them and validates their headers. Interrupted SPUD downloads can resume. Raw source files and large generated artifacts stay outside Git.

The rebuilt union contains **1,309,344 tracks**, **1,199,540 artists**, **18,829 connected playlists**, and **1,480,427 track–artist links**. It includes catalog tracks, unconnected SPUD tracks, and artists without track records. Stable Spotify IDs are identity keys; names are never used to merge artists. Every credited artist is retained, so collaborations appear as shared-track graph paths.

Only **11,265 tracks** overlap between the catalog and observed playlists. Filtering discovery down to playlist membership discards much of the catalog. The new graph keeps those records and exposes missing audio/playlist information explicitly.

Artist inspirations select candidates through exact genre, sourced country, credited-artist or collaboration relationships. The chosen GNN then ranks that candidate set. Artists without credited tracks remain searchable as profiles when their metadata is available. No songs or playlist memberships are invented.

Country is absent from the supplied artist data. To add it, provide `data/raw/artist_countries.csv` with one sourced row per artist:

```csv
spotify_artist_id,country,source
```

Then rerun notebook 02 onward. Country is never inferred from artist names or genre names. Artist profiles with no tracks, genres or country cannot support an artist-only recommendation request; the API asks for a better seed.

## Notebook workflow

| Notebook | What it executes |
|---|---|
| `00_environment_check` | Python, RAM, disk, GPU driver and PyTorch checks |
| `01_preprocessing` | Catalog validation, safe artist-list parsing, audio missingness and correlations |
| `01b_playlist_preprocessing` | Read-only SQLite audit, edge deduplication, orphan removal, duration repair, recomputed sizes |
| `02_joining_and_segmentation` | Full track/artist union, graph relationships, deterministic split and sparse feature propagation |
| `02b_data_analysis` | Coverage, audio correlations, playlist sizes, genre and degree distributions |
| `02c_further_data_analysis` | Catalog audio correlations, playlist-degree relationships, artist-credit paths and graph integrity |
| `03_model_creation` | Three training runs, validation selection, held-out evaluation, checkpoint export and artist-seed demos |

The further analysis finds positive catalog correlations between energy and loudness (Pearson r = 0.765) and danceability and valence (r = 0.528), and a negative correlation between energy and acousticness (r = -0.715). Among the 11,265 catalog tracks observed in playlists, log playlist degree relates more to catalog popularity (r = 0.359) than to any individual audio feature (largest absolute audio r = 0.106). These are descriptive associations; the connected subset is not representative of the whole catalog. Artist credits provide additional graph paths, including 107,357 tracks with multiple credits. Tables and pairwise sample counts are saved in `reports/current/`.

The notebook code calls reusable modules in `scripts/`. Original Spark notebooks are preserved in `notebooks/legacy/` with their old outputs cleared. Old files directly under `reports/` and the old `lightgcn_model_card.json` are historical; **current results live in `reports/current/` and `models/manifest.json`**.

## Three models and shared training settings

| Model | Graph hops | Hidden width | Output dimensions | Nonlinear trunk layers |
|---|---:|---:|---:|---:|
| LightGCN | 2 | 32 | 32 | 0 |
| SIGN | 2 | 128 | 64 | 2 |
| Residual SIGN | 4 | 256 | 128 | 4 |

All use seed 42, 10 epochs, batch size 2,048, Adam, learning rate 0.001, weight decay 0.0001, the same positive ordering/negative sampler, and the same validation/test candidates. Larger capacity is an experimental variable, not a promise of better recommendations.

**LightGCN is an attribute-based variant**, not the canonical free node-ID embedding baseline. With initial embeddings `E0 = XW`, averaging `A^h X` and then applying `W` is algebraically equivalent to linear propagation of `XW`. This lets the model use artist/audio information for tracks without playlist history. The regression tests check this equivalence. See the [original LightGCN paper](https://arxiv.org/abs/2002.02126).

SIGN learns separate nonlinear projections of precomputed graph hops. Residual SIGN adds wider projections, more hops and four residual trunk layers. These are project adaptations of the [SIGN architecture](https://arxiv.org/abs/2004.11198). Sparse propagation is computed once and stored in memory-mapped arrays, avoiding full-graph backpropagation on this laptop. Inference loads one selected model at a time.

### Evaluation protocol

Playlists with at least five tracks reserve 15% of edges each for validation and test (rounded down, at least one each). Smaller playlists supply training context only. The current split is **734,897 training**, **144,907 validation**, and **144,907 test** edges.

Only training playlist edges enter the graph. Static artist, genre, audio and any sourced country metadata can cover the full catalog. Negative sampling excludes every known positive and samples from playlist-observed tracks; catalog-only tracks are not mislabeled as negative preferences.

Training uses BPR over a seed track, another track from its training playlist, and a negative. Evaluation uses the mean of training seed embeddings. Checkpoint, metadata mixture and fit calibration are chosen using validation only. Each frozen model is evaluated once on the test sample.

Metrics are Recall@10, NDCG@10, HitRate@10, MRR@10 and catalog coverage. Evaluation uses all held-out positives plus up to 100 sampled negatives per playlist, on a fixed sample of up to 1,024 playlists per split. Random and training-popularity baselines use the same candidate protocol. These are **sampled-candidate metrics, not full-catalog ranking metrics**, and cannot be directly compared with the old laptop's reported numbers. Artist-only quality and catalog-only songs lack playlist ground truth and are not established by this test.

The fit percentage is a validation-calibrated relative score, not the probability someone will like a song. Artist-only queries and their exact metadata candidate filter are outside the track-seeded calibration protocol; their fit percentages use the same scale but have not been calibrated on artist-seeded ground truth.

## Outputs and validation

- `models/lightgcn_playlist_recommender.pt`
- `models/sign_playlist_recommender.pt`
- `models/residual_sign_playlist_recommender.pt`
- `models/manifest.json`: available models, measured scores, architecture, training settings and dataset fingerprint.
- `models/track_catalog.csv.gz`, `models/artist_catalog.parquet`: shared searchable records.
- `models/*content_features.npy`: shared metadata features.
- `models/artist_metadata_index.npz`, `models/track_artist_index.npz`, `models/metadata_tokens.json`: exact artist discovery evidence.
- `data/processed/portable/`: cleaned tables, train-only graph, split, feature scaling and propagation arrays.
- `reports/current/`: machine/data audits, correlation tables, execution status, figures, training histories and evaluation.
- `reports/current/model_analysis_figures/`: three dissertation-ready PNG/SVG comparisons of architectures, training/validation and held-out results versus CPU cost, with draft captions in `figure_captions.md`. Regenerate from saved results with `& $MusicPython scripts/create_model_figures.py`.

Rebuild notebook 02 and all three models together whenever sources or feature construction change. Bundles are checked against the shared manifest generation; legacy checkpoints are rejected.

```powershell
& $MusicPython -m pytest tests -q
& $MusicPython -m pip check
$env:PATH = (Join-Path (Get-Location) '.tools/node-v22.23.2-win-x64') + ';' + $env:PATH
npm --prefix web run lint
npm --prefix web run build
```

The regression suite checks cold-start preservation, all artist credits, hidden-edge exclusion, LightGCN algebra, increasing model capacity, three-bundle round trips, artist-only seeds, duplicate/exhausted playlists and malformed API inputs. The UI supports add/remove/reset and model switching, and uses current measured model scores.
