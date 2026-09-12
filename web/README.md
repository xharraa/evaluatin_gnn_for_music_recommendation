# Playlist Lab web app

Run `scripts/start_web_demo.ps1` from the repository root after the notebooks have generated all three model bundles. Setup installs Node locally in `.tools/` and records the short-path Python environment in `.tools/environment.json`.

The app supports track search, artist inspirations (including metadata-only artists), editable playlists, three model selectors and dynamically measured model scores. Spotify links open tracks externally; playlists remain local to the current browser session.

The Next.js API routes proxy to `http://127.0.0.1:8000`. Override `RECOMMENDER_API_URL` to use another local model-service address. See the repository README for the complete dataset, training and evaluation protocol.
