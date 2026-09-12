"use client";

import type { CSSProperties } from "react";
import { useEffect, useMemo, useState } from "react";
import { Artwork } from "./artwork";
import {
  Check,
  CircleAlert,
  ExternalLink,
  ListMusic,
  LoaderCircle,
  Music2,
  Plus,
  RotateCcw,
  Search,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";

type Track = {
  trackNodeId: number;
  spotifyTrackId: string;
  title: string;
  artist: string;
  audioAvailable: boolean;
  inPlaylists: boolean;
  inCatalog: boolean;
};
type Artist = {
  spotifyArtistId: string;
  name: string;
  genres: string[];
  trackCount: number;
};
type Suggestion = Track & {
  rank: number;
  matchPercent: number;
  reason: string;
};
type Model = {
  id: string;
  name: string;
  architecture: { hops: number; hidden: number; dim: number; depth: number };
  metrics: { ndcg_at_10: number; parameters: number };
};
type Health = {
  status: string;
  trackCount: number;
  catalogOnlyTrackCount: number;
  models: Model[];
};
const choices = [
  {
    id: "lightgcn",
    name: "LightGCN",
    label: "Light",
    description: "Linear graph propagation · 32 dimensions",
  },
  {
    id: "sign",
    name: "SIGN",
    label: "Medium",
    description: "Nonlinear graph encoder · 64 dimensions",
  },
  {
    id: "residual_sign",
    name: "Residual SIGN",
    label: "Heavy",
    description: "Deeper residual encoder · 128 dimensions",
  },
];

function TrackMark({ track }: { track: Track }) {
  return (
    <div
      className="track-mark"
      style={{ "--track-hue": (track.trackNodeId * 47) % 360 } as CSSProperties}
      aria-hidden="true"
    >
      <span>
        {track.title
          .split(/\s+/)
          .slice(0, 2)
          .map((word) => word[0])
          .join("")
          .toUpperCase()}
      </span>
      <Music2 size={13} />
      <Artwork kind="track" id={track.spotifyTrackId} />
    </div>
  );
}
function AddButton({
  added,
  onClick,
  title,
}: {
  added: boolean;
  onClick: () => void;
  title: string;
}) {
  return (
    <button
      type="button"
      className={`add-button${added ? " is-added" : ""}`}
      disabled={added}
      onClick={onClick}
      aria-label={
        added
          ? `${title} is already in the playlist`
          : `Add ${title} to playlist`
      }
    >
      {added ? <Check size={19} /> : <Plus size={20} />}
    </button>
  );
}

export default function Home() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Track[]>([]);
  const [artistResults, setArtistResults] = useState<Artist[]>([]);
  const [artistSeeds, setArtistSeeds] = useState<Artist[]>([]);
  const [playlist, setPlaylist] = useState<Track[]>([]);
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [modelId, setModelId] = useState("lightgcn");
  const [health, setHealth] = useState<Health | null>(null);
  const [searching, setSearching] = useState(true);
  const [recommending, setRecommending] = useState(false);
  const [searchError, setSearchError] = useState("");
  const [recommendError, setRecommendError] = useState("");
  const [retry, setRetry] = useState(0);
  const playlistIds = useMemo(
    () => new Set(playlist.map((t) => t.spotifyTrackId)),
    [playlist],
  );
  const selectedModel = health?.models.find((m) => m.id === modelId);

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/health", { cache: "no-store", signal: controller.signal })
      .then(async (r) => {
        if (!r.ok) throw new Error("Model service unavailable");
        return r.json();
      })
      .then((data) => {
        if (!controller.signal.aborted) setHealth(data);
      })
      .catch(() => {
        if (!controller.signal.aborted) setHealth(null);
      });
    return () => controller.abort();
  }, [retry]);

  useEffect(() => {
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      setSearching(true);
      try {
        const response = await fetch(
          `/api/search?q=${encodeURIComponent(query)}&limit=8`,
          { cache: "no-store", signal: controller.signal },
        );
        const data = await response.json();
        if (!response.ok) throw new Error(data.error ?? "Search unavailable");
        if (!controller.signal.aborted) {
          setResults(data.results);
          setArtistResults(data.artists ?? []);
          setSearchError("");
        }
      } catch (e) {
        if (!controller.signal.aborted) {
          setResults([]);
          setArtistResults([]);
          setSearchError((e as Error).message);
        }
      } finally {
        if (!controller.signal.aborted) setSearching(false);
      }
    }, 180);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [query, retry]);

  useEffect(() => {
    if (!playlist.length && !artistSeeds.length) return;
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      setRecommending(true);
      try {
        const response = await fetch("/api/recommend", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            modelId,
            trackIds: playlist.map((t) => t.spotifyTrackId),
            artistIds: artistSeeds.map((a) => a.spotifyArtistId),
            limit: 8,
          }),
          signal: controller.signal,
        });
        const data = await response.json();
        if (!response.ok)
          throw new Error(data.error ?? "Recommendations unavailable");
        if (!controller.signal.aborted) {
          setSuggestions(data.results);
          setRecommendError("");
        }
      } catch (e) {
        if (!controller.signal.aborted) {
          setSuggestions([]);
          setRecommendError((e as Error).message);
        }
      } finally {
        if (!controller.signal.aborted) setRecommending(false);
      }
    }, 150);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [playlist, artistSeeds, modelId, retry]);

  function updatePlaylist(next: Track[]) {
    setPlaylist(next);
    setSuggestions([]);
    setRecommendError("");
    setRecommending(next.length + artistSeeds.length > 0);
  }
  function add(track: Track) {
    if (!playlistIds.has(track.spotifyTrackId))
      updatePlaylist([...playlist, track]);
  }
  function selectModel(id: string) {
    if (id !== modelId) {
      setModelId(id);
      setSuggestions([]);
      setRecommendError("");
      setRecommending(playlist.length + artistSeeds.length > 0);
    }
  }
  function updateArtists(next: Artist[]) {
    setArtistSeeds(next);
    setSuggestions([]);
    setRecommendError("");
    setRecommending(next.length + playlist.length > 0);
  }
  function reset() {
    updatePlaylist([]);
    setArtistSeeds([]);
    setRecommending(false);
    setQuery("");
  }

  return (
    <main>
      <header className="site-header">
        <a className="brand" href="#top">
          <span className="brand-icon">
            <Music2 size={22} />
          </span>
          Playlist Lab
        </a>
        <div
          className={`model-status${health?.status === "ready" ? " is-ready" : ""}`}
        >
          <span className="status-dot" />
          {health?.status === "ready"
            ? `${health.models.length} models ready`
            : "Model service offline"}
        </div>
      </header>
      <section
        className="model-selector"
        aria-label="Choose recommendation model"
      >
        {choices.map((choice) => (
          <button
            type="button"
            key={choice.id}
            className={`model-choice${modelId === choice.id ? " selected" : ""}`}
            aria-pressed={modelId === choice.id}
            disabled={!health?.models.some((m) => m.id === choice.id)}
            onClick={() => selectModel(choice.id)}
          >
            <span className="model-tier">
              {choice.label}
              {modelId === choice.id && <Check size={15} />}
            </span>
            <strong>{choice.name}</strong>
            <span>{choice.description}</span>
          </button>
        ))}
      </section>
      <section className="hero" id="top">
        <div>
          <p className="eyebrow">
            <Sparkles size={15} /> Your playlist. Three perspectives.
          </p>
          <h1>
            Find your sound.
            <br />
            <em>Build the next mix.</em>
          </h1>
        </div>
        <p className="hero-copy">
          Start with a song you love. Explore connections through artists,
          genres, audio and shared playlists. Switch models to hear a different
          perspective.
        </p>
      </section>
      {(searchError || recommendError) && (
        <div className="error-banner" role="alert">
          <CircleAlert size={18} />
          <span>{searchError || recommendError}</span>
          <button type="button" onClick={() => setRetry((r) => r + 1)}>
            Retry connection
          </button>
        </div>
      )}
      <section className="workspace" aria-label="Playlist builder">
        <div className="builder-column">
          <section className="panel search-panel">
            <div className="panel-heading">
              <div>
                <span className="step-number">01</span>
                <h2>Find your first note</h2>
              </div>
              <span className="heading-note">Title or artist</span>
            </div>
            <label className="search-box">
              <Search size={22} />
              <input
                type="search"
                value={query}
                maxLength={200}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Try Elvana Gjata, Radiohead, Dreams…"
                aria-label="Search by track title or artist"
                autoComplete="off"
              />
              {query && (
                <button
                  type="button"
                  onClick={() => setQuery("")}
                  aria-label="Clear search"
                >
                  <X size={18} />
                </button>
              )}
            </label>
            <div className="result-caption">
              <span>
                {query ? `Matches for “${query}”` : "Explore the collection"}
              </span>
              <span>
                {searching ? "Searching…" : `${results.length} tracks`}
              </span>
            </div>
            {artistResults.length > 0 && (
              <div className="artist-results" aria-label="Artist results">
                <p className="result-caption">Start from an artist</p>
                {artistResults.map((artist) => (
                  <article className="artist-row" key={artist.spotifyArtistId}>
                    <div className="artist-avatar" aria-hidden="true">
                      <Artwork kind="artist" id={artist.spotifyArtistId} />
                      <Music2 size={19} />
                    </div>
                    <div className="track-copy">
                      <h3>{artist.name}</h3>
                      <p>
                        {artist.genres.slice(0, 3).join(" · ") ||
                          "Artist metadata"}
                      </p>
                      <span className="artist-coverage">
                        {artist.trackCount
                          ? `${artist.trackCount} tracks in the collection`
                          : "Artist profile only · no tracks in these datasets"}
                      </span>
                    </div>
                    <button
                      type="button"
                      className="artist-seed-button"
                      disabled={artistSeeds.some(
                        (a) => a.spotifyArtistId === artist.spotifyArtistId,
                      )}
                      onClick={() => updateArtists([...artistSeeds, artist])}
                    >
                      {artistSeeds.some(
                        (a) => a.spotifyArtistId === artist.spotifyArtistId,
                      )
                        ? "Selected"
                        : "Use as inspiration"}
                    </button>
                  </article>
                ))}
              </div>
            )}
            <div
              className={`track-list${searching ? " is-muted" : ""}`}
              aria-live="polite"
            >
              {!searching &&
                !results.length &&
                !artistResults.length &&
                !searchError && (
                  <div className="empty-search">
                    <Search size={24} />
                    <p>
                      No matches in these datasets. Try fewer words or another
                      artist.
                    </p>
                  </div>
                )}
              {results.map((track) => (
                <article className="track-row" key={track.spotifyTrackId}>
                  <TrackMark track={track} />
                  <div className="track-copy">
                    <h3>{track.title}</h3>
                    <p>{track.artist}</p>
                    {!track.inPlaylists && (
                      <span className="catalog-badge">Catalog discovery</span>
                    )}
                  </div>
                  <AddButton
                    title={track.title}
                    added={playlistIds.has(track.spotifyTrackId)}
                    onClick={() => add(track)}
                  />
                </article>
              ))}
            </div>
          </section>
          <section className="panel playlist-panel">
            <div className="panel-heading">
              <div>
                <span className="step-number">02</span>
                <h2>Your playlist</h2>
              </div>
              <span className="playlist-count">{playlist.length} tracks</span>
            </div>
            {artistSeeds.length > 0 && (
              <div className="artist-inspirations">
                <p>Artist inspirations</p>
                {artistSeeds.map((artist) => (
                  <span key={artist.spotifyArtistId}>
                    {artist.name}
                    <button
                      type="button"
                      aria-label={`Remove artist ${artist.name}`}
                      onClick={() =>
                        updateArtists(
                          artistSeeds.filter(
                            (a) => a.spotifyArtistId !== artist.spotifyArtistId,
                          ),
                        )
                      }
                    >
                      <X size={14} />
                    </button>
                  </span>
                ))}
              </div>
            )}
            {!playlist.length ? (
              <div className="playlist-empty">
                <ListMusic size={36} />
                <h3>
                  {artistSeeds.length
                    ? "Your artist inspirations are ready."
                    : "A good mix starts with one track."}
                </h3>
                <p>
                  {artistSeeds.length
                    ? "Add a suggestion to turn inspiration into a playlist."
                    : "Use + to add a song. Your suggestions will follow."}
                </p>
              </div>
            ) : (
              <div className="selected-list" aria-live="polite">
                {playlist.map((track, i) => (
                  <article className="selected-row" key={track.spotifyTrackId}>
                    <span className="selected-index">
                      {String(i + 1).padStart(2, "0")}
                    </span>
                    <TrackMark track={track} />
                    <div className="track-copy">
                      <h3>{track.title}</h3>
                      <p>{track.artist}</p>
                    </div>
                    <a
                      className="listen-link"
                      href={`https://open.spotify.com/track/${track.spotifyTrackId}`}
                      target="_blank"
                      rel="noreferrer"
                      aria-label={`Open ${track.title} on Spotify`}
                    >
                      <ExternalLink size={16} />
                    </a>
                    <button
                      type="button"
                      className="remove-button"
                      aria-label={`Remove ${track.title}`}
                      onClick={() =>
                        updatePlaylist(
                          playlist.filter(
                            (t) => t.spotifyTrackId !== track.spotifyTrackId,
                          ),
                        )
                      }
                    >
                      <Trash2 size={17} />
                    </button>
                  </article>
                ))}
              </div>
            )}
          </section>
        </div>
        <section className="suggestions-panel" aria-label="Model suggestions">
          <div className="suggestions-heading">
            <div>
              <span className="step-number light">03</span>
              <h2>{selectedModel?.name ?? "Model"} picks</h2>
            </div>
            <div className="live-pill">
              <span />
              {recommending ? "Ranking" : "Ready to explore"}
            </div>
          </div>
          <p className="suggestions-intro">
            A new direction for your mix. Add any pick to your playlist and the
            model will rank again.
          </p>
          {!playlist.length && !artistSeeds.length ? (
            <div className="suggestions-empty">
              <div className="orbit" aria-hidden="true">
                <span className="orbit-note">
                  <Music2 size={27} />
                </span>
                <span className="orbit-dot dot-one" />
                <span className="orbit-dot dot-two" />
              </div>
              <h3>Make room for discovery.</h3>
              <p>
                Add a track to see what connects. Catalog artists work even
                without source playlist history.
              </p>
            </div>
          ) : recommending ? (
            <div className="model-loading" role="status">
              <LoaderCircle size={34} className="spinner" />
              <h3>Finding connections…</h3>
              <p>{selectedModel?.name} is reading your playlist.</p>
            </div>
          ) : (
            <div className="suggestion-list" aria-live="polite">
              {suggestions.map((track) => (
                <article className="suggestion-card" key={track.spotifyTrackId}>
                  <div className="rank">
                    #{String(track.rank).padStart(2, "0")}
                  </div>
                  <TrackMark track={track} />
                  <div className="suggestion-main">
                    <div className="suggestion-title-line">
                      <div className="track-copy">
                        <h3>{track.title}</h3>
                        <p>{track.artist}</p>
                      </div>
                      <AddButton
                        title={track.title}
                        added={playlistIds.has(track.spotifyTrackId)}
                        onClick={() => add(track)}
                      />
                    </div>
                    <div className="fit-row">
                      <div className="fit-track" aria-hidden="true">
                        <span
                          style={{
                            width: `${Math.max(0, Math.min(track.matchPercent, 100))}%`,
                          }}
                        />
                      </div>
                      <strong>{track.matchPercent.toFixed(1)}% fit</strong>
                    </div>
                    <p className="recommendation-reason">{track.reason}</p>
                  </div>
                </article>
              ))}
              {!suggestions.length && !recommendError && (
                <p className="suggestions-intro">
                  No further tracks are available for this playlist.
                </p>
              )}
            </div>
          )}
          <div className="score-note">
            <CircleAlert size={17} />
            <p>
              <strong>About the fit score:</strong> a validation-calibrated
              relative match, not the probability you will like a song.
            </p>
          </div>
        </section>
      </section>
      <section className="demo-footer">
        <button
          className="reset-button"
          type="button"
          onClick={reset}
          disabled={!playlist.length && !artistSeeds.length && !query}
        >
          <RotateCcw size={19} /> Empty playlist &amp; start again
        </button>
        <div className="model-facts">
          <div>
            <strong>{health?.trackCount.toLocaleString() ?? "—"}</strong>
            <span>tracks to explore</span>
          </div>
          <div>
            <strong>{selectedModel?.name ?? "—"}</strong>
            <span>selected model</span>
          </div>
          <div>
            <strong>
              {selectedModel?.metrics.ndcg_at_10.toFixed(4) ?? "—"}
            </strong>
            <span>sampled test NDCG@10</span>
          </div>
        </div>
      </section>
    </main>
  );
}
