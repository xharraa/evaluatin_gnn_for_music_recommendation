import { NextRequest, NextResponse } from "next/server";

// Fetch metadata only; the browser loads the original Spotify image directly.
export async function GET(request: NextRequest) {
  const kind = request.nextUrl.searchParams.get("kind");
  const id = request.nextUrl.searchParams.get("id") ?? "";
  if (!['track', 'artist'].includes(kind ?? '') || !/^[A-Za-z0-9]{22}$/.test(id)) {
    return NextResponse.json({ error: "Invalid Spotify entity" }, { status: 400 });
  }
  const target = new URL("https://open.spotify.com/oembed");
  target.searchParams.set("url", `https://open.spotify.com/${kind}/${id}`);
  try {
    const response = await fetch(target, {
      next: { revalidate: 86400 },
      signal: AbortSignal.timeout(6000),
    });
    if (!response.ok) throw new Error("Artwork unavailable");
    const data = await response.json();
    const url = typeof data.thumbnail_url === "string" ? new URL(data.thumbnail_url) : null;
    const imageUrl = url?.protocol === "https:" &&
      (url.hostname === "i.scdn.co" || url.hostname.endsWith(".spotifycdn.com")) ? url.href : null;
    return NextResponse.json({ imageUrl }, {
      headers: { "Cache-Control": "public, max-age=86400" },
    });
  } catch {
    return NextResponse.json({ imageUrl: null }, {
      headers: { "Cache-Control": "public, max-age=60" },
    });
  }
}
