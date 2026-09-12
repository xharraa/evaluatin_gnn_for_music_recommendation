import { NextRequest, NextResponse } from "next/server";

const apiBase = process.env.RECOMMENDER_API_URL ?? "http://127.0.0.1:8000";

export async function GET(request: NextRequest) {
  const query = request.nextUrl.searchParams.get("q") ?? "";
  const limit = request.nextUrl.searchParams.get("limit") ?? "5";
  const target = new URL("/search", apiBase);
  target.searchParams.set("q", query);
  target.searchParams.set("limit", limit);

  try {
    const response = await fetch(target, { cache: "no-store" });
    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch {
    return NextResponse.json(
      { error: "The local recommendation service is not running." },
      { status: 503 },
    );
  }
}
