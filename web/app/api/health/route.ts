import { NextResponse } from "next/server";

const apiBase = process.env.RECOMMENDER_API_URL ?? "http://127.0.0.1:8000";

export async function GET() {
  try {
    const response = await fetch(`${apiBase}/health`, {
      cache: "no-store",
      signal: AbortSignal.timeout(15000),
    });
    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
  } catch {
    return NextResponse.json(
      { error: "The recommendation service is currently unavailable. Please try again shortly." },
      { status: 503 },
    );
  }
}
