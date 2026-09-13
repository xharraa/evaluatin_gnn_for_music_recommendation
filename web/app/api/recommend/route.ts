import { NextRequest, NextResponse } from "next/server";

// Allow a model switch to load its checkpoint on the CPU backend.
export const maxDuration = 120;

const apiBase = process.env.RECOMMENDER_API_URL ?? "http://127.0.0.1:8000";

export async function POST(request: NextRequest) {
  try {
    const payload = await request.json();
    const response = await fetch(`${apiBase}/recommend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      cache: "no-store",
      signal: AbortSignal.timeout(100000),
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
