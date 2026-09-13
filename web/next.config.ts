import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Vercel packages its own functions; standalone output is for Docker.
  output: process.env.VERCEL ? undefined : "standalone",
};

export default nextConfig;
