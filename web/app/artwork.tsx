"use client";

import Image from "next/image";
import { useEffect, useState } from "react";

export function Artwork({ kind, id }: { kind: "track" | "artist"; id: string }) {
  const [image, setImage] = useState<{ id: string; url: string } | null>(null);
  const [failedUrl, setFailedUrl] = useState<string | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    fetch(`/api/artwork?kind=${kind}&id=${encodeURIComponent(id)}`, { signal: controller.signal })
      .then((response) => response.ok ? response.json() : null)
      .then((data) => {
        if (!controller.signal.aborted && data?.imageUrl) setImage({ id, url: data.imageUrl });
      })
      .catch(() => {});
    return () => controller.abort();
  }, [kind, id]);
  if (!image || image.id !== id || failedUrl === image.url) return null;
  return <Image className="spotify-artwork" src={image.url} alt="" width={80} height={80}
    unoptimized onError={() => setFailedUrl(image.url)} />;
}
