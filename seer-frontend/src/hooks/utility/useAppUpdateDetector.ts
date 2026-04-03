import { useEffect, useRef, useState } from "react";

const POLL_INTERVAL_MS = 5 * 60 * 1000; // 5 minutes
const SCRIPT_SRC_REGEX = /<script[^>]+src="([^"]*\/assets\/[^"]+\.js[^"]*)"/;

function extractBundleUrl(html: string): string | null {
  const match = SCRIPT_SRC_REGEX.exec(html);
  return match ? match[1] : null;
}

function getInitialBundleUrl(): string | null {
  const scripts = document.querySelectorAll<HTMLScriptElement>('script[src*="/assets/"]');
  for (const script of scripts) {
    if (script.src.endsWith(".js")) {
      return script.src;
    }
  }
  return null;
}

export function useAppUpdateDetector(): { updateAvailable: boolean } {
  const [updateAvailable, setUpdateAvailable] = useState(false);
  const initialUrlRef = useRef<string | null>(null);

  useEffect(() => {
    initialUrlRef.current = getInitialBundleUrl();

    const checkForUpdate = async () => {
      if (document.visibilityState === "hidden") return;

      try {
        const response = await fetch("/", { cache: "no-store" });
        const html = await response.text();
        const latestUrl = extractBundleUrl(html);

        if (!latestUrl || !initialUrlRef.current) return;

        // Compare path segments to avoid origin differences (absolute vs relative)
        const initialPath = new URL(initialUrlRef.current, window.location.origin).pathname;
        const latestPath = new URL(latestUrl, window.location.origin).pathname;

        if (initialPath !== latestPath) {
          setUpdateAvailable(true);
          clearInterval(intervalId);
        }
      } catch {
        // Silently ignore network errors — don't disrupt the user
      }
    };

    const intervalId = setInterval(checkForUpdate, POLL_INTERVAL_MS);

    return () => clearInterval(intervalId);
  }, []);

  return { updateAvailable };
}
