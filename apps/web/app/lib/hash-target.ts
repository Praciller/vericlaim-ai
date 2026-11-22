export function hashTargetId(hash: string): string | null {
  if (!hash.startsWith("#") || hash.length < 2) return null;

  try {
    return decodeURIComponent(hash.slice(1)) || null;
  } catch {
    return null;
  }
}
