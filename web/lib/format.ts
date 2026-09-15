export function formatCompactNumber(n: number | null): string {
  if (n == null) return "-";
  return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(n);
}

export function formatBytesToGb(bytes: number | null): string {
  if (bytes == null) return "해당 없음";
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}
