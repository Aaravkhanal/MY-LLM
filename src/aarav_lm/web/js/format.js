/* Number and date formatting shared by the pages. */
export const fmt = (n) => Number(n).toLocaleString("en-US");
export const short = (n) => (n >= 1e9 ? (n / 1e9).toFixed(1).replace(/\.0$/, "") + "B" : n >= 1e6 ? (n / 1e6).toFixed(1).replace(/\.0$/, "") + "M"
  : n >= 1e3 ? (n / 1e3).toFixed(1).replace(/\.0$/, "") + "k" : String(n));
export const pct = (x, digits = 1) => (100 * x).toFixed(digits).replace(/\.0+$/, "") + "%";
export const signed = (x, digits = 1) => (x >= 0 ? "+" : "−") + Math.abs(100 * x).toFixed(digits);
export function when(iso) {
  if (!iso) return "never";
  const d = new Date(iso);
  return isNaN(d) ? "" : d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}
export function duration(seconds) {
  const s = Math.round(seconds);
  if (s < 90) return s + " s";
  if (s < 5400) return Math.round(s / 60) + " min";
  if (s < 172800) return (s / 3600).toFixed(1).replace(/\.0$/, "") + " h";
  return Math.round(s / 86400) + " days";
}
export const ms = (v) => (v === null || v === undefined ? "–" : v >= 1000 ? (v / 1000).toFixed(2) + " s" : Math.round(v) + " ms");
