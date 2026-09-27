/* Charts: hand-built SVG, responsive (they redraw to the width they are given), keyboard reachable, with a table alternative.
   Colours come from the stylesheet through classes (never from here). Series are told apart by colour AND marker shape AND a legend. */
import { el, svg } from "./dom.js";
import { SegmentedControl } from "./ui.js";
import { fmt, short } from "./format.js";

/* Draw `draw(width)` into `host`, and again whenever the host's width changes. */
function responsive(host, draw) {
  let last = 0;
  const run = () => { const w = Math.max(280, Math.round(host.clientWidth || 640)); if (Math.abs(w - last) < 2) return; last = w; draw(w); };
  if (typeof ResizeObserver !== "undefined") new ResizeObserver(run).observe(host); else window.addEventListener("resize", run);
  requestAnimationFrame(run);
}

function tip(host) { const t = el("div", { class: "tip", hidden: true, role: "presentation" }); host.append(t); return t; }

/* Dot-and-interval chart: one row per item; each series is a marker with its 95% interval. `rows[i].cells[key] = {rate, lo, hi, prompts}` in %. */
export function IntervalChart({ rows, series, chance, ariaLabel, valueFormat = (v) => Math.round(v) + "%" }) {
  const host = el("div", { class: "chart" });
  const legend = series.length > 1 ? el("div", { class: "legend" }, series.map((s) => el("span", null, svg("svg", { width: 14, height: 14, viewBox: "0 0 14 14", "aria-hidden": "true" }, marker(s, 7, 7, 5, true)), s.label))) : null;
  const surface = el("div");
  host.append(surface);
  function marker(s, cx, cy, r, forLegend) {
    const cls = "dot" + (s.tone === 2 ? " dot-2" : "");
    return s.shape === "diamond"
      ? svg("path", { class: cls, d: `M${cx},${cy - r - 1} L${cx + r + 1},${cy} L${cx},${cy + r + 1} L${cx - r - 1},${cy} Z` })
      : svg("circle", { class: cls, cx, cy, r: forLegend ? r - 1 : r });
  }
  function draw(W) {
    const narrow = W < 560, L = narrow ? 8 : 190, R = 56, T = 26, B = 34;
    const per = series.length, rowH = (narrow ? 44 : 20) + per * 22;
    const H = T + rows.length * rowH + B, plotW = W - L - R;
    const x = (v) => L + (Math.min(100, Math.max(0, v)) / 100) * plotW;
    const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": ariaLabel });
    const grid = svg("g", { class: "grid" });
    [0, 25, 50, 75, 100].forEach((v) => {
      grid.append(svg("line", { x1: x(v), x2: x(v), y1: T - 6, y2: H - B }), svg("text", { x: x(v), y: H - B + 18, "text-anchor": "middle" }, v + "%"));
    });
    s.append(grid);
    if (chance !== undefined && chance !== null) {
      s.append(svg("line", { class: "chance", x1: x(chance), x2: x(chance), y1: T - 6, y2: H - B }), svg("text", { class: "chance-label", x: x(chance) + 4, y: T - 10 }, `chance ${valueFormat(chance)}`));
    }
    const t = tip(host);
    rows.forEach((r, i) => {
      const top = T + i * rowH, mid = top + rowH / 2;
      const g = svg("g", { class: "row" + (r.base ? " base" : "") });
      g.append(narrow ? svg("text", { class: "row-label", x: L, y: top + 14 }, r.label) : svg("text", { class: "row-label", x: L - 12, y: mid + 4, "text-anchor": "end" }, r.label));
      series.forEach((sr, k) => {
        const c = r.cells[sr.key]; if (!c) return;
        const cy = (narrow ? top + 28 : top + rowH / 2 - ((per - 1) * 11)) + k * 22;
        g.append(svg("line", { class: "ci" + (sr.tone === 2 ? " ci-2" : ""), x1: x(c.lo), x2: x(c.hi), y1: cy, y2: cy }), marker(sr, x(c.rate), cy, 5.5),
          svg("text", { class: "val", x: Math.min(W - 4, x(c.hi) + 8), y: cy + 4 }, valueFormat(c.rate)));
      });
      const desc = `${r.label}: ` + series.map((sr) => { const c = r.cells[sr.key]; return c ? `${sr.label} ${c.rate}%, 95% interval ${c.lo} to ${c.hi}%, ${c.prompts} distinct prompts` : ""; }).filter(Boolean).join("; ");
      const hit = svg("rect", { class: "hit", x: 0, y: top, width: W, height: rowH, tabindex: "0", role: "img", "aria-label": desc });
      const show = () => {
        t.textContent = desc.replace(/^[^:]+: /, r.label + " · ");
        t.style.left = Math.min(W - 120, Math.max(120, x(r.cells[series[0].key].rate))) / W * 100 + "%";
        t.style.top = (mid / H) * 100 + "%"; t.hidden = false;
      };
      hit.addEventListener("mouseenter", show); hit.addEventListener("focus", show);
      hit.addEventListener("mouseleave", () => { t.hidden = true; }); hit.addEventListener("blur", () => { t.hidden = true; });
      s.append(g, hit);
    });
    surface.replaceChildren(s);
    host.querySelectorAll(".tip").forEach((n, i) => { if (i > 0) n.remove(); });
  }
  responsive(host, draw);
  return el("div", null, legend, host);
}

function niceMax(v) {
  if (v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v))), f = v / p;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p;
}

/* Bars per day. `days` = [{date, requests, tokens}]; `metric` is which of the two is drawn. */
export function BarChart({ days, metric = "tokens", unit = "tokens", ariaLabel }) {
  const host = el("div", { class: "chart" });
  const surface = el("div");
  host.append(surface);
  function draw(W) {
    const H = 220, L = 46, B = 28, T = 10, plotH = H - B - T, plotW = W - L;
    const max = niceMax(Math.max(...days.map((d) => d[metric])));
    const slot = plotW / days.length, bw = Math.max(6, Math.min(24, slot * 0.6));
    const y = (v) => T + plotH - (v / max) * plotH;
    const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": ariaLabel || `${unit} per day for the last ${days.length} days; use the table view for the numbers.` });
    const grid = svg("g", { class: "grid" });
    for (let i = 0; i <= 2; i++) {
      const v = (max * i) / 2, yy = y(v);
      grid.append(svg("line", { x1: L, x2: W, y1: yy, y2: yy }), svg("text", { x: L - 8, y: yy + 4, "text-anchor": "end" }, short(Math.round(v))));
    }
    s.append(grid);
    const t = tip(host);
    days.forEach((d, i) => {
      const cx = L + slot * i + slot / 2, top = y(d[metric]), h = T + plotH - top;
      const label = new Date(d.date + "T00:00:00").toLocaleDateString(undefined, { month: "short", day: "numeric" });
      const every = W < 520 ? 4 : 3;
      if (i % every === 0 || i === days.length - 1) s.append(svg("text", { x: cx, y: H - 8, "text-anchor": "middle" }, label));
      const hit = svg("rect", { class: "hit", x: L + slot * i, y: T, width: slot, height: plotH, tabindex: "0", role: "img", "aria-label": `${label}: ${fmt(d.tokens)} tokens, ${fmt(d.requests)} requests` });
      const show = () => { t.textContent = `${label} · ${fmt(d.tokens)} tokens · ${fmt(d.requests)} requests`; t.style.left = Math.min(86, Math.max(14, (cx / W) * 100)) + "%"; t.style.top = (Math.max(top, 36) / H) * 100 + "%"; t.hidden = false; };
      hit.addEventListener("mouseenter", show); hit.addEventListener("focus", show);
      hit.addEventListener("mouseleave", () => { t.hidden = true; }); hit.addEventListener("blur", () => { t.hidden = true; });
      const r = Math.min(3, bw / 2, h);
      const path = d[metric] > 0
        ? `M${cx - bw / 2},${T + plotH} V${top + r} Q${cx - bw / 2},${top} ${cx - bw / 2 + r},${top} H${cx + bw / 2 - r} Q${cx + bw / 2},${top} ${cx + bw / 2},${top + r} V${T + plotH} Z`
        : `M${cx - bw / 2},${T + plotH - 2} H${cx + bw / 2} V${T + plotH} H${cx - bw / 2} Z`;
      s.append(hit, svg("path", { class: "bar" + (d[metric] > 0 ? "" : " zero"), d: path }));
    });
    surface.replaceChildren(s);
    host.querySelectorAll(".tip").forEach((n, i) => { if (i > 0) n.remove(); });
  }
  responsive(host, draw);
  return host;
}

/* A chart with a "Chart | Table" switch; `table` is a node built by DataTable. */
export function ChartWithTable({ chart, table, label = "View" }) {
  const wrap = el("div");
  const show = (v) => { chart.hidden = v !== "chart"; table.hidden = v !== "table"; };
  wrap.append(el("div", { class: "row between" }, SegmentedControl({ ariaLabel: label, value: "chart", onChange: show,
    options: [{ value: "chart", label: "Chart" }, { value: "table", label: "Table" }] })), chart, table);
  table.hidden = true;
  return wrap;
}
