/* The component library. Each component is a function that returns DOM nodes; none of them writes HTML strings. Styling lives in
   css/components.css. Components: Button, Badge, CodeBlock, CodeTabs, Stat, MetricCard, StatusIndicator, Tabs, TabNav, Sidebar,
   DataTable, Input, Textarea, Select, Slider, Dropdown, Modal (confirmDialog), Toast, Skeleton, EmptyState, ErrorState, SegmentedControl,
   Accordion. Charts are in charts.js. */
import { el, icon, append } from "./dom.js";
import { highlight } from "./highlight.js";

let uid = 0;
export const nextId = (p = "id") => `${p}-${++uid}`;

/* ── Button ────────────────────────────────────────────────────────────── */
export function Button({ label, variant = "secondary", size, icon: iconName, iconAfter, href, onclick, type = "button", disabled, ariaLabel, title, id, className, target } = {}) {
  const cls = ["btn", "btn-" + variant, size ? "btn-" + size : "", !label ? "btn-icon" : "", className || ""].filter(Boolean).join(" ");
  const kids = [iconName ? icon(iconName) : null, label ? el("span", { text: label }) : null, iconAfter ? icon(iconAfter) : null];
  if (href) return el("a", { class: cls, href, id, title, "aria-label": ariaLabel, target, rel: target === "_blank" ? "noopener" : false, "aria-disabled": disabled ? "true" : false }, kids);
  return el("button", { class: cls, type, id, title, "aria-label": ariaLabel, disabled: !!disabled, onclick }, kids);
}

export const Badge = (text, tone = "neutral") => el("span", { class: "badge" + (tone === "neutral" ? "" : " badge-" + tone), text });

/* ── StatusIndicator: shape + word; colour is a third cue ──────────────── */
const STATUS = { operational: ["dot", "Operational"], degraded: ["half", "Degraded"], outage: ["cross", "Outage"], unknown: ["ring", "Unknown"] };
export function StatusIndicator(stateName, label) {
  const [shape, word] = STATUS[stateName] || STATUS.unknown;
  return el("span", { class: "status status-" + (STATUS[stateName] ? stateName : "unknown") }, icon(shape, 16), el("span", { text: label || word }));
}

/* ── Stat, MetricCard ──────────────────────────────────────────────────── */
export function Stats(items) {
  return el("dl", { class: "stats" }, items.map(([label, value, note]) => el("div", { class: "stat" }, el("dt", { text: label }), el("dd", null, value, note ? el("small", { text: note }) : null))));
}
export function MetricCard({ label, value, note }) {
  return el("div", { class: "metric" }, el("div", { class: "label", text: label }), el("div", { class: "value", text: value }), note ? el("div", { class: "note", text: note }) : null);
}

/* ── Copy ──────────────────────────────────────────────────────────────── */
export async function copyText(text) {
  try { await navigator.clipboard.writeText(text); return true; } catch (e) {
    try {                                                            // older browsers and non-secure origins: select-and-copy fallback
      const ta = el("textarea", { class: "sr-only", "aria-hidden": "true", tabindex: "-1" }); ta.value = text; document.body.append(ta); ta.select();
      const ok = document.execCommand("copy"); ta.remove(); return ok;
    } catch (x) { return false; }
  }
}

export function CopyButton(getText, { label = "Copy", size = "sm" } = {}) {
  const b = Button({ label, variant: "quiet", size, icon: "copy", ariaLabel: label + " to clipboard" });
  b.addEventListener("click", async () => {
    const ok = await copyText(typeof getText === "function" ? getText() : getText);
    b.replaceChildren(icon(ok ? "check" : "alert"), el("span", { text: ok ? "Copied" : "Press Ctrl+C" }));
    setTimeout(() => b.replaceChildren(icon("copy"), el("span", { text: label })), 1600);
    if (!ok) toast("Copy is blocked here. Select the text and copy it manually.", { tone: "warn" });
  });
  return b;
}

/* ── CodeBlock, CodeTabs ───────────────────────────────────────────────── */
export function CodeBlock({ code, lang = "text", title, copy = true }) {
  const pre = el("pre", { tabindex: "0", "aria-label": title || `${lang} code` }, el("code", null, highlight(code, lang)));
  return el("div", { class: "code" },
    el("div", { class: "code-head" }, el("span", { class: "code-lang", text: title || lang }), copy ? CopyButton(() => code) : null), pre);
}

export function CodeTabs(items, { ariaLabel = "Code language" } = {}) {
  return Tabs({ items: items.map((i) => ({ id: i.id, label: i.label })), ariaLabel,
    render: (id) => { const i = items.find((x) => x.id === id); return CodeBlock({ code: i.code, lang: i.lang }); } });
}

/* ── Tabs: the ARIA tab pattern, arrow keys, Home and End ──────────────── */
export function Tabs({ items, initial, ariaLabel, render, onChange }) {
  const base = nextId("tabs");
  const panel = el("div", { role: "tabpanel", id: base + "-panel", tabindex: "0" });
  const list = el("div", { role: "tablist", class: "tabs", "aria-label": ariaLabel });
  const buttons = new Map();
  function select(id, focus) {
    for (const [k, b] of buttons) { b.setAttribute("aria-selected", String(k === id)); b.tabIndex = k === id ? 0 : -1; }
    panel.setAttribute("aria-labelledby", base + "-" + id);
    panel.replaceChildren(render(id));
    if (focus) buttons.get(id).focus();
    if (onChange) onChange(id);
  }
  items.forEach((item, i) => {
    const b = el("button", { type: "button", role: "tab", class: "tab", id: base + "-" + item.id, "aria-controls": base + "-panel", text: item.label,
      onclick: () => select(item.id, false),
      onkeydown: (e) => {
        let j = null;
        if (e.key === "ArrowRight") j = (i + 1) % items.length;
        else if (e.key === "ArrowLeft") j = (i - 1 + items.length) % items.length;
        else if (e.key === "Home") j = 0;
        else if (e.key === "End") j = items.length - 1;
        if (j !== null) { e.preventDefault(); select(items[j].id, true); }
      } });
    buttons.set(item.id, b); list.append(b);
  });
  select(initial || items[0].id, false);
  const root = el("div", null, list, panel);
  root.select = (id) => select(id, false);
  return root;
}

/* Route tabs: links, not ARIA tabs. The router marks the current one (data-match). */
export function TabNav(items, ariaLabel) {
  return el("nav", { class: "tabnav", "aria-label": ariaLabel }, items.map((i) => el("a", { href: i.href, "data-match": i.match || "exact", text: i.label })));
}

export function Sidebar(groups, ariaLabel) {
  return el("nav", { class: "sidebar", "aria-label": ariaLabel }, groups.map((g) => [
    el("h2", { text: g.title }),
    el("ul", null, g.links.map((l) => el("li", null, el("a", { href: l.href, "data-match": "exact", text: l.label })))),
  ]));
}

/* ── DataTable ─────────────────────────────────────────────────────────── */
export function DataTable({ caption, columns, rows, empty = "Nothing to show yet." }) {
  const head = el("tr", null, columns.map((c) => el("th", { scope: "col", class: c.num ? "num" : false, text: c.label })));
  const body = rows.length ? rows.map((r) => el("tr", { class: r._muted ? "is-muted" : false }, columns.map((c, i) => {
    const v = c.render ? c.render(r) : r[c.key];
    return i === 0 && c.rowHeader ? el("th", { scope: "row", text: v }) : el("td", { class: c.num ? "num" : false }, v);
  }))) : [el("tr", null, el("td", { colspan: columns.length, class: "muted", text: empty }))];
  return el("div", { class: "table-wrap", tabindex: "0", role: "region", "aria-label": caption }, el("table", { class: "table" },
    el("caption", { text: caption }), el("thead", null, head), el("tbody", null, body)));
}

/* ── Form controls ─────────────────────────────────────────────────────── */
function fieldShell(id, label, control, { hint, error, right } = {}) {
  const hintId = hint ? id + "-hint" : null, errId = error ? id + "-err" : null;
  const describedBy = [hintId, errId].filter(Boolean).join(" ");
  if (describedBy) control.setAttribute("aria-describedby", describedBy);
  return el("div", { class: "field" }, el("label", { class: "label", for: id }, el("span", { text: label }), right || null), control,
    hint ? el("span", { class: "hint", id: hintId, text: hint }) : null, error ? el("span", { class: "error-text", id: errId, role: "alert", text: error }) : null);
}

export function Input({ label, id = nextId("in"), type = "text", value = "", hint, error, ...attrs }) {
  const control = el("input", { class: "input", id, type, value, "aria-invalid": error ? "true" : false, ...attrs });
  const f = fieldShell(id, label, control, { hint, error });
  f.input = control;
  return f;
}
export function Textarea({ label, id = nextId("ta"), value = "", hint, rows = 3, ...attrs }) {
  const control = el("textarea", { class: "textarea", id, rows, ...attrs }); control.value = value;
  const f = fieldShell(id, label, control, { hint });
  f.input = control;
  return f;
}
export function Select({ label, id = nextId("sel"), options, value, onChange, hint, disabled }) {
  const control = el("select", { class: "select", id, disabled: !!disabled, onchange: onChange ? (e) => onChange(e.target.value) : false },
    options.map((o) => el("option", { value: o.value, text: o.label, selected: o.value === value })));
  const f = fieldShell(id, label, el("div", { class: "select-wrap" }, control, icon("chevron")), { hint });
  control.setAttribute("aria-describedby", hint ? id + "-hint" : "");
  f.input = control;
  return f;
}
export function Slider({ label, id = nextId("sl"), min, max, step, value, onInput, format = (v) => String(v), hint }) {
  const out = el("output", { class: "slider-value", for: id, text: format(value) });
  const control = el("input", { class: "slider", id, type: "range", min, max, step, value, oninput: (e) => { out.textContent = format(e.target.value); if (onInput) onInput(Number(e.target.value)); } });
  const f = fieldShell(id, label, control, { hint, right: out });
  f.input = control;
  return f;
}
export function SegmentedControl({ options, value, onChange, ariaLabel }) {
  const root = el("div", { class: "seg", role: "group", "aria-label": ariaLabel });
  const paint = (v) => [...root.children].forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.v === v)));
  options.forEach((o) => root.append(el("button", { type: "button", "data-v": o.value, text: o.label, title: o.title, onclick: () => { paint(o.value); onChange(o.value); } })));
  paint(value);
  root.set = paint;
  return root;
}

/* ── Dropdown: a menu button (aria-haspopup=menu) ─────────────────────── */
export function Dropdown({ label, header, items }) {
  const id = nextId("menu");
  const trigger = el("button", { class: "btn btn-quiet btn-sm", type: "button", "aria-haspopup": "menu", "aria-expanded": "false", "aria-controls": id }, el("span", { text: label }), icon("chevron", 14));
  const menu = el("div", { class: "menu", role: "menu", id, hidden: true }, header ? el("div", { class: "menu-label", text: header }) : null,
    items.map((i) => (i.href ? el("a", { role: "menuitem", href: i.href, text: i.label, tabindex: "-1" })
      : el("button", { role: "menuitem", type: "button", text: i.label, tabindex: "-1", onclick: () => { close(true); i.onclick(); } }))));
  const root = el("div", { class: "dropdown" }, trigger, menu);
  const entries = () => [...menu.querySelectorAll("[role=menuitem]")];
  function open() { menu.hidden = false; trigger.setAttribute("aria-expanded", "true"); entries()[0].focus(); document.addEventListener("pointerdown", away); }
  function close(refocus) { menu.hidden = true; trigger.setAttribute("aria-expanded", "false"); document.removeEventListener("pointerdown", away); if (refocus) trigger.focus(); }
  const away = (e) => { if (!root.contains(e.target)) close(false); };
  trigger.addEventListener("click", () => (menu.hidden ? open() : close(true)));
  trigger.addEventListener("keydown", (e) => { if (e.key === "ArrowDown" && menu.hidden) { e.preventDefault(); open(); } });
  menu.addEventListener("keydown", (e) => {
    const list = entries(), i = list.indexOf(document.activeElement);
    if (e.key === "ArrowDown") { e.preventDefault(); list[(i + 1) % list.length].focus(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); list[(i - 1 + list.length) % list.length].focus(); }
    else if (e.key === "Home") { e.preventDefault(); list[0].focus(); }
    else if (e.key === "End") { e.preventDefault(); list[list.length - 1].focus(); }
    else if (e.key === "Escape") { e.preventDefault(); close(true); }
    else if (e.key === "Tab") close(false);
  });
  menu.addEventListener("click", (e) => { if (e.target.closest("a")) close(false); });
  root.close = () => close(false);
  return root;
}

/* ── Modal: a native <dialog> (focus is trapped and Escape closes it by the browser) ─ */
export function confirmDialog({ title, body, confirmLabel = "Confirm", cancelLabel = "Cancel", danger = false }) {
  return new Promise((resolve) => {
    const d = el("dialog", { class: "modal", "aria-labelledby": "dlg-title" });
    const done = (v) => { d.close(); d.remove(); resolve(v); };
    d.append(el("h2", { id: "dlg-title", text: title }), el("p", { class: "muted", text: body }),
      el("div", { class: "actions" }, Button({ label: cancelLabel, variant: "quiet", onclick: () => done(false) }),
        Button({ label: confirmLabel, variant: danger ? "danger" : "primary", onclick: () => done(true) })));
    d.addEventListener("cancel", (e) => { e.preventDefault(); done(false); });
    document.body.append(d);
    d.showModal();
    d.querySelector(".btn-quiet").focus();
  });
}

export function Modal({ title, content, actions }) {
  const d = el("dialog", { class: "modal", "aria-labelledby": "dlg-title" });
  const close = () => { d.close(); d.remove(); };
  d.append(el("h2", { id: "dlg-title", text: title }), content, el("div", { class: "actions" }, actions ? actions(close) : Button({ label: "Close", onclick: close })));
  d.addEventListener("cancel", (e) => { e.preventDefault(); close(); });
  document.body.append(d);
  d.showModal();
  return { dialog: d, close };
}

/* ── Toast (polite live region) ────────────────────────────────────────── */
export function toast(message, { tone = "ok", timeout = 5000 } = {}) {
  const host = document.getElementById("toasts");
  if (!host) return;
  const t = el("div", { class: "toast toast-" + tone, role: tone === "danger" ? "alert" : "status" }, el("div", { class: "toast-body", text: message }),
    Button({ variant: "quiet", size: "sm", ariaLabel: "Dismiss notification", icon: "x", onclick: () => t.remove() }));
  host.append(t);
  let timer = setTimeout(() => t.remove(), timeout);
  t.addEventListener("mouseenter", () => clearTimeout(timer));
  t.addEventListener("mouseleave", () => { timer = setTimeout(() => t.remove(), 2000); });
  return t;
}

/* ── Loading, empty and error states ───────────────────────────────────── */
export const Skeleton = (lines = 3) => el("div", { "aria-hidden": "true" }, Array.from({ length: lines }, (_, i) => el("span", { class: "skeleton " + ["w-80", "", "w-60", "w-40"][i % 4] })));
export const SkeletonBlock = () => el("span", { class: "skeleton block", "aria-hidden": "true" });
export function LoadingRegion(label = "Loading") {
  return el("div", { role: "status", "aria-live": "polite" }, el("span", { class: "sr-only", text: label + "\u2026" }), Skeleton(4));
}
export function EmptyState({ title, body, action }) {
  return el("div", { class: "empty" }, el("h3", { text: title }), body ? el("p", { text: body }) : null, action || null);
}
export function ErrorState({ title = "Something went wrong", message, retry }) {
  return el("div", { class: "error-state", role: "alert" }, el("h3", { text: title }), el("p", { text: message }),
    retry ? Button({ label: "Try again", icon: "refresh", onclick: retry }) : null);
}

/* ── Accordion: numbered, expandable sections (one open at a time) ─────── */
export function Accordion(items) {
  const rows = [];
  const list = el("ol", { class: "accordion" });
  items.forEach((it, i) => {
    const id = nextId("acc");
    const body = el("div", { class: "panel-body", id, hidden: true, role: "region", "aria-labelledby": id + "-t" }, it.body);
    const btn = el("button", { class: "trigger", type: "button", id: id + "-t", "aria-expanded": "false", "aria-controls": id,
      onclick: () => {
        const open = btn.getAttribute("aria-expanded") === "true";
        rows.forEach((r) => { r.btn.setAttribute("aria-expanded", "false"); r.body.hidden = true; });
        if (!open) { btn.setAttribute("aria-expanded", "true"); body.hidden = false; }
      } }, el("span", { class: "n", text: String(i + 1).padStart(2, "0") }), el("span", { class: "title", text: it.title }), icon("chevron", 18));
    btn.lastChild.classList.add("chev");
    rows.push({ btn, body });
    list.append(el("li", null, btn, body));
  });
  list.openFirst = () => rows[0] && rows[0].btn.click();
  list.open = (i) => rows[i] && rows[i].btn.click();
  return list;
}

export { el, append };
