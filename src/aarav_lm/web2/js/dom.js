/* Tiny DOM helpers. Every node is built with DOM calls and textContent, never an HTML string, so model output, server messages and
   user input can never become markup (tests/test_api.py::test_ui_never_writes_model_output_as_html enforces it). */

export function el(tag, props, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v === false || v === null || v === undefined) continue;
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v === true) n.setAttribute(k, "");
    else n.setAttribute(k, v);
  }
  append(n, kids);
  return n;
}

export function append(parent, kids) {
  for (const kid of [kids].flat(Infinity)) {
    if (kid === null || kid === undefined || kid === false) continue;
    parent.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return parent;
}

const SVG_NS = "http://www.w3.org/2000/svg";
export function svg(tag, attrs, ...kids) {
  const n = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v !== false && v !== null && v !== undefined) n.setAttribute(k, v);
  for (const kid of kids.flat(Infinity)) if (kid) n.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return n;
}

/* 16px stroke icons. Each is a list of path/shape specs; colour comes from currentColor, so themes and states recolour them. */
const ICONS = {
  copy: [["rect", { x: 9, y: 9, width: 11, height: 11, rx: 2 }], ["path", { d: "M5 15V6a2 2 0 0 1 2-2h9" }]],
  check: [["path", { d: "M5 12.5l4.5 4.5L19 7.5" }]],
  arrow: [["path", { d: "M5 12h14M13 6l6 6-6 6" }]],
  chevron: [["path", { d: "M6 9l6 6 6-6" }]],
  send: [["path", { d: "M5 12l14-7-5 14-2.5-5.5z" }]],
  stop: [["rect", { x: 6, y: 6, width: 12, height: 12, rx: 2 }]],
  refresh: [["path", { d: "M20 11a8 8 0 1 0-2.3 5.7M20 5v6h-6" }]],
  trash: [["path", { d: "M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V4h6v3" }]],
  key: [["circle", { cx: 8, cy: 15, r: 4 }], ["path", { d: "M11 12l9-9M16 7l3 3" }]],
  sun: [["circle", { cx: 12, cy: 12, r: 4 }], ["path", { d: "M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5L19 19M5 19l1.5-1.5M17.5 6.5L19 5" }]],
  moon: [["path", { d: "M20 14.5A8 8 0 1 1 9.5 4 6.5 6.5 0 0 0 20 14.5z" }]],
  external: [["path", { d: "M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5" }]],
  alert: [["path", { d: "M12 4l9 16H3z" }], ["path", { d: "M12 10v4M12 17.5v.01" }]],
  menu: [["path", { d: "M4 7h16M4 12h16M4 17h16" }]],
  x: [["path", { d: "M6 6l12 12M18 6L6 18" }]],
  logo: [["path", { d: "M5 18V6l7 8 7-8v12" }]],
  dot: [["circle", { cx: 12, cy: 12, r: 5, fill: "currentColor" }]],
  half: [["circle", { cx: 12, cy: 12, r: 6 }], ["path", { d: "M12 6a6 6 0 0 1 0 12z", fill: "currentColor" }]],
  cross: [["circle", { cx: 12, cy: 12, r: 6 }], ["path", { d: "M9.5 9.5l5 5M14.5 9.5l-5 5" }]],
  ring: [["circle", { cx: 12, cy: 12, r: 6 }]],
};

export function icon(name, size = 16) {
  const s = svg("svg", { viewBox: "0 0 24 24", width: size, height: size, fill: "none", stroke: "currentColor", "stroke-width": 1.8,
    "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true", focusable: "false", class: "icon icon-" + name });
  for (const [tag, attrs] of ICONS[name] || []) s.append(svg(tag, attrs));
  return s;
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
