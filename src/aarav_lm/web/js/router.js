/* A small History-API router. Pages are lazy-loaded modules with one contract:
       export default async function page(ctx) -> { title, description?, node, onLeave? }
   Links to known pages are intercepted so navigation is instant and keeps the shell; everything else is left to the browser. The server
   answers each path in ROUTES with the same shell (api/web_static.py PAGE_ROUTES; tests/test_web_routes.py compares the two lists). */
import { state, needsAccount } from "./state.js";
import { LoadingRegion, ErrorState } from "./ui.js";
import { el } from "./dom.js";

export const ROUTES = [
  { path: "/", title: "aarav-lm: a language model built from scratch", load: () => import("./pages/landing.js") },
  { path: "/playground", title: "Playground", load: () => import("./pages/playground.js") },
  { path: "/models", title: "Models", load: () => import("./pages/models.js") },
  { path: "/models/aarav-lm", title: "aarav-lm", load: () => import("./pages/model.js") },
  { path: "/models/aarav-lm/evaluations", title: "aarav-lm evaluations", load: () => import("./pages/evaluations.js") },
  { path: "/models/aarav-lm/architecture", title: "aarav-lm architecture", load: () => import("./pages/architecture.js") },
  { path: "/docs", title: "Documentation", load: () => import("./pages/docs.js") },
  { path: "/docs/quickstart", title: "Quickstart", load: () => import("./pages/docs.js") },
  { path: "/docs/authentication", title: "Authentication", load: () => import("./pages/docs.js") },
  { path: "/docs/api", title: "API overview", load: () => import("./pages/docs.js") },
  { path: "/docs/api/chat-completions", title: "Chat completions", load: () => import("./pages/docs.js") },
  { path: "/docs/models", title: "Models endpoint", load: () => import("./pages/docs.js") },
  { path: "/docs/usage", title: "Usage endpoint", load: () => import("./pages/docs.js") },
  { path: "/docs/errors", title: "Errors", load: () => import("./pages/docs.js") },
  { path: "/docs/sdk/python", title: "Python", load: () => import("./pages/docs.js") },
  { path: "/docs/sdk/javascript", title: "JavaScript", load: () => import("./pages/docs.js") },
  { path: "/docs/sdk/curl", title: "cURL", load: () => import("./pages/docs.js") },
  { path: "/docs/limitations", title: "Limitations", load: () => import("./pages/docs.js") },
  { path: "/dashboard", title: "Dashboard", auth: true, load: () => import("./pages/dashboard.js") },
  { path: "/dashboard/usage", title: "Usage", auth: true, load: () => import("./pages/dashboard.js") },
  { path: "/dashboard/api-keys", title: "API keys", auth: true, load: () => import("./pages/dashboard.js") },
  { path: "/dashboard/settings", title: "Settings", auth: true, load: () => import("./pages/dashboard.js") },
  { path: "/status", title: "Status", load: () => import("./pages/status.js") },
  { path: "/signin", title: "Sign in", load: () => import("./pages/signin.js") },
];
const NOT_FOUND = { path: "*", title: "Page not found", load: () => import("./pages/notfound.js") };

const LEGACY = { home: "/", chat: "/playground", keys: "/dashboard/api-keys", usage: "/dashboard/usage", docs: "/docs", signin: "/signin" };

let renderToken = 0;
let leave = null;
let hooks = { onRoute: () => {} };

export const currentPath = () => location.pathname.replace(/\/+$/, "") || "/";
const find = (p) => ROUTES.find((r) => r.path === p) || null;

export function navigate(to, { replace = false } = {}) {
  const url = new URL(to, location.origin);
  if (url.pathname + url.search === location.pathname + location.search && !url.hash) return render(false);
  history[replace ? "replaceState" : "pushState"](null, "", url.pathname + url.search + url.hash);
  return render(false);
}

/* Mark the link that points at the current page in a subtree: data-match "exact" or "prefix". */
export function markCurrent(root = document) {
  const path = currentPath();
  root.querySelectorAll("a[data-match]").forEach((a) => {
    const href = new URL(a.getAttribute("href"), location.origin).pathname.replace(/\/+$/, "") || "/";
    const on = a.dataset.match === "prefix" ? path === href || path.startsWith(href + "/") : path === href;
    if (on) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
}

function announce(text) { const a = document.getElementById("announcer"); if (a) { a.textContent = ""; setTimeout(() => { a.textContent = text; }, 50); } }

async function render(initial) {
  const token = ++renderToken;
  const path = currentPath();
  const route = find(path) || NOT_FOUND;
  const view = document.getElementById("view");
  if (leave) { try { leave(); } catch (e) { /* a page's cleanup must never block navigation */ } leave = null; }

  if (route.auth && needsAccount() && !state.me) { return navigate("/signin?next=" + encodeURIComponent(path + location.search), { replace: true }); }

  const slow = setTimeout(() => { if (token === renderToken) view.replaceChildren(el("div", { class: "container page" }, LoadingRegion("Loading"))); }, 150);
  let result;
  try {
    const mod = await route.load();
    result = await mod.default({ path, route, query: new URLSearchParams(location.search), navigate });
  } catch (e) {
    clearTimeout(slow);
    if (token !== renderToken) return;
    console.error("page failed", path, e);                                         // for developers; the visitor gets a plain message
    view.replaceChildren(el("div", { class: "container page" }, ErrorState({ title: "This page could not be loaded", message: "Check your connection and try again.", retry: () => render(false) })));
    document.title = "Error · aarav-lm";
    return;
  }
  clearTimeout(slow);
  if (token !== renderToken) return;                                                // a newer navigation won
  view.replaceChildren(...[result.node].flat(Infinity).filter(Boolean));
  leave = result.onLeave || null;
  const title = result.title || route.title;
  document.title = path === "/" ? route.title : `${title} · aarav-lm`;
  const desc = document.querySelector('meta[name="description"]');
  if (desc && result.description) desc.setAttribute("content", result.description);
  markCurrent(document);
  document.body.dataset.page = path.split("/")[1] || "home";
  hooks.onRoute(path);
  if (!initial) {
    const hash = location.hash && document.getElementById(decodeURIComponent(location.hash.slice(1)));
    if (hash) hash.scrollIntoView(); else window.scrollTo(0, 0);
    view.focus({ preventScroll: true });
    announce(title);
  }
}

export function startRouter(h) {
  hooks = { ...hooks, ...h };
  const m = location.hash.match(/^#\/([a-z]+)/);                                    // old hash links (#/chat) still work
  if (m && LEGACY[m[1]]) history.replaceState(null, "", LEGACY[m[1]]);
  window.addEventListener("popstate", () => render(false));
  document.addEventListener("click", (e) => {
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    const a = e.target.closest && e.target.closest("a[href]");
    if (!a || a.target === "_blank" || a.hasAttribute("download") || a.dataset.external !== undefined) return;
    const url = new URL(a.href, location.origin);
    if (url.origin !== location.origin || !find(url.pathname.replace(/\/+$/, "") || "/")) return;
    e.preventDefault();
    navigate(url.pathname + url.search + url.hash);
  });
  return render(true);
}
