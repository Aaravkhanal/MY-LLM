/* The frame around every page: header (brand, navigation, theme toggle, account menu) and footer. Re-rendered when the visitor signs in or out. */
import { el, icon } from "./dom.js";
import { Button, Dropdown, toast } from "./ui.js";
import { state, needsAccount } from "./state.js";
import { json } from "./api.js";
import { navigate, markCurrent } from "./router.js";

const NAV = [["/playground", "Playground"], ["/models", "Models"], ["/docs", "Docs"], ["/status", "Status"]];

function theme() { return document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark"; }
function setTheme(t) {
  document.documentElement.setAttribute("data-theme", t);
  try { localStorage.setItem("aarav_theme", t); } catch (e) { /* not fatal */ }
}

function themeToggle() {
  const b = Button({ variant: "quiet", size: "sm", icon: theme() === "dark" ? "sun" : "moon", ariaLabel: theme() === "dark" ? "Switch to light theme" : "Switch to dark theme" });
  b.addEventListener("click", () => { setTheme(theme() === "dark" ? "light" : "dark"); renderChrome(); });
  return b;
}

export async function signOut() {
  try { await json("/account/logout", { method: "POST", body: {} }); } catch (e) { /* the cookie is cleared or already invalid either way */ }
  state.me = null;
  renderChrome();
  toast("Signed out.");
  navigate("/");
}

function accountArea() {
  if (!needsAccount()) return null;
  if (!state.me) return Button({ label: "Sign in", href: "/signin", size: "sm", variant: "secondary" });
  return Dropdown({ label: state.me.email.length > 22 ? state.me.email.slice(0, 20) + "…" : state.me.email, header: state.me.email,
    items: [{ label: "Dashboard", href: "/dashboard" }, { label: "API keys", href: "/dashboard/api-keys" }, { label: "Usage", href: "/dashboard/usage" },
      { label: "Settings", href: "/dashboard/settings" }, { label: "Sign out", onclick: signOut }] });
}

export function renderChrome() {
  const header = document.getElementById("site-header");
  header.replaceChildren(el("div", { class: "container bar" },
    el("a", { class: "brand", href: "/", "aria-label": "aarav-lm home" }, icon("logo", 20), el("span", { text: "aarav-lm" })),
    el("nav", { class: "primary-nav", "aria-label": "Primary" }, NAV.map(([href, label]) => el("a", { href, "data-match": "prefix", text: label }))),
    el("div", { class: "header-actions" }, themeToggle(), accountArea())));
  const footer = document.getElementById("site-footer");
  const list = (title, items) => el("div", null, el("h2", { text: title }), el("ul", null, items.map(([href, label]) => el("li", null, el("a", { href, text: label })))));
  footer.replaceChildren(el("div", { class: "container" },
    el("div", { class: "footer-grid" },
      el("div", null, el("a", { class: "brand", href: "/" }, icon("logo", 18), el("span", { text: "aarav-lm" })),
        el("p", { class: "small", text: "A 14.2M-parameter language model, trained, evaluated and served from a custom pipeline." })),
      list("Product", [["/playground", "Playground"], ["/models/aarav-lm", "Model"], ["/status", "Status"]]),
      list("Developers", [["/docs/quickstart", "Quickstart"], ["/docs/api/chat-completions", "Chat completions"], ["/docs/errors", "Errors"]]),
      list("Research", [["/models/aarav-lm/evaluations", "Evaluations"], ["/models/aarav-lm/architecture", "Architecture"], ["/docs/limitations", "Limitations"]])),
    el("p", { class: "footer-note xs", text: "Built by Aarav Khanal. Every number on this site comes from the project's own evaluation and run files." })));
  markCurrent(document);
}
