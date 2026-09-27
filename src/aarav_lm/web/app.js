"use strict";
/* aarav web app: chat, keys, usage, docs and sign-in in one small single-page app.
   Everything is built with DOM calls and textContent, never HTML strings: model output and server messages can never
   become markup (tests/test_api.py::test_ui_never_writes_model_output_as_html). Same-origin only, no libraries. */

const state = {
  config: { auth: "none", signup: "closed", model: "", limits: {} },
  me: null,                       // {email, csrf_token} when signed in with an account
  apiKey: "",                     // legacy key mode: kept for this tab only
  history: [],                    // [{role, content}] sent with every request (the server is stateless)
  busy: false,
  controller: null,
  modelInfo: null,
};

try { state.apiKey = sessionStorage.getItem("aarav_key") || ""; } catch (e) { /* private mode: keep it in memory */ }

const SETTINGS_DEFAULTS = { system: "", temperature: 0.8, top_p: 0.95, top_k: 50, max_tokens: 150, repetition_penalty: 1.1, seed: "" };
function loadSettings() {
  try { return { ...SETTINGS_DEFAULTS, ...JSON.parse(localStorage.getItem("aarav_settings") || "{}") }; } catch (e) { return { ...SETTINGS_DEFAULTS }; }
}
function saveSettings(s) { try { localStorage.setItem("aarav_settings", JSON.stringify(s)); } catch (e) { /* not fatal */ } }
let settings = loadSettings();

/* ── tiny DOM helpers ──────────────────────────────────────────────────── */

function el(tag, props, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v === false || v === null || v === undefined) continue;
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v === true) n.setAttribute(k, "");
    else n.setAttribute(k, v);
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid === null || kid === undefined || kid === false) continue;
    n.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return n;
}
const SVG_NS = "http://www.w3.org/2000/svg";
function svg(tag, attrs, ...kids) {
  const n = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v);
  for (const kid of kids.flat(Infinity)) if (kid) n.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return n;
}
const view = () => document.getElementById("view");
function mount(...nodes) { const v = view(); v.replaceChildren(...nodes.flat(Infinity).filter(Boolean)); return v; }
const fmt = (n) => Number(n).toLocaleString("en-US");
const short = (n) => (n >= 1e6 ? (n / 1e6).toFixed(1).replace(/\.0$/, "") + "M" : n >= 1e3 ? (n / 1e3).toFixed(1).replace(/\.0$/, "") + "k" : String(n));
function when(iso) { if (!iso) return "never"; const d = new Date(iso); return isNaN(d) ? "" : d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" }); }

/* ── talking to the server ─────────────────────────────────────────────── */

class ApiError extends Error {
  constructor(status, data, retryAfter) {
    super(describeError(status, data, retryAfter));
    this.status = status;
  }
}

function serverMessage(data) {
  if (!data) return "";
  if (typeof data.detail === "string") return data.detail;
  if (data.error && typeof data.error === "object") return data.error.message || "";
  if (typeof data.error === "string") return data.error;
  return "";
}

function describeError(status, data, retryAfter) {
  const msg = serverMessage(data);
  if (status === 401) return state.config.auth === "account" ? "Please sign in." : "That key was not accepted. Check it and try again.";
  if (status === 403) return msg || "That request was not allowed.";
  if (status === 429) return (msg || "Slow down a little") + (retryAfter ? ` (try again in ${retryAfter}s).` : ".");
  if (status === 503) return "The model is busy with someone else's request. Try again in a few seconds.";
  if (status === 422 || status === 400) return msg || "The request was not valid.";
  if (status === 413) return "That message is too large.";
  if (status === 409 || status === 404) return msg || "That did not work.";
  return msg || `The server returned an error (${status}).`;
}

function headers(hasBody) {
  const h = { Accept: "application/json" };
  if (hasBody) h["Content-Type"] = "application/json";
  if (state.me) h["X-CSRF-Token"] = state.me.csrf_token;
  else if (state.apiKey) h["Authorization"] = "Bearer " + state.apiKey;
  return h;
}

async function call(path, { method = "GET", body, signal } = {}) {
  return fetch(path, { method, headers: headers(body !== undefined), body: body === undefined ? undefined : JSON.stringify(body), signal, credentials: "same-origin" });
}

async function json(path, opts) {
  const res = await call(path, opts);
  let data = null;
  try { data = await res.json(); } catch (e) { /* empty body */ }
  if (!res.ok) throw new ApiError(res.status, data, res.headers.get("retry-after"));
  return data;
}

/* ── router and frame ──────────────────────────────────────────────────── */

const ROUTES = { home: viewHome, chat: viewChat, keys: viewKeys, usage: viewUsage, docs: viewDocs, signin: viewSignin };

function currentRoute() {
  const r = (location.hash.replace(/^#\/?/, "") || "home").split("?")[0];
  return ROUTES[r] ? r : "home";
}

function navigate(name) { location.hash = "#/" + name; }

function needsAccount(r) { return state.config.auth === "account" && !state.me && r !== "docs" && r !== "home"; }

function renderFrame() {
  const r = currentRoute();
  const links = [["home", "home"]];
  if (!needsAccount("chat")) links.push(["chat", "chat"]);
  if (state.config.auth === "account" && state.me) links.push(["keys", "keys"], ["usage", "usage"]);
  links.push(["docs", "docs"]);
  document.getElementById("nav").replaceChildren(...links.map(([name, label]) =>
    el("a", { href: "#/" + name, text: label, "aria-current": r === name ? "page" : false })));
  const who = document.getElementById("who");
  if (state.me) {
    who.replaceChildren(el("span", { text: state.me.email }), el("button", { type: "button", class: "quiet", text: "sign out", onclick: signOut }));
  } else if (state.config.auth === "account") {
    who.replaceChildren(el("a", { href: "#/signin", text: "sign in" }));
  } else {
    who.replaceChildren();
  }
}

async function route() {
  const r = needsAccount(currentRoute()) ? "signin" : currentRoute();
  if (r !== currentRoute()) { history.replaceState(null, "", "#/signin"); }
  renderFrame();
  if (state.controller) { state.controller.abort(); }
  await ROUTES[r]();
  view().focus({ preventScroll: true });
}

async function signOut() {
  try { await json("/account/logout", { method: "POST", body: {} }); } catch (e) { /* the cookie is gone either way */ }
  state.me = null; state.history = [];
  navigate("signin");
  route();
}

/* ── sign in / create account ──────────────────────────────────────────── */

function viewSignin() {
  let creating = false;
  const c = state.config;
  const canCreate = c.signup === "invite" || c.signup === "open";
  const err = el("p", { class: "error", role: "alert", hidden: true });
  const email = el("input", { id: "email", type: "email", autocomplete: "username", required: true, spellcheck: "false" });
  const pass = el("input", { id: "password", type: "password", autocomplete: "current-password", required: true, minlength: "10", maxlength: "128" });
  const code = el("input", { id: "invite", type: "text", autocomplete: "off", spellcheck: "false" });
  const codeLabel = el("label", { hidden: true }, "Invite code", code);
  const heading = el("h1", { text: "Sign in" });
  const submit = el("button", { type: "submit", text: "Sign in →" });
  const toggle = canCreate ? el("button", { type: "button", class: "quiet", text: "Create an account", onclick: () => {
    creating = !creating;
    heading.textContent = creating ? "Create an account" : "Sign in";
    submit.textContent = creating ? "Create account →" : "Sign in →";
    toggle.textContent = creating ? "I already have an account" : "Create an account";
    pass.setAttribute("autocomplete", creating ? "new-password" : "current-password");
    codeLabel.hidden = !(creating && c.signup === "invite");
    err.hidden = true;
  } }) : null;
  const form = el("form", { onsubmit: async (e) => {
    e.preventDefault();
    err.hidden = true; submit.disabled = true;
    try {
      const body = { email: email.value, password: pass.value };
      if (creating) body.invite_code = code.value;
      state.me = await json(creating ? "/account/signup" : "/account/login", { method: "POST", body });
      navigate("chat");
    } catch (ex) {
      err.textContent = ex.message; err.hidden = false;
    } finally { submit.disabled = false; }
  } },
  el("label", null, "Email", email), el("label", null, "Password", pass), codeLabel, err,
  el("p", { class: "small inline" }, submit, toggle));
  mount(heading, el("p", { class: "muted", text: c.signup === "closed" ? "Accounts are by invitation. Ask the owner for one." : c.signup === "invite" ? "Free to use while capacity lasts. Creating an account needs an invite code from the owner." : "Free to use while capacity lasts." }), form);
  email.focus();
}

/* ── chat ──────────────────────────────────────────────────────────────── */

const EXAMPLES = ["Tell me a story about a dragon named Zara.", "Write a short story about a fox and a moon.", "A long story about a robot who feels sad."];

async function viewChat() {
  const list = el("div", { id: "messages", "aria-live": "polite" });
  const banner = el("p", { id: "banner", class: "error", role: "status", hidden: true });
  const info = el("span", { id: "model-info", text: "loading model…" });
  const input = el("textarea", { id: "input", rows: "1", placeholder: "Message aarav…  (Enter to send, Shift+Enter for a new line)", required: true, "aria-label": "Message" });
  const send = el("button", { id: "send", type: "submit", text: "send →" });
  const stop = el("button", { id: "stop", type: "button", class: "quiet", text: "stop", hidden: true });

  const numberField = (id, label, key, min, max, step) => el("label", null, label,
    el("input", { id, type: "number", min, max, step, value: settings[key], oninput: (e) => { settings[key] = e.target.value; saveSettings(settings); } }));
  const rangeField = (id, label, key, min, max, step) => {
    const out = el("output", { id: id + "-out", text: settings[key] });
    return el("label", null, label, out, el("input", { id, type: "range", min, max, step, value: settings[key], oninput: (e) => {
      settings[key] = Number(e.target.value); out.textContent = e.target.value; saveSettings(settings);
    } }));
  };
  const settingsPane = el("div", { id: "settings", hidden: true },
    el("label", null, "System prompt (instruction-tuned models)", el("textarea", { id: "system", rows: "2", placeholder: "You are a friendly storyteller for children.",
      oninput: (e) => { settings.system = e.target.value; saveSettings(settings); } }, "")),
    el("div", { class: "grid" },
      rangeField("temperature", "Temperature", "temperature", 0, 2, 0.05), rangeField("top_p", "Top-p", "top_p", 0.05, 1, 0.05),
      numberField("top_k", "Top-k", "top_k", 1, 2048, 1), numberField("max_tokens", "Max tokens", "max_tokens", 1, 512, 1),
      numberField("repetition_penalty", "Repetition penalty", "repetition_penalty", 0.5, 2, 0.05), numberField("seed", "Seed (blank = random)", "seed", 0, "", 1)));
  settingsPane.querySelector("#system").value = settings.system;

  const toggleSettings = el("button", { id: "settings-toggle", type: "button", class: "quiet", text: "settings", "aria-expanded": "false", "aria-controls": "settings", onclick: () => {
    settingsPane.hidden = !settingsPane.hidden;
    toggleSettings.setAttribute("aria-expanded", String(!settingsPane.hidden));
  } });
  const clear = el("button", { id: "clear", type: "button", class: "quiet", text: "clear", onclick: () => { if (!state.busy) { state.history = []; renderList(); banner.hidden = true; input.focus(); } } });

  const keyForm = state.config.auth === "key" && !state.apiKey ? keyPrompt(() => route()) : null;
  const form = el("form", { id: "composer", autocomplete: "off" }, input, send, stop);

  function renderList() {
    if (!state.history.length) {
      list.replaceChildren(el("div", { class: "empty" }, el("p", { text: "Ask for a story. It writes short children's stories, and it is a small model: it is not a general assistant." }),
        el("ul", null, EXAMPLES.map((t) => el("li", null, el("button", { type: "button", text: t, onclick: () => { input.value = t; autosize(); input.focus(); } }))))));
      return;
    }
    list.replaceChildren(...state.history.map((m) => message(m.role, m.content)));
    window.scrollTo({ top: document.body.scrollHeight });
  }
  const autosize = () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 160) + "px"; };
  input.addEventListener("input", autosize);
  input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); form.requestSubmit(); } });
  stop.addEventListener("click", () => { if (state.controller) state.controller.abort(); });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = input.value.trim();
    if (!text || state.busy) return;
    if (state.config.auth === "key" && !state.apiKey) { banner.textContent = "Enter your API key first."; banner.hidden = false; return; }
    banner.hidden = true;
    state.history.push({ role: "user", content: text });
    input.value = ""; autosize();
    renderList();
    const reply = message("assistant", "");
    reply.querySelector(".body").classList.add("dots");
    list.append(reply);
    window.scrollTo({ top: document.body.scrollHeight });
    setBusy(true);
    let answer = "";
    try {
      answer = await streamChat(state.history, (delta) => {
        const body = reply.querySelector(".body");
        body.classList.remove("dots");
        body.textContent += delta;
        window.scrollTo({ top: document.body.scrollHeight });
      });
      if (answer) state.history.push({ role: "assistant", content: answer });
      else { state.history.pop(); reply.remove(); banner.textContent = "The model returned nothing. Try rephrasing."; banner.hidden = false; }
    } catch (ex) {
      const partial = reply.querySelector(".body").textContent;
      if (ex.name === "AbortError") { if (partial) state.history.push({ role: "assistant", content: partial }); else { state.history.pop(); reply.remove(); } }
      else {
        state.history.pop(); reply.remove();
        banner.textContent = ex instanceof ApiError ? ex.message : "Could not reach the server. Check your connection.";
        banner.hidden = false;
        if (ex.status === 401 && state.config.auth === "account") { state.me = null; navigate("signin"); route(); }
      }
    } finally { setBusy(false); input.focus(); }
  });

  function setBusy(b) {
    state.busy = b; send.hidden = b; stop.hidden = !b; input.disabled = b; clear.disabled = b;
  }

  mount(el("div", { class: "chat-meta" }, info, el("span", { class: "spacer" }), toggleSettings, clear), settingsPane, keyForm, banner, list, form);
  renderList(); autosize(); input.focus();
  try {
    const m = state.modelInfo || (await json("/v1/models")).data[0];
    state.modelInfo = m;
    info.textContent = `${m.id} · ${(m.parameters / 1e6).toFixed(1)}M parameters · context ${m.context_length}${m.instruction_tuned ? " · chat-tuned" : " · text continuation"}`;
  } catch (ex) { info.textContent = ex.status === 401 ? "" : "model unavailable"; }
}

function keyPrompt(done) {
  const key = el("input", { type: "password", autocomplete: "off", spellcheck: "false", placeholder: "aak_…", "aria-label": "API key" });
  return el("form", { onsubmit: (e) => { e.preventDefault(); state.apiKey = key.value.trim(); try { sessionStorage.setItem("aarav_key", state.apiKey); } catch (x) { /* memory only */ } done(); } },
    el("label", null, "API key (kept in this tab only)", key), el("p", { class: "small" }, el("button", { type: "submit", text: "Use this key →" })));
}

function message(role, text) {
  return el("article", { class: "msg " + role }, el("h3", { class: "role", text: role === "user" ? "you" : "aarav" }), el("p", { class: "body", text }));
}

function chatPayload(history) {
  const msgs = [];
  if (settings.system.trim()) msgs.push({ role: "system", content: settings.system.trim() });
  msgs.push(...history);
  const p = { model: state.modelInfo ? state.modelInfo.id : "aarav-lm", messages: msgs, stream: true, temperature: Number(settings.temperature),
    top_p: Number(settings.top_p), top_k: Number(settings.top_k), max_tokens: Number(settings.max_tokens), repetition_penalty: Number(settings.repetition_penalty) };
  if (settings.seed !== "" && settings.seed !== null) p.seed = Number(settings.seed);
  return p;
}

/* Real streaming: each frame is a token the model actually produced (nothing is faked or delayed here). */
async function streamChat(history, onDelta) {
  state.controller = new AbortController();
  try {
    const res = await call("/v1/chat/completions", { method: "POST", body: chatPayload(history), signal: state.controller.signal });
    if (!res.ok) { let d = null; try { d = await res.json(); } catch (e) { /* no body */ } throw new ApiError(res.status, d, res.headers.get("retry-after")); }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "", out = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) !== -1) {
        const frame = buf.slice(0, i).trim(); buf = buf.slice(i + 2);
        if (!frame.startsWith("data:")) continue;
        const payload = frame.slice(5).trim();
        if (payload === "[DONE]") return out;
        let obj; try { obj = JSON.parse(payload); } catch (e) { continue; }
        if (obj.error) throw new ApiError(obj.error.code === "server_busy" ? 503 : 500, obj);
        const delta = obj.choices && obj.choices[0] && obj.choices[0].delta && obj.choices[0].delta.content;
        if (delta) { out += delta; onDelta(delta); }
      }
    }
    return out;
  } finally { state.controller = null; }
}

/* ── keys ──────────────────────────────────────────────────────────────── */

async function viewKeys() {
  const shown = el("div");
  const listEl = el("div", { class: "rows" });
  const err = el("p", { class: "error", role: "alert", hidden: true });
  const label = el("input", { id: "key-label", type: "text", maxlength: "40", placeholder: "what it is for, e.g. my laptop", autocomplete: "off" });
  const create = el("button", { type: "submit", text: "Create key →" });

  async function refresh() {
    try {
      const keys = await json("/account/keys");
      const head = el("div", { class: "row head" }, el("span", { text: "key" }), el("span", { text: "label" }), el("span", { class: "hide-sm", text: "created" }), el("span", { class: "hide-sm", text: "last used" }), el("span"));
      if (!keys.length) { listEl.replaceChildren(el("p", { class: "muted", text: "No keys yet. Create one to use the API from your own code." })); return; }
      listEl.replaceChildren(head, ...keys.slice().reverse().map((k) => el("div", { class: "row" + (k.active ? "" : " gone") },
        el("span", { class: "mono", text: k.prefix }), el("span", { text: k.label || "(no label)" + (k.active ? "" : " · " + (k.revoked_at ? "revoked" : "expired")) }),
        el("span", { class: "hide-sm", text: when(k.created_at) }), el("span", { class: "hide-sm", text: when(k.last_used_at) }),
        el("span", { class: "actions" }, k.active ? [
          el("button", { type: "button", text: "rotate", onclick: () => act(k.id, "rotate") }),
          el("button", { type: "button", class: "danger", text: "revoke", onclick: () => { if (confirm("Revoke this key? Anything using it stops working immediately.")) act(k.id, "revoke"); } }),
        ] : null))));
    } catch (ex) { err.textContent = ex.message; err.hidden = false; }
  }
  async function act(id, what) {
    err.hidden = true;
    try {
      const r = await json(`/account/keys/${id}/${what}`, { method: "POST", body: {} });
      if (what === "rotate") revealKey(r.key, "Here is the replacement. The old key no longer works.");
      await refresh();
    } catch (ex) { err.textContent = ex.message; err.hidden = false; }
  }
  function revealKey(key, note) {
    shown.replaceChildren(el("div", { class: "secret" }, el("p", { class: "small muted", text: note + " It is shown once; copy it now." }),
      el("span", { class: "mono", text: key }), el("p", { class: "small" }, el("button", { type: "button", text: "copy", onclick: (e) => copyText(key, e.target) }))));
  }
  const form = el("form", { onsubmit: async (e) => {
    e.preventDefault(); err.hidden = true; create.disabled = true;
    try { const r = await json("/account/keys", { method: "POST", body: { label: label.value } }); label.value = ""; revealKey(r.key, "Your new key."); await refresh(); }
    catch (ex) { err.textContent = ex.message; err.hidden = false; } finally { create.disabled = false; }
  } }, el("div", { class: "field-row" }, el("label", null, "Label", label), el("span", null, create)));

  const lim = state.config.limits || {};
  mount(el("h1", { text: "API keys" }),
    el("p", { class: "muted", text: "Use a key with any OpenAI-compatible client. Keys are stored only as hashes, so a lost key cannot be recovered; make a new one." }),
    form, err, shown,
    el("h2", { text: "Your keys" }), listEl,
    lim.key_requests_per_minute ? el("p", { class: "small muted", text: `Each key: ${lim.key_requests_per_minute} requests and ${fmt(lim.key_tokens_per_minute)} tokens per minute; up to ${lim.max_keys} active keys.` }) : null,
    el("h2", { text: "Account" }), passwordForm());
  await refresh();
}

function passwordForm() {
  const msg = el("p", { class: "notice", role: "status", hidden: true });
  const cur = el("input", { type: "password", autocomplete: "current-password", required: true, maxlength: "128" });
  const nxt = el("input", { type: "password", autocomplete: "new-password", required: true, minlength: "10", maxlength: "128" });
  return el("form", { onsubmit: async (e) => {
    e.preventDefault(); msg.hidden = true;
    try { await json("/account/password", { method: "POST", body: { current: cur.value, new: nxt.value } }); cur.value = ""; nxt.value = "";
      msg.className = "notice"; msg.textContent = "Password changed. Other devices were signed out."; }
    catch (ex) { msg.className = "error"; msg.textContent = ex.message; }
    msg.hidden = false;
  } }, el("div", { class: "field-row" }, el("label", null, "Current password", cur), el("label", null, "New password (10+ characters)", nxt), el("span", null, el("button", { type: "submit", text: "Change password" }))), msg);
}

async function copyText(text, button) {
  const old = button.textContent;
  try { await navigator.clipboard.writeText(text); button.textContent = "copied"; }
  catch (e) { button.textContent = "select and copy manually"; }
  setTimeout(() => { button.textContent = old; }, 1600);
}

/* ── usage ─────────────────────────────────────────────────────────────── */

async function viewUsage() {
  mount(el("h1", { text: "Usage" }), el("p", { class: "muted", text: "loading…" }));
  let u;
  try { u = await json("/account/usage?days=14"); } catch (ex) { mount(el("h1", { text: "Usage" }), el("p", { class: "error", text: ex.message })); return; }
  const t = u.totals;
  const figure = (n, l) => el("div", { class: "figure" }, el("div", { class: "n", text: fmt(n) }), el("div", { class: "l", text: l }));
  const chart = el("div", { class: "chart" });
  const table = el("div", { hidden: true });
  const toggle = el("button", { type: "button", class: "quiet", text: "view as table", "aria-expanded": "false", onclick: () => {
    table.hidden = !table.hidden; chart.hidden = !table.hidden; toggle.textContent = table.hidden ? "view as table" : "view as chart";
    toggle.setAttribute("aria-expanded", String(!table.hidden));
  } });
  const empty = t.requests === 0;
  if (!empty) drawBars(chart, u.daily);
  table.replaceChildren(el("table", null, el("thead", null, el("tr", null, el("th", { text: "day" }), el("th", { class: "num", text: "requests" }), el("th", { class: "num", text: "tokens" }))),
    el("tbody", null, u.daily.slice().reverse().map((d) => el("tr", null, el("td", { text: d.date }), el("td", { class: "num", text: fmt(d.requests) }), el("td", { class: "num", text: fmt(d.tokens) }))))));
  mount(el("h1", { text: "Usage" }), el("p", { class: "muted", text: "The last 14 days, counted in tokens. Prompts and replies are never stored, only the counts." }),
    el("div", { class: "figures" }, figure(t.requests, "requests"), figure(t.total_tokens, "tokens"), figure(t.errors, "errors")),
    empty ? el("p", { class: "muted", text: "Nothing yet. Send a message in chat, or use a key, and it shows up here." }) : [el("h2", { text: "Tokens per day" }), chart, table, el("p", { class: "small" }, toggle)],
    el("p", { class: "small muted", text: `Browser chat is limited to ${fmt(u.limits.requests_per_minute)} requests and ${fmt(u.limits.tokens_per_minute)} tokens per minute.` }));
}

function niceMax(v) {
  if (v <= 0) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v))), f = v / p;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p;
}

function drawBars(host, days) {
  const W = 640, H = 210, L = 44, B = 26, T = 8, plotH = H - B - T, plotW = W - L;
  const max = niceMax(Math.max(...days.map((d) => d.tokens)));
  const slot = plotW / days.length, bw = Math.max(8, Math.min(22, slot * 0.55));
  const y = (v) => T + plotH - (v / max) * plotH;
  const s = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `Tokens per day for the last ${days.length} days; use "view as table" for the numbers.` });
  const grid = svg("g", { class: "grid" }), axis = svg("g", { class: "axis" });
  for (let i = 0; i <= 2; i++) {                                   // 0, half, top: round numbers because max is 1, 2 or 5 x 10^k
    const v = (max * i) / 2, yy = y(v);
    grid.append(svg("line", { x1: L, x2: W, y1: yy, y2: yy }));
    axis.append(svg("text", { x: L - 8, y: yy + 4, "text-anchor": "end" }, short(Math.round(v))));
  }
  s.append(grid, axis);
  const tip = el("div", { class: "tip", hidden: true });
  days.forEach((d, i) => {
    const cx = L + slot * i + slot / 2, top = y(d.tokens), h = T + plotH - top;
    const label = new Date(d.date + "T00:00:00").toLocaleDateString(undefined, { month: "short", day: "numeric" });
    if (i % 3 === 0 || i === days.length - 1) axis.append(svg("text", { x: cx, y: H - 6, "text-anchor": "middle" }, label));
    const hit = svg("rect", { class: "hit", x: L + slot * i, y: T, width: slot, height: plotH, tabindex: "0", role: "img", "aria-label": `${label}: ${fmt(d.tokens)} tokens, ${fmt(d.requests)} requests` });
    const show = () => { tip.textContent = `${label} · ${fmt(d.tokens)} tokens · ${fmt(d.requests)} requests`; tip.style.left = Math.min(88, Math.max(12, (cx / W) * 100)) + "%"; tip.style.top = (Math.max(top, 30) / H) * 100 + "%"; tip.hidden = false; };
    hit.addEventListener("mouseenter", show); hit.addEventListener("focus", show);
    hit.addEventListener("mouseleave", () => { tip.hidden = true; }); hit.addEventListener("blur", () => { tip.hidden = true; });
    // a bar with 4px rounded data-end, square at the baseline
    const r = Math.min(4, bw / 2, h);
    const path = d.tokens > 0
      ? `M${cx - bw / 2},${T + plotH} V${top + r} Q${cx - bw / 2},${top} ${cx - bw / 2 + r},${top} H${cx + bw / 2 - r} Q${cx + bw / 2},${top} ${cx + bw / 2},${top + r} V${T + plotH} Z`
      : `M${cx - bw / 2},${T + plotH - 2} H${cx + bw / 2} V${T + plotH} H${cx - bw / 2} Z`;
    s.append(hit, svg("path", { class: "bar" + (d.tokens > 0 ? "" : " zero"), d: path }));
  });
  host.replaceChildren(s, tip);
}

/* ── docs ──────────────────────────────────────────────────────────────── */

function apiSnippets() {
  const base = location.origin + "/v1";
  return [
    ["curl", `curl ${base}/chat/completions \\\n  -H "Authorization: Bearer $AARAV_KEY" \\\n  -H "Content-Type: application/json" \\\n  -d '{"model":"aarav-lm","max_tokens":150,\n       "messages":[{"role":"user","content":"Tell me a story about a fox named Kip."}]}'`],
    ["python", `from openai import OpenAI\n\nclient = OpenAI(base_url="${base}", api_key="aak_...")\nr = client.chat.completions.create(\n    model="aarav-lm",\n    messages=[{"role": "user", "content": "Tell me a story about a fox named Kip."}],\n    max_tokens=150,\n)\nprint(r.choices[0].message.content)`],
    ["javascript", `const r = await fetch("${base}/chat/completions", {\n  method: "POST",\n  headers: { Authorization: "Bearer " + KEY, "Content-Type": "application/json" },\n  body: JSON.stringify({ model: "aarav-lm", max_tokens: 150,\n    messages: [{ role: "user", content: "Tell me a story about a fox named Kip." }] }),\n});\nconsole.log((await r.json()).choices[0].message.content);`],
  ];
}

async function viewDocs() {
  const base = location.origin + "/v1";
  const snippets = apiSnippets();
  const lim = state.config.limits || {};
  let m = state.modelInfo;
  if (!m) { try { m = (await json("/v1/models")).data[0]; state.modelInfo = m; } catch (e) { m = null; } }
  const size = m ? `${(m.parameters / 1e6).toFixed(1)}-million-parameter` : "small";
  mount(el("h1", { text: "Docs" }),
    el("p", { class: "muted", text: "An OpenAI-compatible API. Point any OpenAI client at the base URL below and use one of your keys." }),
    el("h2", { text: "Base URL" }), el("pre", null, el("code", { text: base })),
    el("h2", { text: "Quickstart" }),
    state.config.auth === "account" ? el("p", null, state.me ? el("a", { href: "#/keys", text: "Create a key" }) : el("a", { href: "#/signin", text: "Sign in" }), " first, then:") : el("p", { text: "Then:" }),
    snippets.map(([name, code]) => [el("div", { class: "snippet-head" }, el("span", { text: name }), el("button", { type: "button", class: "quiet", text: "copy", onclick: (e) => copyText(code, e.target) })), el("pre", null, el("code", { text: code }))]),
    el("h2", { text: "Endpoints" }),
    el("pre", null, el("code", { text: "GET  /v1/models\nPOST /v1/chat/completions   stream: true supported\nPOST /v1/completions        raw text continuation\nGET  /v1/usage              your key's own totals" })),
    el("h2", { text: "Limits" }),
    el("p", { class: "muted", text: lim.key_requests_per_minute ? `${lim.key_requests_per_minute} requests and ${fmt(lim.key_tokens_per_minute)} tokens per minute per key. Requests over a limit get 429 with Retry-After. One generation runs at a time; a busy server answers 503 with Retry-After. Each generation is cut off after 60 seconds.` : "Rate limits apply; a 429 or 503 response carries Retry-After." }),
    el("h2", { text: "What this model is" }),
    el("p", { class: "muted", text: `A ${size} language model trained from scratch on simple children's stories. It writes short stories and follows simple requests (a name, a topic, a place, a feeling, a length) far better than chance, but not reliably. It has no world knowledge and cannot answer questions, write code or do maths. Treat it as a small, honest research model.` }),
    el("p", { class: "small muted" }, "Errors are OpenAI-shaped: ", el("span", { class: "mono", text: '{"error": {"message", "type", "code"}}' }), "."));
}


/* ── home: the public front page ───────────────────────────────────────── */

let resultsCache = null;
async function loadResults() {
  if (!resultsCache) { const res = await fetch("/results.json", { credentials: "same-origin" }); if (!res.ok) throw new Error("results unavailable"); resultsCache = await res.json(); }
  return resultsCache;
}

/* An accessible tab strip: arrow keys move between tabs; the panel content is rendered by `render(id)`. */
function tabs(items, render, initial) {
  const panel = el("div", { role: "tabpanel" });
  const buttons = new Map();
  const list = el("div", { role: "tablist", class: "tabs" });
  function select(id, focus) {
    for (const [k, b] of buttons) { b.setAttribute("aria-selected", String(k === id)); b.tabIndex = k === id ? 0 : -1; }
    panel.replaceChildren(render(id));
    if (focus) buttons.get(id).focus();
  }
  items.forEach(([id, label], i) => {
    const b = el("button", { type: "button", role: "tab", class: "tab", text: label, onclick: () => select(id, false),
      onkeydown: (e) => {
        const step = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
        if (step) { e.preventDefault(); select(items[(i + step + items.length) % items.length][0], true); }
      } });
    buttons.set(id, b); list.append(b);
  });
  select(initial || items[0][0], false);
  return el("div", null, list, panel);
}

/* The results explorer: pick what to look at and see every model with its 95% interval and the chance level. */
function explorer(data) {
  const st = { set: data.sets[0].id, kind: "all", mode: "greedy" };
  const chartHost = el("div", { class: "chart" });
  const reading = el("p", { class: "reading", "aria-live": "polite" });
  const setBar = el("div", { class: "seg", role: "group", "aria-label": "Which wording" });
  const modeBar = el("div", { class: "seg", role: "group", "aria-label": "Decoding" });
  const kindSel = el("select", { id: "kind", "aria-label": "Kind of request", onchange: (e) => { st.kind = e.target.value; draw(); } },
    el("option", { value: "all", text: "All kinds of request" }), data.kinds.map((k) => el("option", { value: k.id, text: k.label })));
  const pressed = (bar, value) => [...bar.children].forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.v === value)));
  data.sets.forEach((s2) => setBar.append(el("button", { type: "button", "data-v": s2.id, text: s2.id === "final" ? "New wording" : "Trained wording", title: s2.label,
    onclick: () => { st.set = s2.id; draw(); } })));
  [["greedy", "Greedy"], ["sampled", "Sampled"]].forEach(([v, t]) => modeBar.append(el("button", { type: "button", "data-v": v, text: t,
    onclick: () => { st.mode = v; draw(); } })));

  function draw() {
    pressed(setBar, st.set); pressed(modeBar, st.mode);
    const set = data.sets.find((x) => x.id === st.set);
    const rows = data.models.map((m) => ({ m, c: set.results[m.id][st.mode][st.kind] }));
    const W = 640, rowH = 46, L = 168, R = 60, T = 14, B = 34, H = T + rowH * rows.length + B, plotW = W - L - R;
    const x = (v) => L + (v / 100) * plotW;
    const svgEl = svg("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `Pass rate for each model. ${rows.map((r) => `${r.m.label} ${r.c.rate}%`).join("; ")}.` });
    const grid = svg("g", { class: "grid" }), axis = svg("g", { class: "axis" });
    [0, 25, 50, 75, 100].forEach((v) => { grid.append(svg("line", { x1: x(v), x2: x(v), y1: T, y2: H - B })); axis.append(svg("text", { x: x(v), y: H - 12, "text-anchor": "middle" }, v + "%")); });
    svgEl.append(grid, axis);
    const chance = rows[0].c.chance;
    svgEl.append(svg("line", { class: "chance", x1: x(chance), x2: x(chance), y1: T, y2: H - B }),
      svg("text", { class: "chance-label", x: x(chance) + 5, y: T + 10 }, `chance ${chance}%`));
    const tip = el("div", { class: "tip", hidden: true });
    rows.forEach((r, i) => {
      const cy = T + rowH * i + rowH / 2, base = r.m.id === "base";
      const g = svg("g", { class: "row" + (base ? " base" : "") });
      g.append(svg("text", { class: "row-label", x: L - 12, y: cy + 4, "text-anchor": "end" }, r.m.label),
        svg("line", { class: "ci", x1: x(r.c.lo), x2: x(r.c.hi), y1: cy, y2: cy }),
        svg("circle", { class: "dot", cx: x(r.c.rate), cy, r: 5.5 }),
        svg("text", { class: "val", x: x(r.c.hi) + 8, y: cy + 4 }, Math.round(r.c.rate) + "%"));
      const label = `${r.m.label}: ${r.c.rate}%, 95% interval ${r.c.lo} to ${r.c.hi}%, ${r.c.prompts} distinct prompts`;
      const hit = svg("rect", { class: "hit", x: 0, y: cy - rowH / 2, width: W, height: rowH, tabindex: "0", role: "img", "aria-label": label });
      const show = () => { tip.textContent = `${r.m.label} · ${r.c.rate}% (${r.c.lo}–${r.c.hi}%) · ${r.c.prompts} distinct prompts`; tip.style.left = Math.min(70, Math.max(30, (x(r.c.rate) / W) * 100)) + "%"; tip.style.top = (cy / H) * 100 + "%"; tip.hidden = false; };
      hit.addEventListener("mouseenter", show); hit.addEventListener("focus", show);
      hit.addEventListener("mouseleave", () => { tip.hidden = true; }); hit.addEventListener("blur", () => { tip.hidden = true; });
      svgEl.append(g, hit);
    });
    chartHost.replaceChildren(svgEl, tip);
    const best = rows.slice(1).reduce((a, b) => (b.c.rate > a.c.rate ? b : a));
    const base = rows[0].c, few = rows[0].c.prompts < 10;
    const overlap = best.c.lo <= base.hi;
    reading.textContent = `${best.m.label} passes ${best.c.rate}% (95% interval ${best.c.lo}–${best.c.hi}%); the untuned base passes ${base.rate}%; chance is ${chance}%. ` +
      (overlap ? "The intervals overlap, so this is not evidence of a real difference. " : "The intervals do not overlap. ") +
      (few ? `Only ${base.prompts} distinct prompts here, so treat this kind as anecdotal.` : "");
  }
  draw();
  return el("div", null,
    el("div", { class: "controls" }, el("label", null, "Wording", setBar), el("label", null, "Decoding", modeBar), el("label", { for: "kind" }, "Kind of request", kindSel)),
    chartHost, reading,
    el("p", { class: "small muted", text: "Each dot is a model's pass rate on held-out requests; the line is a 95% interval over distinct prompts. The fine-tuned models are far above chance and above the untuned model on wording they were trained on. On new wording the gain is modest and, for most kinds, not statistically established." }));
}

const PIPELINE = [
  ["Data", "TinyStories, simple stories written by a larger model (licence CDLA-Sharing-1.0). About 507,000 stories were cleaned, exact duplicates removed, and split by a hash of each story's content, so a held-out story can never move into training. A larger 299-million-token corpus is used for the longer pretraining run. Held-out stories were checked against a sample of the training stories: none shares even half of its 8-word sequences."],
  ["Tokenizer", "A byte-level BPE tokenizer with 4,096 tokens, trained on the training split only. It averages 3.96 characters per token on this text, can represent any text without an unknown token, and cannot be tricked into producing control tokens from user input."],
  ["Model", "A Llama-style decoder-only transformer: 8 layers, width 384, 6 attention heads sharing 2 key/value heads, rotary position embeddings, RMS normalisation, SwiGLU feed-forward layers and tied embeddings. 14,162,304 parameters, context 512. The parameter count was recomputed by hand and matches."],
  ["Training", "AdamW with a cosine schedule on a fanless 8 GB MacBook Air: 90 million tokens in 4.1 hours. Resuming from a checkpoint reproduces an uninterrupted run bit for bit. The loss under fine-tuning is weighted by the tokens actually scored, and health diagnostics watch gradients and loss without ever changing training."],
  ["Evaluation", "Every score is shown next to chance and a confidence interval, models are compared on identical prompts, and the final test set was committed before any model was scored on it. Seven flaws in the project's own pipeline were found, fixed and tested along the way, and negative results are reported as such."],
  ["Serving", "An OpenAI-compatible API (checked with the official client), accounts with self-service API keys, per-key limits, request timeouts, and exact batched decoding so that several users share each model step. Usage records keep counts only."],
];

function pipeline() {
  const panels = [];
  const wrap = el("ol", { class: "steps" });
  PIPELINE.forEach(([title, body], i) => {
    const panel = el("p", { class: "step-body", hidden: true, id: "step-" + i, text: body });
    const btn = el("button", { type: "button", "aria-expanded": "false", "aria-controls": "step-" + i, onclick: () => {
      const open = btn.getAttribute("aria-expanded") === "true";
      panels.forEach((p2) => { p2.btn.setAttribute("aria-expanded", "false"); p2.panel.hidden = true; });
      if (!open) { btn.setAttribute("aria-expanded", "true"); panel.hidden = false; }
    } }, el("span", { class: "step-n", text: String(i + 1).padStart(2, "0") }), title);
    panels.push({ btn, panel });
    wrap.append(el("li", null, btn, panel));
  });
  return wrap;
}

async function viewHome() {
  const c = state.config, mi = c.model_info || {};
  const running = mi.parameters ? `Running now: ${(mi.parameters / 1e6).toFixed(1)}M parameters · context ${mi.context_length} · ${mi.instruction_tuned ? "chat-tuned" : "text continuation"}` : "";
  const signedIn = c.auth !== "account" || state.me;
  const results = el("div", { class: "muted", text: "loading results…" });
  mount(
    el("section", { class: "hero" },
      el("h1", { class: "display", text: "A small language model, built from scratch." }),
      el("p", { class: "lede", text: "aarav writes short children's stories and follows simple requests. It was trained on a laptop, evaluated with unusual care, and is served here with accounts and an OpenAI-compatible API." }),
      el("p", { class: "inline cta" }, el("a", { href: signedIn ? "#/chat" : "#/signin", text: signedIn ? "Try it →" : "Sign in to try it →" }), el("a", { href: "#/docs", class: "quiet-link", text: "Use the API" })),
      running ? el("p", { class: "small muted status", text: running }) : null),
    el("h2", { text: "What it does" }),
    el("ul", { class: "plain" },
      el("li", null, el("em", { text: "“Tell me a story about a dragon named Zara.”" }), " A short story, usually with the name you gave it."),
      el("li", null, el("em", { text: "“A long story about a robot who feels sad.”" }), " It tries to honour the topic, the feeling and the length, and often does."),
      el("li", null, "It is a small model. It has no world knowledge and cannot answer questions, write code or do maths.")),
    el("h2", { text: "How good is it" }),
    results,
    el("h2", { text: "How it was built" }),
    pipeline(),
    el("h2", { text: "Use it from code" }),
    el("p", { class: "muted", text: "Any OpenAI-compatible client works. Point it at this address and use one of your keys." }),
    tabs(apiSnippets().map(([id]) => [id, id]), (id) => {
      const code = apiSnippets().find(([n]) => n === id)[1];
      return el("div", null, el("pre", null, el("code", { text: code })), el("p", { class: "small" }, el("button", { type: "button", class: "quiet", text: "copy", onclick: (e) => copyText(code, e.target) })));
    }),
    el("h2", { text: "What is stored" }),
    el("ul", { class: "plain" },
      el("li", null, el("strong", { text: "Stored: " }), "your email, a salted hash of your password, hashes of your API keys, and for each request its time, token counts and outcome."),
      el("li", null, el("strong", { text: "Not stored: " }), "your prompts and the model's replies. Chat history lives in your browser tab only.")),
    el("h2", { text: "Honest limits" }),
    el("ul", { class: "plain" },
      el("li", { text: "It is a 14-million-parameter story model, not an assistant." }),
      el("li", { text: "On request wording it never saw in training it is only modestly better than the untuned model, and length control is weak." }),
      el("li", { text: "One generation runs at a time unless batching is switched on; a busy server answers 503 and tells you when to retry." })),
    el("footer", { class: "foot small muted" }, "Built by Aarav Khanal. Every number on this page comes from the project's own evaluation files."));
  try { results.replaceWith(explorer(await loadResults())); } catch (e) { results.textContent = "The results could not be loaded."; }
}

/* ── start ─────────────────────────────────────────────────────────────── */

async function init() {
  try { state.config = await json("/app/config"); } catch (e) { /* keep the defaults: no accounts */ }
  if (state.config.auth === "account") {
    try { const s = await json("/account/session"); state.me = s.signed_in ? { email: s.email, csrf_token: s.csrf_token } : null; } catch (e) { state.me = null; }
  }
  window.addEventListener("hashchange", route);
  await route();
}

init();
