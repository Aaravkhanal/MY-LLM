/* The signed-in dashboard: overview, usage, API keys, settings. Data comes from the account endpoints (/account/usage, /account/keys, ...);
   usage rows hold counts only, and a new key's secret is shown once, at creation, and never again. */
import { el, icon, fill } from "../dom.js";
import { Button, Badge, CopyButton, DataTable, EmptyState, ErrorState, Input, LoadingRegion, MetricCard, Select, Sidebar, StatusIndicator, confirmDialog, toast } from "../ui.js";
import { BarChart, ChartWithTable } from "../charts.js";
import { state } from "../state.js";
import { json, publicJson } from "../api.js";
import { fmt, ms, when } from "../format.js";

const NAV = [{ title: "Dashboard", links: [{ href: "/dashboard", label: "Overview" }, { href: "/dashboard/usage", label: "Usage" }, { href: "/dashboard/api-keys", label: "API keys" }, { href: "/dashboard/settings", label: "Settings" }] }];

export default async function page({ path }) {
  const body = el("div", { class: "dash-body" });
  const view = { "/dashboard": overview, "/dashboard/usage": usage, "/dashboard/api-keys": keys, "/dashboard/settings": settings }[path] || overview;
  if (state.config.auth !== "account") {
    body.append(EmptyState({ title: "Accounts are not enabled on this server", body: "The dashboard needs accounts. This server uses an API key or no sign-in at all.", action: Button({ label: "Open the playground", href: "/playground", variant: "primary" }) }));
  } else { view(body); }
  const side = el("div", { class: "dash-side" }, el("p", { class: "xs muted", text: state.me ? state.me.email : "" }), Sidebar(NAV, "Dashboard sections"));
  return { title: "Dashboard", node: el("div", { class: "container page dash" }, side, body) };
}

/* Load something, showing a skeleton while waiting and a retryable error if it fails. */
async function load(host, what, fetcher, draw) {
  host.replaceChildren(LoadingRegion(`Loading ${what}`));
  try { draw(await fetcher()); }
  catch (e) { host.replaceChildren(ErrorState({ title: `Your ${what} could not be loaded`, message: e.message, retry: () => load(host, what, fetcher, draw) })); }
}

/* ── overview ──────────────────────────────────────────────────────────── */
function overview(body) {
  load(body, "overview", () => Promise.all([json("/account/usage?days=14"), publicJson("/status.json").catch(() => null)]), ([u, s]) => {
    const t = u.totals, rate = t.requests ? (100 * t.errors) / t.requests : 0;
    const chart = BarChart({ days: u.daily, metric: "tokens" });
    fill(body, el("h1", { text: "Overview" }), el("p", { class: "muted", text: "The last 14 days for your account." }),
      el("div", { class: "metrics" },
        MetricCard({ label: "Requests", value: fmt(t.requests), note: "in the last 14 days" }),
        MetricCard({ label: "Tokens", value: fmt(t.total_tokens), note: `${fmt(t.prompt_tokens)} read, ${fmt(t.completion_tokens)} generated` }),
        MetricCard({ label: "Average latency", value: t.requests ? ms(t.mean_duration_ms) : "–", note: t.latency_p95_ms ? `median ${ms(t.latency_p50_ms)}, 95th percentile ${ms(t.latency_p95_ms)}` : "of successful requests" }),
        MetricCard({ label: "Error rate", value: t.requests ? rate.toFixed(1) + "%" : "–", note: t.requests ? `${fmt(t.errors)} of ${fmt(t.requests)} requests` : "no requests yet" }),
        el("div", { class: "metric" }, el("div", { class: "label", text: "Model status" }), el("div", { class: "value small-value" }, s ? StatusIndicator(s.overall) : StatusIndicator("unknown", "Cannot reach the server")),
          el("div", { class: "note" }, el("a", { href: "/status", text: "System status" })))),
      t.requests === 0
        ? EmptyState({ title: "No requests yet", body: "Send a message in the playground, or call the API with a key, and your usage appears here.", action: el("div", { class: "row" }, Button({ label: "Open the playground", href: "/playground", variant: "primary" }), Button({ label: "Read the quickstart", href: "/docs/quickstart" })) })
        : [el("h2", { class: "sub-h", text: "Tokens per day" }), chart,
          el("p", { class: "small" }, el("a", { href: "/dashboard/usage", text: "See usage in detail" }), icon("arrow", 14))]);
  });
}

/* ── usage ─────────────────────────────────────────────────────────────── */
function usage(body) {
  let days = 14, metric = "tokens";
  const draw = (u) => {
    const t = u.totals;
    const table = DataTable({ caption: `Requests and tokens per day (UTC), last ${days} days`, columns: [{ key: "date", label: "Day", rowHeader: true }, { key: "requests", label: "Requests", num: true, render: (d) => fmt(d.requests) }, { key: "tokens", label: "Tokens", num: true, render: (d) => fmt(d.tokens) }], rows: u.daily.slice().reverse() });
    const sel = Select({ label: "Period", id: "usage-days", value: String(days), options: [7, 14, 30, 90].map((d) => ({ value: String(d), label: `Last ${d} days` })), onChange: (v) => { days = Number(v); fetchAndDraw(); } });
    const chart = t.requests === 0 ? null : el("div", null, BarChart({ days: u.daily, metric, unit: metric }));
    fill(body, el("h1", { text: "Usage" }), el("p", { class: "muted", text: "Counted in tokens. Prompts and replies are never stored, only these counts." }),
      el("div", { class: "row between" }, sel), el("div", { class: "metrics" },
        MetricCard({ label: "Requests", value: fmt(t.requests) }), MetricCard({ label: "Tokens", value: fmt(t.total_tokens) }),
        MetricCard({ label: "Median latency", value: t.latency_p50_ms === null ? "–" : ms(t.latency_p50_ms), note: "successful requests" }),
        MetricCard({ label: "Errors", value: fmt(t.errors), note: t.requests ? (100 * t.errors / t.requests).toFixed(1) + "% of requests" : "" })),
      t.requests === 0 ? EmptyState({ title: "Nothing in this period", body: "Send a message in the playground or use a key, and it shows up here." })
        : [el("h2", { class: "sub-h", text: metric === "tokens" ? "Tokens per day" : "Requests per day" }), ChartWithTable({ chart, table, label: "Show as" })],
      el("p", { class: "small muted", text: `Browser chat is limited to ${fmt(u.limits.requests_per_minute)} requests and ${fmt(u.limits.tokens_per_minute)} tokens per minute.` }));
  };
  const fetchAndDraw = () => load(body, "usage", () => json(`/account/usage?days=${days}`), draw);
  fetchAndDraw();
}

/* ── API keys ──────────────────────────────────────────────────────────── */
function keys(body) {
  const lim = state.config.limits || {};
  const reveal = el("div", { "aria-live": "polite" });
  const list = el("div");
  const err = el("p", { class: "alert", role: "alert", hidden: true });
  const label = Input({ label: "Name", id: "key-label", maxlength: "40", placeholder: "what it is for, e.g. my laptop", autocomplete: "off" });
  const create = Button({ label: "Create key", type: "submit", variant: "primary", icon: "key" });

  function showSecret(secret, note) {
    reveal.replaceChildren(el("div", { class: "secret" }, el("p", { class: "small", text: note }), el("p", { class: "mono secret-key", text: secret }),
      el("div", { class: "row" }, CopyButton(() => secret, { label: "Copy key", size: "sm" }), Button({ label: "I have copied it", variant: "quiet", size: "sm", onclick: () => reveal.replaceChildren() })),
      el("p", { class: "xs muted", text: "It is shown once and cannot be shown again. If you lose it, make a new key." })));
    reveal.querySelector(".secret").scrollIntoView({ block: "nearest" });
  }
  async function refresh() {
    try {
      const all = await json("/account/keys");
      if (!all.length) { list.replaceChildren(EmptyState({ title: "No keys yet", body: "Create one to use the API from your own code." })); return; }
      list.replaceChildren(DataTable({ caption: "Your API keys. The secret is never shown again after creation.", columns: [
        { key: "prefix", label: "Key", rowHeader: true, render: (k) => el("span", { class: "mono", text: k.prefix + "…" }) },
        { key: "label", label: "Name", render: (k) => k.label || "(no name)" }, { key: "created", label: "Created", nowrap: true, render: (k) => when(k.created_at) }, { key: "used", label: "Last used", nowrap: true, render: (k) => when(k.last_used_at) },
        { key: "s", label: "Status", render: (k) => (k.active ? Badge("active", "ok") : Badge(k.revoked_at ? "revoked" : "expired", "danger")) },
        { key: "a", label: "", nowrap: true, render: (k) => (k.active ? el("div", { class: "row" }, Button({ label: "Rotate", size: "sm", variant: "quiet", icon: "refresh", onclick: () => act(k, "rotate") }), Button({ label: "Revoke", size: "sm", variant: "danger", icon: "trash", onclick: () => act(k, "revoke") })) : "") },
      ], rows: all.slice().reverse().map((k) => ({ ...k, _muted: !k.active })) }));
    } catch (e) { list.replaceChildren(ErrorState({ title: "Your keys could not be loaded", message: e.message, retry: refresh })); }
  }
  async function act(k, what) {
    err.hidden = true;
    const ok = await confirmDialog(what === "revoke"
      ? { title: "Revoke this key?", body: `${k.prefix}… (${k.label || "no name"}) stops working immediately, for everything that uses it.`, confirmLabel: "Revoke", danger: true }
      : { title: "Rotate this key?", body: "A replacement key is created and the old one stops working immediately. Update whatever uses it.", confirmLabel: "Rotate" });
    if (!ok) return;
    try {
      const r = await json(`/account/keys/${k.id}/${what}`, { method: "POST", body: {} });
      if (what === "rotate") showSecret(r.key, "Here is the replacement key. The old one no longer works.");
      toast(what === "rotate" ? "Key rotated." : "Key revoked.");
      await refresh();
    } catch (e) { err.textContent = e.message; err.hidden = false; }
  }
  const form = el("form", { class: "key-form", onsubmit: async (e) => {
    e.preventDefault(); err.hidden = true; create.disabled = true;
    try { const r = await json("/account/keys", { method: "POST", body: { label: label.input.value } }); label.input.value = ""; showSecret(r.key, "Your new key."); toast("Key created."); await refresh(); }
    catch (ex) { err.textContent = ex.message; err.hidden = false; } finally { create.disabled = false; }
  } }, label, create);

  fill(body, el("h1", { text: "API keys" }),
    el("p", { class: "muted prose", text: "Use a key with any OpenAI-compatible client. Keys are stored only as hashes, so a lost key cannot be recovered; make a new one." }),
    form, err, reveal, el("h2", { class: "sub-h", text: "Your keys" }), list,
    lim.key_requests_per_minute ? el("p", { class: "small muted", text: `Each key: ${lim.key_requests_per_minute} requests and ${fmt(lim.key_tokens_per_minute)} tokens per minute; up to ${lim.max_keys} active keys.` }) : null);
  list.append(LoadingRegion("Loading keys"));
  refresh();
}

/* ── settings ──────────────────────────────────────────────────────────── */
function settings(body) {
  const msg = el("p", { class: "notice", role: "status", hidden: true });
  const cur = Input({ label: "Current password", type: "password", autocomplete: "current-password", required: true, maxlength: "128", id: "pw-cur" });
  const nxt = Input({ label: "New password", type: "password", autocomplete: "new-password", required: true, minlength: "10", maxlength: "128", id: "pw-new", hint: "At least 10 characters." });
  const submit = Button({ label: "Change password", type: "submit" });
  const form = el("form", { class: "auth-form", onsubmit: async (e) => {
    e.preventDefault(); msg.hidden = true; submit.disabled = true;
    try { await json("/account/password", { method: "POST", body: { current: cur.input.value, new: nxt.input.value } }); cur.input.value = ""; nxt.input.value = "";
      msg.className = "notice"; msg.textContent = "Password changed. Your other devices were signed out."; toast("Password changed."); }
    catch (ex) { msg.className = "alert"; msg.textContent = ex.message; }
    msg.hidden = false; submit.disabled = false;
  } }, cur, nxt, submit, msg);
  fill(body, el("h1", { text: "Settings" }),
    el("h2", { class: "sub-h", text: "Account" }), el("p", null, "Signed in as ", el("strong", { text: state.me ? state.me.email : "" }), "."),
    el("h2", { class: "sub-h", text: "Password" }), form,
    el("h2", { class: "sub-h", text: "What is stored about you" }),
    el("p", { class: "muted prose", text: "Your email, a salted hash of your password, hashes of your API keys, and for each request its time, token counts, duration and outcome. Prompts and replies are not stored; chat history lives in your browser tab only. Account deletion is not built yet; ask the owner of this site." }));
}
