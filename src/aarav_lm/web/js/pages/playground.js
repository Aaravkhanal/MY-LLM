/* The playground: settings on the left, the conversation in the middle, the composer at the bottom. Conversation and settings live in
   the browser tab only (the settings in localStorage); the server is stateless about them. */
import { el, icon } from "../dom.js";
import { Button, CopyButton, Select, Slider, Input, Textarea, StatusIndicator, Badge, EmptyState, toast, confirmDialog } from "../ui.js";
import { state, signedIn, needsAccount, loadSettings, saveSettings, SETTINGS_DEFAULTS } from "../state.js";
import { ApiError, NetworkError, streamChat, chatPayload, loadModels, publicJson } from "../api.js";
import { modelFacts, millions } from "../data.js";
import { EXAMPLE_PROMPTS } from "../snippets.js";
import { fmt } from "../format.js";

let conversation = [];            // [{role, content, meta?}] kept while the tab is open (not stored anywhere)
let settings = loadSettings();
let lastUsage = null;

export default async function page({ navigate }) {
  const f = modelFacts();
  let models = [{ id: f.name }];
  if (signedIn() && !(state.config.auth === "key" && !state.apiKey)) { try { models = await loadModels(); } catch (e) { /* the sidebar shows the configured model instead */ } }
  let modelId = models[0].id;
  let controller = null, timer = null, busy = false;

  const announcer = el("div", { class: "sr-only", role: "status", "aria-live": "polite" });
  const log = el("div", { class: "pg-log", role: "log", "aria-label": "Conversation", "aria-relevant": "additions" });
  const gate = el("div", { class: "pg-banner", id: "pg-gate", hidden: true });          // sign-in / key prompt (state of the visitor)
  const banner = el("div", { class: "pg-banner", id: "pg-error", hidden: true });        // the last error (state of the last request)
  const usageEl = el("span", { class: "xs muted", id: "pg-usage" });
  const statusHost = el("span", { class: "pg-status" });

  /* ── composer ── */
  const input = el("textarea", { class: "textarea pg-input", id: "pg-input", rows: "1", placeholder: "Message aarav-lm…", "aria-label": "Message", "aria-describedby": "pg-hint" });
  const send = Button({ label: "Send", variant: "primary", icon: "send", type: "submit" });
  const stop = Button({ label: "Stop", icon: "stop" }); stop.hidden = true;
  const form = el("form", { class: "pg-composer", autocomplete: "off", onsubmit: (e) => { e.preventDefault(); submit(); } },
    el("div", { class: "pg-composer-row" }, input, el("div", { class: "row" }, stop, send)),
    el("p", { id: "pg-hint", class: "xs muted", text: "Enter to send · Shift+Enter for a new line" }));
  const autosize = () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 168) + "px"; };
  input.addEventListener("input", autosize);
  input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); form.requestSubmit(); } });
  stop.addEventListener("click", () => controller && controller.abort());

  const locked = () => (needsAccount() && !state.me) || (state.config.auth === "key" && !state.apiKey);
  function paintGate() {
    gate.replaceChildren(); gate.hidden = true;
    if (needsAccount() && !state.me) {
      gate.hidden = false;
      gate.append(el("p", { text: "Sign in to run the model. It is free while capacity lasts." }), el("div", { class: "row" }, Button({ label: "Sign in", href: "/signin?next=/playground", variant: "primary" })));
    } else if (state.config.auth === "key" && !state.apiKey) {
      gate.hidden = false;
      const key = Input({ label: "API key (kept in this tab only)", type: "password", autocomplete: "off", spellcheck: "false", placeholder: "aak_…", id: "pg-key" });
      gate.append(el("form", { onsubmit: (e) => { e.preventDefault(); state.apiKey = key.input.value.trim(); try { sessionStorage.setItem("aarav_key", state.apiKey); } catch (x) { /* memory only */ } navigate("/playground", { replace: true }); } },
        key, Button({ label: "Use this key", type: "submit", variant: "primary" })));
    }
    const off = locked() || busy;
    input.disabled = off; send.disabled = off;
  }

  /* ── messages ── */
  function messageNode(m, isLast) {
    const body = el("p", { class: "body", text: m.content });
    const isUser = m.role === "user";
    const meta = !isUser ? el("div", { class: "msg-meta" },
      el("span", { class: "xs muted", text: metaText(m.meta) }),
      CopyButton(() => m.content, { label: "Copy" }),
      isLast ? Button({ label: "Regenerate", variant: "quiet", size: "sm", icon: "refresh", onclick: regenerate, disabled: busy || locked() }) : null) : null;
    return el("article", { class: "msg msg-" + m.role }, el("h3", { class: "role", text: isUser ? "You" : "aarav-lm" }), body, meta);
  }
  const metaText = (meta) => !meta ? "" : [meta.tokens ? `${meta.tokens} tokens` : "", meta.finish === "length" ? "stopped at the max-token limit" : meta.finish === "stop" ? "ended naturally" : ""].filter(Boolean).join(" · ");

  function paintLog() {
    if (!conversation.length) {
      log.replaceChildren(el("div", { class: "pg-empty" }, el("h2", { text: "Ask for a story" }),
        el("p", { class: "muted", text: "It writes short children's stories and follows a few kinds of request (a name, a topic, a place, a feeling, a length). It is a small model, not a general assistant." }),
        el("div", { class: "chips", role: "group", "aria-label": "Example prompts" }, EXAMPLE_PROMPTS.map((t) => el("button", { type: "button", class: "chip", text: t, onclick: () => { input.value = t; autosize(); input.focus(); } })))));
    } else {
      log.replaceChildren(...conversation.map((m, i) => messageNode(m, i === conversation.length - 1 && m.role === "assistant")));
    }
    log.scrollTop = log.scrollHeight; window.scrollTo({ top: document.body.scrollHeight });
    usageEl.textContent = lastUsage ? `Last request: ${fmt(lastUsage.prompt_tokens)} prompt + ${fmt(lastUsage.completion_tokens)} generated · ${fmt(lastUsage.total_tokens)} of ${f.context || "?"} tokens of context` : `Context: ${f.context || "?"} tokens`;
  }

  /* ── generating ── */
  function setBusy(b) { busy = b; send.hidden = b; stop.hidden = !b; paintGate(); clearBtn.disabled = b; log.setAttribute("aria-busy", String(b)); }

  async function generate() {
    banner.hidden = true;
    const bodyNode = el("p", { class: "body" }, el("span", { class: "dots", "aria-label": "Generating" }));
    const pending = el("article", { class: "msg msg-assistant" }, el("h3", { class: "role", text: "aarav-lm" }), bodyNode);
    log.append(pending); log.scrollTop = log.scrollHeight;
    controller = new AbortController();
    setBusy(true); announcer.textContent = "Generating a reply.";
    let got = "";
    try {
      const r = await streamChat("/v1/chat/completions", chatPayload(conversation, settings, modelId), (d) => {
        if (!got) bodyNode.replaceChildren();
        got += d; bodyNode.textContent = got; log.scrollTop = log.scrollHeight; window.scrollTo({ top: document.body.scrollHeight });
      }, controller.signal);
      if (r.usage) lastUsage = r.usage;
      if (got) conversation.push({ role: "assistant", content: got, meta: { finish: r.finish, tokens: r.usage ? r.usage.completion_tokens : null } });
      else showError("The model returned nothing.", "Try rephrasing your request.");
      announcer.textContent = "Reply complete.";
    } catch (e) {
      if (e.name === "AbortError") { if (got) conversation.push({ role: "assistant", content: got, meta: { finish: "stopped" } }); announcer.textContent = "Stopped."; }
      else {
        if (got) conversation.push({ role: "assistant", content: got, meta: { finish: "error" } });
        const msg = e instanceof ApiError || e instanceof NetworkError ? e.message : "Something went wrong while generating. Try again.";
        showError(msg, null, e);
        if (e instanceof ApiError && e.status === 401 && needsAccount()) { state.me = null; navigate("/signin?next=/playground"); }
        announcer.textContent = "The request failed.";
      }
    } finally { controller = null; setBusy(false); paintLog(); input.focus({ preventScroll: true }); }
  }

  function showError(message, hint, ex) {
    banner.hidden = false;
    banner.replaceChildren(el("p", { role: "alert", class: "pg-error" }, icon("alert", 16), el("span", { text: message + (hint ? " " + hint : "") })),
      el("div", { class: "row" }, Button({ label: "Try again", icon: "refresh", onclick: () => { banner.hidden = true; if (conversation.length && conversation[conversation.length - 1].role === "user") generate(); } }),
        Button({ label: "Dismiss", variant: "quiet", onclick: () => { banner.hidden = true; } })));
    if (ex && ex.retryAfter) toast(`Try again in ${ex.retryAfter} seconds.`, { tone: "warn" });
  }

  function submit() {
    const text = input.value.trim();
    if (!text || busy || locked()) return;
    conversation.push({ role: "user", content: text });
    input.value = ""; autosize(); paintLog(); generate();
  }
  function regenerate() {
    if (busy || locked()) return;
    while (conversation.length && conversation[conversation.length - 1].role === "assistant") conversation.pop();
    if (conversation.length) { paintLog(); generate(); }
  }
  const clearBtn = Button({ label: "Clear", variant: "quiet", size: "sm", icon: "trash", onclick: async () => {
    if (!conversation.length) return;
    if (await confirmDialog({ title: "Clear the conversation?", body: "The messages in this tab will be removed. Nothing is stored on the server.", confirmLabel: "Clear", danger: true })) { conversation = []; lastUsage = null; paintLog(); input.focus(); }
  } });

  /* ── sidebar ── */
  const set = (k) => (v) => { settings[k] = v; saveSettings(settings); };
  const controls = el("div");
  function buildControls() {
    controls.replaceChildren(
      el("h2", { class: "side-h", text: "Model" }),
      Select({ label: "Model", id: "pg-model", options: models.map((m) => ({ value: m.id, label: m.id })), value: modelId, onChange: (v) => { modelId = v; }, hint: models.length < 2 ? "This server runs one model." : undefined }),
      el("p", { class: "xs muted" }, `${millions(f.params)} parameters · context ${f.context || "?"} · ${f.tuned ? "chat tuned" : "text continuation"}. `, el("a", { href: "/models/aarav-lm", text: "Model page" })),
      el("h2", { class: "side-h", text: "Sampling" }),
      Slider({ label: "Temperature", id: "pg-temp", min: 0, max: 2, step: 0.05, value: settings.temperature, onInput: set("temperature"), format: (v) => Number(v).toFixed(2), hint: "0 is greedy and repeatable; higher is more varied." }),
      Slider({ label: "Top-p", id: "pg-topp", min: 0.05, max: 1, step: 0.05, value: settings.top_p, onInput: set("top_p"), format: (v) => Number(v).toFixed(2) }),
      Slider({ label: "Max tokens", id: "pg-max", min: 16, max: 512, step: 1, value: settings.max_tokens, onInput: set("max_tokens"), format: (v) => String(v) }),
      el("details", { class: "advanced" }, el("summary", { text: "Advanced" }),
        Textarea({ label: "System prompt", id: "pg-system", value: settings.system, rows: 3, placeholder: "You are a friendly storyteller for children.", oninput: (e) => set("system")(e.target.value) }),
        Input({ label: "Top-k", id: "pg-topk", type: "number", min: "1", max: "2048", step: "1", value: settings.top_k, oninput: (e) => set("top_k")(e.target.value) }),
        Input({ label: "Repetition penalty", id: "pg-rep", type: "number", min: "0.5", max: "2", step: "0.05", value: settings.repetition_penalty, oninput: (e) => set("repetition_penalty")(e.target.value) }),
        Input({ label: "Seed", id: "pg-seed", type: "number", min: "0", step: "1", value: settings.seed, hint: "Blank is random. The same seed repeats a reply.", oninput: (e) => set("seed")(e.target.value) })),
      Button({ label: "Reset settings", variant: "quiet", size: "sm", onclick: () => { settings = { ...SETTINGS_DEFAULTS }; saveSettings(settings); buildControls(); toast("Settings reset."); } }));
  }
  buildControls();
  const side = el("details", { class: "pg-side", open: true }, el("summary", { text: "Settings" }), controls);
  const mq = window.matchMedia("(min-width: 52rem)");
  const syncOpen = () => { side.open = mq.matches; };
  syncOpen(); mq.addEventListener("change", syncOpen);

  /* ── status ── */
  async function pollStatus() {
    try { const s = await publicJson("/status.json"); statusHost.replaceChildren(StatusIndicator(s.overall)); }
    catch (e) { statusHost.replaceChildren(StatusIndicator("unknown", "Cannot reach the server")); }
  }
  pollStatus(); timer = setInterval(pollStatus, 30000);

  const head = el("div", { class: "pg-head" }, el("div", { class: "row" }, el("h1", { class: "pg-title", text: "Playground" }), Badge(modelId), statusHost), el("div", { class: "row" }, usageEl, clearBtn));
  const main = el("section", { class: "pg-main", "aria-label": "Conversation and message" }, head, gate, banner, log, form, announcer);
  paintGate(); paintLog(); autosize();
  setTimeout(() => input.focus({ preventScroll: true }), 0);
  return { title: "Playground", node: el("div", { class: "playground" }, side, main), onLeave: () => { clearInterval(timer); mq.removeEventListener("change", syncOpen); if (controller) controller.abort(); } };
}
