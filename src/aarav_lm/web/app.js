"use strict";
// aarav-lm web chat. Talks to the same-origin API. Model output is only ever inserted with
// textContent, never as HTML, so generated text cannot inject markup or script.

const $ = (id) => document.getElementById(id);
const els = {
  messages: $("messages"), input: $("input"), send: $("send"), stop: $("stop"), form: $("composer"),
  banner: $("banner"), info: $("model-info"), settings: $("settings"), toggle: $("settings-toggle"),
  clear: $("clear"), key: $("api-key"), system: $("system"),
};
const numFields = ["temperature", "top_p", "top_k", "max_tokens", "repetition_penalty", "seed"];

const state = { history: [], busy: false, controller: null, instructionTuned: false, maxTokensCap: 512, baseNote: "" };

function banner(text, kind) {
  els.banner.hidden = !text;
  els.banner.textContent = text || "";
  els.banner.className = "banner" + (kind ? " " + kind : "");
}

// Clear transient errors but keep the standing "base model" notice visible.
function resetBanner() { banner(state.baseNote, state.baseNote ? "warn" : ""); }

function authHeaders() {
  const h = { "Content-Type": "application/json" };
  const key = els.key.value.trim();
  if (key) h["Authorization"] = "Bearer " + key;
  return h;
}

function addMessage(role, text, extraClass) {
  const div = document.createElement("div");
  div.className = "msg " + role + (extraClass ? " " + extraClass : "");
  div.textContent = text;
  els.messages.appendChild(div);
  els.messages.scrollTop = els.messages.scrollHeight;
  return div;
}

function readSettings() {
  const num = (id, int) => {
    const raw = $(id).value.trim();
    if (raw === "") return undefined;
    const v = int ? parseInt(raw, 10) : parseFloat(raw);
    return Number.isFinite(v) ? v : undefined;
  };
  const body = {
    temperature: num("temperature"), top_p: num("top_p"), top_k: num("top_k", true),
    max_tokens: Math.min(num("max_tokens", true) || 150, state.maxTokensCap),
    repetition_penalty: num("repetition_penalty"), seed: num("seed", true),
  };
  for (const k of Object.keys(body)) if (body[k] === undefined) delete body[k];
  return body;
}

function describeError(status, detail, retryAfter) {
  if (status === 401) return "The server needs a valid API key. Open Settings and paste one (scripts/create_key.py makes keys).";
  if (status === 429) return "Rate limit reached. Try again in " + (retryAfter || "a few") + " seconds.";
  if (status === 503) return "The model is busy with another request. Try again in a moment.";
  if (status === 413) return "That message is too large.";
  if (status === 422) return "The request was rejected: " + (typeof detail === "string" ? detail : JSON.stringify(detail));
  return "Server error (" + status + ")" + (detail ? ": " + detail : "") + ".";
}

async function loadModelInfo() {
  try {
    const r = await fetch("/model", { headers: authHeaders() });
    if (r.status === 401) { els.info.textContent = "API key required"; banner(describeError(401), "warn"); return; }
    if (!r.ok) throw new Error("status " + r.status);
    const m = await r.json();
    state.instructionTuned = !!m.instruction_tuned;
    state.maxTokensCap = m.max_tokens_cap || 512;
    $("max_tokens").max = state.maxTokensCap;
    els.info.textContent = `${m.name} · ${(m.params / 1e6).toFixed(1)}M params · ctx ${m.context_length} · ${m.device}` +
      (m.instruction_tuned ? "" : " · base model");
    state.baseNote = m.instruction_tuned ? "" :
      "Base model (no instruction tuning): each message is treated as the start of a text to continue, e.g. \"Once upon a time, there was a little dog named Max.\"";
    resetBanner();
  } catch (e) {
    els.info.textContent = "model unavailable";
    banner("Could not reach the server: " + e.message, "error");
  }
}

function setBusy(busy) {
  state.busy = busy;
  els.send.hidden = busy; els.stop.hidden = !busy; els.input.disabled = busy;
  if (!busy) els.input.focus();
}

async function send(text) {
  resetBanner();
  state.history.push({ role: "user", content: text });
  addMessage("user", text);
  const bubble = addMessage("assistant", "", "dots");
  setBusy(true);
  state.controller = new AbortController();
  let reply = "", finish = "", usage = null, failed = false;

  const messages = [];
  const sys = els.system.value.trim();
  if (sys && state.instructionTuned) messages.push({ role: "system", content: sys });
  messages.push(...state.history);

  try {
    const r = await fetch("/chat", {
      method: "POST", headers: authHeaders(), signal: state.controller.signal,
      body: JSON.stringify({ messages, stream: true, ...readSettings() }),
    });
    if (!r.ok) {
      let detail = ""; try { detail = (await r.json()).detail; } catch (_) { /* not JSON */ }
      throw new Error(describeError(r.status, detail, r.headers.get("Retry-After")));
    }
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const line = buffer.slice(0, idx).trim(); buffer = buffer.slice(idx + 2);
        if (!line.startsWith("data:")) continue;
        const evt = JSON.parse(line.slice(5));
        if (evt.error) throw new Error(describeError(evt.status || 500, evt.error));
        if (evt.delta) {
          reply += evt.delta;
          bubble.classList.remove("dots");
          bubble.textContent = reply;
          els.messages.scrollTop = els.messages.scrollHeight;
        }
        if (evt.done) { finish = evt.finish_reason; usage = evt.usage; }
      }
    }
  } catch (e) {
    failed = true;
    if (e.name === "AbortError") banner("Stopped.", "warn");
    else banner(e.message || String(e), "error");
  } finally {
    bubble.classList.remove("dots");
    if (!reply) bubble.textContent = failed ? "(no reply)" : "(empty reply)";
    if (usage) {
      const meta = document.createElement("span");
      meta.className = "meta";
      meta.textContent = `${usage.completion_tokens} tokens · ${finish}`;
      bubble.appendChild(meta);
    }
    if (reply) state.history.push({ role: "assistant", content: reply.trim() });
    else state.history.pop();            // drop the unanswered user message so the next request is valid
    state.controller = null;
    setBusy(false);
  }
}

els.form.addEventListener("submit", (ev) => {
  ev.preventDefault();
  const text = els.input.value.trim();
  if (!text || state.busy) return;
  els.input.value = "";
  send(text);
});
els.input.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); els.form.requestSubmit(); }
});
els.stop.addEventListener("click", () => state.controller && state.controller.abort());
els.clear.addEventListener("click", () => {
  if (state.controller) state.controller.abort();
  state.history = []; els.messages.replaceChildren(); resetBanner();
});
els.toggle.addEventListener("click", () => {
  els.settings.hidden = !els.settings.hidden;
  els.toggle.setAttribute("aria-expanded", String(!els.settings.hidden));
});
for (const id of ["temperature", "top_p"]) {
  const out = $(id + "-out"), input = $(id);
  const sync = () => { out.textContent = input.value; };
  input.addEventListener("input", sync); sync();
}
try { els.key.value = sessionStorage.getItem("aarav_key") || ""; } catch (_) { /* storage blocked */ }
document.getElementById("key-form").addEventListener("submit", (ev) => ev.preventDefault());
els.key.addEventListener("change", () => {
  try { sessionStorage.setItem("aarav_key", els.key.value.trim()); } catch (_) { /* ignore */ }
  loadModelInfo();
});
loadModelInfo();
