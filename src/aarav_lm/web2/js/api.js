/* Talking to the server: fetch with the session cookie and CSRF token (or a legacy API key), friendly errors, real SSE streaming. */
import { state } from "./state.js";

export class ApiError extends Error {
  constructor(status, data, retryAfter) {
    super(describeError(status, data, retryAfter));
    this.status = status;
    this.retryAfter = retryAfter ? Number(retryAfter) : null;
  }
}

function serverMessage(data) {
  if (!data) return "";
  if (typeof data.detail === "string") return data.detail;
  if (data.error && typeof data.error === "object") return data.error.message || "";
  if (typeof data.error === "string") return data.error;
  return "";
}

/* Plain-language messages. Server text is shown only for statuses where it is written for people; a raw stack trace never reaches here
   because the server never sends one, and 5xx bodies are replaced by a generic line. */
export function describeError(status, data, retryAfter) {
  const msg = serverMessage(data);
  if (status === 401) return state.config.auth === "account" ? "Please sign in." : "That key was not accepted. Check it and try again.";
  if (status === 403) return msg || "That request was not allowed.";
  if (status === 429) return (msg || "Slow down a little") + (retryAfter ? ` (try again in ${retryAfter}s).` : ".");
  if (status === 503) return "The model is busy with someone else's request. Try again in a few seconds.";
  if (status === 422 || status === 400) return msg || "The request was not valid.";
  if (status === 413) return "That message is too large.";
  if (status === 409 || status === 404) return msg || "That did not work.";
  if (status >= 500) return "Something went wrong on the server. Try again in a moment.";
  return msg || `The server returned an error (${status}).`;
}

function headers(hasBody) {
  const h = { Accept: "application/json" };
  if (hasBody) h["Content-Type"] = "application/json";
  if (state.me) h["X-CSRF-Token"] = state.me.csrf_token;
  else if (state.apiKey) h.Authorization = "Bearer " + state.apiKey;
  return h;
}

export function call(path, { method = "GET", body, signal } = {}) {
  return fetch(path, { method, headers: headers(body !== undefined), body: body === undefined ? undefined : JSON.stringify(body), signal, credentials: "same-origin" });
}

export async function json(path, opts) {
  let res;
  try { res = await call(path, opts); } catch (e) {
    if (e.name === "AbortError") throw e;
    throw new NetworkError();
  }
  let data = null;
  try { data = await res.json(); } catch (e) { /* empty body */ }
  if (!res.ok) throw new ApiError(res.status, data, res.headers.get("retry-after"));
  return data;
}

export class NetworkError extends Error {
  constructor() { super("Could not reach the server. Check your connection and try again."); this.name = "NetworkError"; }
}

/* Public, no credentials needed (the status page, results, config). */
export async function publicJson(path) {
  let res;
  try { res = await fetch(path, { headers: { Accept: "application/json" }, credentials: "same-origin", cache: "no-store" }); } catch (e) { throw new NetworkError(); }
  if (!res.ok) throw new ApiError(res.status, null, null);
  return res.json();
}

/* Real streaming: each frame is a token the model actually produced; nothing is faked or delayed here. Resolves to the full text.
   `path` is /v1/chat/completions for signed-in use and /demo/chat for the public demo. */
export async function streamChat(path, payload, onDelta, signal) {
  let res;
  try { res = await call(path, { method: "POST", body: payload, signal }); } catch (e) {
    if (e.name === "AbortError") throw e;
    throw new NetworkError();
  }
  if (!res.ok) {
    let d = null;
    try { d = await res.json(); } catch (e) { /* no body */ }
    throw new ApiError(res.status, d, res.headers.get("retry-after"));
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "", out = "", finish = null, usage = null;
  for (;;) {
    let chunk;
    try { chunk = await reader.read(); } catch (e) { if (e.name === "AbortError") throw e; throw new NetworkError(); }
    if (chunk.done) break;
    buf += dec.decode(chunk.value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, i).trim();
      buf = buf.slice(i + 2);
      if (!frame.startsWith("data:")) continue;
      const payloadText = frame.slice(5).trim();
      if (payloadText === "[DONE]") return { text: out, finish, usage };
      let obj;
      try { obj = JSON.parse(payloadText); } catch (e) { continue; }
      if (obj.error) throw new ApiError(obj.error.code === "server_busy" ? 503 : 500, obj);
      const choice = obj.choices && obj.choices[0];
      const delta = choice && choice.delta && choice.delta.content;
      if (choice && choice.finish_reason) finish = choice.finish_reason;
      if (obj.usage) usage = obj.usage;
      if (delta) { out += delta; onDelta(delta); }
    }
  }
  return { text: out, finish, usage };
}

/* Chat request body from the playground settings. */
export function chatPayload(history, settings, modelId) {
  const msgs = [];
  if (settings.system && settings.system.trim()) msgs.push({ role: "system", content: settings.system.trim() });
  msgs.push(...history);
  const p = { model: modelId || "aarav-lm", messages: msgs, stream: true, temperature: Number(settings.temperature), top_p: Number(settings.top_p),
    top_k: Number(settings.top_k), max_tokens: Number(settings.max_tokens), repetition_penalty: Number(settings.repetition_penalty) };
  if (settings.seed !== "" && settings.seed !== null && settings.seed !== undefined) p.seed = Number(settings.seed);
  return p;
}

let modelsPromise = null;
export async function loadModels() {
  if (state.models) return state.models;
  const d = await json("/v1/models");
  state.models = d.data;
  return state.models;
}
export function forgetModels() { state.models = null; modelsPromise = null; }

let resultsPromise = null;
export function loadResults() {
  if (state.results) return Promise.resolve(state.results);
  if (!resultsPromise) resultsPromise = publicJson("/results.json").then((r) => { state.results = r; return r; }).catch((e) => { resultsPromise = null; throw e; });
  return resultsPromise;
}
