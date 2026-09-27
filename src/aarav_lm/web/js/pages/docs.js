/* The documentation. One module renders every /docs page inside a shared layout (sidebar, previous and next). Examples are built from
   the address the site is served from and the real request schema (api/openai_compat.py); limits and the model name come from the
   server's config, so the docs cannot drift from what is running. */
import { el, icon } from "../dom.js";
import { Sidebar, CodeBlock, CodeTabs, DataTable, Badge, Button } from "../ui.js";
import { state, needsAccount } from "../state.js";
import { modelFacts, millions } from "../data.js";
import { chatSnippets, streamSnippets, baseUrl } from "../snippets.js";
import { fmt } from "../format.js";

const NAV = [
  { title: "Getting started", links: [["/docs", "Introduction"], ["/docs/quickstart", "Quickstart"], ["/docs/authentication", "Authentication"]] },
  { title: "API", links: [["/docs/api", "Overview"], ["/docs/api/chat-completions", "Chat completions"], ["/docs/models", "Models"], ["/docs/usage", "Usage"], ["/docs/errors", "Errors"]] },
  { title: "SDK", links: [["/docs/sdk/python", "Python"], ["/docs/sdk/javascript", "JavaScript"], ["/docs/sdk/curl", "cURL"]] },
  { title: "Model", links: [["/models/aarav-lm/architecture", "Architecture"], ["/models/aarav-lm/evaluations", "Evaluation"], ["/docs/limitations", "Limitations"]] },
];
const FLAT = NAV.flatMap((g) => g.links).filter(([h]) => h.startsWith("/docs"));

const P = (t) => el("p", null, t);
const H2 = (t, id) => el("h2", { class: "doc-h2", id }, t);
const Callout = (...kids) => el("div", { class: "callout" }, ...kids);
const Endpoint = (method, path, note) => el("p", { class: "endpoint" }, el("span", { class: "method", text: method }), el("code", { text: path }), note ? el("span", { class: "muted", text: note }) : null);
const json = (o) => JSON.stringify(o, null, 2);
const table = (caption, cols, rows) => DataTable({ caption, columns: cols.map(([key, label, num]) => ({ key, label, num, rowHeader: key === cols[0][0] })), rows });

function facts() {
  const f = modelFacts(), lim = state.config.limits || {};
  return { f, lim, name: f.name, alias: "aarav-lm", base: baseUrl(), created: 1790481168 };
}

/* ── the pages ─────────────────────────────────────────────────────────── */

const PAGES = {
  "/docs": { title: "Introduction", render: () => { const { f, base } = facts(); return [
    el("h1", { text: "Documentation" }),
    el("p", { class: "lede", text: `aarav-lm is a ${millions(f.params)}-parameter language model behind an OpenAI-compatible HTTP API. Point an OpenAI client at ${base}, use one of your API keys, and call it as you would any chat model.` }),
    H2("What you can do"),
    el("ul", { class: "plain" }, el("li", { text: "Chat completions, with server-sent-event streaming." }), el("li", { text: "Text completions (raw continuation, no chat template)." }),
      el("li", { text: "List the model and read your own usage." }), el("li", { text: "Manage keys, limits and usage from the dashboard." })),
    H2("What to expect"),
    P(`The model writes short children's stories and follows a few kinds of request (a name, a topic, a place, a feeling, a length). It has a ${f.context || 512}-token context, no world knowledge, and no tool use. It is a small research model, not a general assistant; read the `),
    el("p", null, el("a", { href: "/docs/limitations", text: "limitations" }), " before building on it."),
    Callout(el("strong", { text: "Base URL " }), el("code", { text: base })),
    el("div", { class: "row" }, Button({ label: "Quickstart", href: "/docs/quickstart", variant: "primary", iconAfter: "arrow" }), Button({ label: "Try it in the playground", href: "/playground" })),
  ]; } },

  "/docs/quickstart": { title: "Quickstart", render: () => { const { name } = facts(); return [
    el("h1", { text: "Quickstart" }),
    el("p", { class: "lede", text: "From nothing to a first reply in three steps." }),
    H2("1. Get an API key"),
    needsAccount() ? el("p", null, state.me ? [el("a", { href: "/dashboard/api-keys", text: "Create a key" }), " in the dashboard. It is shown once; copy it now."] : [el("a", { href: "/signin?next=/dashboard/api-keys", text: "Sign in" }), " (or create an account), then create a key in the dashboard. It is shown once."]) : P("This server does not use accounts; use the key its owner gave you."),
    H2("2. Keep it out of your code"),
    CodeBlock({ lang: "bash", code: `export AARAV_KEY="aak_..."` }),
    H2("3. Make a request"),
    CodeTabs(chatSnippets(), { ariaLabel: "Language" }),
    P("The reply is an ordinary OpenAI chat completion. To see tokens as they are produced, ask for a stream:"),
    CodeTabs(streamSnippets(), { ariaLabel: "Language (streaming)" }),
    Callout(el("strong", { text: "Model name. " }), `Use "aarav-lm" (an alias that always works) or the full name "${name}".`),
    el("p", null, "Next: ", el("a", { href: "/docs/api/chat-completions", text: "the chat completions reference" }), ", or ", el("a", { href: "/docs/errors", text: "how errors look" }), "."),
  ]; } },

  "/docs/authentication": { title: "Authentication", render: () => { const { lim } = facts(); return [
    el("h1", { text: "Authentication" }),
    el("p", { class: "lede", text: "API requests carry a bearer key. Browsers use a session cookie instead." }),
    CodeBlock({ lang: "bash", code: "Authorization: Bearer aak_..." }),
    H2("API keys"),
    el("ul", { class: "plain" },
      el("li", { text: "Create, name, rotate and revoke keys in the dashboard. A key is shown once, at creation; only a hash is stored, so a lost key cannot be recovered. Make a new one." }),
      el("li", { text: lim.max_keys ? `Up to ${lim.max_keys} active keys per account.` : "Keys per account are limited." }),
      el("li", { text: lim.key_requests_per_minute ? `Each key: ${lim.key_requests_per_minute} requests and ${fmt(lim.key_tokens_per_minute)} tokens per minute. A key can also send at most ten requests in the same instant.` : "Each key has a per-minute request and token limit." }),
      el("li", { text: "Rotating a key creates its replacement and stops the old key at once. Revoking stops it at once." })),
    Callout(el("strong", { text: "Never put a key in front-end code. " }), "Anyone who can read the page can read the key. Call the API from a server you control."),
    H2("Browser sessions"),
    P("The web app signs you in with an HttpOnly, SameSite=Strict cookie and sends an X-CSRF-Token header on every state-changing request. That is for this site only; scripts and servers should use a key."),
    H2("Failure"),
    P("A missing or invalid key returns 401 with a WWW-Authenticate: Bearer header and the standard error body."),
    CodeBlock({ lang: "json", code: json({ error: { message: "missing or invalid API key", type: "authentication_error", param: null, code: "invalid_api_key" } }) }),
  ]; } },

  "/docs/api": { title: "API overview", render: () => { const { base, lim } = facts(); return [
    el("h1", { text: "API overview" }),
    el("p", { class: "lede", text: `An OpenAI-compatible API at ${base}. The official OpenAI SDKs work when pointed at it.` }),
    table("Endpoints", [["m", "Method"], ["p", "Path"], ["d", "Purpose"], ["a", "Auth"]], [
      { m: "POST", p: "/v1/chat/completions", d: "Chat completion, optionally streamed", a: "key" }, { m: "POST", p: "/v1/completions", d: "Raw text continuation", a: "key" },
      { m: "GET", p: "/v1/models", d: "The served model", a: "key" }, { m: "GET", p: "/v1/models/{id}", d: "One model", a: "key" }, { m: "GET", p: "/v1/usage", d: "Your key's own usage totals", a: "key" },
      { m: "GET", p: "/health", d: "Liveness", a: "none" }, { m: "GET", p: "/status.json", d: "Per-service status", a: "none" }]),
    H2("Conventions"),
    el("ul", { class: "plain" },
      el("li", null, "JSON in, JSON out. Streaming uses server-sent events (", el("code", { text: "stream: true" }), ") and ends with ", el("code", { text: "data: [DONE]" }), "."),
      el("li", { text: "Errors use the OpenAI shape; see Errors." }),
      el("li", { text: "One reply is generated at a time unless the server batches; a busy server answers 503 with Retry-After." }),
      el("li", { text: "Every generation is cut off after a time limit and returns what it has, with finish_reason \"length\"." }),
      el("li", { text: "Prompts and replies are never stored; usage rows hold counts, duration and outcome." })),
    H2("Limits"),
    P(lim.key_requests_per_minute ? `${lim.key_requests_per_minute} requests and ${fmt(lim.key_tokens_per_minute)} tokens per minute per key; a 429 carries Retry-After.` : "Rate limits apply; a 429 or 503 carries Retry-After."),
  ]; } },

  "/docs/api/chat-completions": { title: "Chat completions", render: () => { const { name, created } = facts(); return [
    el("h1", { text: "Chat completions" }),
    Endpoint("POST", "/v1/chat/completions"),
    el("p", { class: "lede", text: "Generate a reply to a conversation. Compatible with the OpenAI request and response shapes." }),
    H2("Request body"),
    table("Parameters", [["name", "Parameter"], ["type", "Type"], ["desc", "Description"]], [
      { name: "messages", type: "array, required", desc: "1 to 64 messages with role system, developer, user or assistant. Content is a string or text parts. Each message is at most 20,000 characters." },
      { name: "model", type: "string", desc: `"aarav-lm" or "${name}". Anything else returns 404 model_not_found.` },
      { name: "max_tokens", type: "integer ≥ 1", desc: "Most tokens to generate (alias max_completion_tokens). The server caps it (512 by default). Default 128." },
      { name: "temperature", type: "0 to 2", desc: "0 is greedy and repeatable; higher is more varied." },
      { name: "top_p", type: "(0, 1]", desc: "Nucleus sampling." },
      { name: "top_k", type: "1 to 100000", desc: "Extension: sample from the k most likely tokens." },
      { name: "repetition_penalty", type: "0.5 to 2", desc: "Extension: values above 1 discourage repeating tokens." },
      { name: "stop", type: "string or up to 4 strings", desc: "Generation stops before any of these appears." },
      { name: "seed", type: "integer ≥ 0", desc: "With the same seed and settings the reply repeats." },
      { name: "stream", type: "boolean", desc: "Send server-sent events as tokens are produced." },
      { name: "n", type: "must be 1", desc: "Only one choice per request." }]),
    P("Fields the server does not know (tools, user, logprobs and so on) are ignored, not rejected, so existing client code keeps working."),
    H2("Response"),
    CodeBlock({ lang: "json", title: "200 OK", code: json({ id: "chatcmpl-…", object: "chat.completion", created, model: name,
      choices: [{ index: 0, message: { role: "assistant", content: "Once upon a time, a little fox named Kip…" }, finish_reason: "stop" }], usage: { prompt_tokens: 15, completion_tokens: 112, total_tokens: 127 } }) }),
    table("finish_reason", [["v", "Value"], ["m", "Meaning"]], [{ v: "stop", m: "The model finished the story or a stop sequence appeared." }, { v: "length", m: "max_tokens or the time limit was reached; the reply is cut off." }]),
    H2("Streaming"),
    P("With stream: true the reply arrives as chunks. The first carries the role, then one per piece of text, then a closing chunk with the finish_reason and the usage (an extension over the standard), then [DONE]."),
    CodeBlock({ lang: "bash", title: "text/event-stream", code: [
      `data: {"object":"chat.completion.chunk","model":"${name}","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}`, "",
      `data: {"object":"chat.completion.chunk","model":"${name}","choices":[{"index":0,"delta":{"content":"Once"},"finish_reason":null}]}`, "",
      `data: {"object":"chat.completion.chunk","model":"${name}","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":15,"completion_tokens":112,"total_tokens":127}}`, "", "data: [DONE]"].join("\n") }),
    P("If generation fails after the stream has started, the last frame is an error object followed by [DONE]; the HTTP status is already 200 by then, so streaming clients must check for it."),
    H2("Example"),
    CodeTabs(chatSnippets(), { ariaLabel: "Language" }),
  ]; } },

  "/docs/models": { title: "Models", render: () => { const { f, name, created } = facts(); return [
    el("h1", { text: "Models" }),
    Endpoint("GET", "/v1/models"), Endpoint("GET", "/v1/models/{id}"),
    el("p", { class: "lede", text: "The model this server runs. There is one; its checkpoint is chosen by the server's owner." }),
    CodeBlock({ lang: "json", title: "200 OK", code: json({ object: "list", data: [{ id: name, object: "model", created, owned_by: "aarav-lm", context_length: f.context, parameters: f.params, instruction_tuned: f.tuned }] }) }),
    table("Fields", [["k", "Field"], ["d", "Meaning"]], [{ k: "id", d: "The model's name. The alias aarav-lm is accepted everywhere a model is named." }, { k: "context_length", d: "Tokens of prompt plus reply the model can hold." },
      { k: "parameters", d: "Number of trainable parameters." }, { k: "instruction_tuned", d: "True if it was fine-tuned for chat; false means it only continues text." }]),
    CodeBlock({ lang: "bash", code: `curl ${baseUrl()}/models -H "Authorization: Bearer $AARAV_KEY"` }),
  ]; } },

  "/docs/usage": { title: "Usage", render: () => [
    el("h1", { text: "Usage" }), Endpoint("GET", "/v1/usage?days=30"),
    el("p", { class: "lede", text: "Totals for the key you call it with, over the last N days (1 to 366). Prompts and replies are never stored, only these counts." }),
    CodeBlock({ lang: "json", title: "200 OK", code: json({ object: "usage", key_id: 3, days: 30, requests: 42, prompt_tokens: 630, completion_tokens: 4180, total_tokens: 4810, errors: 1, mean_duration_ms: 812.4, latency_p50_ms: 790, latency_p95_ms: 1310 }) }),
    table("Fields", [["k", "Field"], ["d", "Meaning"]], [{ k: "requests, errors", d: "Requests made, and how many ended with a status of 400 or more." }, { k: "prompt_tokens, completion_tokens", d: "Tokens read and generated." },
      { k: "mean_duration_ms", d: "Mean wall-clock time per request." }, { k: "latency_p50_ms, latency_p95_ms", d: "Median and 95th-percentile time of successful requests (null when there are none)." }]),
    P("A key must be used to call this; a browser session gets its usage on the dashboard."),
  ] },

  "/docs/errors": { title: "Errors", render: () => [
    el("h1", { text: "Errors" }),
    el("p", { class: "lede", text: "Errors on the /v1 routes have the OpenAI shape, so SDKs raise their usual exceptions." }),
    CodeBlock({ lang: "json", code: json({ error: { message: "rate limit exceeded", type: "rate_limit_error", param: null, code: "rate_limit_exceeded" } }) }),
    table("Statuses", [["s", "Status"], ["t", "type / code"], ["m", "When"], ["a", "What to do"]], [
      { s: "400", t: "invalid_request_error", m: "A parameter is out of range, n is not 1, too many stop sequences.", a: "Fix the request; do not retry unchanged." },
      { s: "401", t: "authentication_error / invalid_api_key", m: "Missing, wrong, revoked or expired key.", a: "Check the key; create a new one." },
      { s: "404", t: "invalid_request_error / model_not_found", m: "Unknown model name.", a: 'Use "aarav-lm".' },
      { s: "413", t: "invalid_request_error / request_too_large", m: "The body is over 1 MB.", a: "Send less." },
      { s: "429", t: "rate_limit_error / rate_limit_exceeded", m: "Per-key request or token limit, or too many at once.", a: "Wait for Retry-After seconds, then retry." },
      { s: "500", t: "server_error", m: "Generation failed on the server.", a: "Retry once after a moment." },
      { s: "503", t: "server_error / server_busy", m: "The queue is full or your request waited too long.", a: "Wait for Retry-After seconds, then retry." }]),
    H2("Retrying"),
    P("Honour the Retry-After header on 429 and 503. Otherwise back off exponentially (1 s, 2 s, 4 s) and stop after a few tries. Do not retry 400, 401, 404 or 413: they will fail the same way."),
    H2("While streaming"),
    P("Errors that occur before the first token are ordinary HTTP errors. An error after streaming has begun arrives as a final data frame containing an error object, then [DONE]."),
  ] },

  "/docs/sdk/python": { title: "Python", render: () => { const { base } = facts(); return [
    el("h1", { text: "Python" }), el("p", { class: "lede", text: "The official OpenAI package works unchanged." }),
    CodeBlock({ lang: "bash", code: "pip install openai" }),
    H2("A reply"), CodeBlock({ lang: "python", code: chatSnippets().find((s) => s.id === "python").code }),
    H2("Streaming"), CodeBlock({ lang: "python", code: streamSnippets().find((s) => s.id === "python").code }),
    H2("Handling errors"),
    CodeBlock({ lang: "python", code: `import time
import openai
from openai import OpenAI

client = OpenAI(base_url="${base}", api_key="aak_...", max_retries=0)

def ask(prompt, tries=3):
    for attempt in range(tries):
        try:
            r = client.chat.completions.create(
                model="aarav-lm", max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            return r.choices[0].message.content
        except (openai.RateLimitError, openai.InternalServerError) as e:
            wait = float(e.response.headers.get("retry-after", 2 ** attempt))
            time.sleep(wait)
    raise RuntimeError("the model stayed busy")` }),
  ]; } },

  "/docs/sdk/javascript": { title: "JavaScript", render: () => { const { base } = facts(); return [
    el("h1", { text: "JavaScript" }), el("p", { class: "lede", text: "Plain fetch works in browsers and Node 18+; the openai package works too." }),
    H2("A reply"), CodeBlock({ lang: "javascript", code: chatSnippets().find((s) => s.id === "javascript").code }),
    H2("Streaming"), CodeBlock({ lang: "javascript", code: streamSnippets().find((s) => s.id === "javascript").code }),
    H2("With the openai package"),
    CodeBlock({ lang: "javascript", code: `import OpenAI from "openai";

const client = new OpenAI({ baseURL: "${base}", apiKey: process.env.AARAV_KEY });
const r = await client.chat.completions.create({
  model: "aarav-lm",
  max_tokens: 150,
  messages: [{ role: "user", content: "Tell me a story about a fox named Kip." }],
});
console.log(r.choices[0].message.content);` }),
    H2("Handling errors"),
    CodeBlock({ lang: "javascript", code: `async function ask(prompt, tries = 3) {
  for (let attempt = 0; attempt < tries; attempt++) {
    const res = await fetch("${base}/chat/completions", {
      method: "POST",
      headers: { Authorization: "Bearer " + process.env.AARAV_KEY, "Content-Type": "application/json" },
      body: JSON.stringify({ model: "aarav-lm", max_tokens: 150, messages: [{ role: "user", content: prompt }] }),
    });
    if (res.ok) return (await res.json()).choices[0].message.content;
    if (res.status !== 429 && res.status !== 503) throw new Error((await res.json()).error.message);
    const wait = Number(res.headers.get("retry-after")) || 2 ** attempt;
    await new Promise((r) => setTimeout(r, wait * 1000));
  }
  throw new Error("the model stayed busy");
}` }),
  ]; } },

  "/docs/sdk/curl": { title: "cURL", render: () => { const b = baseUrl(); return [
    el("h1", { text: "cURL" }), el("p", { class: "lede", text: "Everything from the shell." }),
    H2("List the model"), CodeBlock({ lang: "bash", code: `curl ${b}/models -H "Authorization: Bearer $AARAV_KEY"` }),
    H2("A reply"), CodeBlock({ lang: "bash", code: chatSnippets().find((s) => s.id === "curl").code }),
    H2("Streaming"), CodeBlock({ lang: "bash", code: streamSnippets().find((s) => s.id === "curl").code }),
    H2("Your usage"), CodeBlock({ lang: "bash", code: `curl "${b}/usage?days=7" -H "Authorization: Bearer $AARAV_KEY"` }),
    H2("Show the headers (rate limits)"), CodeBlock({ lang: "bash", code: `curl -i ${b}/models -H "Authorization: Bearer $AARAV_KEY"` }),
  ]; } },

  "/docs/limitations": { title: "Limitations", render: () => { const { f, lim } = facts(); return [
    el("h1", { text: "Limitations" }), el("p", { class: "lede", text: "What this model and this service cannot do, stated plainly." }),
    H2("The model"),
    el("ul", { class: "plain" },
      el("li", { text: `It is a ${millions(f.params)}-parameter story model. It has no world knowledge and cannot answer questions, reason, summarise, translate, write code or do arithmetic.` }),
      el("li", { text: "Story logic drifts, names and pronouns sometimes disagree, and greedy decoding can loop." }),
      el("li", { text: "Length requests (“short”, “long”) and requests for two topics are unreliable." }),
      el("li", { text: "It has not been audited for bias, toxicity or safety and it has no moderation. The training text was generated by another model. English only." })),
    H2("The evidence"),
    el("ul", { class: "plain" },
      el("li", null, "On request wording it never saw in training, the fine-tuned versions are only modestly better than the untuned model and are statistically tied with each other. See the ", el("a", { href: "/models/aarav-lm/evaluations", text: "evaluations" }), "."),
      el("li", { text: "The evaluation measures rule-checkable request following, not how good a story is to read." })),
    H2("The service"),
    el("ul", { class: "plain" },
      el("li", { text: `Context is ${f.context || 512} tokens for the prompt and the reply together.` }),
      el("li", { text: lim.key_requests_per_minute ? `Per key: ${lim.key_requests_per_minute} requests and ${fmt(lim.key_tokens_per_minute)} tokens per minute.` : "Requests are rate limited per key." }),
      el("li", { text: "One generation runs at a time unless batching is on; when the queue is full the answer is 503 with Retry-After." }),
      el("li", { text: "There is no function calling, embeddings, image input or multiple choices (n must be 1)." }),
      el("li", { text: "There is one model; it cannot be switched per request." })),
  ]; } },
};

/* ── layout ────────────────────────────────────────────────────────────── */

export default async function page({ path }) {
  const p = PAGES[path] || PAGES["/docs"];
  const content = el("article", { class: "doc-body" }, ...p.render().flat(Infinity).filter(Boolean));
  const i = FLAT.findIndex(([h]) => h === path);
  const pager = el("nav", { class: "pager", "aria-label": "Previous and next pages" },
    i > 0 ? el("a", { href: FLAT[i - 1][0], class: "prev" }, icon("arrow", 14), el("span", null, el("small", { text: "Previous" }), el("br"), FLAT[i - 1][1])) : el("span"),
    i >= 0 && i < FLAT.length - 1 ? el("a", { href: FLAT[i + 1][0], class: "next" }, el("span", null, el("small", { text: "Next" }), el("br"), FLAT[i + 1][1]), icon("arrow", 14)) : el("span"));
  const side = el("details", { class: "doc-nav", open: true }, el("summary", { text: "Documentation menu" }), Sidebar(NAV.map((g) => ({ title: g.title, links: g.links.map(([href, label]) => ({ href, label })) })), "Documentation"));
  const mq = window.matchMedia("(min-width: 52rem)");
  const sync = () => { side.open = mq.matches; };
  sync(); mq.addEventListener("change", sync);
  return { title: p.title, description: `aarav-lm documentation: ${p.title}.`, node: el("div", { class: "container page docs" }, side, el("div", { class: "doc-main" }, content, pager)),
    onLeave: () => mq.removeEventListener("change", sync) };
}
