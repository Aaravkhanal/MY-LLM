/* The home page: what it is, a live demo, the evidence, how it was built, the API, the limits. Concise on purpose: the detail lives on the
   model and docs pages. Every number is read from the server's config or results.json. */
import { el, icon } from "../dom.js";
import { Button, Stats, CodeTabs, ErrorState, Skeleton } from "../ui.js";
import { IntervalChart } from "../charts.js";
import { state, signedIn } from "../state.js";
import { ApiError, NetworkError, streamChat, chatPayload } from "../api.js";
import { modelFacts, millions, mtok, results, defaultModel, baseModel, setById, readDifference, pct } from "../data.js";
import { chatSnippets, baseUrl, EXAMPLE_PROMPTS } from "../snippets.js";
import { fmt, duration } from "../format.js";

export default async function page() {
  const f = modelFacts();
  const root = el("div");
  root.append(hero(f), demoSection(f));
  const evidence = el("section", { class: "section container", "aria-labelledby": "eval-h" }, el("h2", { id: "eval-h", text: "Evaluation" }), Skeleton(3));
  const built = el("section", { class: "section container", "aria-labelledby": "built-h" }, el("h2", { id: "built-h", text: "How it was built" }), Skeleton(3));
  root.append(whatItDoes(), evidence, built, apiSection(f), limits(f), finalCta());
  results().then((r) => { fillEvidence(evidence, r); fillBuilt(built, r, f); })
    .catch(() => { evidence.replaceChildren(el("h2", { id: "eval-h", text: "Evaluation" }), ErrorState({ title: "The results could not be loaded", message: "Check your connection and reload the page." }));
      built.replaceChildren(el("h2", { id: "built-h", text: "How it was built" }), el("p", { class: "muted", text: "The build facts could not be loaded." })); });
  return { title: "aarav-lm: a language model built from scratch", description: "aarav-lm is a 14.2M-parameter language model built from scratch, with a playground, an OpenAI-compatible API and open evaluation.", node: root };
}

function hero(f) {
  return el("section", { class: "hero container" },
    el("p", { class: "eyebrow", text: "Language model · built from scratch" }),
    el("h1", { class: "display", text: "aarav-lm" }),
    el("p", { class: "hero-sub", text: "A language model built from scratch." }),
    el("p", { class: "lede", text: `A ${millions(f.params)}-parameter model trained, evaluated and served from a custom pipeline: tokenizer, transformer, training loop, evaluation harness and an OpenAI-compatible API, all written for this project.` }),
    el("div", { class: "row cta" }, Button({ label: "Try the model", href: "/playground", variant: "primary", iconAfter: "arrow" }), Button({ label: "Read the docs", href: "/docs" })),
    Stats([["Parameters", millions(f.params)], ["Context", f.context ? `${f.context} tokens` : "–"], ["Tuning", f.tuned === null ? "–" : f.tuned ? "Chat tuned" : "Text continuation"], ["API", "OpenAI-compatible"]]));
}

/* ── the live demo ─────────────────────────────────────────────────────── */

function demoMode(f) {
  if (state.config.auth === "key" && !state.apiKey) return "key";
  if (signedIn()) return "account";
  return f.demo && f.demo.enabled ? "demo" : "locked";
}

function demoSection(f) {
  const mode = demoMode(f);
  const input = el("textarea", { class: "textarea", id: "demo-prompt", rows: "2", maxlength: "300", placeholder: "Ask for a short story…", "aria-describedby": "demo-help", disabled: mode === "locked" || mode === "key" });
  const out = el("div", { class: "demo-out", "aria-live": "polite", "aria-busy": "false" });
  const help = el("p", { id: "demo-help", class: "xs muted" });
  const go = Button({ label: "Generate", variant: "primary", icon: "send", type: "submit", disabled: mode === "locked" || mode === "key" });
  const stop = Button({ label: "Stop", variant: "quiet", icon: "stop" });
  stop.hidden = true;
  let controller = null;

  help.textContent = mode === "demo" ? `Free demo: short stories, a few tries a minute (up to ${f.demo.max_tokens} tokens). Sign in for more.`
    : mode === "locked" ? "Sign in to run the model. It is free while capacity lasts."
    : mode === "key" ? "This server needs an API key. Enter it in the playground."
    : "Signed in: replies use your account's limits.";
  const setOut = (...n) => out.replaceChildren(...n);
  setOut(el("p", { class: "muted small", text: mode === "locked" || mode === "key" ? "" : "The reply appears here, token by token." }));

  async function run() {
    const text = input.value.trim();
    if (!text || controller) return;
    controller = new AbortController();
    go.hidden = true; stop.hidden = false; input.disabled = true; out.setAttribute("aria-busy", "true");
    const body = el("p", { class: "reply" });
    setOut(body);
    body.append(el("span", { class: "dots", "aria-label": "Generating" }));
    let got = "";
    try {
      const path = mode === "demo" ? "/demo/chat" : "/v1/chat/completions";
      const payload = mode === "demo" ? { prompt: text } : { ...chatPayload([{ role: "user", content: text }], { temperature: 0.7, top_p: 0.95, top_k: 50, max_tokens: 150, repetition_penalty: 1.1 }, f.name) };
      const r = await streamChat(path, payload, (d) => { if (!got) body.replaceChildren(); got += d; body.textContent = got; }, controller.signal);
      if (!r.text) setOut(el("p", { class: "muted", text: "The model returned nothing. Try rephrasing." }));
      else if (r.finish === "length") body.after(el("p", { class: "xs muted", text: "Stopped at the length limit." }));
    } catch (e) {
      if (e.name === "AbortError") { if (!got) setOut(el("p", { class: "muted", text: "Stopped." })); }
      else setOut(ErrorState({ title: e instanceof ApiError && e.status === 429 ? "Slow down a little" : "That did not work", message: e instanceof ApiError || e instanceof NetworkError ? e.message : "Something went wrong. Try again.", retry: run }));
    } finally { controller = null; go.hidden = false; stop.hidden = true; input.disabled = false; out.setAttribute("aria-busy", "false"); input.focus({ preventScroll: true }); }
  }
  stop.addEventListener("click", () => controller && controller.abort());
  input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); form.requestSubmit(); } });
  const form = el("form", { class: "demo-form", onsubmit: (e) => { e.preventDefault(); run(); } },
    el("label", { class: "sr-only", for: "demo-prompt", text: "Your request" }), input,
    el("div", { class: "row between" },
      el("div", { class: "chips", role: "group", "aria-label": "Example prompts" }, EXAMPLE_PROMPTS.map((t) => el("button", { type: "button", class: "chip", text: t, disabled: mode === "locked" || mode === "key",
        onclick: () => { input.value = t; input.focus(); } }))),
      el("div", { class: "row" }, stop, go)),
    help);
  const lock = mode === "locked" ? el("p", { class: "small" }, el("a", { href: "/signin", text: "Sign in" }), " to generate text.") : mode === "key" ? el("p", { class: "small" }, el("a", { href: "/playground", text: "Open the playground" }), " and enter your key.") : null;
  return el("section", { class: "section container", "aria-labelledby": "demo-h" }, el("h2", { id: "demo-h", text: "Try it" }),
    el("p", { class: "lede", text: "Ask for a short children's story. It follows a few kinds of request (a name, a topic, a place, a feeling, a length) and is not a general assistant." }),
    form, lock, el("div", { class: "demo-result" }, out));
}

/* ── what it does ──────────────────────────────────────────────────────── */

function whatItDoes() {
  const item = (prompt, text) => el("li", null, el("span", { class: "quote", text: `“${prompt}”` }), el("span", { class: "muted", text }));
  return el("section", { class: "section container", "aria-labelledby": "does-h" }, el("h2", { id: "does-h", text: "What it does" }),
    el("ul", { class: "examples" },
      item("Tell me a story about a dragon named Zara.", "A short story, usually with the name you gave it."),
      item("A long story about a robot who feels sad.", "It tries to honour the topic, the feeling and the length, and often does."),
      item("Write a story set at the beach.", "Requests it saw during training work best; new wording is harder.")),
    el("p", { class: "muted small", text: "It is a small model. It has no world knowledge and cannot answer questions, write code or do maths." }));
}

/* ── evaluation (from results.json) ────────────────────────────────────── */

function fillEvidence(host, r) {
  const set = setById(r, "final"), base = baseModel(r), def = defaultModel(r);
  const rows = [base, def].map((m) => ({ label: m.label.replace(/ \(.*\)$/, ""), base: m.id === "base", cells: {
    greedy: set.results[m.id].greedy.all, sampled: set.results[m.id].sampled.all } }));
  const chance = set.results.base.greedy.all.chance;
  const diff = readDifference(set.vs_base[def.id].sampled.ALL);
  host.replaceChildren(el("h2", { id: "eval-h", text: "Evaluation" }),
    el("p", { class: "lede", text: `Requests are scored by rule (name present, at most 90 words, has dialogue…) on wording the model has never seen. Chance level is ${chance}%: a story that ignores the request passes about one time in ten.` }),
    IntervalChart({ rows, chance, ariaLabel: `Pass rate on unseen wording: ${rows.map((x) => `${x.label} ${x.cells.greedy.rate}% greedy, ${x.cells.sampled.rate}% sampled`).join("; ")}.`,
      series: [{ key: "greedy", label: "Greedy decoding", shape: "circle" }, { key: "sampled", label: "Sampled (temperature 0.7)", shape: "diamond", tone: 2 }] }),
    el("p", { text: `Sampled, the default model passes ${pct(set.results[def.id].sampled.all.rate / 100, 0)} of requests against ${pct(set.results.base.sampled.all.rate / 100, 0)} for the untuned base: ${diff.text}. ${diff.verdict}` }),
    el("p", { class: "muted small", text: `Retraining the same recipe with three different seeds gave ${r.seeds.sets.final.greedy.values.map((v) => v.toFixed(0) + "%").join(", ")} greedy on this set, so gaps of a few points between versions are within training noise.` }),
    el("p", { class: "small" }, el("a", { href: "/models/aarav-lm/evaluations", text: "Open the evaluation dashboard" }), icon("arrow", 14)));
}

/* ── how it was built (from the run files) ─────────────────────────────── */

function fillBuilt(host, r, f) {
  const stages = r.training.stages, pre = stages.filter((s) => s.kind.startsWith("pretrain"));
  const tokens = pre.reduce((n, s) => n + s.tokens_seen, 0), secs = pre.reduce((n, s) => n + s.seconds, 0);
  const a = f.arch;
  const full = r.training.full_split_loss.exp_010_small_long_tinystories.val;
  const steps = [
    ["Data", "TinyStories: simple stories written by a larger model, split by a hash of each story so a held-out story can never reach training."],
    ["Tokenizer", `A byte-level BPE vocabulary of ${a ? fmt(a.vocab_size) : fmt(f.vocab || 4096)} tokens, trained on the training split only.`],
    ["Model", a ? `${a.layers} layers, width ${a.width}, ${a.heads} attention heads sharing ${a.kv_heads} key/value heads, rotary positions, RMSNorm and SwiGLU; ${fmt(a.parameters)} parameters.` : "A Llama-style decoder-only transformer."],
    ["Training", `${mtok(tokens)} tokens of pretraining in ${duration(secs)} on an 8 GB MacBook Air, then supervised fine-tuning on rule-checkable story requests.`],
    ["Evaluation", `Held-out loss ${full.loss.toFixed(3)} (perplexity ${full.perplexity.toFixed(2)}); requests scored by rule, next to chance, with confidence intervals and a final set written before any model was scored.`],
    ["Serving", "An OpenAI-compatible API, accounts with self-service keys, per-key limits and exact batched decoding."],
  ];
  host.replaceChildren(el("h2", { id: "built-h", text: "How it was built" }),
    el("ol", { class: "steps" }, steps.map(([t, d], i) => el("li", null, el("span", { class: "n", text: String(i + 1).padStart(2, "0") }), el("div", null, el("strong", { text: t }), el("span", { class: "muted", text: " — " + d }))))),
    el("p", { class: "small" }, el("a", { href: "/models/aarav-lm/architecture", text: "See the architecture" }), icon("arrow", 14)));
}

/* ── API ───────────────────────────────────────────────────────────────── */

function apiSection(f) {
  return el("section", { class: "section container", "aria-labelledby": "api-h" }, el("h2", { id: "api-h", text: "Use it from code" }),
    el("p", { class: "lede" }, "Any OpenAI-compatible client works. Point it at ", el("code", { text: baseUrl() }), " and use one of your keys."),
    CodeTabs(chatSnippets(), { ariaLabel: "Example language" }),
    el("p", { class: "small" }, el("a", { href: "/docs/quickstart", text: "Read the quickstart" }), icon("arrow", 14)));
}

function limits(f) {
  return el("section", { class: "section container", "aria-labelledby": "lim-h" }, el("h2", { id: "lim-h", text: "Limitations" }),
    el("ul", { class: "plain prose" },
      el("li", { text: `It is a ${millions(f.params)}-parameter story model, not an assistant: no world knowledge, no reasoning, no code, no maths.` }),
      el("li", { text: "On request wording it never saw in training it is only modestly better than the untuned model, and the fine-tuned versions are statistically tied with each other." }),
      el("li", { text: "Length requests (“short”, “long”) and requests for two topics are unreliable." }),
      el("li", { text: "It has not been audited for bias or safety, and it has no moderation." }),
      el("li", { text: "One generation runs at a time unless batching is on; a busy server answers 503 with Retry-After." })),
    el("p", { class: "small" }, el("a", { href: "/docs/limitations", text: "Read the details" }), icon("arrow", 14)));
}

function finalCta() {
  return el("section", { class: "section container final-cta" }, el("h2", { text: "Try it, or build on it." }),
    el("div", { class: "row" }, Button({ label: "Try the model", href: "/playground", variant: "primary", iconAfter: "arrow" }), Button({ label: "Read the docs", href: "/docs" }), StatusLink()));
}
function StatusLink() { return el("a", { class: "quiet-link small", href: "/status" }, "System status"); }
