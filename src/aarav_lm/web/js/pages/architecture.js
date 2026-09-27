/* The architecture page. Every value is read from the running model's configuration (/app/config -> model_info.architecture, which the
   server builds from the loaded weights), and the parameter breakdown is computed from those values and checked against the model's
   own parameter count. Nothing here is typed in except the prose. */
import { el, icon } from "../dom.js";
import { Accordion, DataTable, Skeleton, ErrorState, Badge } from "../ui.js";
import { modelHeader } from "../model_shared.js";
import { modelFacts, results, mtok } from "../data.js";
import { fmt, duration } from "../format.js";

export default async function page() {
  const f = modelFacts();
  const host = el("div", { class: "arch" }, Skeleton(6));
  if (!f.arch) {
    host.replaceChildren(ErrorState({ title: "The architecture is not available", message: "The server did not report its configuration. Reload the page, or try again in a moment." }));
  } else {
    results().then((r) => host.replaceChildren(...content(f.arch, r, f))).catch(() => host.replaceChildren(...content(f.arch, null, f)));
  }
  return { title: "aarav-lm architecture", description: "The architecture of aarav-lm, read from the running model.", node: el("div", { class: "container page" }, modelHeader(), host) };
}

/* Parameter counts from the configuration, with no bias terms (the model has none). */
export function breakdown(a) {
  const qOut = a.heads * a.head_size, kvOut = a.kv_heads * a.head_size;
  const rows = [
    ["Token embedding (also the output head)", a.vocab_size * a.width, "vocabulary × width, counted once because the weights are tied"],
    ["Attention, per layer", a.width * qOut + 2 * a.width * kvOut + qOut * a.width, "query, key, value and output projections"],
    ["Feed-forward (SwiGLU), per layer", 3 * a.width * a.ffn_hidden, "gate, up and down projections"],
    ["Normalisation, per layer", 2 * a.width, "two RMSNorm scales"],
  ];
  const layer = rows[1][1] + rows[2][1] + rows[3][1];
  const total = rows[0][1] + a.layers * layer + a.width;
  return { rows, layer, total, final: a.width };
}

function content(a, r, f) {
  const b = breakdown(a);
  const matches = b.total === a.parameters;
  const bytesPerToken = r ? (() => { const v = r.training.full_split_loss.exp_010_small_long_tinystories.val; return v.loss / Math.LN2 / v.bits_per_byte; })() : null;

  const stage = (title, spec, text, extra) => el("li", { class: "stage" }, el("div", { class: "stage-title" }, el("h3", { text: title }), el("span", { class: "mono spec", text: spec })), el("p", { class: "muted", text }), extra || null);
  const block = el("ol", { class: "sublist", "aria-label": `Inside each of the ${a.layers} blocks` },
    ["RMSNorm", `Grouped-query attention: ${a.heads} query heads share ${a.kv_heads} key/value heads (head size ${a.head_size}); rotary positions, θ = ${fmt(a.rope_theta)}`, "Residual add", "RMSNorm", `SwiGLU feed-forward: ${a.width} → ${a.ffn_hidden} → ${a.width}`, "Residual add"].map((t) => el("li", { text: t })));

  const flow = el("ol", { class: "flow", "aria-label": "Data flow, top to bottom" },
    stage("Data", "TinyStories", "Short stories, split into train, validation and test by a hash of each story."),
    stage("Tokenizer", `${fmt(a.vocab_size)} tokens`, "Byte-level BPE trained on the training split only; any text can be represented."),
    stage("Embeddings", `${fmt(a.vocab_size)} × ${a.width}`, a.positions === "rope" ? "One vector per token. There is no learned position table: positions enter as rotations inside attention." : "Token and learned position vectors."),
    stage(`Transformer blocks × ${a.layers}`, `${a.norm}, ${a.activation}`, "Each block reads the sequence, mixes information across positions, then transforms each position.", block),
    stage("LM head", `${a.width} → ${fmt(a.vocab_size)}`, `A final ${a.norm}, then logits over the vocabulary using the ${a.tied_embeddings ? "tied embedding matrix" : "output matrix"}.`),
    stage("Token generation", `context ${a.context_length}`, "Pick the next token (greedy, or by temperature, top-k, top-p and a repetition penalty), append it to the key/value cache, repeat until the model ends the story, a stop sequence appears or the limit is reached."));

  const config = DataTable({ caption: "Configuration of the running model", columns: [{ key: "k", label: "Setting", rowHeader: true }, { key: "v", label: "Value" }], rows: [
    ["Family", `${a.family === "modern" ? "Llama-style" : a.family} decoder-only transformer`], ["Layers", a.layers], ["Width", a.width], ["Query heads", a.heads], ["Key/value heads", `${a.kv_heads} (${a.heads / a.kv_heads} query heads share each)`],
    ["Head size", a.head_size], ["Feed-forward hidden size", a.ffn_hidden], ["Vocabulary", fmt(a.vocab_size)], ["Context length", `${a.context_length} tokens`], ["Normalisation", a.norm],
    ["Activation", a.activation], ["Positions", a.positions === "rope" ? `rotary (RoPE), θ = ${fmt(a.rope_theta)}` : "learned"], ["Tied input/output embeddings", a.tied_embeddings ? "yes" : "no"], ["Bias terms", a.bias ? "yes" : "no"],
  ].map(([k, v]) => ({ k, v })) });

  const params = DataTable({ caption: "Where the parameters are", columns: [{ key: "k", label: "Component", rowHeader: true }, { key: "n", label: "Parameters", num: true }, { key: "note", label: "Note" }], rows: [
    ...b.rows.map(([k, n, note]) => ({ k, n: fmt(n), note })), { k: `× ${a.layers} layers`, n: fmt(a.layers * b.layer), note: "the three per-layer rows, times the layer count" },
    { k: "Final normalisation", n: fmt(b.final), note: "one RMSNorm scale" }, { k: "Total, computed", n: fmt(b.total), note: "" }, { k: "Total, reported by the model", n: fmt(a.parameters), note: `outside the embedding table: ${fmt(a.parameters_body)}` }] });

  const sections = Accordion([
    { title: "Data", body: el("div", null, el("p", { text: "TinyStories V2: short, simple stories written by a larger model (licence CDLA-Sharing-1.0). Every story is assigned to train, validation or test by a hash of its content, so a story can never sit in two splits and adding more data cannot move a held-out story into training. Held-out stories were checked against a sample of the training stories: none shares even half of its 8-word sequences." }),
      el("p", { text: "Fine-tuning data are synthetic story requests paired with real stories that satisfy each request by rule, made from the training split only." })) },
    { title: "Tokenizer", body: el("p", { text: `A byte-level BPE tokenizer with ${fmt(a.vocab_size)} tokens, trained on the training split only. It can represent any text without an unknown token${bytesPerToken ? `, and averages about ${bytesPerToken.toFixed(2)} bytes of text per token (computed from the held-out loss and bits per byte)` : ""}. Control tokens typed by a user are read as ordinary text, so they cannot steer the model.` }) },
    { title: "Model", body: el("div", null, el("p", { text: `${a.layers} layers of width ${a.width}, ${a.heads} query heads sharing ${a.kv_heads} key/value heads, ${a.norm} normalisation, ${a.activation} feed-forward layers and ${a.positions === "rope" ? "rotary position embeddings" : "learned positions"}: ${fmt(a.parameters)} parameters.` }),
      el("p", { class: "row" }, matches ? icon("check", 16) : icon("alert", 16), el("span", { text: matches ? "The parameter breakdown below adds up to the model's own count." : "The breakdown below does not match the model's own count: the configuration and the weights disagree." })), params) },
    { title: "Training", body: trainingBody(r) },
    { title: "Evaluation", body: el("div", null, el("p", { text: "Every score is shown next to chance and a confidence interval, versions are compared on identical prompts, and the final set was committed before any model was scored on it. Language-modelling loss is computed over every window of the held-out split, with a bootstrap standard error." }),
      el("p", null, el("a", { href: "/models/aarav-lm/evaluations", text: "Open the evaluation dashboard" }))) },
    { title: "Serving", body: el("p", { text: `An OpenAI-compatible API (checked with the official client), accounts with self-service API keys, per-key rate and token limits, a time limit on every generation and exact batched decoding, so that several users share each model step${f.batching > 1 ? ` (this server batches up to ${f.batching})` : ""}. Usage records keep counts only; prompts and replies are never stored.` }) },
  ]);
  sections.openFirst();

  return [
    el("section", null, el("h2", { text: "Architecture" }), el("p", { class: "lede", text: "Read from the running model's configuration, not from a description." }), flow),
    el("h2", { class: "sub-h", text: "Configuration" }), config,
    el("h2", { class: "sub-h", text: "How it was built" }), sections,
  ];
}

function trainingBody(r) {
  if (!r) return el("p", { class: "muted", text: "The training records could not be loaded." });
  const rows = r.training.stages.map((s) => ({ e: s.experiment, kind: s.kind, steps: fmt(s.steps), tok: mtok(s.tokens_seen), t: duration(s.seconds), v: s.best_val_loss.toFixed(4) }));
  return el("div", null, el("p", { text: "AdamW with a warm-up and cosine schedule, gradient clipping, fp32 on the Apple GPU, batches of 32 sequences by gradient accumulation, deterministic seeds and atomic checkpoints. Resuming a run reproduces an uninterrupted one bit for bit. Health monitors watch the loss and gradients without ever changing training." }),
    DataTable({ caption: "Training runs, read from the run folders (fine-tuning loss is on the fine-tuning data, so it is not comparable with pretraining loss)", columns: [
      { key: "e", label: "Run", rowHeader: true }, { key: "kind", label: "Stage" }, { key: "steps", label: "Steps", num: true }, { key: "tok", label: "Tokens", num: true }, { key: "t", label: "Time", num: true }, { key: "v", label: "Best validation loss", num: true }], rows }));
}
