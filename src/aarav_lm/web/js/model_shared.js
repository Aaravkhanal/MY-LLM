/* The header shared by the model pages: name, live status, key facts and the route tabs. */
import { el } from "./dom.js";
import { Stats, StatusIndicator, TabNav } from "./ui.js";
import { publicJson } from "./api.js";
import { modelFacts, millions } from "./data.js";

export function modelHeader() {
  const f = modelFacts();
  const status = el("span", null, StatusIndicator("unknown", "Checking…"));
  publicJson("/status.json").then((s) => status.replaceChildren(StatusIndicator(s.overall))).catch(() => status.replaceChildren(StatusIndicator("unknown", "Cannot reach the server")));
  return el("div", { class: "model-head" },
    el("p", { class: "eyebrow" }, el("a", { href: "/models", text: "Models" }), " / aarav-lm"),
    el("div", { class: "row between" }, el("h1", { text: "aarav-lm" }), status),
    el("p", { class: "lede", text: `${millions(f.params)} parameter language model` }),
    Stats([["Parameters", millions(f.params)], ["Context length", f.context ? `${f.context} tokens` : "–"], ["Architecture", "Transformer", f.arch ? `decoder-only, ${f.arch.layers} layers, width ${f.arch.width}` : "decoder-only"],
      ["Training", "From scratch", "pretrained, then fine-tuned"]]),
    TabNav([{ href: "/models/aarav-lm", label: "Overview" }, { href: "/playground", label: "Playground" }, { href: "/models/aarav-lm/evaluations", label: "Evaluations" },
      { href: "/models/aarav-lm/architecture", label: "Architecture" }, { href: "/docs/api/chat-completions", label: "API" }], "Model sections"));
}
