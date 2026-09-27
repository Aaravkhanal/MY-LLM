import { el } from "../dom.js";
import { Button, StatusIndicator, Skeleton, DataTable, ErrorState, ModelCard } from "../ui.js";
import { publicJson } from "../api.js";
import { modelFacts, millions, results } from "../data.js";

export default async function page() {
  const f = modelFacts();
  const status = el("span", null, StatusIndicator("unknown", "Checking…"));
  publicJson("/status.json").then((s) => status.replaceChildren(StatusIndicator(s.overall))).catch(() => status.replaceChildren(StatusIndicator("unknown", "Cannot reach the server")));
  const versions = el("div", null, Skeleton(4));
  results().then((r) => {
    versions.replaceChildren(DataTable({ caption: "Versions that have been evaluated. The server runs one at a time.", columns: [
      { key: "label", label: "Version", rowHeader: true, render: (m) => m.label + (m.id === r.default_model ? "  (served by default)" : "") },
      { key: "stage", label: "Stage" }, { key: "about", label: "What it is" }], rows: r.models }));
  }).catch(() => versions.replaceChildren(ErrorState({ title: "The versions could not be loaded", message: "Check your connection and reload the page." })));
  return { title: "Models", node: el("div", { class: "container page" },
    el("div", { class: "page-head" }, el("h1", { text: "Models" }), el("p", { text: "Models served by this deployment." })),
    ModelCard({ name: "aarav-lm", href: "/models/aarav-lm", status,
      description: `${millions(f.params)}-parameter decoder-only language model trained from scratch on simple children's stories, then fine-tuned to follow a few kinds of story request.`,
      badges: [f.context ? `${f.context} token context` : "context unknown", f.tuned ? "chat tuned" : "text continuation", "OpenAI-compatible"],
      actions: el("div", { class: "row" }, Button({ label: "Open", href: "/models/aarav-lm", variant: "primary", size: "sm" }), Button({ label: "Playground", href: "/playground", size: "sm" })) }),
    el("h2", { class: "sub-h", text: "Versions" }),
    el("p", { class: "muted prose", text: "Each fine-tuning recipe is kept and evaluated so the differences, and the lack of them, stay visible. Only one checkpoint is loaded by the server, so versions cannot be switched from the playground." }),
    versions) };
}
