/* Facts about the model and the evaluation, read from the server's config and results.json. Nothing here is typed in by hand: when a
   value is unknown the helpers return null and the pages show a dash. */
import { state } from "./state.js";
import { loadResults } from "./api.js";
import { pct, signed } from "./format.js";

export function modelFacts() {
  const mi = state.config.model_info || {};
  const arch = mi.architecture || null;
  return { params: mi.parameters ?? null, context: mi.context_length ?? null, tuned: mi.instruction_tuned ?? null, vocab: mi.vocab_size ?? null,
    batching: mi.batching ?? 1, arch, name: state.config.model || "aarav-lm", demo: mi.demo || { enabled: false } };
}
export const mtok = (n) => (n / 1e6).toFixed(1) + "M";
export const millions = (n) => (n === null || n === undefined ? "–" : (n / 1e6).toFixed(1) + "M");

export async function results() { return loadResults(); }

export const defaultModel = (r) => r.models.find((m) => m.id === r.default_model);
export const baseModel = (r) => r.models.find((m) => m.id === "base");
export const setById = (r, id) => r.sets.find((s) => s.id === id);

/* A plain-English reading of a paired difference, decided by its interval (never by eye). */
export function readDifference(d) {
  const text = `${signed(d.diff)} points (95% interval ${signed(d.lo)} to ${signed(d.hi)})`;
  if (d.lo > 0) return { text, verdict: "The interval is above zero: a gain that is unlikely to be chance." };
  if (d.hi < 0) return { text, verdict: "The interval is below zero: a loss that is unlikely to be chance." };
  return { text, verdict: "The interval includes zero, so this is not evidence of a real difference." };
}

export { pct, signed };
