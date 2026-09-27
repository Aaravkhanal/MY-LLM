/* Model overview and model card. Prose is written once here; every number in it is read from the results and the running model. */
import { el } from "../dom.js";
import { Skeleton, ErrorState, DataTable } from "../ui.js";
import { modelHeader } from "../model_shared.js";
import { modelFacts, millions, mtok, results, defaultModel, baseModel, setById, readDifference, pct } from "../data.js";
import { fmt, duration } from "../format.js";

export default async function page() {
  const f = modelFacts();
  const body = el("div", { class: "overview-body" }, Skeleton(6));
  results().then((r) => body.replaceChildren(...card(r, f))).catch(() => body.replaceChildren(ErrorState({ title: "The model card could not be loaded", message: "Check your connection and reload the page." })));
  return { title: "aarav-lm", description: "Model card for aarav-lm: a 14.2M-parameter language model built from scratch.", node: el("div", { class: "container page" }, modelHeader(), body) };
}

const H = (t) => el("h2", { class: "sub-h", text: t });

function card(r, f) {
  const def = defaultModel(r), base = baseModel(r);
  const final = setById(r, "final"), tmpl = setById(r, "template");
  const stages = r.training.stages, pre = stages.filter((s) => s.kind.startsWith("pretrain"));
  const tokens = pre.reduce((n, s) => n + s.tokens_seen, 0), secs = pre.reduce((n, s) => n + s.seconds, 0);
  const full = r.training.full_split_loss.exp_010_small_long_tinystories.val;
  const dSamp = readDifference(final.vs_base[def.id].sampled.ALL), dGreedy = readDifference(final.vs_base[def.id].greedy.ALL);
  const a = f.arch;
  return [
    el("p", { class: "prose", text: `aarav-lm is a ${millions(f.params)}-parameter decoder-only language model written and trained from scratch, with no pretrained weights and no borrowed model code, on one 8 GB MacBook Air. It writes short, simple children's stories, and the fine-tuned version follows a few kinds of story request. It is a research and portfolio artefact, not a general assistant.` }),
    H("Intended use"),
    el("ul", { class: "plain prose" }, el("li", { text: "Learning, demonstration and study of a complete from-scratch pipeline: data, tokenizer, training, evaluation, serving." }),
      el("li", { text: "Generating simple children's-story text on a laptop or a small server." })),
    H("Training"),
    el("p", { class: "prose", text: `Pretrained on TinyStories V2 (short stories written by a larger model, CDLA-Sharing-1.0) for ${mtok(tokens)} tokens in ${duration(secs)}, then fine-tuned on synthetic story requests paired with real stories that satisfy each request by rule. Training, validation and test stories are separated by a hash of each story's content, and the tokenizer saw only the training split.` }),
    H("Results"),
    DataTable({ caption: "Held-out language-modelling loss, every window of the validation split scored (lower is better)", columns: [
      { key: "m", label: "Stage", rowHeader: true }, { key: "loss", label: "Validation loss", num: true }, { key: "ppl", label: "Perplexity", num: true }, { key: "top1", label: "Next-token accuracy", num: true }],
      rows: ["exp_004_small_tinystories", "exp_010_small_long_tinystories"].map((e, i) => { const v = r.training.full_split_loss[e].val; return { m: i ? `Pretrained, ${mtok(tokens)} tokens` : `Pretrained, ${mtok(pre[0].tokens_seen)} tokens`, loss: `${v.loss.toFixed(3)} ± ${v.stderr.toFixed(3)}`, ppl: v.perplexity.toFixed(2), top1: pct(v.top1) }; }) }),
    el("p", { class: "prose", text: `Request following, scored by rule on wording the model has never seen, sampled decoding: the default version passes ${pct(final.results[def.id].sampled.all.rate / 100, 0)} against ${pct(final.results.base.sampled.all.rate / 100, 0)} for the untuned base, a paired difference of ${dSamp.text}. ${dSamp.verdict} With greedy decoding the difference is ${dGreedy.text}: ${dGreedy.verdict.toLowerCase()} On wording it was trained on the fine-tuned model passes ${pct(tmpl.results[def.id].sampled.all.rate / 100, 0)} against ${pct(tmpl.results.base.sampled.all.rate / 100, 0)}.` }),
    el("p", { class: "prose", text: `The same recipe trained with three seeds scored ${r.seeds.sets.final.greedy.values.map((v) => v.toFixed(1) + "%").join(", ")} (greedy, new wording): the seed alone moves the score by ${r.seeds.sets.final.greedy.range} points, more than the differences between fine-tuning recipes.` }),
    el("p", { class: "small" }, el("a", { href: "/models/aarav-lm/evaluations", text: "Open the evaluation dashboard" }), " for every version, every kind of request and the paired comparisons."),
    H("Limitations"),
    el("ul", { class: "plain prose" },
      el("li", { text: "Not a general assistant: no world knowledge, no reasoning, no code, no arithmetic." }),
      el("li", { text: "Story logic drifts, names and pronouns sometimes disagree, and greedy decoding can loop." }),
      el("li", { text: "Length requests and requests for two topics are unreliable; the fine-tuned versions are statistically tied with each other." }),
      el("li", { text: "Not audited for bias, toxicity or safety; the training text was generated by another model and there is no moderation. English only." }),
      el("li", { text: "The evaluation covers rule-checkable request kinds only; story quality as a person would judge it was not measured." })),
    H("Licences"),
    el("p", { class: "prose", text: "Code: MIT. Training data: CDLA-Sharing-1.0. A licence for the weights has not been chosen and should be decided before they are distributed." }),
    a ? el("p", { class: "small muted", text: `Running now: ${a.layers} layers, width ${a.width}, ${fmt(a.parameters)} parameters, context ${a.context_length}.` }) : null,
  ].filter(Boolean);
}
