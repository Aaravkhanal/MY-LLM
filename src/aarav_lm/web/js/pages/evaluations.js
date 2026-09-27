/* The evaluation dashboard. Everything is drawn from results.json, which is generated from the committed evaluation files
   (scripts/build_site_data.py; a test proves it matches). Three things are kept apart on the page: what was MEASURED (pass rates),
   how UNCERTAIN it is (95% intervals over distinct prompts) and how it is INTERPRETED (paired difference against the untuned base). */
import { el } from "../dom.js";
import { Badge, DataTable, ErrorState, SegmentedControl, Select, Skeleton, Stats } from "../ui.js";
import { IntervalChart, ChartWithTable } from "../charts.js";
import { modelHeader } from "../model_shared.js";
import { results, readDifference, baseModel, defaultModel } from "../data.js";
import { pct, signed, fmt } from "../format.js";

const SET_NAMES = { final: "New wording", template: "Trained wording", dev: "Development" };
const MODE_NAMES = { greedy: "Greedy", sampled: "Sampled", both: "Both" };
const shortLabel = (l) => l.replace(/ \(.*\)$/, "");

export default async function page({ query }) {
  const host = el("div", { class: "eval" }, Skeleton(6));
  results().then((r) => host.replaceChildren(...dashboard(r, query))).catch(() => host.replaceChildren(ErrorState({ title: "The results could not be loaded", message: "Check your connection and reload the page." })));
  return { title: "aarav-lm evaluations", description: "Pass rates, confidence intervals and paired comparisons for every version of aarav-lm.", node: el("div", { class: "container page" }, modelHeader(), host) };
}

/* The verdict on one paired difference, decided by its interval. */
function verdict(d) {
  if (d.lo > 0) return { tone: "ok", word: "gain", long: "the interval is above zero" };
  if (d.hi < 0) return { tone: "danger", word: "loss", long: "the interval is below zero" };
  return { tone: "neutral", word: "not established", long: "the interval includes zero" };
}

function dashboard(r, query) {
  const state = { set: SET_NAMES[query.get("set")] ? query.get("set") : "final", mode: MODE_NAMES[query.get("mode")] ? query.get("mode") : "sampled",
    kind: r.kinds.some((k) => k.id === query.get("kind")) ? query.get("kind") : "all" };
  const surface = el("div", { "aria-live": "polite" });
  const setBar = SegmentedControl({ ariaLabel: "Which wording", value: state.set, onChange: (v) => { state.set = v; draw(); },
    options: r.sets.map((s) => ({ value: s.id, label: SET_NAMES[s.id], title: s.label })) });
  const modeBar = SegmentedControl({ ariaLabel: "Decoding method", value: state.mode, onChange: (v) => { state.mode = v; draw(); },
    options: Object.entries(MODE_NAMES).map(([value, label]) => ({ value, label })) });
  const kindSel = Select({ label: "Kind of request", id: "eval-kind", value: state.kind, onChange: (v) => { state.kind = v; draw(); },
    options: [{ value: "all", label: "All kinds" }, ...r.kinds.map((k) => ({ value: k.id, label: k.label }))] });

  function sync() {
    const q = new URLSearchParams({ set: state.set, mode: state.mode, kind: state.kind });
    history.replaceState(null, "", location.pathname + "?" + q.toString());
    setBar.set(state.set); modeBar.set(state.mode);
  }

  function draw() {
    sync();
    const set = r.sets.find((s) => s.id === state.set);
    const modes = state.mode === "both" ? ["greedy", "sampled"] : [state.mode];
    const cellOf = (m, mode) => set.results[m.id][mode][state.kind];
    const kindLabel = state.kind === "all" ? "all kinds of request" : r.kinds.find((k) => k.id === state.kind).label.toLowerCase();
    const pairedOf = (m, mode) => (m.id === "base" ? null : set.vs_base[m.id][mode][state.kind === "all" ? "ALL" : state.kind]);
    const series = [{ key: "greedy", label: "Greedy decoding", shape: "circle" }, { key: "sampled", label: "Sampled (temperature 0.7)", shape: "diamond", tone: 2 }].filter((s) => modes.includes(s.key));
    const rows = r.models.map((m) => ({ label: shortLabel(m.label), base: m.id === "base", cells: Object.fromEntries(modes.map((mode) => [mode, cellOf(m, mode)])) }));
    const chance = cellOf(r.models[0], modes[0]).chance;
    const chart = IntervalChart({ rows, series, chance, ariaLabel: `Pass rate for ${kindLabel} on ${SET_NAMES[state.set].toLowerCase()}: ${rows.map((x) => `${x.label} ` + modes.map((mo) => `${x.cells[mo].rate}% ${mo}`).join(", ")).join("; ")}.` });

    const tableRows = [];
    for (const m of r.models) for (const mode of modes) {
      const c = cellOf(m, mode), d = pairedOf(m, mode), v = d ? verdict(d) : null;
      tableRows.push({ m, mode, c, d, v, _muted: c.prompts < 10 });
    }
    const table = DataTable({ caption: `${set.label}. Kind: ${kindLabel}. Measured, uncertain, interpreted.`, columns: [
      { key: "label", label: "Version", rowHeader: true, render: (x) => shortLabel(x.m.label) },
      ...(modes.length > 1 ? [{ key: "mode", label: "Decoding", render: (x) => MODE_NAMES[x.mode] }] : []),
      { key: "rate", label: "Measured pass rate", num: true, render: (x) => `${x.c.rate.toFixed(1)}%` },
      { key: "ci", label: "95% interval", num: true, render: (x) => `${x.c.lo.toFixed(1)}–${x.c.hi.toFixed(1)}%` },
      { key: "n", label: "Distinct prompts", num: true, render: (x) => fmt(x.c.prompts) },
      { key: "d", label: "Versus untuned base (paired)", num: true, render: (x) => (x.d ? `${signed(x.d.diff)} [${signed(x.d.lo)}, ${signed(x.d.hi)}]` : "reference") },
      { key: "v", label: "Interpretation", render: (x) => (x.v ? Badge(x.v.word, x.v.tone) : Badge("reference")) },
    ], rows: tableRows });

    surface.replaceChildren(
      el("p", { class: "muted small", text: set.label }),
      el("h2", { class: "sub-h", text: `Pass rate by version: ${kindLabel}` }),
      ChartWithTable({ chart, table, label: "Show as" }),
      reading(r, set, state, modes, pairedOf, cellOf),
      dataset(r, set));
  }
  draw();

  return [
    el("section", { class: "eval-intro" }, el("h2", { text: "Evaluations" }),
      el("p", { class: "lede", text: "How well each version follows story requests, scored by rule on held-out prompts and always shown next to chance." })),
    el("div", { class: "eval-controls" }, el("div", null, el("span", { class: "label-s", text: "Wording" }), setBar), el("div", null, el("span", { class: "label-s", text: "Decoding" }), modeBar), kindSel),
    surface, seedSection(r), howToRead(), versions(r),
  ];
}

/* What the numbers say, in words, decided by the intervals. */
function reading(r, set, st, modes, pairedOf, cellOf) {
  const def = defaultModel(r), base = baseModel(r);
  const lines = modes.map((mode) => {
    const d = pairedOf(def, mode), c = cellOf(def, mode), b = cellOf(base, mode), rd = readDifference(d);
    return el("p", null, el("strong", { text: `${MODE_NAMES[mode]}: ` }),
      `${shortLabel(def.label)} passes ${c.rate.toFixed(1)}% (interval ${c.lo.toFixed(1)}–${c.hi.toFixed(1)}%); the untuned base passes ${b.rate.toFixed(1)}%; chance is ${b.chance}%. Paired on the same prompts the difference is ${rd.text}. ${rd.verdict}`);
  });
  const few = cellOf(base, modes[0]).prompts < 10;
  return el("div", { class: "reading prose", "aria-live": "polite" }, el("h3", { class: "sub-h2", text: "Reading" }), lines,
    few ? el("p", { class: "alert-soft", text: `Only ${cellOf(base, modes[0]).prompts} distinct prompts of this kind, so treat it as anecdotal.` }) : null);
}

function dataset(r, set) {
  const all = set.results.base.greedy.all;
  return el("div", { class: "eval-dataset" }, el("h3", { class: "sub-h2", text: "Evaluation dataset and decoding" }),
    Stats([["Prompts", fmt(set.prompts), `${fmt(set.scored)} scored`], ["Distinct prompts", fmt(all.prompts)], ["Chance level", `${all.chance}%`, "a story that ignores the request"], ["File hash", set.sha256.slice(0, 12) + "…", "SHA-256, checked before comparing"]]),
    el("p", { class: "small muted prose" }, `Greedy decoding: temperature 0. Sampled: temperature ${r.decoding.sampled.temperature}, top-k ${r.decoding.sampled.top_k}, top-p ${r.decoding.sampled.top_p}. Up to ${r.max_new_tokens} new tokens. Intervals are 95% bootstrap intervals that resample distinct prompts, because repeated draws of one prompt are not independent evidence.`));
}

function howToRead() {
  return el("section", { class: "howto prose" }, el("h2", { class: "sub-h", text: "How to read this" }),
    el("ul", { class: "plain" },
      el("li", null, el("strong", { text: "Measured: " }), "the share of requests whose story passed the rule for that request (a name is present, at most 90 words, has dialogue, and so on)."),
      el("li", null, el("strong", { text: "Uncertain: " }), "the 95% interval. Overlapping intervals neither prove nor rule out a real difference; they only say the evaluation is too small to tell."),
      el("li", null, el("strong", { text: "Interpreted: " }), "the paired difference compares two versions on identical prompts, which is much more sensitive than comparing two separate intervals. “Not established” means its interval includes zero.")),
    el("p", { class: "muted small", text: "Three sets are shown. The new-wording set was written and committed before any model was scored, but it has since scored several candidates; the trained-wording set uses the training templates on held-out stories; the development set was used while tuning the recipes, so it is not held back. Kinds with few distinct prompts are anecdotal." }));
}

function versions(r) {
  return el("section", { class: "prose" }, el("h2", { class: "sub-h", text: "Versions" }),
    DataTable({ caption: "The versions compared above", columns: [{ key: "label", label: "Version", rowHeader: true, render: (m) => m.label + (m.id === r.default_model ? " \u00b7 default" : "") }, { key: "about", label: "What it is" }], rows: r.models }));
}


/* Both recipes trained with three seeds each: the spread a single row's interval does not include, and whether the
   two recipes actually differ once that spread is accounted for. */
function seedSection(r) {
  const setName = { final: "New wording", template: "Trained wording", dev: "Development" };
  const ids9 = r.seeds.epoch1.models, ids12 = r.seeds.epoch2.models;
  const seedOf = { exp009: 1337, exp015: 1338, exp016: 1339, exp012: 1337, exp013: 1338, exp014: 1339 };

  function table(models, sd, caption) {
    const cols = models.map((id, i) => ({ key: "abc"[i], label: `Seed ${seedOf[id]}`, num: true }));
    const rows = [];
    for (const sid of ["final", "template", "dev"]) for (const mode of ["greedy", "sampled"]) {
      const x = sd.sets[sid][mode];
      const row = { set: `${setName[sid]}, ${mode}`, mean: x.mean.toFixed(1) + "%", sd: x.sd.toFixed(1), range: x.range.toFixed(1) };
      x.values.forEach((v, i) => { row["abc"[i]] = v.toFixed(1) + "%"; });
      rows.push(row);
    }
    return DataTable({ caption, columns: [{ key: "set", label: "Set and decoding", rowHeader: true }, ...cols,
      { key: "mean", label: "Mean", num: true }, { key: "sd", label: "Std. dev.", num: true }, { key: "range", label: "Range", num: true }], rows });
  }
  function lengthTable(models, sd, caption) {
    const cols = models.map((id, i) => ({ key: "abc"[i], label: `Seed ${seedOf[id]}`, num: true }));
    const rows = ["long", "short", "long_topic", "short_topic"].map((k) => {
      const row = { k: r.kinds.find((x) => x.id === k).label };
      sd.kinds[k].greedy.forEach((v, i) => { row["abc"[i]] = v.toFixed(1) + "%"; });
      return row;
    });
    return DataTable({ caption, columns: [{ key: "k", label: "Kind", rowHeader: true }, ...cols], rows });
  }

  const fin12 = r.seeds.epoch2.sets.final.greedy, fin9 = r.seeds.epoch1.sets.final.greedy;
  const diffMean = (fin12.mean - fin9.mean).toFixed(1);

  return el("section", { class: "prose" }, el("h2", { class: "sub-h", text: "Both recipes, three training seeds each" }),
    el("p", { text: `Each recipe (exp_009's one epoch, exp_012's two epochs) was trained three times with different random seeds and nothing else changed, to measure how much a rerun differs before comparing the recipes to each other.` }),
    el("h3", { class: "sub-h2", text: "exp_012's recipe (two epochs)" }),
    el("p", { text: `On new wording, greedy, the three seeds pass ${r.seeds.epoch2.sets.final.greedy.values.map((v) => v.toFixed(1) + "%").join(", ")}: a spread of ${fin12.range} points (standard deviation ${fin12.sd}). Fine-tuning validation loss is the same to three decimals across seeds, so the loss cannot tell them apart.` }),
    table(ids12, r.seeds.epoch2, "Pass rate by training seed, exp_012's recipe"),
    el("h3", { class: "sub-h2", text: "exp_009's recipe (one epoch)" }),
    el("p", { text: `The same measurement for exp_009's recipe: new wording, greedy, passes ${r.seeds.epoch1.sets.final.greedy.values.map((v) => v.toFixed(1) + "%").join(", ")}, a smaller spread of ${fin9.range} points (standard deviation ${fin9.sd}).` }),
    table(ids9, r.seeds.epoch1, "Pass rate by training seed, exp_009's recipe"),
    el("h3", { class: "sub-h2", text: "Comparing the two recipes" }),
    el("p", { text: `Three-seed means on new wording, greedy: exp_012's recipe ${fin12.mean.toFixed(1)}%, exp_009's recipe ${fin9.mean.toFixed(1)}% (difference ${diffMean} points) — close relative to either recipe's own seed spread, so this evaluation cannot say one recipe beats the other. What is not close: the two recipes fail differently. exp_009's recipe never learns "long" well in any seed (at most 20.8% in-template) but reliably keeps "short" working (75 to 100%); exp_012's recipe sometimes learns "long" well (up to 79%) at the cost of "short" collapsing toward 0% in the same run, and the reverse in another. Two epochs did not make length control more reliable, it made it more seed-dependent.` }),
    lengthTable(ids9, r.seeds.epoch1, "Length-related kinds by training seed, exp_009's recipe (trained wording, greedy)"),
    lengthTable(ids12, r.seeds.epoch2, "Length-related kinds by training seed, exp_012's recipe (trained wording, greedy)"),
    el("p", { class: "muted small", text: "What this means generally: the 95% interval on a single row measures only how many prompts were asked, not how much a retrained model would differ. A gap between two versions smaller than the seed-to-seed spread, or a skill that appears in one training run, cannot be told apart from luck. Every version here except these six was trained once." }));
}
