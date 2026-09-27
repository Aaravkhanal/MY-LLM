/* System status, from GET /status.json: the state of each service as the server process itself sees it. It cannot see beyond the
   process (a server that is down cannot answer), so an unreachable server is shown as exactly that, never as "operational". */
import { el } from "../dom.js";
import { Button, StatusIndicator, Stats, Skeleton, ErrorState } from "../ui.js";
import { publicJson } from "../api.js";
import { duration } from "../format.js";

const SERVICES = [
  ["api", "API", "The HTTP API and the OpenAI-compatible routes"],
  ["inference", "Inference", "The model and its request queue"],
  ["authentication", "Authentication", "Accounts, sessions and API keys"],
  ["playground", "Playground", "This web app"],
];
const HEADLINE = { operational: "All systems operational", degraded: "Some systems are degraded", outage: "A service is down", unknown: "Cannot reach the server" };

export default async function page() {
  const host = el("div");
  let timer = null, alive = true;

  async function load(manual) {
    if (manual) host.setAttribute("aria-busy", "true");
    let s = null, failed = null;
    try { s = await publicJson("/status.json"); } catch (e) { failed = e; }
    if (!alive) return;
    host.removeAttribute("aria-busy");
    paint(s, failed);
  }

  function paint(s, failed) {
    const overall = s ? s.overall : "unknown";
    const summary = el("div", { class: "status-summary", role: "status", "aria-live": "polite" }, StatusIndicator(overall, HEADLINE[overall]));
    const rows = SERVICES.map(([key, name, blurb]) => {
      const svc = s && s.services[key];
      return el("li", { class: "service" }, el("div", null, el("h3", { text: name }), el("p", { class: "xs muted", text: blurb })),
        el("div", { class: "service-state" }, StatusIndicator(svc ? svc.state : "unknown"), el("p", { class: "xs muted", text: svc ? svc.detail : "No answer from the server." })));
    });
    const facts = s ? Stats([["Model", "aarav-lm", s.model], ["Server uptime", duration(s.uptime_seconds)], ["Queue", `${s.queue.waiting} waiting of ${s.queue.limit}`, `${s.queue.in_flight} generating now`]]) : null;
    const checked = s ? `Last checked ${new Date(s.checked_at).toLocaleTimeString()}. Refreshes every 30 seconds.` : "The last check failed. Refreshes every 30 seconds.";
    host.replaceChildren(...[summary,
      failed ? ErrorState({ title: "The status could not be loaded", message: "The server did not answer. It may be restarting, or your connection may be down.", retry: () => load(true) }) : null,
      el("ul", { class: "services" }, rows), facts,
      el("div", { class: "row status-foot" }, el("span", { class: "xs muted", text: checked }), Button({ label: "Refresh", variant: "quiet", size: "sm", icon: "refresh", onclick: () => load(true) })),
      el("h2", { class: "sub-h", text: "What this page can and cannot see" }),
      el("p", { class: "muted prose", text: "The server checks itself: the request queue against its limit, the batch scheduler thread, a round trip to the account database, and whether the web app files are being served. It cannot check anything from outside, and it keeps no history, so there is no incident log. If the server is down this page says it cannot reach it." })].filter(Boolean));
  }

  host.append(el("div", { role: "status" }, el("span", { class: "sr-only", text: "Loading status" }), Skeleton(5)));
  load(false);
  timer = setInterval(() => load(false), 30000);
  return { title: "Status", node: el("div", { class: "container page" }, el("div", { class: "page-head" }, el("h1", { text: "System status" }), el("p", { text: "Live state of each part of the service." })), host),
    onLeave: () => { alive = false; clearInterval(timer); } };
}
