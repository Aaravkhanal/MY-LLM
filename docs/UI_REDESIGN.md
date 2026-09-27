# Frontend redesign: audit and plan

This is the audit that was done before any frontend code was changed, the decisions taken where the brief and the existing
project disagreed, and the implementation plan. The implementation audit (files, routes, tests) is appended at the end when the work
is done.

## 1. Audit of what exists

**Frontend** (`src/aarav_lm/web/`): one hand-written single-page app with no build step and no libraries. `index.html` (22 lines),
`style.css` (287), `app.js` (672), `results.json` (generated). Every element is created with DOM calls and `textContent`, never
HTML strings, so model output and server messages cannot become markup (a test enforces it). Routing is by URL hash
(`#/home|chat|keys|usage|docs|signin`). State is a module-level object; chat history lives in the tab only.

**Serving** (`api/app.py`): FastAPI. Static files are an exact-match allowlist (`WEB_FILES`: `/`, `/app.js`, `/style.css`,
`/results.json`), each answered with a strict Content-Security-Policy (`default-src 'none'; script-src 'self'; style-src 'self';
connect-src 'self'; img-src 'self' data:`), `nosniff`, `no-referrer`, `no-cache`. There is no path-based routing for pages, no
inline script or style is allowed, and no external font or script can load.

**API** (all real, all kept unchanged unless listed in section 4):

| Area | Endpoints |
|---|---|
| Health / info | `GET /health`, `GET /model`, `GET /stats` (key), `GET /app/config` (public) |
| Generation | `POST /v1/chat/completions` (+SSE), `POST /v1/completions`, `GET /v1/models`, `GET /v1/models/{id}`, `GET /v1/usage`, native `POST /generate`, `POST /chat` |
| Accounts | `POST /account/signup|login|logout|password`, `GET /account/session|me|keys|usage`, `POST /account/keys`, `/keys/{id}/revoke`, `/keys/{id}/rotate` |

**Authentication:** SQLite `AccountStore`; scrypt passwords; HttpOnly `SameSite=Strict` session cookie plus an `X-CSRF-Token`
header on every state-changing request; self-service API keys shown once, stored as hashes, with per-key rpm/tpm limits;
invite-gated or open sign-up. Usage rows hold counts, duration and outcome, never text.

**Model serving:** one engine behind a lock, or `BatchScheduler` waves with `--batch-size`; a busy server answers 503 with
Retry-After; 429 for per-key limits. Streaming frames are real tokens.

**Evaluation data:** committed JSON under `experiments/*/eval/` with per-prompt records; `scripts/build_site_data.py` turns
them into `web/results.json` (7 models x 2 wording sets x greedy/sampled x 12 kinds, each cell with rate, clustered 95%
interval, chance, distinct prompts). A test proves the file is exactly what the evaluation files say.

**Reusable:** the DOM helper `el()`, `svg()`, the tab strip (arrow-key navigation), the results explorer, the SVG bar chart with
table alternative, the copy button, the error-message mapper. **Styling architecture:** CSS custom properties, warm neutrals, one green,
no blue or purple, WCAG AA checked by `tests/test_web_design.py`.

**Must stay unchanged:** model weights and training code; the OpenAI-compatible API and its error shape; cookie + CSRF sessions;
the "no HTML from model output" guarantee; the strict CSP (so: no inline style or script, no external assets); the "counts only"
privacy rule; every number coming from files, never typed by hand.

## 2. Where the brief and the project disagree, and what I decided

| Brief | Existing | Decision |
|---|---|---|
| Path routes (`/playground`, `/docs/quickstart`, ...) | hash routes | Path routes with the History API. The server returns the app shell for an exact list of page paths (anything else stays a real 404). Old `#/chat` style links are redirected on load. |
| Dark, near-black, off-white text, subtle green | light by default, follows the OS | Dark is the default; a light theme remains as a toggle, remembered in the browser. |
| "Restrained cards", "consistent border radius", Modal/Toast components | earlier instruction: no blocks, no boxes | Sections stay unboxed (type, whitespace, hairlines). Bordered, unfilled elements only where a control needs an edge: inputs, code blocks, the composer, dialogs, tables. One radius token (6px) for those. No filled or tinted panels, no shadows, no gradients, no blur. `MetricCard` and `ModelCard` are hairline-delimited, not boxes. |
| No purple, no blue (earlier instruction), "subtle green accent" (now) | same | Unchanged: the hue test still forbids blue and purple everywhere. |
| Syntax highlighting, charts, icons | no libraries, strict CSP | In-house: a small tokenizer for bash/python/javascript/json that builds `<span>` nodes via the DOM helper; SVG charts; icons as inline SVG. No dependencies, so nothing new can break the CSP. |
| Fonts | system fonts | System font stack (a web font would need a `font-src` and a file to host). |
| Model selector | one served model | The selector lists what `/v1/models` returns. It shows one entry today, honestly. Other versions appear on the evaluation page only; they are not switchable because the server loads a single checkpoint. |
| "Average latency, error rate" on the dashboard | usage rows already store duration and status | Added to the API response (section 4). |
| Status page from real health data | `/health` says only `ok` | New public `/status.json` built from in-process checks (section 4). It cannot see outside the process; the page says so. |

## 3. Information architecture

Routes, all served as the app shell by the backend (exact list; the client renders a not-found page for anything unknown):

    /                                  landing (hero, live demo, what/eval/built/API/limits, CTA)
    /playground                        full-page playground
    /models                            model list
    /models/aarav-lm                   overview + model card
    /models/aarav-lm/evaluations       evaluation dashboard
    /models/aarav-lm/architecture      architecture (pipeline, real values, six expandable sections)
    /docs                              introduction
    /docs/quickstart, /docs/authentication
    /docs/api, /docs/api/chat-completions, /docs/models, /docs/usage, /docs/errors
    /docs/sdk/python, /docs/sdk/javascript, /docs/sdk/curl
    /docs/limitations
    /dashboard, /dashboard/usage, /dashboard/api-keys, /dashboard/settings
    /status
    /signin

The docs sidebar follows the brief (Getting started / API / SDK / Model); its Architecture and Evaluation entries link to the model
pages so the numbers exist in exactly one place.

## 4. Backend changes (small, additive, tested)

1. Serve every file under `web/` from an exact map built at start-up (still no listing, no traversal) and the app shell for the page
   paths above.
2. `GET /status.json` (public): per-service state (`operational | degraded | outage`) with the check that produced it. API: the
   process is answering. Inference: model loaded, scheduler thread alive, queue depth against its limit. Authentication: a database
   round trip. Playground: the shell and its script are present. Also uptime and queue depth; no user data.
3. `/app/config` model info gains the real architecture (layers, width, heads, KV heads, feed-forward size, vocabulary, context,
   norm, activation, position encoding, tied embeddings) read from the loaded model, so the architecture page shows what is running.
4. `/account/usage` totals gain median and 95th-percentile latency (mean and error count already exist).
5. `results.json` gains: the development wording set, paired differences against the untuned base and against the default model
   (from the per-prompt records, fixed seed), training facts (steps, tokens, seconds, validation loss per stage) and evaluation-set
   descriptions (prompt counts) read from the run and eval files.

## 5. Front-end structure (no build step, ES modules, same-origin)

    web/index.html            shell: skip link, header, main, footer, one module script
    web/css/tokens.css        colour, space, type, radius, motion tokens (dark, light)
    web/css/base.css          reset, typography, layout
    web/css/components.css    the component library
    web/css/pages.css         page-specific layout
    web/js/main.js            start-up, config and session, router
    web/js/router.js          History API router, link interception, redirects, focus and title management
    web/js/api.js             fetch, errors, streaming
    web/js/dom.js             el(), svg(), icons
    web/js/ui.js              Button, Badge, CodeBlock, Stat, StatusIndicator, Tabs, Sidebar, DataTable, MetricCard, Chart, Input,
                              Slider, Dropdown, Modal, Toast, Skeleton, EmptyState, ErrorState
    web/js/highlight.js       tokenizer for code blocks
    web/js/pages/*.js         one module per page group

## 6. Phases (each ends with tests, a commit and a push; nothing lands half-working)

1. Backend and data (section 4) with tests.
2. Design system and shell: tokens, components, router, header and footer, theme toggle, toasts.
3. Landing, playground, status, sign-in.
4. Model pages: overview, evaluations, architecture.
5. Documentation.
6. Dashboard: overview, usage, API keys, settings.
7. Quality pass: keyboard and screen-reader checks, responsive checks at 390, 768 and 1280, console and network errors, broken links,
   loading, empty and error states, performance; then this file's implementation audit.

## 7. How it is tested

* Python: page-route table, asset serving and traversal, CSP on every page, `/status.json` truth (degraded when the queue is full,
  outage when the model is gone), architecture values equal to the model's, usage percentiles, results data equals the eval files,
  design rules (hue, contrast, no shadows/gradients/filled panels, single radius, colours only in CSS), no HTML from model output,
  every internal link resolves to a route, JavaScript syntax with `node --check`.
* Browser (Playwright with real Chrome): every route at three widths in both themes, console and page errors, failed requests,
  a real streamed generation, sign-up, key creation, rotation and revocation, keyboard-only navigation of the tab strips and dialogs.

---

# Implementation audit

Written after the work, from the repository and the test runs; the numbers are counts of what exists.

## What was built, by phase

1. **Backend and data** (commit `05a3a0c`): page routes and an exact asset map (`api/web_static.py`), the public `/status.json`, the running model's
   architecture in `/app/config`, latency percentiles, and richer `results.json` (development set, paired comparisons, training facts).
2. **Public demo, API fixes, foundation** (`6e2ddb2`): `POST /demo/chat`, the `finish_reason` fix, usage in the closing stream frame, `--web-dir`, the design tokens,
   the component library, charts, the API layer with streaming, the router and the frame.
3. **Pages, cut-over, tests, docs** (this commit): the landing page, playground, status, sign-in, models, model card, evaluations, architecture, documentation,
   dashboard; the new frontend replaced the old one; tests rewritten; gzip for static files; screenshots and documentation.

## Files changed

* **Frontend, replaced:** `web/app.js` (672 lines), `web/style.css` (287) and `web/index.html` were removed.
* **Frontend, added under `src/aarav_lm/web/`:** `index.html`, `favicon.svg`, `css/` (4 files: tokens, base, components, pages), `js/` (13 modules: api, charts,
  chrome, data, dom, format, highlight, main, model_shared, router, snippets, state, theme, ui) and `js/pages/` (11 modules: landing, playground, models, model,
  evaluations, architecture, docs, dashboard, status, signin, notfound); about 2,800 lines in all. `results.json` is generated.
* **Backend:** `api/web_static.py` (new), `api/app.py`, `api/openai_compat.py`, `api/store.py`, `inference/scheduler.py`, `scripts/serve.py`,
  `scripts/build_site_data.py`.
* **Tests:** `tests/web_sources.py`, `test_web_routes.py`, `test_demo.py` (new); `test_web_design.py` (rewritten), `test_site.py`, `test_api.py` (updated).
* **Docs:** `UI_REDESIGN.md`, `PLATFORM.md` (rewritten), `API.md`, `SECURITY.md`, `HOSTING.md`, `CORRECTNESS.md`, `README.md`; ten screenshots in `docs/img/`.

## Routes added

24 page routes (section 3), all served as one shell and rendered by the client router: `/`, `/playground`, `/models`, `/models/aarav-lm`, `/models/aarav-lm/evaluations`,
`/models/aarav-lm/architecture`, 12 under `/docs`, 4 under `/dashboard`, `/status`, `/signin`. Old `#/chat`-style links redirect on load. **API routes added:**
`GET /status.json`, `POST /demo/chat` (opt-in). Nothing existing was removed or renamed.

## Components added

Button, Badge, CodeBlock and CodeTabs (with an in-house highlighter), Stat, MetricCard, ModelCard, StatusIndicator (shape plus word), Tabs (ARIA) and TabNav (routes),
Sidebar, DataTable, Input, Textarea, Select, Slider, SegmentedControl, Dropdown (menu button), Modal (native dialog, `confirmDialog`), Toast, Skeleton and LoadingRegion,
EmptyState, ErrorState (with retry), Accordion, CopyButton; charts: IntervalChart (dots and 95% intervals, responsive layout) and BarChart, each with a table alternative.

## Backend and API changes

* Page paths and an exact asset map; static files gzip-compressed once at start-up and negotiated by `Accept-Encoding`.
* `/status.json`; architecture and demo info in `/app/config`; p50 and p95 latency in `/account/usage` and `/v1/usage`; usage in the closing stream frame.
* `POST /demo/chat` with `--public-demo` and `--demo-per-minute`; `serve.py --web-dir`.
* **Fixes found on the way:** streams that failed after the 200 header under load (CORRECTNESS 8); `finish_reason: "eot"` on the OpenAI routes (CORRECTNESS 9).
* Not changed: model weights, training code, the request and response shapes of existing endpoints (additions only), authentication.

## UX improvements

Path URLs and instant navigation; focus and a screen-reader announcement on every page change; a live demo on the landing page; a full-page playground with sampling controls,
regenerate, copy, real token counts and a live status light; evaluation filters (wording, decoding, kind) that persist in the URL, a chart and a table for the same data, and each
comparison labelled measured, uncertain or interpreted; an architecture page whose parameter breakdown is computed and checked against the model; documentation with tabs, copy buttons and
highlighting, every example run against a live server; a dashboard with real metrics and key management where a secret is shown once; a status page that admits what it cannot see; light and dark themes;
skeleton, empty, error and retry states everywhere data loads.

## Tests performed

* **Automated:** the suite went from 482 tests before the redesign to 558 (all pass; the count includes tests that were rewritten); new or rewritten files: `test_web_routes.py` (page list, traversal, unknown paths,
  asset map, status truth including a stopped scheduler, a dead database and a full queue, architecture equals the model, latency percentiles, gzip, client and server route lists agree,
  every import and link resolves), `test_demo.py` (off by default, streaming, rate limit, cap, no usage row, 503 when busy, closing usage frame, finish reasons),
  `test_web_design.py` (hue, warmth, contrast AA, validated chart colours, colours only in tokens, no shadows, gradients, blur or cards, icon-only buttons named, motion),
  `test_site.py` (results equal the evaluation files; the documented paired numbers), `test_api.py` (CSP, no HTML from model output, JavaScript parses, no third-party host).
* **Real Chrome:** 24 routes x 3 widths (390, 768, 1280) x 2 themes: no console errors, no horizontal overflow, no empty pages, no broken internal links. Flows: sign-up and
  sign-in with return, streamed chat, regenerate, clear (dialog focus trapped and returned), the demo and its rate limit, key create/rotate/revoke (secret shown once and gone after reload),
  password form, status degraded/outage/unreachable (mocked), playground 503, 429, network failure and a raw-traceback 500 (never shown), evaluation results failing and slow (skeleton, error state),
  and keyboard-only use of the skip link, the account menu, tab strips (arrows, End) and the accordion.
* **API examples:** the cURL, Python (OpenAI SDK) and Node examples for chat and streaming, plus models, usage and the error shapes, were run against a live server with a real key.
* **Performance:** the home page is 192 KB uncompressed and about 47 KB gzipped over 23 requests (first contentful paint about 0.6 s cold on localhost); the playground 104 KB / 32 KB.

## Bugs the browser testing found (all fixed)

A chat error banner that was erased the moment it appeared; table cells that printed `[object HTMLSpanElement]`; arrays passed to `replaceChildren` printed as text; a toast covering the Send button;
the page content padded to zero because a later CSS rule overrode the container's padding; a stray `null` printed under the status summary.

## Remaining issues and known limitations

* Docker, Caddy and cloudflared are still unrun here; the public-hosting steps are documented, not exercised.
* **One model at a time.** The model selector shows what `/v1/models` returns, which is one entry. Other versions are on the evaluation page only.
* Dashboard charts show tokens per day only; there is no per-endpoint or per-key breakdown, and keys cannot be renamed after creation (the name is set at creation).
* Account deletion, email verification and password reset are not built.
* Status is self-reported from inside the process, keeps no history, and cannot detect a crashed server (the page reports it cannot reach it).
* No automated visual regression tests: the browser runs were manual scripts, not part of CI (CI has no Chrome); the design rules that can be checked without a browser are tests.
* Colour-blind separation of the two chart colours is a validator warning (deutan/protan about 7); the charts add marker shapes, direct labels and a legend, and there is a table view.
* Older screenshots in git history show the previous interface.
