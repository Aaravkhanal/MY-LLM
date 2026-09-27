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
