/* Start-up: read the public config and the session, draw the frame, start the router. */
import { state } from "./state.js";
import { json, publicJson } from "./api.js";
import { renderChrome } from "./chrome.js";
import { startRouter } from "./router.js";

async function init() {
  try { state.config = await publicJson("/app/config"); state.configLoaded = true; } catch (e) { /* keep the defaults; pages show their own error states */ }
  if (state.config.auth === "account") {
    try { const s = await json("/account/session"); state.me = s.signed_in ? { email: s.email, csrf_token: s.csrf_token } : null; } catch (e) { state.me = null; }
  }
  renderChrome();
  await startRouter({});
}

init();
