/* Shared application state. Chat history and settings live in the browser tab only; the server is stateless about them. */

export const state = {
  config: { auth: "none", signup: "closed", model: "", limits: {}, model_info: {} },
  me: null,                       // {email, csrf_token} when signed in with an account
  apiKey: "",                     // legacy key mode: kept for this tab only
  models: null,                   // /v1/models data, loaded on demand
  results: null,                  // /results.json, loaded on demand
  configLoaded: false,
  online: true,
};

try { state.apiKey = sessionStorage.getItem("aarav_key") || ""; } catch (e) { /* private mode: memory only */ }

export const SETTINGS_DEFAULTS = { system: "", temperature: 0.8, top_p: 0.95, top_k: 50, max_tokens: 150, repetition_penalty: 1.1, seed: "" };

export function loadSettings() {
  try { return { ...SETTINGS_DEFAULTS, ...JSON.parse(localStorage.getItem("aarav_settings") || "{}") }; } catch (e) { return { ...SETTINGS_DEFAULTS }; }
}
export function saveSettings(s) { try { localStorage.setItem("aarav_settings", JSON.stringify(s)); } catch (e) { /* not fatal */ } }

export const needsAccount = () => state.config.auth === "account";
export const signedIn = () => state.config.auth !== "account" || !!state.me;
