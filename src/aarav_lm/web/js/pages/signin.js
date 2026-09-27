import { el } from "../dom.js";
import { Button, Input, EmptyState, toast } from "../ui.js";
import { state } from "../state.js";
import { json } from "../api.js";
import { renderChrome } from "../chrome.js";

export default async function page({ query, navigate }) {
  if (state.config.auth !== "account") {
    return { title: "Sign in", node: el("div", { class: "container page" }, EmptyState({ title: "Accounts are not enabled here",
      body: state.config.auth === "key" ? "This server uses an API key. Enter it in the playground." : "This server does not need an account.",
      action: Button({ label: "Open the playground", href: "/playground", variant: "primary" }) })) };
  }
  if (state.me) { navigate(safeNext(query.get("next")) || "/dashboard", { replace: true }); return { title: "Sign in", node: null }; }

  const signup = state.config.signup;                                   // "invite" | "open" | "closed"
  let creating = false;
  const heading = el("h1", { text: "Sign in" });
  const sub = el("p", { class: "muted" });
  const email = Input({ label: "Email", type: "email", autocomplete: "username", required: true, spellcheck: "false", id: "email" });
  const password = Input({ label: "Password", type: "password", autocomplete: "current-password", required: true, minlength: "10", maxlength: "128", id: "password", hint: "At least 10 characters." });
  const invite = Input({ label: "Invite code", type: "text", autocomplete: "off", spellcheck: "false", id: "invite", hint: "Ask the owner of this site for one." });
  invite.hidden = true;
  const err = el("p", { class: "alert", role: "alert", hidden: true });
  const submit = Button({ label: "Sign in", type: "submit", variant: "primary", iconAfter: "arrow" });
  const toggle = signup === "closed" ? null : Button({ label: "Create an account", variant: "quiet", onclick: () => setMode(!creating) });

  function setMode(c) {
    creating = c;
    heading.textContent = c ? "Create an account" : "Sign in";
    submit.querySelector("span").textContent = c ? "Create account" : "Sign in";
    toggle.querySelector("span").textContent = c ? "I already have an account" : "Create an account";
    password.input.setAttribute("autocomplete", c ? "new-password" : "current-password");
    invite.hidden = !(c && signup === "invite");
    err.hidden = true;
    paintSub();
  }
  function paintSub() {
    sub.textContent = signup === "closed" ? "Accounts are by invitation. Ask the owner for one."
      : creating && signup === "invite" ? "Creating an account needs an invite code from the owner."
      : "Free to use while capacity lasts.";
  }
  paintSub();

  const form = el("form", { class: "auth-form", novalidate: false, onsubmit: async (e) => {
    e.preventDefault();
    err.hidden = true; submit.disabled = true;
    try {
      const body = { email: email.input.value, password: password.input.value };
      if (creating) body.invite_code = invite.input.value;
      state.me = await json(creating ? "/account/signup" : "/account/login", { method: "POST", body });
      renderChrome();
      toast(creating ? "Account created. Welcome." : "Signed in.");
      navigate(safeNext(query.get("next")) || "/playground");
    } catch (ex) {
      err.textContent = ex.message; err.hidden = false;
    } finally { submit.disabled = false; }
  } }, email, password, invite, err, el("div", { class: "row" }, submit, toggle));

  return { title: "Sign in", node: el("div", { class: "container page narrow" }, heading, sub, form,
    el("p", { class: "xs muted", text: "Only your email, a salted hash of your password, hashes of your API keys and per-request token counts are stored. Prompts and replies are not." })) };
}

/* Only same-site paths: a ?next= pointing at another site must never be followed. */
function safeNext(n) { return n && n.startsWith("/") && !n.startsWith("//") ? n : null; }
