import { el } from "../dom.js";
import { Button, EmptyState } from "../ui.js";

export default async function page() {
  return { title: "Page not found", node: el("div", { class: "container page" },
    EmptyState({ title: "Page not found", body: "That address does not match a page on this site. It may have moved.",
      action: el("div", { class: "row" }, Button({ label: "Go to the home page", href: "/", variant: "primary" }), Button({ label: "Read the docs", href: "/docs" })) })) };
}
