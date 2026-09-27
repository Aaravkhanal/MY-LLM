/* A small syntax highlighter for the code blocks (bash, python, javascript, json). It tokenises with regular expressions and returns
   DOM nodes: text is only ever set with textContent, so a code sample can never become markup. Not a parser; it is right for the
   short, well-formed samples in the docs. */
import { el } from "./dom.js";

const KEYWORDS = {
  python: "from import as def return class if elif else for while in not and or is None True False with try except raise lambda yield pass break continue",
  javascript: "const let var function return if else for while of in new await async import from export default class extends try catch throw null undefined true false typeof",
  bash: "if then else fi for do done while in export",
  json: "true false null",
};

const RULES = {
  bash: [
    ["comment", /#[^\n]*/],
    ["string", /"(?:\\.|[^"\\])*"|'[^']*'/],
    ["variable", /\$\{[^}]*\}|\$[A-Za-z_][A-Za-z0-9_]*/],
    ["keyword", /(?<![\w-])--?[A-Za-z][\w-]*/],
    ["function", /(?:^|(?<=[\n|&;]\s*))(?:curl|pip|python3?|export|echo|cat|npm|node|git|docker)(?=\s)/],
    ["number", /\b\d+(?:\.\d+)?\b/],
    ["punct", /\\(?=\n)|[|&;<>]/],
  ],
  python: [
    ["comment", /#[^\n]*/],
    ["string", /"""[\s\S]*?"""|'''[\s\S]*?'''|[fbr]?"(?:\\.|[^"\\\n])*"|[fbr]?'(?:\\.|[^'\\\n])*'/],
    ["number", /\b\d+(?:\.\d+)?\b/],
    ["keyword", null],
    ["function", /\b[A-Za-z_][A-Za-z0-9_]*(?=\()/],
    ["punct", /[()[\]{}:,.=+\-*/<>]/],
  ],
  javascript: [
    ["comment", /\/\/[^\n]*|\/\*[\s\S]*?\*\//],
    ["string", /`(?:\\.|[^`\\])*`|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'/],
    ["number", /\b\d+(?:\.\d+)?\b/],
    ["keyword", null],
    ["function", /\b[A-Za-z_$][A-Za-z0-9_$]*(?=\()/],
    ["punct", /[()[\]{}:;,.=+\-*/<>!?|&]/],
  ],
  json: [
    ["function", /"(?:\\.|[^"\\])*"(?=\s*:)/],
    ["string", /"(?:\\.|[^"\\])*"/],
    ["number", /-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b/],
    ["keyword", null],
    ["punct", /[{}[\],:]/],
  ],
};

const cache = {};
function compile(lang) {
  if (cache[lang]) return cache[lang];
  const words = (KEYWORDS[lang] || "").split(" ").filter(Boolean);
  const parts = RULES[lang].map(([name, re]) => {
    const src = re ? re.source : `\\b(?:${words.join("|")})\\b`;
    return `(?<${name}>${src})`;
  });
  return (cache[lang] = new RegExp(parts.join("|"), "gm"));
}

const ALIASES = { sh: "bash", shell: "bash", curl: "bash", py: "python", js: "javascript", node: "javascript" };

export function highlight(code, lang) {
  const l = ALIASES[lang] || lang;
  if (!RULES[l]) return [document.createTextNode(code)];
  const re = compile(l);
  re.lastIndex = 0;
  const out = [];
  let last = 0, m;
  while ((m = re.exec(code)) !== null) {
    if (m[0] === "") { re.lastIndex++; continue; }
    if (m.index > last) out.push(document.createTextNode(code.slice(last, m.index)));
    const type = Object.keys(m.groups).find((k) => m.groups[k] !== undefined);
    out.push(el("span", { class: "tok-" + type, text: m[0] }));
    last = m.index + m[0].length;
  }
  if (last < code.length) out.push(document.createTextNode(code.slice(last)));
  return out;
}
