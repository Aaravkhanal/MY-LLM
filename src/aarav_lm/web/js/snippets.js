/* The API examples shown on the landing page and in the docs. Built from the address the page is served from. */
export const baseUrl = () => location.origin + "/v1";

export function chatSnippets(model = "aarav-lm") {
  const base = baseUrl();
  return [
    { id: "curl", label: "cURL", lang: "bash", code:
`curl ${base}/chat/completions \\
  -H "Authorization: Bearer $AARAV_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${model}",
    "max_tokens": 150,
    "messages": [{"role": "user", "content": "Tell me a story about a fox named Kip."}]
  }'` },
    { id: "python", label: "Python", lang: "python", code:
`from openai import OpenAI

client = OpenAI(base_url="${base}", api_key="aak_...")
r = client.chat.completions.create(
    model="${model}",
    messages=[{"role": "user", "content": "Tell me a story about a fox named Kip."}],
    max_tokens=150,
)
print(r.choices[0].message.content)` },
    { id: "javascript", label: "JavaScript", lang: "javascript", code:
`const r = await fetch("${base}/chat/completions", {
  method: "POST",
  headers: { Authorization: "Bearer " + KEY, "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "${model}",
    max_tokens: 150,
    messages: [{ role: "user", content: "Tell me a story about a fox named Kip." }],
  }),
});
console.log((await r.json()).choices[0].message.content);` },
  ];
}

export function streamSnippets(model = "aarav-lm") {
  const base = baseUrl();
  return [
    { id: "curl", label: "cURL", lang: "bash", code:
`curl -N ${base}/chat/completions \\
  -H "Authorization: Bearer $AARAV_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"model": "${model}", "stream": true, "max_tokens": 150,
       "messages": [{"role": "user", "content": "A short story about a moon."}]}'` },
    { id: "python", label: "Python", lang: "python", code:
`from openai import OpenAI

client = OpenAI(base_url="${base}", api_key="aak_...")
stream = client.chat.completions.create(
    model="${model}",
    messages=[{"role": "user", "content": "A short story about a moon."}],
    max_tokens=150,
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)` },
    { id: "javascript", label: "JavaScript", lang: "javascript", code:
`const res = await fetch("${base}/chat/completions", {
  method: "POST",
  headers: { Authorization: "Bearer " + KEY, "Content-Type": "application/json" },
  body: JSON.stringify({ model: "${model}", stream: true, max_tokens: 150,
    messages: [{ role: "user", content: "A short story about a moon." }] }),
});
const reader = res.body.getReader(), decoder = new TextDecoder();
let buffer = "";
for (;;) {
  const { value, done } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });
  let i;
  while ((i = buffer.indexOf("\\n\\n")) !== -1) {
    const frame = buffer.slice(0, i).trim(); buffer = buffer.slice(i + 2);
    if (!frame.startsWith("data:") || frame.includes("[DONE]")) continue;
    process.stdout.write(JSON.parse(frame.slice(5)).choices[0].delta.content ?? "");
  }
}` },
  ];
}

export const EXAMPLE_PROMPTS = [
  "Tell me a story about a dragon named Zara.",
  "Write a short story about a fox and the moon.",
  "A story about a robot who feels sad.",
];
