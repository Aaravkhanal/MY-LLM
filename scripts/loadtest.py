"""Load-test a running server: how many users can it serve at once, and how does it fail when it is full?

    python scripts/loadtest.py --url http://127.0.0.1:8000 --key aak_... --concurrency 4 --requests 40 --max-tokens 120
    python scripts/loadtest.py --url ... --key ... --concurrency 8 --requests 80 --stream      # also time-to-first-token
    python scripts/loadtest.py ... --json                                                      # machine-readable

Each of `--concurrency` workers sends chat requests back to back until `--requests` have been made. Reported: the mix of HTTP
statuses (200 served, 429 rate limited, 503 busy, anything else a fault), latency percentiles of the successful requests,
completion tokens per second across all workers, and (with --stream) time to first token. The server generates one reply at a
time, so beyond a small concurrency the useful outcome is orderly 503s with Retry-After, never hangs or errors: this shows which.
Use a key whose limits are high enough not to be what you measure (manage_keys.py key create ... --rpm 6000 --tpm 10000000).
"""

import argparse
import http.client
import json
import statistics
import threading
import time
import urllib.parse
from collections import Counter

PROMPTS = ["Tell me a story about a dragon named Zara.", "Write a short story about a fox and a moon.",
           "A story about a robot who feels sad.", "Tell me a story about a girl named Mia and a red ball."]


def _one(host: str, port: int, scheme: str, key: str, body: dict, stream: bool, timeout: float) -> dict:
    t0 = time.perf_counter()
    conn = (http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection)(host, port, timeout=timeout)
    try:
        conn.request("POST", "/v1/chat/completions", json.dumps(body),
                     {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            resp.read()
            return {"status": resp.status, "latency": time.perf_counter() - t0, "retry_after": resp.getheader("Retry-After")}
        if not stream:
            data = json.loads(resp.read())
            return {"status": 200, "latency": time.perf_counter() - t0, "tokens": data["usage"]["completion_tokens"]}
        ttft, tokens, buf = None, 0, b""
        while True:
            chunk = resp.read1(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                frame, buf = buf.split(b"\n\n", 1)
                if frame.startswith(b"data: {"):
                    obj = json.loads(frame[6:])
                    if obj.get("error"):
                        return {"status": 500, "latency": time.perf_counter() - t0, "error": obj["error"].get("message", "")}
                    if obj["choices"][0]["delta"].get("content"):
                        tokens += 1
                        ttft = ttft if ttft is not None else time.perf_counter() - t0
        return {"status": 200, "latency": time.perf_counter() - t0, "tokens": tokens, "ttft": ttft}
    except (OSError, http.client.HTTPException) as e:
        return {"status": "connection_error", "latency": time.perf_counter() - t0, "error": type(e).__name__}
    finally:
        conn.close()


def pct(values, p):
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))], 3)


def run(url: str, key: str, concurrency: int, requests: int, max_tokens: int = 100, stream: bool = False,
        timeout: float = 120.0, model: str = "aarav-lm") -> dict:
    u = urllib.parse.urlparse(url)
    host, port, scheme = u.hostname, u.port or (443 if u.scheme == "https" else 80), u.scheme
    results, lock, counter = [], threading.Lock(), {"n": 0}

    def worker(wid: int) -> None:
        while True:
            with lock:
                if counter["n"] >= requests:
                    return
                i = counter["n"]
                counter["n"] += 1
            body = {"model": model, "stream": stream, "max_tokens": max_tokens, "temperature": 0.8, "seed": i,
                    "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}]}
            r = _one(host, port, scheme, key, body, stream, timeout)
            with lock:
                results.append(r)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(w,)) for w in range(concurrency)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    wall = time.perf_counter() - t0
    ok = [r for r in results if r["status"] == 200]
    lat = [r["latency"] for r in ok]
    tokens = sum(r.get("tokens", 0) for r in ok)
    return {
        "concurrency": concurrency, "requests": len(results), "wall_seconds": round(wall, 2),
        "statuses": dict(Counter(str(r["status"]) for r in results)),
        "served_per_second": round(len(ok) / wall, 2) if wall else 0, "completion_tokens_per_second": round(tokens / wall, 1) if wall else 0,
        "latency_seconds": {"p50": pct(lat, 50), "p95": pct(lat, 95), "max": round(max(lat), 3) if lat else None,
                            "mean": round(statistics.mean(lat), 3) if lat else None},
        "time_to_first_token_seconds": ({"p50": pct([r["ttft"] for r in ok if r.get("ttft") is not None], 50),
                                         "p95": pct([r["ttft"] for r in ok if r.get("ttft") is not None], 95)} if stream else None),
        "retry_after_seen": sorted({r["retry_after"] for r in results if r.get("retry_after")}),
        "faults": [r for r in results if r["status"] not in (200, 429, 503)][:5],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--requests", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--model", default="aarav-lm")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = run(a.url, a.key, a.concurrency, a.requests, a.max_tokens, a.stream, model=a.model)
    if a.json:
        print(json.dumps(r, indent=2))
        return
    lat = r["latency_seconds"]
    print(f"{r['requests']} requests, concurrency {r['concurrency']}, {r['wall_seconds']} s")
    print(f"  statuses: {r['statuses']}   (200 served, 429 rate limited, 503 busy)")
    print(f"  served {r['served_per_second']}/s, {r['completion_tokens_per_second']} completion tokens/s")
    print(f"  latency p50 {lat['p50']} s, p95 {lat['p95']} s, max {lat['max']} s")
    if r["time_to_first_token_seconds"]:
        print(f"  time to first token p50 {r['time_to_first_token_seconds']['p50']} s, p95 {r['time_to_first_token_seconds']['p95']} s")
    if r["retry_after_seen"]:
        print(f"  Retry-After values seen on refusals: {r['retry_after_seen']}")
    if r["faults"]:
        print(f"  FAULTS (not 200/429/503): {r['faults']}")


if __name__ == "__main__":
    main()
