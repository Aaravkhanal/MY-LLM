# Hosting it for other people

The aim: other people use the model at a public address, safely, without you renting anything you don't need. This guide
is the plan and the exact steps. **What has and has not been verified is stated in each section;** nothing here has been
deployed to a public address yet, because that needs your accounts, a domain and your decision on cost.

What "hosting" is here: one Python process (the model, the API and the web app, [PLATFORM.md](PLATFORM.md)), a SQLite file
of accounts, and a reverse proxy that provides HTTPS. The model is tiny, so this is a small CPU workload. **No GPU is needed**
(measured: a 5M-parameter model decodes about 5x faster on the CPU than on the M3's GPU, [PERFORMANCE.md](PERFORMANCE.md)).

## Choose a path

| | A. Your Mac + a tunnel | B. A small rented server | C. Managed containers |
|---|---|---|---|
| Cost | free (Cloudflare's free tunnel) | a small monthly VM fee (see below) | pay per use |
| Effort | 10 minutes | an hour, once | more: no persistent disk, so SQLite must go |
| Uptime | only while the Mac is awake and online | always | always, scales to zero |
| Best for | showing it to people this week, a demo link for a CV | anyone can use it, any time | later, with Postgres |
| Risk | the Mac shares CPU and thermals with your training; sleeps; home IP | small | needs the database work |

**Recommendation:** start with **A** to get a link you can share today, and move to **B** when you want it always on. Skip **C**
until there is a reason to (it needs a database that outlives a container, which is one more piece to build).

## Capacity: what it can serve

The server generates **one reply at a time** (a lock around the model) and answers everyone else with a queue, then a polite
`503 Retry-After`. So capacity is the model's speed:

    replies per second  ≈  tokens per second ÷ tokens per reply        (a story is about 120-250 tokens)

Measured on the M3 (single stream through the HTTP API, CPU): the 5M-parameter model decodes about **380 tokens/s**
([DEPLOYMENT.md](DEPLOYMENT.md)), i.e. roughly one story per half a second, or 120 stories a minute for everyone together.
The 14M model is slower per token; **its number has not been measured yet** (the machine was training). Measure it on the
machine you will host on, with the tool in this repo:

```bash
python scripts/manage_keys.py key create you@example.com --label load --rpm 6000 --tpm 10000000
python scripts/loadtest.py --url http://127.0.0.1:8000 --key aak_... --concurrency 4 --requests 40 --max-tokens 150 --stream
```

It reports the mix of 200 / 429 / 503, latency percentiles, completion tokens per second, and time to first token. That the
server refuses cleanly when full (503 with Retry-After, never a hang or a 500) is covered by `tests/test_loadtest.py`.

Per-user limits keep one person from using it all: a signed-in browser or a self-made key gets **20 requests and 30,000 tokens
per minute** by default (`--web-rpm/--web-tpm/--key-rpm/--key-tpm`), five active keys per person. Twenty requests a minute is a
new story every three seconds, so with the model producing roughly one story per second or two, **a handful of people chatting at
the same time is comfortable; dozens is not** without batching. Batched decoding across users is built (`--batch-size 8`, exact and
tested; about 2-2.5x throughput at 16 concurrent users in a contended measurement, [PERFORMANCE.md](PERFORMANCE.md)); it is off by
default until measured on an idle machine.

## Path A: your Mac and a Cloudflare quick tunnel

1. Keep the Mac awake and plugged in for as long as you want it reachable; do not train at the same time (both slow by ~40%).
2. Start the server on localhost (it stays private to the Mac; the tunnel is the only way in):
   ```bash
   python scripts/manage_keys.py user add you@example.com --password      # your own login
   AARAV_SIGNUP_CODE='a-long-random-invite' caffeinate -i python scripts/serve.py \
       --db secrets/accounts.db --checkpoint models/tiny-tinystories-v1-sft-int8.pt --trust-proxy --cookie-secure
   ```
3. In another terminal: `brew install cloudflared`, then `cloudflared tunnel --url http://127.0.0.1:8000`. It prints a public
   `https://….trycloudflare.com` address. Share that address and the invite code.
4. Stop it (Ctrl-C) when you are done; the address dies with the process. A permanent address needs a Cloudflare account and a
   domain (a named tunnel).

*Not verified here:* `cloudflared` is not installed on this machine, so the tunnel itself was not run. The parts that matter for
correctness (`--trust-proxy`, secure cookies behind HTTPS, per-client rate limiting from `X-Forwarded-For`) are covered by tests.
`--trust-proxy` is only safe when a proxy really is in front; never use it on a port people can reach directly.

## Path B: a small server with Docker and automatic HTTPS

Any small Linux VM works (2 vCPU and 2-4 GB RAM is plenty for the model, which needs well under 1 GB). Prices vary by provider and
change; a small instance is typically a few to about twenty US dollars a month (check current pricing; this is an estimate, not
a quote), and you can stop paying by deleting it. Set a budget alert before creating anything.

1. Point a DNS name (an `A` record) at the VM's address. Open ports 80 and 443 only.
2. Install Docker, clone the repository, and copy the model you want into `models/` (the packaged 5M models are already there).
3. `cp deploy/env.example .env`, set `DOMAIN` and a long random `AARAV_SIGNUP_CODE`, and choose `AARAV_CHECKPOINT`.
4. `docker compose up -d --build`. Caddy fetches and renews the HTTPS certificate by itself; the app is reachable only through it.
5. Create your login: `docker compose exec app python scripts/manage_keys.py --db /data/accounts.db user add you@example.com --password`.
6. Open `https://your-domain`, sign in, and try the chat. Check `https://your-domain/health` from an uptime monitor.

What the files do: `Dockerfile` (CPU-only PyTorch, non-root user, health check, models mounted rather than baked in);
`docker-compose.yml` (the app is not published, only Caddy is; memory limit; no extra privileges); `deploy/Caddyfile`
(automatic TLS, HSTS, and `flush_interval -1` so streamed tokens reach the browser as they are produced).

*Not verified here:* Docker is not installed on the development machine, so **the image and compose file have not been built or
run**. The YAML parses and every flag they pass exists; treat the first `docker compose up` as the test, and check the log with
`docker compose logs app`.

## Operating it

* **Back up the accounts** (they are the only state): `python scripts/backup_db.py --db /data/accounts.db --out /backups --keep 14`
  from a daily cron (inside Docker: `docker compose exec app python scripts/backup_db.py --db /data/accounts.db --out /data/backups
  --keep 14`, then copy them off the machine). The copy is consistent while the server runs, private (0600), and integrity-checked
  (tested). It contains password and key *hashes*: store it like a secret.
* **Change the model:** put the new checkpoint in `models/`, set `AARAV_CHECKPOINT`, `docker compose up -d`. Sessions and keys
  survive; usage rows record the model id. Never overwrite a model file in place ([AUDIT.md](AUDIT.md): model versioning).
* **See what is happening:** `GET /stats` (with a key) gives counters; `usage` rows give per-person token counts; `docker compose
  logs`. Watch `usage_write_failures` in `/stats`; it should stay 0.
* **Stop abuse:** revoke a key (`manage_keys.py key revoke ID`) or disable a person (`user disable EMAIL`); both take effect on the
  next request. Change `AARAV_SIGNUP_CODE` to stop new signups; existing accounts keep working.
* **Update the code:** `git pull && docker compose up -d --build`.

## Before you share the address

- [ ] HTTPS works and `http://` redirects (Caddy does both) and cookies are `Secure` (`--cookie-secure`).
- [ ] The invite code is long and random, and signup is closed if you do not want strangers.
- [ ] You have logged in, created a key, used it with the `openai` client, revoked it, and seen it stop working.
- [ ] `loadtest.py` shows orderly 503s, not errors, when you push past capacity.
- [ ] A backup ran and you have opened the copy.
- [ ] Budget alert set (path B).
- [ ] The page says what the model is: a small story model, not a general assistant (the chat and docs pages already say so).
- [ ] You know what you keep: an email, a password hash, key hashes, and per-request token counts. **No prompts or replies are
      stored.** Tell users this in one sentence; if you add analytics or logging of text, that changes.
- [ ] Licences: the training data is CDLA-Sharing-1.0 ([DATASETS.md](DATASETS.md)); choose a licence for the weights before you
      distribute them.

## Not done, and when it matters

| Missing | Matters when |
|---|---|
| Continuous batching (new requests joining a wave already running) and vectorised sampling | many users at once, when waves are long |
| Postgres and a shared rate limiter | more than one server process, or managed containers |
| Email verification and password reset | you open signup to strangers (an email service is needed) |
| Metrics and alerting beyond `/health` and `/stats` | anything you depend on |
| A per-user daily quota | you want a hard cap on what one person can consume |
