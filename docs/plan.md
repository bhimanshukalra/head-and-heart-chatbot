# Head & Heart Chatbot — Production Hardening Plan

A LangGraph chatbot that classifies each message as **emotional** (therapist persona) or **logical** (fact-focused persona) and routes it to the matching agent. This plan upgrades the base script into a production-grade service.

Stack: Python ≥3.12, `uv`, LangGraph, LangChain, Anthropic.

---

## 1. Problems in the base script

### Bugs / correctness
| # | Issue | Impact |
|---|-------|--------|
| 1 | Agents only see `messages[-1]` | No conversation memory — each reply ignores history. The biggest functional flaw. |
| 2 | `router` is a *node* returning `{"next": ...}`, but `next` is not in `State` | State key silently dropped; routing works only because of the lambda re-reading it. Should be a conditional-edge function returning the node name directly. |
| 3 | Classifier has no error handling | One API/parse failure crashes the whole loop. |
| 4 | `state.get("message_type", "logical")` default is unreachable (key always present, may be `None`) | `None` silently falls through to logical. |
| 5 | `message_type` persists across turns in the loop's `state` | Stale value; harmless today, fragile later. |
| 6 | Model `claude-3-5-sonnet-latest` is outdated/deprecated (project now uses Groq) | Move to a current model ID, configurable. |
| 7 | `print(last_message.content)` assumes `str` | Content can be a list of blocks. |
| 8 | Reply is appended as a raw dict, then read as an object | Works via coercion, but inconsistent; return `AIMessage`. |

### Production gaps
- No config management, logging, tracing, retries, timeouts, tests, packaging, CI, or safety handling.
- Single hard-coded binary classification; borderline messages (mixed emotional + practical) are forced into one bucket.
- **Safety:** an emotional/therapy-style bot must handle crisis content (self-harm, abuse) and avoid presenting itself as a licensed clinician.

---

## 2. Target architecture

```
START → safety_check ──crisis──► crisis_response ──► END
            │
            └─ok──► classify ──► (conditional) ──► therapist ─┐
                                      ├────────► logical ─────┤
                                      └──(mixed)─► blended ───┴──► END
```

- **Checkpointer** (`MemorySaver` → `SqliteSaver`/`PostgresSaver`) keyed by `thread_id` for real multi-turn memory.
- **Router** as a pure conditional-edge function (no dummy node).
- **Classifier** returns `emotional | logical | mixed` plus a confidence score; low confidence → `mixed`/default path.
- **Safety node** runs first (cheap keyword/regex prefilter, then LLM check on suspicious input) and short-circuits to a crisis response with hotline resources.
- **History trimming** via `trim_messages` / summarization to stay inside the context window.

### Proposed layout
```
head-and-heart-chatbot/
├── pyproject.toml
├── .env.example
├── README.md
├── src/head_and_heart/
│   ├── __init__.py
│   ├── config.py        # pydantic-settings: model, temps, timeouts, paths
│   ├── logging.py       # structured logging (structlog/JSON)
│   ├── llm.py           # model factory, retries, timeouts
│   ├── state.py         # State TypedDict
│   ├── prompts.py       # system prompts as constants / templates
│   ├── schemas.py       # MessageClassifier etc. (pydantic)
│   ├── nodes/
│   │   ├── safety.py
│   │   ├── classifier.py
│   │   ├── therapist.py
│   │   └── logical.py
│   ├── graph.py         # build_graph(checkpointer)
│   ├── cli.py           # interactive loop
│   └── api.py           # FastAPI (phase 5)
└── tests/
    ├── unit/            # nodes with fake LLM
    ├── integration/     # graph routing end-to-end
    └── evals/           # classifier accuracy dataset
```

---

## 3. Phased roadmap

### Phase 0 — Project foundation
- [x] Add dependencies via `uv add`: `langgraph`, `langchain`, `langchain-groq`, `python-dotenv`, `pydantic-settings`, `tenacity`; dev: `pytest`, `pytest-asyncio`, `ruff`, `mypy`, `pre-commit`. _(`tenacity` added but unused; Groq client retries cover Phase 2.)_
- [x] Restructure into `src/head_and_heart/` package; keep `main.py` as thin entrypoint (or `[project.scripts]` console command).
- [x] `.env.example` (`GROQ_API_KEY`, `LLM_PROVIDER`, `MODEL_NAME`, `LOG_LEVEL`); ensure `.env` in `.gitignore`.
- [x] `config.py` with `pydantic-settings`; fail fast if API key missing.
- [x] Ruff + mypy config in `pyproject.toml`; pre-commit hooks. _(pre-commit config written; hooks not installed, versions may need `pre-commit autoupdate`.)_

### Phase 1 — Fix correctness (highest value, smallest diff)
- [x] Pass **full (trimmed) history** to agents: `[system] + state["messages"]`.
- [x] Replace `router` node with conditional-edge function returning `"therapist" | "logical"`; remove `next`.
- [x] Return `AIMessage` from agents; read content safely (`message.text()` / join text blocks).
- [x] Use a current model ID from config; set `temperature` per node (classifier `0`, therapist ~0.7, logical ~0.2). _(Groq model IDs not yet verified against a live key.)_
- [x] Reset/overwrite `message_type` each turn; typed `Literal` field in `State`.
- [x] Wrap classifier in try/except; on failure default to `logical` and log a warning.

### Phase 2 — Reliability
- [x] `max_retries` + `timeout` on the model; `tenacity` for backoff on rate-limit/5xx. _(built-in client retries via `max_retries`; `tenacity` not used.)_
- [x] Graceful CLI error handling (`KeyboardInterrupt`, `EOFError`, API errors → friendly message, loop continues).
- [x] Context management: `trim_messages(max_tokens=…, strategy="last")`; optional rolling summary node for long chats. _(trimming done; rolling summary skipped.)_
- [x] Persistence: `SqliteSaver` with `thread_id` so sessions survive restarts.
- [x] Streaming responses (`graph.stream(..., stream_mode="messages")`) for better UX.
- [ ] Use async (`ainvoke`/`astream`) in service path. _(deferred to Phase 5.)_

### Phase 3 — Quality of routing & responses
- [x] Add `mixed` class + confidence; blended agent (empathize first, then practical steps). _(low confidence or classifier failure now falls back to `mixed`, not `logical`.)_
- [x] Give classifier recent context (last 2–3 turns), not just the last message ("yes, exactly" is unclassifiable alone).
- [x] Move prompts to `prompts.py`; version them; add few-shot examples to classifier. _(prompts versioned via `PROMPT_VERSION`; few-shot examples added.)_
- [x] Soften the "purely logical / no emotions" prompt so it stays polite and doesn't ignore clear distress.
- [x] Add a short disclaimer that the bot is not a licensed therapist (first turn / on request). _(printed once at the start of a new CLI thread, plus a prompt rule to say so when asked.)_

### Phase 4 — Safety & compliance
- [x] `safety_check` node: self-harm, suicide, abuse, medical emergency detection.
- [x] Crisis response template with region-appropriate resources (e.g. 988 in US, configurable), no LLM improvisation on the critical path. _(see `crisis.py`; hotline numbers must be verified per region before launch.)_
- [x] Input length limit and basic prompt-injection hygiene (treat user text as data; never let it alter system prompts). _(prompt rules + 4000-char limit; regex/LLM guardrails are not injection-proof.)_
- [x] PII handling: redact in logs; documented retention policy for stored conversations; make persistence opt-in/deletable. _(log redaction, opt-in persistence via `PERSIST_HISTORY`, `--delete-thread`; policy in `docs/privacy.md`.)_
- [ ] Review Groq usage policies for mental-health-adjacent products. _(manual task, not done; checklist in `docs/privacy.md`.)_

### Phase 5 — Observability & serving
- [x] Structured JSON logging with `thread_id`, node name, latency, token usage (no raw message content at INFO). _(`LOG_FORMAT=json`; per-node latency and tokens logged, never message text.)_
- [x] Tracing with LangSmith (env-flag controlled) or OpenTelemetry. _(LangSmith via env vars only, off by default; it sends message content to LangSmith.)_
- [x] Metrics: classification distribution, crisis triggers, error rate, p50/p95 latency, cost per conversation. _(Prometheus `/metrics`; latency histograms give p50/p95, tokens are counted but cost per conversation is not computed.)_
- [x] FastAPI service: `POST /chat` (SSE streaming), `GET /health`, per-user `thread_id`, request validation, rate limiting, auth. _(auth via `API_KEYS`, in-process rate limit, `DELETE /threads/{id}`; single-process limiter, no CORS.)_
- [x] Dockerfile (multi-stage, `uv`), non-root user, health check. _(written but not built: Docker daemon was not running.)_

### Phase 6 — Testing & CI
- [x] **Unit tests:** each node with a fake chat model (`GenericFakeChatModel`) — no network.
- [x] **Graph tests:** routing for emotional / logical / mixed / crisis / classifier-failure paths.
- [x] **Memory test:** follow-up message referencing earlier turn resolves correctly.
- [x] **Eval set:** ~100 labeled messages; track classifier accuracy/F1 in a separate, opt-in (`-m eval`) run. _(100 classifier rows plus a 49-row safety set in `tests/evals/`; the classifier eval is written but has never been run against real Groq. Labels are my own judgment.)_
- [x] GitHub Actions: `ruff`, `mypy`, `pytest` on PR; eval job on schedule/manual. _(`.github/workflows/ci.yml`, `eval.yml`; needs a `GROQ_API_KEY` repo secret; workflows have not run on GitHub yet.)_
- [x] Coverage gate (e.g. ≥85% on non-LLM code). _(98% measured; `fail_under = 85` in `pyproject.toml`.)_

### Phase 7 — Docs & release
- [x] README: setup, env vars, run, test, architecture diagram, safety notes.
- [x] `CHANGELOG.md`, semantic versioning, tagged releases. _(changelog written under `[Unreleased]`; **no tag created**, tag `v0.1.0` after the pre-release checks in `docs/runbook.md`.)_
- [x] Runbook: common failures (rate limits, key rotation), how to read traces. _(`docs/runbook.md`; written from the code, not from live operations.)_

---

### Phase 8 — Live validation (blocks the `v0.1.0` tag)
Everything before this phase was tested against fake models only. This phase used a real `GROQ_API_KEY` (checked 2026-11-24).
- [x] Confirm the Groq model IDs in `.env.example` exist, and that the classifier model supports structured output. _(The original Llama IDs had been retired; defaults are now `openai/gpt-oss-120b` / `openai/gpt-oss-20b`; structured output works.)_
- [x] Smoke test the graph: one emotional, one logical, one mixed and one crisis message. _(All four routed correctly, 1–3 s per reply. The ambiguous-safety LLM check leans toward crisis replies, e.g. "Write a story about an abusive relationship" and "How much paracetamol is an overdose?" were flagged.)_
- [x] Run `pytest -m eval tests/evals/test_classifier_eval.py -s`; review the confusion matrix and misclassified rows. _(116/116, including 16 harder cases. The labels are mine and the model agrees with all of them, so the set is easy; an independently labeled, harder set is still worth building.)_
- [x] Verify every hotline number in `crisis.py` for the supported regions. _(US, CA, UK, IN, AU checked against official sites; 999 and 112 were not searched separately.)_
- [ ] Review Groq's usage, data-retention and acceptable-use policies for mental-health-adjacent use (see `docs/privacy.md`). _(Manual; not done.)_
- [ ] Add the `GROQ_API_KEY` repo secret so `eval.yml` can run. _(CI itself passes on every push; the secret is yours to add.)_
- [ ] `docker build .`, run the container and confirm the health check. _(Not done: the Docker daemon was not running.)_
- [ ] Tag `v0.1.0`: bump the version, move the changelog entries out of `[Unreleased]`.

### Phase 9 — Demo UI (Streamlit)
A thin front end for demos and internal testing. It is not the production UI: Streamlit re-runs the script on every interaction and does not isolate users.
- [x] Add `streamlit` as an optional dependency group (not in the core install).
- [x] `demo/app.py`: chat box with streaming, calling the existing `/chat` SSE endpoint (preferred, so auth, rate limits and safety all apply) or the graph directly. _(calls `/chat`; `demo/client.py` is tested end to end against the real app.)_
- [x] Show the not-a-therapist disclaimer, show crisis replies clearly, and keep conversations in memory only (no persistence by default). _(crisis flag comes from a new `meta` SSE event.)_
- [x] Per-session `thread_id` in `st.session_state`; a "New conversation" button that calls `DELETE /threads/{id}`.
- [x] Optional debug panel (off by default) showing the route taken and the classifier confidence, never raw logs. _(debug panel uses the new `debug` request flag.)_
- [x] README section: how to run it locally; note that it must not be exposed publicly as-is. _(only tested headlessly with Streamlit's AppTest; not viewed in a browser.)_

### Phase 10 — Hardening for real users (later)
Known limitations from the earlier phases, to pick up once Phase 8 passes and there is a real audience.
- [ ] Multi-instance support: shared rate limiting (Redis or a gateway) and Postgres checkpointing instead of SQLite.
- [ ] Authenticate or network-restrict `/metrics`; add CORS configuration for browser clients.
- [ ] Broaden the safety patterns: slang, misspellings and other languages, driven by new cases in `tests/evals/safety_dataset.jsonl`.
- [ ] Data retention: automatic expiry of stored conversations, and optional encryption at rest.
- [ ] Native async nodes (`ainvoke`) so API concurrency does not depend on worker threads.
- [ ] Optional rolling summary of older turns instead of dropping them.
- [ ] Cost per conversation (token counts times per-model prices) as a metric.
- [ ] Prompt-injection red-team cases added to the eval set.

---

## 4. Key design decisions (to confirm)

| Decision | Recommendation | Alternatives |
|----------|----------------|--------------|
| Model | **Decided: Groq.** Llama 3.3 70B for agents, Llama 3.1 8B for classifier and safety (IDs unverified) | Single model for all nodes |
| Persistence | **Implemented:** opt-in SQLite, in-memory by default. Postgres not built | In-memory only |
| Delivery surface | **Implemented:** CLI and FastAPI (SSE) | Web UI / Slack bot |
| Safety check | **Implemented:** regex prefilter + LLM confirm, fail-safe to crisis | LLM-only (slower/costlier), keywords-only (misses nuance) |
| Mixed messages | **Implemented:** dedicated `blended` agent | Run both agents and merge |

## 5. Definition of done

- [x] Multi-turn conversations retain context across restarts (tested with SQLite; needs `PERSIST_HISTORY`).
- [x] API failures degrade gracefully and are logged (tested with fake failing models only).
- [x] Explicit crisis inputs always get the vetted reply. Ambiguous ones depend on an LLM check that fails safe.
- [x] ≥85% test coverage on deterministic code (98% measured).
- [ ] Classifier eval accuracy ≥90% on the labeled set (eval written, never run against Groq).
- [ ] CI green on GitHub (workflows never run).
- [ ] Docker image builds and passes its health check (never built).
- [x] No secrets or raw user content in logs (by design, with redaction tests; not audited against a live run).

## 6. Suggested order of work

1. Phase 0 + Phase 1 → one PR (functional, correct, packaged).
2. Phase 6 (tests for what exists) → locks in behavior before adding features.
3. Phase 2 → Phase 3 → Phase 4 (safety before any public exposure).
4. Phase 5 → Phase 7.
5. Phase 8 (live validation) next, then tag `v0.1.0`; Phase 9 (demo UI) and Phase 10 (hardening) after that.
