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
- [ ] Add `mixed` class + confidence; blended agent (empathize first, then practical steps).
- [ ] Give classifier recent context (last 2–3 turns), not just the last message ("yes, exactly" is unclassifiable alone).
- [ ] Move prompts to `prompts.py`; version them; add few-shot examples to classifier.
- [ ] Soften the "purely logical / no emotions" prompt so it stays polite and doesn't ignore clear distress.
- [ ] Add a short disclaimer that the bot is not a licensed therapist (first turn / on request).

### Phase 4 — Safety & compliance
- [ ] `safety_check` node: self-harm, suicide, abuse, medical emergency detection.
- [ ] Crisis response template with region-appropriate resources (e.g. 988 in US, configurable), no LLM improvisation on the critical path.
- [ ] Input length limit and basic prompt-injection hygiene (treat user text as data; never let it alter system prompts).
- [ ] PII handling: redact in logs; documented retention policy for stored conversations; make persistence opt-in/deletable.
- [ ] Review Anthropic usage policies for mental-health-adjacent products.

### Phase 5 — Observability & serving
- [ ] Structured JSON logging with `thread_id`, node name, latency, token usage (no raw message content at INFO).
- [ ] Tracing with LangSmith (env-flag controlled) or OpenTelemetry.
- [ ] Metrics: classification distribution, crisis triggers, error rate, p50/p95 latency, cost per conversation.
- [ ] FastAPI service: `POST /chat` (SSE streaming), `GET /health`, per-user `thread_id`, request validation, rate limiting, auth.
- [ ] Dockerfile (multi-stage, `uv`), non-root user, health check.

### Phase 6 — Testing & CI
- [ ] **Unit tests:** each node with a fake chat model (`GenericFakeChatModel`) — no network.
- [ ] **Graph tests:** routing for emotional / logical / mixed / crisis / classifier-failure paths.
- [ ] **Memory test:** follow-up message referencing earlier turn resolves correctly.
- [ ] **Eval set:** ~100 labeled messages; track classifier accuracy/F1 in a separate, opt-in (`-m eval`) run.
- [ ] GitHub Actions: `ruff`, `mypy`, `pytest` on PR; eval job on schedule/manual.
- [ ] Coverage gate (e.g. ≥85% on non-LLM code).

### Phase 7 — Docs & release
- [ ] README: setup, env vars, run, test, architecture diagram, safety notes.
- [ ] `CHANGELOG.md`, semantic versioning, tagged releases.
- [ ] Runbook: common failures (rate limits, key rotation), how to read traces.

---

## 4. Key design decisions (to confirm)

| Decision | Recommendation | Alternatives |
|----------|----------------|--------------|
| Model | Configurable; default to a current Claude Sonnet, classifier on a cheaper/faster model (Haiku) | Single model for all nodes |
| Persistence | SQLite for CLI/dev, Postgres for service | In-memory only |
| Delivery surface | CLI first, then FastAPI | Web UI / Slack bot |
| Safety check | Keyword prefilter + LLM confirm | LLM-only (slower/costlier), keywords-only (misses nuance) |
| Mixed messages | Dedicated `blended` agent | Run both agents and merge |

## 5. Definition of done

- Multi-turn conversations retain context across restarts.
- No unhandled exceptions from API failures; all failures degrade gracefully and are logged.
- Crisis inputs always produce the vetted crisis response.
- ≥85% test coverage on deterministic code; classifier eval accuracy tracked and ≥90% on the labeled set.
- CI green (lint, types, tests); Docker image builds and passes health check.
- No secrets or raw user content in logs.

## 6. Suggested order of work

1. Phase 0 + Phase 1 → one PR (functional, correct, packaged).
2. Phase 6 (tests for what exists) → locks in behavior before adding features.
3. Phase 2 → Phase 3 → Phase 4 (safety before any public exposure).
4. Phase 5 → Phase 7.
