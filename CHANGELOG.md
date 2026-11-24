# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). While the version is `0.x`, minor releases may change
configuration or behavior.

## [Unreleased]

### Changed
- GitHub Actions use `actions/checkout@v7` and `astral-sh/setup-uv@v7` (Node 24 runtime).
- Pre-commit runs ruff and mypy through `uv run`, so hook versions always match `uv.lock` (they
  were pinned to ruff 0.8.6 and mypy 1.14.1 against 0.17.6 and 2.4.1 in the project).

## [0.1.0] - 2026-11-24

First tagged release. It was checked against the live Groq API, but two pre-release items were
still open when it was tagged: the Groq policy review and the `GROQ_API_KEY` repository secret
for the weekly eval workflow (the secret was added afterwards and the workflow passed). Treat it as an early release, not as ready for
unsupervised public use.

### Added
- LangGraph pipeline: safety check, classifier, and therapist, logical and blended agents.
- `mixed` classification with a confidence threshold, and a blended agent.
- Classifier context from the last few turns, few-shot prompt, and versioned prompts.
- Safety check (regex plus LLM confirmation) and fixed regional crisis replies with no LLM.
- Conversation memory with token-budget trimming, optional SQLite persistence and thread deletion.
- Streaming CLI and a FastAPI service with SSE streaming, API-key auth, per-user rate limiting,
  per-user thread isolation and `DELETE /threads/{id}`.
- JSON logging with PII redaction, Prometheus metrics, per-node latency and token tracking.
- Dockerfile (multi-stage, non-root, health check).
- Test suite (98% coverage, 85% gate), labeled classifier and safety datasets, opt-in live
  classifier eval, and GitHub Actions workflows.
- README, privacy notes and runbook.
- Optional Streamlit demo UI (`uv sync --extra demo`) and a `meta` SSE event (`crisis`, plus
  `message_type` and `confidence` when `debug` is requested).

### Changed
- Provider is Groq (`langchain-groq`) instead of Anthropic. Region aliases (`GB`, `IND`, ...) are
  accepted for `CRISIS_REGION`.
- Classifier failures and low-confidence results route to the blended agent instead of the logical one.
- Persistence is opt-in (`PERSIST_HISTORY`); conversations are in memory by default.

### Fixed
- Default model IDs (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`) no longer exist on Groq;
  defaults are now `openai/gpt-oss-120b` and `openai/gpt-oss-20b` (verified live).
- The live classifier eval did not read `.env`, so it skipped even with a key configured.
- Agents saw only the latest message, so they had no memory of earlier turns.
- The router was a dummy node writing a state key that did not exist; it is now a conditional edge.
- A classifier error crashed the whole chat loop.
- The log redactor masked timestamps as phone numbers.
- Safety patterns missed "hurting/harming myself" and curly apostrophes.

### Known issues
- `docker run --env-file` keeps quotes around values, which breaks a quoted `GROQ_API_KEY` (documented).
- Groq policy review is outstanding.
- The live safety confirmation errs toward crisis replies on some informational messages.
- The classifier eval scores 100%, which mainly shows the labeled set is easy.
