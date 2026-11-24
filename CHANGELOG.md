# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). While the version is `0.x`, minor releases may change
configuration or behavior.

## [Unreleased]

Everything below is the work toward `0.1.0`. **Do not tag `0.1.0` until the live checks in the
runbook's "Before the first release" list pass.**

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
- Provider is Groq (`langchain-groq`) instead of Anthropic.
- Classifier failures and low-confidence results route to the blended agent instead of the logical one.
- Persistence is opt-in (`PERSIST_HISTORY`); conversations are in memory by default.

### Fixed
- Agents saw only the latest message, so they had no memory of earlier turns.
- The router was a dummy node writing a state key that did not exist; it is now a conditional edge.
- A classifier error crashed the whole chat loop.
- The log redactor masked timestamps as phone numbers.
- Safety patterns missed "hurting/harming myself" and curly apostrophes.

### Known issues
- Not yet validated against the live Groq API (model IDs, structured output, classifier accuracy).
- Docker image and GitHub workflows have never run.
- Hotline numbers in `crisis.py` are unverified.
