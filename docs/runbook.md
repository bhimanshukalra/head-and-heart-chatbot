# Runbook

Operational notes for the HTTP service. Commands assume the metric and log names in this repo;
nothing here has been exercised against a live deployment yet.

## Before the first release

Do these once, then tag `0.1.0` and move the `[Unreleased]` changelog entries under it.

1. Confirm the Groq model IDs in `.env.example` exist and that the classifier model supports
   structured output. Run the CLI with a real key and try an emotional, a logical and a mixed
   message.
2. Run `pytest -m eval tests/evals/test_classifier_eval.py -s`. Review the confusion matrix and
   the misclassified list, and adjust the prompt or the dataset labels if the labels were wrong.
3. Verify every hotline number in `src/head_and_heart/crisis.py` for the regions you support.
4. Read Groq's current terms, data retention and acceptable-use policies for mental-health use.
5. `docker build .` and run the container with the health check.
6. Push and confirm both GitHub workflows pass (add a `GROQ_API_KEY` repository secret first).

Release steps: update `version` in `pyproject.toml` and `__version__` in
`src/head_and_heart/__init__.py`, edit the changelog, then tag (`git tag v0.1.0`).

## Reading logs and metrics

With `LOG_FORMAT=json`, each line has `time`, `level`, `logger`, `message`, and when relevant
`thread_id`, `node`, `latency_ms`, `input_tokens`, `output_tokens`, `outcome`. Message text is
never logged. Treat logs as sensitive anyway.

Useful Prometheus queries (scrape `/metrics`):

| Question | Query |
|----------|-------|
| p95 latency by node | `histogram_quantile(0.95, sum by (le, node) (rate(node_latency_seconds_bucket[5m])))` |
| Node error rate | `sum by (node) (rate(node_errors_total[5m]))` |
| Classifier fallbacks | `rate(classifier_fallbacks_total[5m])` |
| Classification mix | `sum by (message_type) (rate(messages_classified_total[1h]))` |
| Crisis replies | `sum by (category) (increase(crisis_triggers_total[1d]))` |
| Token use | `sum by (node, direction) (rate(llm_tokens_total[1h]))` |
| Request outcomes | `sum by (outcome) (rate(chat_requests_total[5m]))` |

Token counts are only recorded for agent nodes. Cost per conversation is not computed.

## Tracing (optional)

LangSmith tracing is off by default. Setting `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` in
the environment enables it. **Traces contain full message text**, including crisis conversations,
so enable it only if your privacy policy allows sending that to LangSmith.

## Common failures

| Symptom | Likely cause | What to do |
|---------|--------------|------------|
| Service exits at startup with a validation error | `GROQ_API_KEY` missing, or `API_KEYS` unset | Set the variable. For local work only, use `API_AUTH_DISABLED=true`. |
| `API_KEYS must look like 'user:key,...'` | Malformed key list | Fix the format. |
| `error` SSE events and rising `node_errors_total` | Groq outage, bad key, or a model ID that no longer exists | Check the application logs for the underlying error, then Groq's status page. |
| Rate-limit errors from Groq | Too much traffic for your Groq tier | The client retries `MAX_RETRIES` times with backoff. Lower `RATE_LIMIT_PER_MINUTE`, add capacity, or raise your Groq limits. |
| HTTP 429 from this service | A user exceeded `RATE_LIMIT_PER_MINUTE` | Expected behavior. The limit is per process, so replicas each count separately. |
| `classifier_fallbacks_total` climbing, most replies are `blended` | Classifier model failing or not supporting structured output | Check logs for "Classification failed", verify `CLASSIFIER_MODEL_NAME`. |
| Many messages classified `mixed` | Confidence scores are low or uncalibrated | Run the eval. Adjust `CLASSIFIER_CONFIDENCE_THRESHOLD` or the prompt. |
| Long chats become incoherent | History is trimmed to `MAX_CONTEXT_TOKENS` | Raise it (within the model's window) or add summarization. |
| `database is locked` or disk errors with persistence | SQLite file on slow or shared storage | Use local disk with a persistent volume. For multiple replicas, move to Postgres (not implemented). |
| Health check failing in Docker | App crashed at startup | `docker logs <container>` and read the first error. |

## Key rotation

- **Groq key:** create a new key in the Groq console, update `GROQ_API_KEY` in your secret store,
  restart the service, confirm requests succeed, then revoke the old key.
- **Service API keys:** `API_KEYS` allows several keys for one user, so rotate without downtime:
  set `alice:NEWKEY,alice:OLDKEY`, move the client to the new key, then remove the old pair.
  Keys are compared in constant time. Use long random values (for example `openssl rand -hex 24`).
- **After a suspected leak:** rotate immediately, then check logs for unexpected `thread_id`s.

## Data requests

- Delete one conversation: `DELETE /threads/{thread_id}` with that user's key, or
  `head-and-heart --delete-thread <id>` for the CLI.
- Delete everything: stop the service and remove the SQLite file.
- Remember that logs and any LangSmith traces may also hold conversation metadata or content.

## When the crisis reply fires

Crisis replies are fixed text, so wording does not drift. Watch `crisis_triggers_total` for
unexpected spikes (possible false positives from a new phrasing) and review misses reported by
users. To add a phrase, add a case to `tests/evals/safety_dataset.jsonl` first, then extend the
patterns in `src/head_and_heart/nodes/safety.py`.
