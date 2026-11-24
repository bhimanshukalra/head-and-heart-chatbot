# Privacy, retention and safety notes

## What is stored
- **Default:** conversations live in memory only and are gone when the process exits.
- **Opt-in:** with `PERSIST_HISTORY=true`, messages are written to the SQLite file at
  `CHECKPOINT_DB_PATH` (git-ignored). By default they are **unencrypted**: protect the file with
  disk encryption and file permissions.
- **Encryption at rest:** set `CHECKPOINT_ENCRYPTION_KEY` (16, 24 or 32 bytes) to AES-encrypt the
  stored checkpoints. Keep the key in a secret store. If it is lost the history cannot be read, and
  changing it makes older conversations unreadable. Only message content in checkpoints is
  covered, not the file's existence or size.

## Retention and deletion
- Stored conversations are kept until deleted, unless `RETENTION_DAYS` is set: the API then sweeps
  idle conversations every `RETENTION_CHECK_INTERVAL_S` seconds (and the CLI does so at startup),
  deleting any whose latest activity is older than that. With persistence off, this also stops
  in-memory conversations from growing without bound.
- Purge on demand: `head-and-heart --purge-older-than <days>`.
- Delete one conversation: `head-and-heart --delete-thread <thread-id>`.
- Delete everything: remove the SQLite file.

## Logs
- Message text is not logged at INFO. Log output is passed through a redactor that masks emails,
  phone-like numbers and long digit strings. Treat logs as potentially sensitive anyway.

## Data sent to the provider
- Message text is sent to Groq to generate replies and run the classifier and safety checks.
  Review Groq's current terms, data-retention and acceptable-use policies before any public launch,
  especially for mental-health-adjacent use. (Manual task, not automated.)

## Safety
- Explicit crisis statements are detected by regex and answered with a fixed template, with no LLM
  involved. Ambiguous wording is confirmed by an LLM; if that check fails, the crisis reply is used.
- Hotline numbers in `src/head_and_heart/crisis.py` must be verified for each supported region.
- This is an AI assistant, not a substitute for professional care. It is not a clinical product.
