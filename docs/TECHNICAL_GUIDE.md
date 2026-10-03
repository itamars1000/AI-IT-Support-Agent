# Technical Guide

## Local setup

Requirements: Docker with Compose. For local tests and evaluation scripts, use Python 3.12.

1. Copy `.env.example` to `.env` (`Copy-Item .env.example .env` in PowerShell, or `cp .env.example .env` in Bash).
2. Set `VOYAGE_API_KEY` and `ANTHROPIC_API_KEY` for document questions. The model defaults to the Claude Haiku model used in the recorded checks; change `ANTHROPIC_MODEL` for another model available to your account.
3. The example database credentials are for local development. If changing the password, update both `POSTGRES_PASSWORD` and `DATABASE_URL` consistently.
4. Start the app:

```powershell
docker compose up -d --build --wait app
```

The chat is at http://127.0.0.1:8001/ and Swagger at http://127.0.0.1:8001/docs. PostgreSQL is available locally on port 5433. Compose overrides the app's database host/port so it reaches the database container. Initialization applies the SQL migrations automatically; an existing database volume is retained.

## Load policy documents

1. Use `POST /documents` in Swagger to upload a PDF from [`evaluation/fixtures/pdfs`](../evaluation/fixtures/pdfs).
2. Embed the saved chunks:

```powershell
docker compose exec app python -m app.embed_chunks
```

Then ask a document question through the chat or `/agent`. Rebuild the app with the startup command after code changes; containers do not automatically copy changes from the checkout.

## Chat and API behavior

The chat sends ordinary questions to `/agent` with write actions disabled. Click a citation or source card to read the retrieved excerpt and page number. Ask the agent to open a ticket: its read-only `prepare_ticket` tool summarizes the issue and user-reported troubleshooting in a proposal card. Review it, optionally **Edit details**, then click **Confirm ticket** or reply **yes** / **כן** while that proposal is pending. **Cancel proposal** or **no** / **לא** dismisses it; a new conversation message makes an unconfirmed proposal inactive.

Confirmation sends `/agent` an `approved_ticket`, `allowed_actions: ["create_ticket"]` and an `idempotency_key` UUID. The agent invokes `create_ticket` with no field arguments; the server binds it to the reviewed payload and rejects model-supplied changes. The UI shows a receipt only from a successful creation tool result, including when the final LLM answer fails. Retries reuse the exact approved fields and UUID. If confirmation is lost, resolve that request before continuing or starting a new chat; page reloads discard the browser's retry identifier.

The **Still need help?** form remains available for manual ticket creation through `POST /tickets`, with the same UUID supplied in an `Idempotency-Key` header. That endpoint accepts requests without the optional header for existing clients. No ticket is saved by preparing or cancelling a proposal.

Follow-up questions use up to three recent completed question/answer pairs (at most 6,000 characters per message and 12,000 characters in total). A short reply such as "open it" or "תפתח" can prepare a ticket proposal using the issue and attempted steps already in that context; creation still requires reviewing and confirming the proposal. Older pairs are removed as the limits are reached. The browser keeps context in memory; **New chat** or a page reload clears it. Failed or incomplete exchanges are excluded, and retries reuse their original context. The server does not persist chat history. Keep the chat open if ticket creation needs a retry, so its request identifier can be reused.

Open **Request activity** beneath an answer or agent ticket receipt to inspect the recorded model calls and tools in execution order, durations, request ID, models, reported token usage and failures. Search steps include the actual search query and links to returned excerpts. The panel appears after a request finishes; it does not stream progress. `/agent` includes an optional `trace` with content-free metadata, also returned for partial workflows and model-provider errors. Traces are not stored in a new database table. Retrieval time includes query embedding and database search and is part of the search tool duration; these durations should not be added together. Processing time is measured before response serialization and excludes browser/network time. Token counts include the usage reported by the provider; unavailable values are shown explicitly. The existing structured logs continue to record request summaries.

API clients can include an optional `history` array in `/agent`, for example `{"message":"I restarted it; the same error persists.","history":[{"role":"user","content":"My VPN shows error 809."},{"role":"assistant","content":"Try restarting the VPN client."}]}`. Only complete user/assistant text pairs are accepted. History grants no write permissions; policy evidence is retrieved again and citation references are checked against the current request.

To try hybrid retrieval, send `{"question":"VPN connection procedure","retrieval_mode":"hybrid"}` to `/search` or `/rag`. Set `RETRIEVAL_MODE=hybrid` in `.env` and recreate the app to use it for all document searches, including the agent. Database initialization creates and updates the full-text search index automatically.

In hybrid results, `retrieval_score` determines the order; `distance` remains the cosine distance. `vector_rank` and `lexical_rank` show each branch's contribution.

## Tests

Create a Python environment and install the runtime and test dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

On Linux/macOS, use `.venv/bin/python` instead. The default command skips PostgreSQL integration tests. To include them, start the local database with Compose and run:

```powershell
$env:RUN_DATABASE_TESTS='1'
.\.venv\Scripts\python.exe -m app.init_db
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Database test fixtures use transactions that are rolled back. The tests mock external model/embedding calls and do not need API keys. GitHub Actions installs `requirements-dev.txt`, starts a disposable pgvector database, applies the migrations and runs the complete suite including database tests.

## Evaluation and scope

See the [evaluation report](../evaluation/REPORT.md) for recorded results, limits and reproduction steps. Evaluation scripts are separate from the automated test suite; provider-backed evaluation needs API keys and can consume quota. The corpus is synthetic and its runbooks are not vendor guidance.

This is a local demo API. Authentication and per-user authorization are not implemented. `allowed_actions` gates tools within an agent request; it is not a user identity or access-control system.
