# AI Assistant + Private RAG Document Intelligence (Multi-User)

A secure, multi-user, personal RAG (Retrieval-Augmented Generation) chatbot.
Every person who signs up gets their own account, their own uploaded
documents, their own vector search results, and their own private chat
history — one user can never see, search, or touch another user's data.

Built on:

- **FastAPI** + **Uvicorn** — backend/API
- **Groq** — LLM answer generation
- **Pinecone** — vector database (per-user namespaces)
- **Google Gemini Embedding API** — text embeddings
- **Tavily** — real-time web search
- **Pollinations.ai** — image generation (no API key required)
- **SQLite** (via SQLAlchemy) — user accounts, sessions, document metadata, chat history
- **Jinja2 + vanilla HTML/CSS/JS** — frontend

---

## 1. What changed from the single-user version

The original app worked, but had no accounts: every visitor shared one
Pinecone index and one on-disk JSON registry. This version adds:

| Area | Before | After |
|---|---|---|
| Accounts | None | Register/login/logout, bcrypt-hashed passwords |
| Identity | N/A | Server-issued session cookie; user id is **never** trusted from the frontend |
| Document storage | Shared registry.json + shared Pinecone index | Per-user rows in SQLite + **per-user Pinecone namespace** |
| RAG retrieval | Searched the whole index | Searches only the caller's own Pinecone namespace |
| Chat history | Client-side only (lost on refresh) | Persisted server-side per user, ownership-checked on every read/delete |
| Routes | All public | `/chat`, `/upload`, `/documents*`, `/chat-history*`, `/image/generate` require login |

RAG pipeline internals (PDF extraction → cleaning → chunking → embedding →
Pinecone → retrieval → Groq), the Tavily web-search integration, and the
image-generation feature are all unchanged in behavior — they're just now
wrapped with authentication and scoped to the logged-in user.

---

## 2. How authentication works

1. **Register** (`POST /api/auth/register`) — email + password (+ confirm).
   Password is hashed with **bcrypt** before it ever touches the database;
   the plain password is never stored or logged. Each user gets an opaque
   UUID as their internal id — the email is *not* used as the primary key.
2. **Login** (`POST /api/auth/login`) — verifies the password hash, creates
   a row in the `auth_sessions` table, and signs a JWT whose `jti` claim
   points at that row. The JWT is set as an **HttpOnly, SameSite=Lax**
   cookie — frontend JavaScript can never read or exfiltrate it via
   `document.cookie` or `localStorage`.
3. **Every protected request** — the `get_current_user` FastAPI dependency
   reads the cookie, verifies the JWT signature/expiry, and checks that the
   matching `auth_sessions` row still exists. Only then is the request
   allowed to proceed, using **that resolved user id** — never a `user_id`
   read from the request body, query string, or a header.
4. **Logout** (`POST /api/auth/logout`) — deletes the `auth_sessions` row
   server-side (so the token is dead immediately, not just once it
   naturally expires) and clears the cookie.
5. **Page protection** — visiting `/` while logged out gets a server-side
   302 redirect to `/login` before any chatbot HTML is ever sent. Browser
   back/forward navigation can't bypass this since it's enforced per-request
   on the server, not just hidden in JS.

## 3. How user-specific document isolation works

- **Pinecone namespace per user.** Every user's chunks are upserted to
  (and every query is restricted to) a Pinecone namespace derived from
  their user id (`user_<id>`). Namespaces give hard, engine-level
  isolation — a query scoped to one namespace cannot return vectors from
  another, even if a metadata filter were accidentally left off somewhere.
- **Document metadata in SQLite**, with an `owner_user_id` column. Every
  list/get/delete query filters on `owner_user_id == current_user.id`.
  If a document doesn't belong to you, the API returns `404 Not Found` —
  identical to the response for a document that doesn't exist at all, so
  you can't even confirm another user's document id is valid.
- **Chat sessions & messages** follow the exact same pattern
  (`owner_user_id` column, ownership-checked on every read/delete).
- **Uploads always use the authenticated user's id**, resolved server-side
  from the session cookie — the frontend never sends, and the backend
  never trusts, a client-supplied owner id.

---

## 4. Project structure

```
rag-bot/
├── app/
│   ├── main.py           # FastAPI app, routes (auth, chat, docs, history, image-gen)
│   ├── auth.py            # Password hashing, JWT, get_current_user dependency
│   ├── database.py        # SQLAlchemy engine/session setup
│   ├── db_models.py        # User, AuthSession, Document, ChatSession, ChatMessageRow
│   ├── config.py          # Settings (.env loading)
│   ├── models.py          # Pydantic request/response schemas
│   ├── rag.py              # RAG orchestration, now user-scoped + chat history
│   ├── pinecone_db.py      # Pinecone client, namespaced per user
│   ├── embeddings.py       # Gemini embeddings
│   ├── groq_client.py      # Groq LLM calls
│   ├── web_search.py       # Tavily web search
│   ├── image_gen.py        # Pollinations.ai image generation
│   ├── pdf_processor.py    # PDF text extraction
│   └── chunking.py         # Text chunking
├── templates/
│   ├── index.html          # Main chatbot UI (protected)
│   ├── login.html          # Login page
│   └── register.html       # Registration page
├── static/
│   ├── script.js           # Main app frontend logic
│   ├── auth.js              # Shared login/register form handler
│   └── style.css
├── data/                    # Local SQLite DB + upload scratch dir (gitignored)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 5. Setup instructions

### 5.1 Prerequisites

- Python 3.11+
- A [Groq](https://console.groq.com/keys) API key
- A [Pinecone](https://app.pinecone.io) API key + a serverless index
- A [Google Gemini](https://aistudio.google.com/apikey) API key (for embeddings)
- (Optional) A [Tavily](https://tavily.com) API key for Web Search mode

### 5.2 Install

```bash
cd rag-bot
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 5.3 Configure environment variables

```bash
cp .env.example .env
```

Then edit `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=rag-chatbot-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

GEMINI_API_KEY=your_gemini_api_key_here
EMBEDDING_MODEL=gemini-embedding-001

TAVILY_API_KEY=your_tavily_api_key_here     # optional — Web Search mode

# REQUIRED — generate with: openssl rand -hex 32
SECRET_KEY=replace_with_a_long_random_value
```

**Never commit your real `.env` file.** Only `.env.example` (with empty/
placeholder values) belongs in version control — this is already handled
by `.gitignore`.

#### Pinecone setup
Create a serverless index in the [Pinecone console](https://app.pinecone.io)
matching your embedding model's output dimension (Gemini `gemini-embedding-001`
defaults to 3072 dimensions — check the model card if you change it), and
set `PINECONE_INDEX_NAME`, `PINECONE_CLOUD`, and `PINECONE_REGION` to match.
No manual namespace setup is needed — the app creates a namespace per user
automatically the first time they upload a document.

#### Groq setup
Create an API key at [console.groq.com](https://console.groq.com/keys) and
set `GROQ_API_KEY`. `GROQ_MODEL` can be any currently available Groq-hosted
chat model.

#### Tavily (web search) setup
Create a free API key at [tavily.com](https://tavily.com) and set
`TAVILY_API_KEY`. If left blank, Web Search mode returns a clear error but
the rest of the app (General AI + your documents' RAG + image generation)
keeps working normally.

#### Image generation
Uses [Pollinations.ai](https://pollinations.ai), which requires **no API
key** — it works out of the box. `IMAGE_GEN_MODEL` in `.env` selects the
underlying model (`flux` for quality, `turbo` for speed).

### 5.4 Run the app

```bash
uvicorn app.main:app --reload
```

Then open **http://localhost:8000** — you'll be redirected to `/login`.
Click "Create one" to register your first account.

---

## 6. Creating an account & how it feels day-to-day

1. Go to `/register`, enter an email + password (min. 8 characters) + confirm.
2. You're immediately logged in and redirected to the main chatbot.
3. Upload PDFs — they're indexed under your account only.
4. Ask questions in **Auto**, **General AI**, **Document (RAG)**, or **Web
   Search** mode.
5. Your conversations appear in the **Chats** list in the sidebar; click one
   to reopen it, or **+ New Chat** to start fresh.
6. **Log out** clears your session both in the browser and on the server.
7. Log back in any time — your documents and chat history are exactly as
   you left them, and still invisible to any other account.

---

## 7. Database

By default the app uses a local SQLite file at `data/app.db` (created
automatically on first run — no manual migration step needed). To use
Postgres/MySQL instead in production, just point `DATABASE_URL` at it, e.g.:

```env
DATABASE_URL=postgresql://user:password@host:5432/dbname
```

and install the matching driver (e.g. `psycopg2-binary`) — the SQLAlchemy
models themselves are database-agnostic.

---

## 8. Security notes

- Passwords: bcrypt-hashed, never logged or returned by any endpoint.
- Sessions: JWT + HttpOnly/SameSite cookie, individually revocable server-side.
- Authorization: every document/chat-history lookup is filtered by
  `owner_user_id == current_user.id` at the database layer, and by
  Pinecone namespace at the vector-search layer — not just hidden in the UI.
- Errors: internal exceptions are logged server-side but never returned to
  the client with stack traces or secret values.
- Secrets: `SECRET_KEY`, `GROQ_API_KEY`, `PINECONE_API_KEY`,
  `GEMINI_API_KEY`, and `TAVILY_API_KEY` all stay server-side in `.env` and
  are never sent to, or readable from, the frontend.
- Set `COOKIE_SECURE=true` once you deploy behind HTTPS.

## 9. Manually verifying the isolation guarantees

1. Register **User A**, upload `A.pdf`, ask a question about it — confirm
   you get an answer grounded in `A.pdf`.
2. Log out. Register **User B**. Confirm the documents list is empty and a
   question that would only be answerable from `A.pdf` returns "couldn't
   find relevant information," not an answer from `A.pdf`.
3. Upload `B.pdf` as User B, confirm you can retrieve only `B.pdf`.
4. Open your browser dev tools and try changing a document id or chat
   session id in a request — every such attempt returns `404 Not Found`,
   whether the id belongs to another user or doesn't exist at all.
5. Confirm User A's chat history never appears while logged in as User B.
