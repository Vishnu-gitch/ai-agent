# AI Agent Studio

A simple AI agent with a FastAPI backend (GitHub Models / OpenAI-compatible API) and a static HTML/JS frontend.

## Security fixes applied (2026-07-01)
- Removed the real `backend/.env` file containing a live token from the repo (it must **never** be committed — only `.env.example` is tracked).
- Replaced the unrestricted `eval()` calculator with a safe AST-based parser that only allows numbers and basic math operators.
- Added rate limiting (10 requests/minute per IP) on `/api/chat` and `/api/test` to prevent abuse of your API key.
- Removed internal debug fields (`github_token_set`, etc.) from public API responses.
- Fixed CORS to use an explicit allow-list (`ALLOWED_ORIGINS` env var) plus a regex for Netlify preview URLs, instead of a non-functional wildcard.
- Added input length validation on chat messages.

## Project structure
```
AI-AGENT/
├── backend/          # FastAPI app -> deploy to Railway
│   ├── agent.py
│   ├── requirements.txt
│   ├── Procfile
│   ├── runtime.txt
│   └── .env.example
└── frontend/         # Static site -> deploy to Netlify
    └── index.html
```

## Local development
```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # then edit .env and add your real GITHUB_TOKEN
uvicorn agent:app --reload
```

## Deployment
See the step-by-step guide provided separately. In short:
1. Push this repo to GitHub.
2. Deploy `backend/` to Railway, set `GITHUB_TOKEN` and `ALLOWED_ORIGINS` as environment variables there (never in code).
3. Deploy `frontend/` to Netlify.
4. Update `API_URL` in `frontend/index.html` to your Railway backend URL, then redeploy the frontend.

## Rotating your token
If you ever suspect a key has leaked (e.g. committed to git, shared in a chat, pasted somewhere public), revoke it immediately at
GitHub → Settings → Developer settings → Personal access tokens, then generate a new one and update it only as an environment variable on Railway.
