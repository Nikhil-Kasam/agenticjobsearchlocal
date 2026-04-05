# 🤖 Multi-Agent Job Search Orchestrator

<div align="center">

**A fully local, privacy-first AI agent that searches real job boards, scores listings against your resume with RAG, writes tailored cover letters, and auto-fills applications — pausing for your review before submitting.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Offline-orange?logo=ollama)](https://ollama.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Checkpointed-1C3C3C?logo=langchain)](https://langchain-ai.github.io/langgraph/)
[![Celery](https://img.shields.io/badge/Celery-Redis-37814A?logo=celery)](https://docs.celeryq.dev/)

</div>

---

## ✨ Features

| Feature | Details |
|---------|---------|
| 🔍 **Automated Scraping** | Searches 35+ ATS portals and direct company career pages via DuckDuckGo |
| 🧠 **RAG Evaluation** | Scores every job against your resume using ChromaDB retrieval + `qwen3-coder:30b` |
| 📋 **Structured Outputs** | [Instructor](https://python.useinstructor.com/) + Pydantic enforces valid JSON — no hallucinated scores |
| ✉️ **Cover Letter Generation** | Tailored, markdown-free cover letters (<300 words) for each qualifying job |
| ⏸️ **Human-in-the-Loop** | Dashboard approval gate before any application is submitted |
| 🌐 **Web Dashboard** | Real-time FastAPI + WebSocket dashboard (dark mode, live feed, Approve/Skip) |
| 🤖 **Browser Automation** | [browser-use](https://github.com/browser-use/browser-use) + Playwright fills and submits forms autonomously |
| ♻️ **Crash Recovery** | LangGraph + PostgreSQL checkpoints resume mid-application on worker restart |
| 🛡️ **Domain Whitelist** | 45+ allowlisted ATS/company domains — blocks LLM hallucinated navigation |
| ⚡ **Ollama Optimized** | Uses pre-quantized 30B models that fit effortlessly within a 24GB VRAM GPU with zero tuning |
| 🔒 **100% Local** | No data sent to OpenAI or Anthropic — everything runs on your own hardware |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Docker Compose Stack                            │
│                                                                     │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────────┐   │
│  │  FastAPI  │  │   Redis   │  │ Postgres  │  │    Flower     │   │
│  │  :8080    │  │   :6379   │  │   :5432   │  │    :5555      │   │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └───────────────┘   │
│        │              │              │                              │
│  ┌─────▼──────────────▼──────────────▼──────────────────────────┐  │
│  │              Celery Workers (3 queues)                        │  │
│  │  [search-worker]   [evaluate-worker×3]   [apply-worker]      │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
        ▲
        │  http://host.docker.internal:11434
        ▼
┌───────────────────────────────────────┐
│     Ollama Server (Host Machine)      │
│  LLM:  qwen3-coder:30b                │
│  Emb:  nomic-embed-text:latest        │
│  API:  OpenAI-Compatible endpoint     │
└───────────────────────────────────────┘
```

**Data flow:**
1. **Dashboard → Search worker** → DuckDuckGo scrapes 35+ job boards → Postgres
2. **Search worker → Evaluate worker** → RAG + Instructor LLM scoring → Postgres
3. **Score ≥ threshold** → cover letter generated → status: `pending_review`
4. **Dashboard "Pending Review"** → user clicks **Approve** → apply worker fires
5. **Apply worker** → LangGraph browser agent fills form → `submitted` in Postgres

---

## 🖥️ System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | 16 GB VRAM (RTX 4080) | 24 GB+ VRAM (RTX 3090/4090/5090) |
| **RAM** | 32 GB | 64 GB |
| **Disk** | 50 GB free | 100 GB SSD |
| **OS** | Ubuntu 22.04+ / WSL2 | Ubuntu 24.04 |
| **Docker** | Latest standard Docker Compose runtime | Latest |

---

## 🚀 Quick Start

### Step 1 — Start the Local LLM (Ollama)

Current deployment works with and extremely stable **Ollama** setup utilizing highly optimized `.gguf` quantizations(working with vllm too).

1. **Install Ollama**
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
2. **Expose Ollama to Docker Containers**
   By default, Ollama only listens on `127.0.0.1`. We need it to listen on all interfaces so the Docker Compose pipeline can communicate with it:
   ```bash
   sudo mkdir -p /etc/systemd/system/ollama.service.d
   echo -e "[Service]\nEnvironment=\"OLLAMA_HOST=0.0.0.0\"" | sudo tee /etc/systemd/system/ollama.service.d/override.conf
   sudo systemctl daemon-reload
   sudo systemctl restart ollama
   ```
3. **Pull Target Models**
   ```bash
   ollama pull qwen3-coder:30b
   ollama pull nomic-embed-text:latest
   ```

Verify Ollama is fully exposed:
```bash
curl http://localhost:11434/v1/models
# Will output: {"data":[{"id":"qwen3-coder:30b",...}, {"id":"nomic-embed-text:latest",...}]}
```

---

### Step 2 — Configure Your Profile

Edit these files before running anything:

**`profile.json`** — Your personal details for form auto-fill:
```json
{
  "name": "Your Full Name",
  "email": "you@example.com",
  "phone": "555-555-5555",
  "linkedin": "https://linkedin.com/in/yourprofile",
  "github": "https://github.com/yourusername",
  "city": "San Francisco", "state": "CA",
  "work_authorization": "US Citizen",
  "resume_path": "YourResume.pdf"
}
```

**`search_config.json`** — Job search criteria (already targets 35+ sources):
```json
{
  "job_titles_query": "(\"Machine Learning\" OR \"MLE\" OR \"AI Engineer\")",
  "match_threshold": 60,
  "auto_submit": false
}
```

Place your resume PDF in the project root and update `resume_path` in `profile.json`.

---

### Step 3 — Start the Application Stack

```bash
# Build and start all services (Redis, Postgres, FastAPI, Celery workers, Flower)
sudo docker compose up --build -d

# Check all containers are healthy
sudo docker compose ps
```

Expected output:
```
NAME                                   STATUS
jobagent_web                           Up (healthy)  → :8080
jobagent_redis                         Up (healthy)  → :6379
jobagent_postgres                      Up (healthy)  → :5432
jobagent_flower                        Up            → :5555
jobagent_search_worker                 Up
jobagent_apply_worker                  Up
agenticjobsearchlocal-celery-evaluate  Up
```

---

### Step 4 — Open the Dashboard

| Interface | URL | Purpose |
|-----------|-----|---------|
| **Job Dashboard** | http://localhost:8080 | Search, review, approve applications |
| **Flower (Celery)** | http://localhost:5555 | Monitor background worker queues |

Click **"Start Search Pipeline"** in the dashboard to begin. Jobs will flow through the pipeline automatically.

---

## 🐳 Docker Commands Reference

### Starting & Stopping

```bash
# Start everything
sudo docker compose up -d

# Stop all containers (data preserved)
sudo docker compose stop

# Stop + remove ALL data (full reset — WARNING: deletes job history and embeddings)
sudo docker compose down -v
sudo rm -rf ./data/postgres
sudo rm -rf ./chroma_db
```

### Monitoring

```bash
# View all container statuses
sudo docker compose ps

# Live logs from all services
sudo docker compose logs -f

# Logs for a specific service
sudo docker compose logs -f fastapi
sudo docker compose logs -f celery-search
sudo docker compose logs -f celery-evaluate
sudo docker compose logs -f celery-apply
```

---

## 📂 Project Structure

```
agenticjobsearchlocal/
│
├── 🐳 Docker & Infrastructure
│   ├── docker-compose.yml       # Full stack: Redis, Postgres, FastAPI, Celery, Flower
│   ├── Dockerfile               # Base image (FastAPI + search/evaluate workers)
│   └── Dockerfile.apply         # Apply worker image (includes Playwright browsers)
│
├── 🧠 AI Pipeline
│   ├── main.py                  # Standalone pipeline entry point (no Celery)
│   ├── supervisor.py            # Multi-agent parallel orchestrator
│   ├── workflow.py              # LangGraph state machine (node definitions)
│   ├── llm_client.py            # Ollama client + Instructor structured outputs
│   ├── browser_agent.py         # LangGraph browser agent + Postgres checkpointing
│   ├── database.py              # ChromaDB vector store + resume embeddings
│   └── agent_pool.py            # Async worker pool for parallel evaluation
│
├── ⚙️ Celery Task Queues
│   ├── celery_app.py            # Celery app config (3 queues: search/evaluate/apply)
│   └── tasks/
│       ├── search_tasks.py      # Queue 1: DuckDuckGo scraping → Postgres
│       ├── evaluate_tasks.py    # Queue 2: RAG scoring → Postgres
│       └── apply_tasks.py       # Queue 3: Cover letter + browser submission
│
├── 🗄️ Database Layer
│   └── db/
│       ├── models.py            # SQLAlchemy async ORM (Job, ApplicationAttempt)
│       └── session.py           # Async engine, session factory, init_db()
│
├── 🛡️ Safety
│   └── safety/
│       └── domain_whitelist.py  # 55+ allowed ATS/company domains + LinkedIn rule
│
├── 🌐 Web Dashboard
│   ├── web_app.py               # FastAPI app (HITL endpoints + WebSocket)
│   ├── templates/index.html     # Dashboard UI (dark mode, live feed, approve/skip)
│   └── static/                  # CSS/JS assets
│
├── ⚙️ Configuration
│   ├── profile.json             # Your personal info for form auto-fill
│   ├── search_config.json       # Job titles, locations, sources, match threshold
│   ├── .env                     # Environment variables (DB URL, OLLAMA URL, HF token)
│   └── requirements.txt         # Python dependencies
│
└── 📊 Data (gitignored)
    ├── data/postgres/           # Persistent Postgres volume
    └── chroma_db/               # ChromaDB vector store (resume embeddings)
```

---

## ⚙️ Configuration Reference

### `.env` — Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://redis:6379/0` | Celery broker URL |
| `POSTGRES_URL` | `postgresql+asyncpg://jobagent:jobagent@postgres:5432/jobagent` | Async DB URL |
| `POSTGRES_URL_SYNC` | `postgresql://jobagent:jobagent@postgres:5432/jobagent` | Sync DB URL (LangGraph) |
| `OLLAMA` | `http://host.docker.internal:11434/v1` | Ollama compatible OpenAI Endpoint |
| `LLM` | `qwen3-coder:30b` | Ollama model handle |
| `EMBEDDING_MODEL` | `nomic-embed-text:latest` | Dedicated 768d embedding model |
| `MATCH_THRESHOLD` | `60` | Minimum score (0–100) to trigger cover letter + apply |

### `search_config.json` — Search Criteria

| Field | Description |
|-------|-------------|
| `job_titles_query` | Boolean search string for job titles |
| `match_threshold` | Score (0–100) above which jobs are auto-queued for application |
| `auto_submit` | `false` = always pause for dashboard approval before submitting |
| `region_queries` | US regions to search across (West Coast, East Coast, etc.) |
| `job_sources` | List of ATS portals and company career sites to scrape |
| `max_results_per_source` | Results per source per search run |
| `search_delay_seconds` | Delay between searches (avoid rate limiting) |

---

## 🛡️ Safety Features

- **Domain Whitelist**: 45+ pre-approved ATS and company domains. The browser agent is blocked from navigating anywhere else — even if the LLM hallucinates a URL.
- **LinkedIn Scope**: Only `/jobs/` pages are accessible. The agent cannot access your profile, feed, or messages.
- **Human Gate**: `auto_submit: false` in `search_config.json` (default) ensures every application needs your explicit Approve click before submission.
- **Instructor Enforcement**: All LLM outputs for scoring and cover letters are validated against Pydantic schemas — malformed responses trigger automatic retries (max 3×) before falling back.

---

## 🐛 Troubleshooting

| Problem | Fix |
|---------|-----|
| `APIConnectionError('Connection error')` | Ollama is unreachable. Ensure you bound it to `0.0.0.0` as per Quick Start Step 1! |
| Dashboard returns 500 / Connection Refused | Ensure `REDIS_URL` matches `.env` precisely. Run `sudo docker compose restart` |
| Jobs not appearing after pipeline start | Check search worker logs: `sudo docker compose logs -f celery-search` |
| `Collection expecting embedding with dimension...` | Delete the old ChromaDB folder `sudo rm -rf chroma_db` completely & restart stack |
| Browser agent crashes mid-form | Restart the apply worker — LangGraph resumes from last Postgres checkpoint |

---

## 📖 Key Technologies

| Technology | Role | Link |
|-----------|------|------|
| **Ollama** | Ultra-efficient LLM host serving qwen3 and nomic models | [Docs](https://ollama.com/) |
| **qwen3-coder:30b** | Reasoning model (scoring, cover letters) | [HuggingFace](https://huggingface.co/Qwen) |
| **Instructor** | Pydantic-enforced structured LLM outputs | [Docs](https://python.useinstructor.com/) |
| **LangGraph** | Browser agent state machine + crash recovery | [Docs](https://langchain-ai.github.io/langgraph/) |
| **browser-use** | Chromium browser automation | [GitHub](https://github.com/browser-use/browser-use) |
| **Celery + Redis** | Distributed task queues (search/evaluate/apply) | [Docs](https://docs.celeryq.dev/) |
| **PostgreSQL** | Job state persistence + LangGraph checkpoints | [Official](https://www.postgresql.org/) |
| **ChromaDB** | Vector store for resume RAG | [Docs](https://docs.trychroma.com/) |
| **FastAPI** | Web dashboard + WebSocket API | [Docs](https://fastapi.tiangolo.com/) |

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---
