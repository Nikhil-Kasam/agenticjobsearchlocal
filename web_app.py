"""
web_app.py — FastAPI web dashboard with HITL approval endpoints + WebSocket.

Endpoints:
  GET  /                         — Dashboard HTML
  GET  /api/jobs/found           — All found jobs (from Postgres)
  GET  /api/jobs/evaluated       — All evaluated jobs sorted by score
  GET  /api/jobs/pending         — Jobs awaiting user review
  POST /api/jobs/{id}/approve    — Approve: triggers submit_application Celery task
  POST /api/jobs/{id}/skip       — Skip: marks job as skipped
  POST /api/pipeline/start       — Start a new search pipeline run
  GET  /api/pipeline/status      — Current Celery queue depths
  WS   /ws/updates               — Real-time updates via Redis pub/sub
"""

import asyncio
import json
import os
import psycopg2
import redis.asyncio as aioredis
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from celery_app import celery_app

load_dotenv()

DB_URL = os.getenv("POSTGRES_URL_SYNC", "postgresql://jobagent:jobagent@localhost:5432/jobagent")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


# ── Startup: init DB tables ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    from db.session import init_db
    await init_db()
    yield


app = FastAPI(title="AI Job Application Dashboard", lifespan=lifespan)

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


# ── Database helpers (sync psycopg2 for simplicity in FastAPI routes) ─────────

def _query(sql: str, params=()) -> list[dict]:
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        cur.execute(sql, params)
        cols = [desc[0] for desc in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        cur.close(); conn.close()
        # Serialize datetime objects
        for row in rows:
            for k, v in row.items():
                if hasattr(v, "isoformat"):
                    row[k] = v.isoformat()
                elif isinstance(v, (list, dict)):
                    pass  # already serializable
        return rows
    except Exception as e:
        print(f"  [WebApp] DB query error: {e}")
        return []


def _execute(sql: str, params=()):
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
        cur.close(); conn.close()
    except Exception as e:
        print(f"  [WebApp] DB execute error: {e}")


# ── WebSocket manager ──────────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active = [w for w in self.active if w != ws]

    async def broadcast(self, message: str):
        disconnected = []
        for ws in self.active:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.disconnect(ws)


manager = ConnectionManager()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/jobs/found")
async def get_jobs_found():
    jobs = _query("SELECT * FROM jobs ORDER BY scraped_at DESC")
    return jobs


@app.get("/api/jobs/evaluated")
async def get_jobs_evaluated():
    jobs = _query(
        "SELECT * FROM jobs WHERE status != 'FOUND' ORDER BY match_score DESC NULLS LAST"
    )
    return jobs


@app.get("/api/jobs/pending")
async def get_jobs_pending():
    """Jobs waiting for human approval in the dashboard."""
    jobs = _query(
        "SELECT * FROM jobs WHERE status = 'PENDING_REVIEW' ORDER BY match_score DESC"
    )
    return jobs


@app.post("/api/jobs/{job_id}/approve")
async def approve_job(job_id: str):
    """User clicked Approve — trigger browser agent submission."""
    from tasks.apply_tasks import submit_application

    _execute("UPDATE jobs SET status = 'APPROVED', updated_at = now() WHERE id = %s", (job_id,))
    submit_application.delay(job_id)

    await manager.broadcast(json.dumps({
        "type": "job_approved",
        "job_id": job_id,
    }))
    return {"status": "approved", "job_id": job_id}


@app.post("/api/jobs/{job_id}/skip")
async def skip_job(job_id: str):
    """User clicked Skip — mark as skipped."""
    _execute("UPDATE jobs SET status = 'SKIPPED', updated_at = now() WHERE id = %s", (job_id,))
    await manager.broadcast(json.dumps({
        "type": "job_skipped",
        "job_id": job_id,
    }))
    return {"status": "skipped", "job_id": job_id}


@app.post("/api/pipeline/start")
async def start_pipeline():
    """Trigger a fresh search pipeline run from search_config.json."""
    from tasks.search_tasks import run_search_pipeline

    try:
        with open("search_config.json", "r") as f:
            config = json.load(f)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Config read error: {e}"})

    task = run_search_pipeline.delay(config)
    return {"status": "search_started", "task_id": task.id}


@app.get("/api/pipeline/status")
async def pipeline_status():
    """Get current job counts by status."""
    counts = _query("""
        SELECT status, COUNT(*) as count
        FROM jobs
        GROUP BY status
        ORDER BY status
    """)
    return {row["status"]: row["count"] for row in counts}


@app.get("/api/stats")
async def get_stats():
    """Dashboard stats summary."""
    rows = _query("""
        SELECT
            COUNT(*) FILTER (WHERE status = 'FOUND') as found,
            COUNT(*) FILTER (WHERE status = 'EVALUATED') as evaluated,
            COUNT(*) FILTER (WHERE status = 'PENDING_REVIEW') as pending,
            COUNT(*) FILTER (WHERE status = 'SUBMITTED') as submitted,
            COUNT(*) FILTER (WHERE status = 'SKIPPED') as skipped,
            COUNT(*) FILTER (WHERE status = 'FAILED') as failed,
            AVG(match_score) FILTER (WHERE match_score >= 0) as avg_score
        FROM jobs
    """)
    return rows[0] if rows else {}


# ── WebSocket: Real-time dashboard updates ────────────────────────────────────

@app.websocket("/ws/updates")
async def websocket_updates(websocket: WebSocket):
    """
    Subscribes to Redis 'dashboard_updates' channel.
    Broadcasts messages to connected dashboard clients.
    """
    await manager.connect(websocket)
    r = aioredis.from_url(REDIS_URL)
    pubsub = r.pubsub()
    await pubsub.subscribe("dashboard_updates")

    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                data = message["data"]
                if isinstance(data, bytes):
                    data = data.decode()
                await manager.broadcast(data)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"  [WS] Error: {e}")
        manager.disconnect(websocket)
    finally:
        await pubsub.unsubscribe("dashboard_updates")
        await r.aclose()


# ── Legacy JSON file endpoints (backward compat) ───────────────────────────────

@app.get("/api/legacy/found")
async def legacy_found():
    """Read from jobs_found.json if it exists (backward compat)."""
    try:
        with open("jobs_found.json", "r") as f:
            jobs = json.load(f)
        jobs.sort(key=lambda x: x.get("scraped_at", ""), reverse=True)
        return jobs
    except Exception:
        return []


@app.get("/api/legacy/evaluated")
async def legacy_evaluated():
    """Read from jobs_evaluated.json if it exists (backward compat)."""
    try:
        with open("jobs_evaluated.json", "r") as f:
            return json.load(f)
    except Exception:
        return []


if __name__ == "__main__":
    import uvicorn
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    print("Starting Job Application Dashboard on http://127.0.0.1:8080")
    print("(Note: port 8080 avoids conflict with vLLM server on port 8000)")
    uvicorn.run("web_app:app", host="127.0.0.1", port=8080, reload=True)
