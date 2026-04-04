"""
tasks/apply_tasks.py — Celery apply worker.

Two tasks:
  prepare_application  — Generates cover letter, sets pending_review,
                         notifies dashboard via Redis pub/sub.
                         Mirrors cover_letter_generator_node from workflow.py.

  submit_application   — Called when user clicks Approve in dashboard.
                         Resumes LangGraph browser agent from Postgres checkpoint.
                         Mirrors application_filler_node from workflow.py.
"""

import asyncio
import json
import os
import psycopg2
import redis
from celery import shared_task
from dotenv import load_dotenv

load_dotenv()

DB_URL_SYNC = os.getenv("POSTGRES_URL_SYNC", "postgresql://jobagent:jobagent@localhost:5432/jobagent")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
POSTGRES_URL = os.getenv("POSTGRES_URL_SYNC", "postgresql://jobagent:jobagent@localhost:5432/jobagent")


def _get_job_full(job_id: str) -> dict | None:
    try:
        conn = psycopg2.connect(DB_URL_SYNC)
        cur = conn.cursor()
        cur.execute("""
            SELECT id, title, company, url, description, source,
                   match_score, top_matching_skills, missing_skills, recommendation
            FROM jobs WHERE id = %s
        """, (job_id,))
        row = cur.fetchone()
        cur.close(); conn.close()
        if not row:
            return None
        return {
            "id": row[0], "title": row[1], "company": row[2],
            "url": row[3], "description": row[4], "source": row[5],
            "match_score": row[6],
            "top_matching_skills": row[7] or [],
            "missing_skills": row[8] or [],
            "recommendation": row[9],
        }
    except Exception as e:
        print(f"  [Apply] DB read error: {e}")
        return None


def _set_status(job_id: str, status: str, extra: dict = None) -> None:
    try:
        conn = psycopg2.connect(DB_URL_SYNC)
        cur = conn.cursor()

        if extra and "cover_letter" in extra:
            cur.execute("""
                UPDATE jobs SET status = %s, cover_letter = %s,
                    cover_letter_subject = %s, updated_at = now()
                WHERE id = %s
            """, (status, extra["cover_letter"], extra.get("cover_letter_subject", ""), job_id))
        elif extra and "langgraph_thread_id" in extra:
            cur.execute("""
                UPDATE jobs SET status = %s, langgraph_thread_id = %s, updated_at = now()
                WHERE id = %s
            """, (status, extra["langgraph_thread_id"], job_id))
        elif extra and "error_message" in extra:
            cur.execute("""
                UPDATE jobs SET status = %s, error_message = %s, updated_at = now()
                WHERE id = %s
            """, (status, extra["error_message"][:500], job_id))
        else:
            cur.execute(
                "UPDATE jobs SET status = %s, updated_at = now() WHERE id = %s",
                (status, job_id)
            )

        conn.commit()
        cur.close(); conn.close()
    except Exception as e:
        print(f"  [Apply] DB set_status error: {e}")


def _notify_dashboard(event_type: str, payload: dict) -> None:
    """Push a real-time notification to the FastAPI WebSocket via Redis pub/sub."""
    try:
        r = redis.from_url(REDIS_URL)
        message = json.dumps({"type": event_type, **payload})
        r.publish("dashboard_updates", message)
    except Exception as e:
        print(f"  [Apply] Redis notify error: {e}")


def _load_profile() -> dict:
    try:
        with open("profile.json", "r") as f:
            return json.load(f)
    except Exception:
        return {}


async def _generate_cover_letter(job: dict) -> object:
    import sys
    if "/app" not in sys.path:
        sys.path.append("/app")
    
    from llm_client import LocalLLM
    import PyPDF2

    llm = LocalLLM()
    profile = _load_profile()

    resume_text = ""
    resume_path = profile.get("resume_path", "Nikhil_Resume.pdf")
    try:
        with open(resume_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            resume_text = "".join([p.extract_text() or "" for p in reader.pages])
    except Exception as e:
        print(f"  [Apply] Resume read error: {e}")

    return await llm.async_generate_cover_letter(job, resume_text, profile)


@shared_task(bind=True, queue="apply", max_retries=2, default_retry_delay=30)
def prepare_application(self, job_id: str):
    """
    Generate cover letter for an approved job.
    Sets status to pending_review and notifies the dashboard.
    User must click Approve in the dashboard to trigger submit_application.
    """
    print(f"  [Apply] Preparing application for job {job_id}")

    job = _get_job_full(job_id)
    if not job:
        print(f"  [Apply] Job {job_id} not found")
        return

    try:
        cover = asyncio.run(_generate_cover_letter(job))

        _set_status(job_id, "PENDING_REVIEW", {
            "cover_letter": cover.body,
            "cover_letter_subject": cover.subject_line,
        })

        _notify_dashboard("new_pending_review", {
            "job_id": job_id,
            "title": job["title"],
            "company": job["company"],
            "match_score": job["match_score"],
            "cover_letter_subject": cover.subject_line,
        })

        print(f"  [Apply] ✓ {job['title']} @ {job['company']} → pending_review")
        print(f"           Subject: {cover.subject_line}")

    except Exception as exc:
        print(f"  [Apply] Error preparing {job_id}: {exc}")
        _set_status(job_id, "FAILED", {"error_message": str(exc)})
        raise self.retry(exc=exc)


async def _run_browser_agent(job: dict, profile: dict) -> str:
    """
    Runs the LangGraph browser agent with Postgres checkpointing.
    Returns final status string.
    """
    import sys
    if "/app" not in sys.path:
        sys.path.append("/app")
        
    from browser_agent import BrowserAgent
    from safety.domain_whitelist import SafeBrowserController

    controller = SafeBrowserController(job_id=job["id"])

    # Domain whitelist check before even launching browser
    if not controller.check_navigation(job["url"]):
        return "DOMAIN_BLOCKED"

    agent = BrowserAgent(job_id=job["id"])

    # Get cover letter from DB
    conn = psycopg2.connect(DB_URL_SYNC)
    cur = conn.cursor()
    cur.execute("SELECT cover_letter FROM jobs WHERE id = %s", (job["id"],))
    row = cur.fetchone()
    cur.close(); conn.close()

    cover_letter = row[0] if row else ""

    result = await agent.fill_and_submit(
        job_url=job["url"],
        profile=profile,
        cover_letter=cover_letter,
        controller=controller,
    )

    return result


@shared_task(bind=True, queue="apply", max_retries=1, default_retry_delay=60)
def submit_application(self, job_id: str):
    """
    Called when user clicks Approve in the dashboard.
    Resumes LangGraph browser agent (from Postgres checkpoint if crashed).
    """
    print(f"  [Apply] Submitting application for job {job_id}")

    job = _get_job_full(job_id)
    if not job:
        return

    profile = _load_profile()
    _set_status(job_id, "APPROVED")

    try:
        result = asyncio.run(_run_browser_agent(job, profile))

        if result == "SUBMITTED":
            _set_status(job_id, "SUBMITTED")
            _notify_dashboard("application_submitted", {
                "job_id": job_id,
                "title": job["title"],
                "company": job["company"],
            })
            print(f"  [Apply] ✅ Application submitted: {job['title']} @ {job['company']}")

        elif result == "DOMAIN_BLOCKED":
            _set_status(job_id, "DOMAIN_BLOCKED")
            print(f"  [Apply] 🚫 Domain blocked: {job['url']}")

        else:
            _set_status(job_id, "FAILED", {"error_message": result})

    except Exception as exc:
        print(f"  [Apply] Error submitting {job_id}: {exc}")
        _set_status(job_id, "FAILED", {"error_message": str(exc)})
        raise self.retry(exc=exc)
