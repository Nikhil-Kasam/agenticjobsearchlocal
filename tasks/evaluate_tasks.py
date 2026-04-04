"""
tasks/evaluate_tasks.py — Celery evaluate worker.

Loads a job from Postgres, runs RAG retrieval + Instructor-enforced
LLM scoring, then enqueues prepare_application if score >= threshold.
Mirrors the job_evaluator_node logic from workflow.py.
"""

import asyncio
import os
import psycopg2
from celery import shared_task
from dotenv import load_dotenv

load_dotenv()

MATCH_THRESHOLD = int(os.getenv("MATCH_THRESHOLD", "60"))
DB_URL_SYNC = os.getenv("POSTGRES_URL_SYNC", "postgresql://jobagent:jobagent@localhost:5432/jobagent")


def _get_job(job_id: str) -> dict | None:
    try:
        conn = psycopg2.connect(DB_URL_SYNC)
        cur = conn.cursor()
        cur.execute(
            "SELECT id, title, company, url, description, source FROM jobs WHERE id = %s",
            (job_id,)
        )
        row = cur.fetchone()
        cur.close(); conn.close()
        if not row:
            return None
        return {
            "id": row[0], "title": row[1], "company": row[2],
            "url": row[3], "description": row[4], "source": row[5],
        }
    except Exception as e:
        print(f"  [Eval] DB read error: {e}")
        return None


def _save_evaluation(job_id: str, evaluation) -> None:
    """Persist JobEvaluation to Postgres and update status."""
    from db.models import JobStatus
    try:
        conn = psycopg2.connect(DB_URL_SYNC)
        cur = conn.cursor()

        import json
        status = (
            JobStatus.EVALUATED.value
            if evaluation.recommendation != "apply"
            else JobStatus.EVALUATED.value
        )

        cur.execute("""
            UPDATE jobs SET
                match_score = %s,
                top_matching_skills = %s,
                missing_skills = %s,
                recommendation = %s,
                eval_reasoning = %s,
                status = 'EVALUATED',
                updated_at = now()
            WHERE id = %s
        """, (
            evaluation.match_score,
            json.dumps(evaluation.top_matching_skills),
            json.dumps(evaluation.missing_skills),
            evaluation.recommendation,
            evaluation.reasoning,
            job_id,
        ))
        conn.commit()
        cur.close(); conn.close()
    except Exception as e:
        print(f"  [Eval] DB save error: {e}")


def _mark_failed(job_id: str, error: str) -> None:
    try:
        conn = psycopg2.connect(DB_URL_SYNC)
        cur = conn.cursor()
        cur.execute(
            "UPDATE jobs SET status = 'FAILED', error_message = %s, updated_at = now() WHERE id = %s",
            (error[:500], job_id)
        )
        conn.commit()
        cur.close(); conn.close()
    except Exception as e:
        print(f"  [Eval] DB mark_failed error: {e}")


async def _run_evaluation(job: dict) -> object:
    """Async evaluation: RAG retrieval + Instructor LLM scoring."""
    import sys, os
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")

    from database import VectorDBClient
    from llm_client import LocalLLM

    db = VectorDBClient()
    llm = LocalLLM()

    job_text = f"{job['title']} at {job['company']}: {job.get('description', '')}"

    # RAG: retrieve most relevant resume chunks
    resume_context = db.search_resume(job_text, k=4)

    # Instructor-enforced structured evaluation
    evaluation = await llm.async_score_match(job_text, resume_context)
    return evaluation


@shared_task(bind=True, queue="evaluate", max_retries=2, default_retry_delay=15)
def evaluate_job(self, job_id: str):
    """
    Evaluate a single job against the resume using RAG + LLM.
    Saves structured evaluation to Postgres.
    Enqueues prepare_application if recommendation is 'apply'.
    """
    from tasks.apply_tasks import prepare_application

    print(f"  [Eval] Evaluating job {job_id}")

    job = _get_job(job_id)
    if not job:
        print(f"  [Eval] Job {job_id} not found in DB")
        return

    try:
        evaluation = asyncio.run(_run_evaluation(job))
        _save_evaluation(job_id, evaluation)

        print(
            f"  [Eval] {job['title']} @ {job['company']}: "
            f"score={evaluation.match_score} recommendation={evaluation.recommendation}"
        )

        # Enqueue apply if above threshold AND recommended
        if evaluation.match_score >= MATCH_THRESHOLD and evaluation.recommendation == "apply":
            prepare_application.delay(job_id)
            print(f"  [Eval] → Enqueued prepare_application for job {job_id}")

    except Exception as exc:
        print(f"  [Eval] Error evaluating {job_id}: {exc}")
        _mark_failed(job_id, str(exc))
        raise self.retry(exc=exc)
