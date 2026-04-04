"""
tasks/search_tasks.py — Celery search worker.

Reuses the DuckDuckGo search logic from main.py.
For each newly discovered job, upserts to Postgres and
enqueues an evaluate task.
"""

import asyncio
import json
import os
import random
import uuid
import urllib.parse
from datetime import datetime
from celery import shared_task
from ddgs import DDGS
from dotenv import load_dotenv

load_dotenv()

# ── Junk filters (from main.py) ───────────────────────────────────────────────
JUNK_DOMAINS = [
    "bing.com", "google.com/aclick", "doubleclick.net",
    "forbes.com", "coursiv.io", "interviewkickstart.com",
    "professional-education-gl.mit.edu", "bamboohr.com/blog",
    "bamboohr.com/resources", "bamboohr.com/legal",
    "bamboohr.com/careers", "bamboohr.com/job-description",
]

JUNK_TITLE_KEYWORDS = [
    "certification program", "certification course",
    "14-week data science", "top ai certification",
    "best ai certificate", "top 7 ai courses",
    "artificial intelligence class", "start learning today",
    "flexible learning schedule", "trusted by 20000",
    "machine learning classes", "machine learning course online",
    "online machine learning course", "learn in-demand ai",
]


def _is_valid_job(url: str, title: str) -> bool:
    url_lower = url.lower()
    title_lower = title.lower()
    if "bing.com/aclick" in url_lower or "doubleclick.net" in url_lower:
        return False
    for jd in JUNK_DOMAINS:
        if jd in url_lower:
            return False
    for kw in JUNK_TITLE_KEYWORDS:
        if kw in title_lower:
            return False
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.strip("/")
    if not path or path in ("jobs", "careers", "search", ""):
        return False
    return len(url) >= 20


def _extract_company(url: str, title: str = "") -> str:
    try:
        domain = urllib.parse.urlparse(url).netloc
        path = urllib.parse.urlparse(url).path.strip("/").split("/")
        if "greenhouse.io" in domain and path:
            return path[0].replace("-", " ").title()
        if "lever.co" in domain and path:
            return path[0].replace("-", " ").title()
        if "ashbyhq.com" in domain and path:
            return path[0].replace("-", " ").title()
        if "myworkdayjobs" in domain:
            return domain.split(".")[0].replace("-", " ").title()
        if " at " in title:
            return title.split(" at ")[-1].strip()
        if " - " in title:
            parts = title.split(" - ")
            return parts[-1].strip() if len(parts) > 1 else parts[0]
        return domain
    except Exception:
        return "Unknown"


def _upsert_job_sync(job_data: dict) -> str | None:
    """
    Synchronous Postgres upsert (Celery tasks run in sync context).
    Returns the job id if inserted (new job), None if already exists.
    """
    import psycopg2
    from psycopg2.extras import Json

    db_url = os.getenv(
        "POSTGRES_URL_SYNC",
        "postgresql://jobagent:jobagent@localhost:5432/jobagent"
    )

    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()

        job_id = str(uuid.uuid4())

        cur.execute("""
            INSERT INTO jobs (id, title, company, url, description, source, scraped_at, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'FOUND')
            ON CONFLICT (url) DO NOTHING
            RETURNING id
        """, (
            job_id,
            job_data["title"],
            job_data["company"],
            job_data["url"],
            job_data.get("description", ""),
            job_data.get("source", "ddgs"),
            datetime.now(),
        ))

        row = cur.fetchone()
        conn.commit()
        cur.close()
        conn.close()

        return row[0] if row else None  # None means URL was duplicate
    except Exception as e:
        print(f"  [Search] DB upsert error: {e}")
        return None


@shared_task(bind=True, queue="search", max_retries=3, default_retry_delay=30)
def run_search_pipeline(self, config: dict):
    """
    Main search task. Runs all DuckDuckGo queries from search_config.json,
    upserts new jobs to Postgres, and enqueues evaluate tasks.
    """
    from tasks.evaluate_tasks import evaluate_job

    title_query = config.get("job_titles_query", '("Machine Learning" OR "MLE" OR "AI Engineer")')
    region_queries = config.get("region_queries", {})
    sources = config.get("job_sources", [])
    delay = max(config.get("search_delay_seconds", 5), 15)

    all_regions_query = " OR ".join(region_queries.values())
    combined_location = f"({all_regions_query})"

    ddgs = DDGS()
    total_new = 0

    for n, source in enumerate(sources, 1):
        query = f'{source["site_query"]} {title_query} {combined_location}'
        print(f"  [{n}/{len(sources)}] Searching: {source['name']}")

        for attempt in range(3):
            try:
                results = ddgs.text(query, timelimit="w") or []
                break
            except Exception as e:
                msg = str(e)
                if "429" in msg or "403" in msg or "RatelimitException" in msg:
                    wait = 30 * (2 ** attempt)
                    print(f"    ⚠ Rate limited — waiting {wait}s")
                    import time; time.sleep(wait)
                else:
                    results = []
                    break

        for r in results:
            url = r.get("href", "")
            title = r.get("title", "Unknown")
            if not _is_valid_job(url, title):
                continue

            job_data = {
                "title": title,
                "url": url,
                "description": r.get("body", ""),
                "company": _extract_company(url, title),
                "source": source["name"],
            }

            job_id = _upsert_job_sync(job_data)
            if job_id:
                # New job — enqueue for evaluation
                evaluate_job.delay(job_id)
                total_new += 1

        if n < len(sources):
            import time
            time.sleep(delay + random.uniform(1, 5))

    print(f"  ✓ Search complete: {total_new} new jobs queued for evaluation")
    return {"total_new_jobs": total_new}
