"""
main.py — Job search and evaluation pipeline.

Stage 1 SEARCH: Uses ddgs (DuckDuckGo Search) → saves to jobs_found.json
Stage 2 EVALUATE: Async LLM + RAG scores jobs concurrently → saves to jobs_evaluated.json
(Stage 3 APPLY is separate — run only after reviewing the JSON files)
"""

import asyncio
import json
import os
import random
import urllib.parse
import logging
import PyPDF2
from datetime import datetime
from ddgs import DDGS

from database import VectorDBClient
from llm_client import LocalLLM
from agent_pool import WorkerPool, AgentTask, TaskStatus

# Suppress noisy logs
logging.getLogger("browser_use").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

JOBS_FOUND_FILE = "jobs_found.json"
JOBS_EVALUATED_FILE = "jobs_evaluated.json"


def extract_company(url: str, title: str = "") -> str:
    """Extract company name from job portal URL."""
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
        if "careers" in domain:
            parts = domain.split(".")
            return parts[1].title() if len(parts) > 1 else domain

        if " at " in title:
            return title.split(" at ")[-1].strip()
        if " - " in title:
            parts = title.split(" - ")
            return parts[-1].strip() if len(parts) > 1 else parts[0]

        return domain
    except:
        return "Unknown"


# ─── Junk URL / title filter ───
JUNK_DOMAINS = [
    "bing.com", "google.com/aclick", "doubleclick.net",
    "forbes.com", "coursiv.io", "interviewkickstart.com",
    "professional-education-gl.mit.edu", "bamboohr.com/blog",
    "bamboohr.com/resources", "bamboohr.com/legal",
    "bamboohr.com/careers", "bamboohr.com/job-description",
]

JUNK_TITLE_KEYWORDS = [
    "certification program", "certification course", "certification programs",
    "14-week data science", "top ai certification", "best ai certificate",
    "top 7 ai courses", "artificial intelligence class",
    "start learning today", "flexible learning schedule",
    "trusted by 20000", "machine learning classes",
    "machine learning course online", "online machine learning course",
    "learn in-demand ai",
]


def is_valid_job(url: str, title: str) -> bool:
    """Filter out ads, courses, and non-job pages."""
    url_lower = url.lower()
    title_lower = title.lower()

    if "bing.com/aclick" in url_lower:
        return False
    if "doubleclick.net" in url_lower:
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

    if not url or len(url) < 20:
        return False

    return True


async def search_one(ddgs: DDGS, query: str) -> list:
    """Run one ddgs search with retry on 429/403. No result limit."""
    for attempt in range(3):
        try:
            results = ddgs.text(query, timelimit="w")
            return results or []
        except Exception as e:
            msg = str(e)
            if "429" in msg or "403" in msg or "RatelimitException" in msg:
                wait = 30 * (2 ** attempt)
                print(f"\n    ⚠ Rate limited. Waiting {wait}s...")
                await asyncio.sleep(wait)
            else:
                raise
    return []


async def search_jobs(config: dict) -> list[dict]:
    """
    Search using ddgs — combines all regions into one query per source.
    No result count limits. Saves progressively to jobs_found.json.
    """
    title_query = config["job_titles_query"]
    region_queries = config["region_queries"]
    sources = config["job_sources"]
    delay = max(config.get("search_delay_seconds", 5), 15)

    all_regions_query = " OR ".join(region_queries.values())
    combined_location = f"({all_regions_query})"

    total = len(sources)
    all_jobs = []

    ddgs = DDGS()

    for n, source in enumerate(sources, 1):
        query = f'{source["site_query"]} {title_query} {combined_location}'
        print(f"  [{n}/{total}] {source['name']}", end="", flush=True)

        try:
            results = await search_one(ddgs, query)

            if results:
                added = 0
                for r in results:
                    url = r.get("href", "")
                    title = r.get("title", "Unknown")
                    if is_valid_job(url, title):
                        all_jobs.append({
                            "title": title,
                            "url": url,
                            "description": r.get("body", ""),
                            "company": extract_company(url, title),
                            "source": source["name"],
                            "scraped_at": datetime.now().isoformat()
                        })
                        added += 1
                print(f" → {added}/{len(results)} valid")
            else:
                print(f" → 0")

        except Exception as e:
            print(f" ✗ {type(e).__name__}: {str(e)[:60]}")

        _save_deduped(all_jobs, JOBS_FOUND_FILE)

        if n < total:
            await asyncio.sleep(delay + random.uniform(1, 5))

    unique = _deduplicate(all_jobs)
    _save_json(unique, JOBS_FOUND_FILE)
    return unique


def _deduplicate(jobs: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for j in jobs:
        u = j.get("url", "")
        if u and u not in seen:
            seen.add(u)
            unique.append(j)
    return unique


def _save_deduped(jobs: list[dict], filepath: str):
    unique = _deduplicate(jobs)
    _save_json(unique, filepath)


def _save_json(data: list[dict], filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


async def main():
    print("╔══════════════════════════════════════════════════╗")
    print("║  🤖 AI Job Application Automator                ║")
    print("║  ddgs Search · Qwen 2.5 Coder 32B · CUDA       ║")
    print("╚══════════════════════════════════════════════════╝\n")

    # ─── Load Config ───
    with open("search_config.json", "r") as f:
        config = json.load(f)
    print("✓ search_config.json")

    with open("profile.json", "r") as f:
        profile = json.load(f)
    print(f"✓ profile.json ({profile.get('name')})")

    # ─── Resume + Vector DB ───
    resume_path = profile.get("resume_path", "Nikhil_Resume.pdf")
    db = None
    if os.path.exists(resume_path):
        with open(resume_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            resume_text = "".join([p.extract_text() or "" for p in reader.pages])
        print(f"✓ Resume: {len(resume_text)} chars")
        db = VectorDBClient()
        db.ingest_pdf(resume_path)
    else:
        print(f"⚠ Resume not found at {resume_path}")
        return

    llm = LocalLLM()

    # ═══════════════════════════════════════════
    #  STAGE 1: SEARCH → jobs_found.json
    # ═══════════════════════════════════════════
    print(f"\n{'═'*55}")
    print(f"  STAGE 1: SEARCH (ddgs — no browser, no CAPTCHA)")
    print(f"  Output: {JOBS_FOUND_FILE}")
    print(f"{'═'*55}\n")

    unique_jobs = await search_jobs(config)
    print(f"\n  📊 {len(unique_jobs)} unique jobs found")
    print(f"  💾 Saved to {JOBS_FOUND_FILE}")

    if not unique_jobs:
        print("  ⚠ No jobs found. Adjust search_config.json.")
        return

    # ═══════════════════════════════════════════
    #  STAGE 2: EVALUATE → jobs_evaluated.json
    #  Uses RAG (vector DB) + async WorkerPool
    # ═══════════════════════════════════════════
    threshold = config.get("match_threshold", 60)
    print(f"\n{'═'*55}")
    print(f"  STAGE 2: EVALUATE ({len(unique_jobs)} jobs, RAG + async)")
    print(f"  Output: {JOBS_EVALUATED_FILE}")
    print(f"{'═'*55}\n")

    # Build evaluation tasks
    eval_tasks = []
    for i, job in enumerate(unique_jobs):
        eval_tasks.append(AgentTask(
            task_id=f"eval-{i:04d}",
            task_type="evaluate",
            payload={
                "job_index": i,
                "job": job,
            },
            max_attempts=2,
        ))

    # Define the async handler: RAG retrieval + async LLM scoring
    async def evaluate_job(payload: dict) -> dict:
        job = payload["job"]
        job_text = f"{job['title']} at {job['company']}: {job.get('description', '')}"

        # RAG: retrieve relevant resume chunks via vector similarity
        resume_context = db.search_resume(job_text, k=3)

        # Async LLM scoring
        score = await llm.async_score_match(job_text, resume_context)

        return {"job_index": payload["job_index"], "score": score}

    # Run all evaluations concurrently via WorkerPool
    pool = WorkerPool(max_workers=3)
    reports = await pool.run_all(eval_tasks, handler=evaluate_job)

    # Merge scores back into jobs
    score_map = {}
    succeeded = 0
    failed_tasks = 0
    for r in reports:
        if r.status == TaskStatus.SUCCESS and r.result:
            score_map[r.result["job_index"]] = r.result["score"]
            succeeded += 1
        else:
            failed_tasks += 1
            err = r.error if r else "no report"
            print(f"  ❌ Task {r.task_id if r else '?'} failed: {err}")

    print(f"\n  📈 WorkerPool stats: {succeeded} succeeded, {failed_tasks} failed, {len(score_map)} scores collected")

    all_evaluated = []
    for i, job in enumerate(unique_jobs):
        raw_score = score_map.get(i, -1)
        job["match_score"] = raw_score if raw_score >= 0 else 0  # -1 → 0 (failed eval)
        tag = "✓" if job["match_score"] >= threshold else "·"
        print(f"  [{i+1}/{len(unique_jobs)}] {tag} {job['match_score']:>3} | {job['title'][:35]} @ {job['company'][:15]}")
        all_evaluated.append(job)

    # Sort by score, save ALL
    all_evaluated.sort(key=lambda x: x["match_score"], reverse=True)
    _save_json(all_evaluated, JOBS_EVALUATED_FILE)

    qualified = [j for j in all_evaluated if j["match_score"] >= threshold]

    print(f"\n  📊 {len(all_evaluated)} jobs evaluated, {len(qualified)} above threshold (≥{threshold})")
    print(f"  💾 Saved ALL {len(all_evaluated)} jobs to {JOBS_EVALUATED_FILE}")

    # ═══════════════════════════════════════════
    #  DONE
    # ═══════════════════════════════════════════
    print(f"\n{'═'*55}")
    print(f"  ✅ COMPLETE")
    print(f"{'═'*55}")
    print(f"  📄 {JOBS_FOUND_FILE:30s} → {len(unique_jobs)} unique jobs")
    print(f"  📄 {JOBS_EVALUATED_FILE:30s} → {len(all_evaluated)} scored jobs")
    print(f"  🎯 Jobs above threshold (≥{threshold}): {len(qualified)}")
    print(f"\n  Top 10 matches:")
    for i, j in enumerate(all_evaluated[:10], 1):
        print(f"    {i:>2}. [{j['match_score']:>3}] {j['title'][:35]} @ {j['company'][:15]}")
    print(f"\n  Review the JSON files, then run Stage 3 (apply) separately.")
    print(f"{'═'*55}")


if __name__ == "__main__":
    asyncio.run(main())

