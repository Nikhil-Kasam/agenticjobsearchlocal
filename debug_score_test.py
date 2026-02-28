"""Quick debug script to test LLM scoring on a few jobs."""
import asyncio
import json
from llm_client import LocalLLM
from database import VectorDBClient

async def test():
    db = VectorDBClient()
    db.ingest_pdf("Nikhil_Resume.pdf")
    llm = LocalLLM()

    with open("jobs_found.json", "r", encoding="utf-8") as f:
        jobs = json.load(f)[:3]

    for i, job in enumerate(jobs):
        title = job.get("title", "Unknown")
        company = job.get("company", "Unknown")
        desc = job.get("description", "")
        job_text = f"{title} at {company}: {desc}"
        resume_ctx = db.search_resume(job_text, k=3)
        print(f"\n--- Job {i+1}: {title[:50]} @ {company[:20]} ---")
        print(f"  Job text length: {len(job_text)} chars")
        print(f"  Resume context length: {len(resume_ctx)} chars")
        print(f"  Resume context preview: {resume_ctx[:200]}...")
        score = await llm.async_score_match(job_text, resume_ctx)
        print(f"  FINAL SCORE: {score}")

asyncio.run(test())
