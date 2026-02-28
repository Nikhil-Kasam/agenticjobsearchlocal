"""Test the WorkerPool + LLM scoring path directly (bypasses Stage 1 search)."""
import asyncio
import json
from llm_client import LocalLLM
from database import VectorDBClient
from agent_pool import WorkerPool, AgentTask, TaskStatus

async def test():
    db = VectorDBClient()
    db.ingest_pdf("Nikhil_Resume.pdf")
    llm = LocalLLM()

    with open("jobs_found.json", "r", encoding="utf-8") as f:
        jobs = json.load(f)

    print(f"Testing {len(jobs)} jobs through WorkerPool...\n")

    # Build eval tasks (same as main.py)
    eval_tasks = []
    for i, job in enumerate(jobs):
        eval_tasks.append(AgentTask(
            task_id=f"eval-{i:04d}",
            task_type="evaluate",
            payload={"job_index": i, "job": job},
            max_attempts=2,
        ))

    # Same handler as main.py
    async def evaluate_job(payload: dict) -> dict:
        job = payload["job"]
        job_text = f"{job['title']} at {job['company']}: {job.get('description', '')}"
        resume_context = db.search_resume(job_text, k=3)
        score = await llm.async_score_match(job_text, resume_context)
        return {"job_index": payload["job_index"], "score": score}

    pool = WorkerPool(max_workers=3)
    reports = await pool.run_all(eval_tasks, handler=evaluate_job)

    # Merge scores
    print("\n--- RESULTS ---")
    for r in reports:
        if r.status == TaskStatus.SUCCESS and r.result:
            idx = r.result["job_index"]
            score = r.result["score"]
            title = jobs[idx]["title"][:50]
            print(f"  Job {idx}: score={score} | {title}")
        else:
            print(f"  FAILED: {r.task_id} - {r.error}")

    scores = [r.result["score"] for r in reports if r.status == TaskStatus.SUCCESS and r.result]
    print(f"\nScores: {scores}")
    print(f"All same? {len(set(scores)) == 1}")

asyncio.run(test())
