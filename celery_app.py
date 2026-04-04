"""
celery_app.py — Celery application instance and task routing.

Three queues:
  search   — DuckDuckGo scraping (1 concurrent worker)
  evaluate — RAG + LLM scoring  (3 concurrent workers)
  apply    — Browser automation + HITL  (1 concurrent worker)
"""

import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "jobagent",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        "tasks.search_tasks",
        "tasks.evaluate_tasks",
        "tasks.apply_tasks",
    ],
)

celery_app.conf.update(
    # Task routing — each type goes to its dedicated queue
    task_routes={
        "tasks.search_tasks.*":   {"queue": "search"},
        "tasks.evaluate_tasks.*": {"queue": "evaluate"},
        "tasks.apply_tasks.*":    {"queue": "apply"},
    },

    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Reliability settings
    task_acks_late=True,           # Only ack after task completes (crash-safe)
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,  # One task at a time per worker slot

    # Result TTL (keep results 24h for dashboard status checks)
    result_expires=86400,

    # Timezone
    timezone="UTC",
    enable_utc=True,
)
