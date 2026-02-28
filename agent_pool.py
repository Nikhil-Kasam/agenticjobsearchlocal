"""
agent_pool.py — Worker pool for multi-agent parallel execution.

Implements a Claude Code-style multi-agent system where:
- Workers execute tasks independently and in parallel
- Each worker self-evaluates its results and retries on failure
- Workers report status back to the supervisor
- Failed workers are automatically restarted with adjusted strategies
"""

import asyncio
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class AgentTask:
    """Represents a unit of work for a worker agent."""
    task_id: str
    task_type: str  # "search", "evaluate", "cover_letter", "fill_form"
    payload: dict   # Task-specific data
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str = ""
    attempts: int = 0
    max_attempts: int = 3
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str = ""


@dataclass
class WorkerReport:
    """Status report from a worker back to the supervisor."""
    task_id: str
    worker_id: str
    status: TaskStatus
    result: Any = None
    error: str = ""
    step_log: list = field(default_factory=list)


class AgentWorker:
    """
    A self-managing worker agent that:
    1. Executes an assigned task
    2. Self-evaluates the result
    3. Retries with adjusted strategy on failure
    4. Reports detailed status back to supervisor
    """

    def __init__(self, worker_id: str, task_handler: Callable, evaluator: Callable = None):
        self.worker_id = worker_id
        self.task_handler = task_handler  # async fn(payload) -> result
        self.evaluator = evaluator       # async fn(result) -> (bool, reason)
        self.step_log = []

    def _log(self, message: str):
        entry = f"[{self.worker_id}] {message}"
        self.step_log.append(entry)
        print(f"  {entry}")

    async def execute(self, task: AgentTask) -> WorkerReport:
        """Execute a task with retry logic and self-evaluation."""

        for attempt in range(1, task.max_attempts + 1):
            task.attempts = attempt
            task.status = TaskStatus.RUNNING if attempt == 1 else TaskStatus.RETRYING

            self._log(f"Attempt {attempt}/{task.max_attempts}: {task.task_type} ({task.task_id})")

            try:
                # ── Step 1: Execute the task ──
                result = await self.task_handler(task.payload)

                # ── Step 2: Self-evaluate ──
                if self.evaluator:
                    self._log("Self-evaluating result...")
                    is_valid, reason = await self.evaluator(result)

                    if not is_valid:
                        self._log(f"❌ Evaluation failed: {reason}")
                        if attempt < task.max_attempts:
                            # Adjust payload for retry
                            task.payload["_retry_reason"] = reason
                            task.payload["_attempt"] = attempt + 1
                            wait = 2 ** attempt  # Exponential backoff
                            self._log(f"⏳ Waiting {wait}s before retry...")
                            await asyncio.sleep(wait)
                            continue
                        else:
                            task.status = TaskStatus.FAILED
                            task.error = f"Exhausted retries. Last failure: {reason}"
                            self._log(f"💀 All {task.max_attempts} attempts exhausted")
                            return WorkerReport(
                                task_id=task.task_id,
                                worker_id=self.worker_id,
                                status=TaskStatus.FAILED,
                                error=task.error,
                                step_log=self.step_log
                            )

                # ── Step 3: Success ──
                task.status = TaskStatus.SUCCESS
                task.result = result
                task.completed_at = datetime.now().isoformat()
                self._log(f"✅ Completed successfully")

                return WorkerReport(
                    task_id=task.task_id,
                    worker_id=self.worker_id,
                    status=TaskStatus.SUCCESS,
                    result=result,
                    step_log=self.step_log
                )

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                self._log(f"💥 Exception: {error_msg}")

                if attempt < task.max_attempts:
                    wait = 2 ** attempt
                    self._log(f"⏳ Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    task.status = TaskStatus.FAILED
                    task.error = error_msg
                    return WorkerReport(
                        task_id=task.task_id,
                        worker_id=self.worker_id,
                        status=TaskStatus.FAILED,
                        error=error_msg,
                        step_log=self.step_log
                    )

        # Should never reach here, but safety net
        return WorkerReport(
            task_id=task.task_id,
            worker_id=self.worker_id,
            status=TaskStatus.FAILED,
            error="Unknown failure",
            step_log=self.step_log
        )


class WorkerPool:
    """
    Manages a pool of concurrent AgentWorkers.
    
    - Limits concurrency with a semaphore
    - Dispatches tasks to available workers
    - Collects and aggregates results
    - Tracks overall progress
    """

    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)
        self.active_tasks = 0
        self.completed = 0
        self.failed = 0
        self.total = 0
        self.results: list[WorkerReport] = []
        self._lock = asyncio.Lock()

    async def _run_worker(self, worker: AgentWorker, task: AgentTask) -> WorkerReport:
        """Run a single worker with concurrency control."""
        async with self.semaphore:
            async with self._lock:
                self.active_tasks += 1

            report = await worker.execute(task)

            async with self._lock:
                self.active_tasks -= 1
                self.completed += 1
                if report.status == TaskStatus.FAILED:
                    self.failed += 1
                self.results.append(report)

            # Progress update
            print(f"  📊 Progress: {self.completed}/{self.total} "
                  f"(active: {self.active_tasks}, failed: {self.failed})")

            return report

    async def run_all(
        self, 
        tasks: list[AgentTask], 
        handler: Callable, 
        evaluator: Callable = None
    ) -> list[WorkerReport]:
        """
        Execute all tasks in parallel with the worker pool.
        
        Args:
            tasks: List of AgentTask objects
            handler: Async function(payload) -> result
            evaluator: Optional async function(result) -> (bool, reason)
        
        Returns:
            List of WorkerReport objects
        """
        self.total = len(tasks)
        self.completed = 0
        self.failed = 0
        self.results = []

        print(f"\n  🏭 Worker Pool: {len(tasks)} tasks, {self.max_workers} concurrent workers")

        # Create workers and dispatch
        coroutines = []
        for i, task in enumerate(tasks):
            worker = AgentWorker(
                worker_id=f"worker-{i:03d}",
                task_handler=handler,
                evaluator=evaluator
            )
            coroutines.append(self._run_worker(worker, task))

        # Run all concurrently (semaphore limits actual parallelism)
        await asyncio.gather(*coroutines, return_exceptions=True)

        # Summary
        successes = sum(1 for r in self.results if r.status == TaskStatus.SUCCESS)
        failures = sum(1 for r in self.results if r.status == TaskStatus.FAILED)
        print(f"\n  ✅ Pool complete: {successes} succeeded, {failures} failed out of {self.total}")

        return self.results

    def get_successful_results(self) -> list[Any]:
        """Get only the successful results."""
        return [r.result for r in self.results if r.status == TaskStatus.SUCCESS]

    def get_failed_tasks(self) -> list[WorkerReport]:
        """Get reports for failed tasks."""
        return [r for r in self.results if r.status == TaskStatus.FAILED]
