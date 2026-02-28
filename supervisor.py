"""
supervisor.py — Central orchestrator for the multi-agent job application system.

Implements a Claude Code-style supervisor pattern:
- Breaks the pipeline into discrete parallelizable stages
- Dispatches worker agents for each stage
- Monitors progress, handles failures, redistributes failed work
- Aggregates results and drives the pipeline forward
"""

import asyncio
import json
import random
import urllib.parse
from datetime import datetime
from langchain_openai import ChatOpenAI
from browser_use import Agent

from agent_pool import AgentTask, WorkerPool, TaskStatus
from llm_client import LocalLLM
from database import VectorDBClient


class Supervisor:
    """
    The Supervisor breaks the job application pipeline into stages and
    runs each stage using a pool of parallel worker agents:
    
    Stage 1: SEARCH   — Multiple workers scrape different portals in parallel
    Stage 2: EVALUATE — Multiple workers score job matches in parallel
    Stage 3: APPLY    — Sequential (one at a time, review-gated)
    """

    def __init__(self, config: dict, profile: dict, resume_text: str):
        self.config = config
        self.profile = profile
        self.resume_text = resume_text
        self.llm = LocalLLM()
        self.db = VectorDBClient()
        self.model_name = "qwen2.5-coder:32b"

        # Results
        self.all_jobs = []
        self.qualified_jobs = []
        self.application_results = []

    # ═══════════════════════════════════════════════════════════════
    #  STAGE 1: PARALLEL JOB SEARCH
    # ═══════════════════════════════════════════════════════════════

    async def stage_search(self) -> list[dict]:
        """
        Dispatch parallel search workers — one per (portal × region) combo.
        Uses DuckDuckGo instead of Google to avoid CAPTCHAs.
        Uses the user's exact OR-based query syntax.
        """
        print("\n╔══════════════════════════════════════════════════╗")
        print("║  STAGE 1: PARALLEL JOB SEARCH (DuckDuckGo)      ║")
        print("╚══════════════════════════════════════════════════╝")

        title_query = self.config.get("job_titles_query",
            '("MLE" OR "Machine Learning" OR "AI Engineer")')
        region_queries = self.config.get("region_queries", {})
        sources = self.config.get("job_sources", [])
        delay = self.config.get("search_delay_seconds", 5)

        # Build individual search tasks: one per (source × region)
        tasks = []
        for source in sources:
            for region_name, region_query in region_queries.items():
                # Construct exact query: site:portal + title OR-query + region OR-query
                full_query = f'{source["site_query"]} {title_query} {region_query}'

                tasks.append(AgentTask(
                    task_id=f"search-{source['name']}-{region_name}",
                    task_type="search",
                    payload={
                        "query": full_query,
                        "source_name": source["name"],
                        "region": region_name,
                        "delay": delay
                    },
                    max_attempts=2
                ))

        print(f"  📋 {len(tasks)} search combos ({len(sources)} portals × {len(region_queries)} regions)")
        print(f"  🔍 Using DuckDuckGo (no CAPTCHA)")
        print(f"  📝 Title query: {title_query[:80]}...")

        # Run with worker pool (3 concurrent browsers)
        pool = WorkerPool(max_workers=3)
        reports = await pool.run_all(
            tasks=tasks,
            handler=self._search_handler,
            evaluator=self._search_evaluator
        )

        # Aggregate and deduplicate
        all_jobs = []
        for r in pool.get_successful_results():
            if isinstance(r, list):
                all_jobs.extend(r)

        seen = set()
        unique = []
        for job in all_jobs:
            url = job.get("url", "")
            if url and url not in seen:
                seen.add(url)
                unique.append(job)

        self.all_jobs = unique
        print(f"\n  📊 Stage 1 complete: {len(unique)} unique jobs found")

        failures = pool.get_failed_tasks()
        if failures:
            print(f"  ⚠ {len(failures)} search(es) failed after retries:")
            for f in failures[:5]:
                print(f"    · {f.task_id}: {f.error[:80]}")

        return unique

    async def _search_handler(self, payload: dict) -> list[dict]:
        """
        Handler for a single search task.
        Uses DuckDuckGo with direct URL navigation to avoid CAPTCHA and
        the agent getting stuck on the search bar.
        """
        query = payload["query"]
        source_name = payload["source_name"]
        region = payload["region"]
        delay = payload.get("delay", 5)

        # URL-encode the query for DuckDuckGo direct navigation
        encoded_query = urllib.parse.quote(query)
        # DuckDuckGo search URL — df:w means past week (closest to 3 days)
        search_url = f"https://duckduckgo.com/?q={encoded_query}&df=w&ia=web"

        task_prompt = f"""
        STEP 1: Navigate to this EXACT URL by using the go_to_url action:
        {search_url}

        STEP 2: Wait for the page to fully load. You should see DuckDuckGo search results.

        STEP 3: Read ALL the search result entries on the page. For each result, extract:
        - title: The blue link heading text (this is the job title)
        - company: The company name (usually visible in the URL or description)
        - url: The full URL of the job listing link
        - description: The snippet/description text

        STEP 4: Use the "done" action with ALL results as a JSON array:
        [{{"title": "ML Engineer", "company": "Databricks", "url": "https://boards.greenhouse.io/databricks/jobs/123", "description": "Looking for..."}}]

        CRITICAL RULES:
        - Do NOT type anything into any search bar. The URL already has the search query.
        - Do NOT click on any job links. Only READ the search results page.
        - Do NOT navigate away from the search results page.
        - If there are zero results, use "done" with: []
        - If you see an error page, use "done" with: [{{"error": "page_error"}}]
        """

        llm = ChatOpenAI(model=self.model_name, base_url="http://localhost:11434/v1", api_key="ollama")
        agent = Agent(task=task_prompt, llm=llm, max_actions_per_step=1, max_failures=3)

        # Stagger starts to avoid rate limits
        await asyncio.sleep(random.uniform(1, delay))

        result = await agent.run()
        return self._parse_search_result(result, source_name, region)

    async def _search_evaluator(self, result) -> tuple[bool, str]:
        """Self-evaluate search results — check they're valid."""
        if result is None:
            return False, "Result is None"
        if not isinstance(result, list):
            return False, f"Expected list, got {type(result).__name__}"
        for item in result:
            if isinstance(item, dict) and item.get("error"):
                return False, f"Search error: {item['error']}"
        return True, "OK"

    def _parse_search_result(self, result, source_name: str, region: str) -> list[dict]:
        """Parse browser-use output into structured job listings using LLM."""
        import re
        result_text = str(result)

        if "captcha" in result_text.lower() or "unusual traffic" in result_text.lower():
            return [{"error": "captcha"}]

        parse_prompt = f"""Extract job listings from this browser output. Return ONLY a valid JSON array.
Each listing must have: title, company, url, description.
Only include actual job postings (not ads or unrelated links).
If no job listings found, return [].

Browser output:
{result_text[:4000]}

JSON array:"""

        parsed = self.llm.generate(parse_prompt,
            system_prompt="Return ONLY a valid JSON array. No markdown code blocks. No explanation text.")

        listings = []
        json_match = re.search(r'\[.*\]', parsed, re.DOTALL)
        if json_match:
            try:
                raw = json.loads(json_match.group())
                for item in raw:
                    if isinstance(item, dict) and item.get("url") and not item.get("error"):
                        listings.append({
                            "title": item.get("title", "Unknown"),
                            "company": item.get("company", source_name),
                            "url": item.get("url", ""),
                            "description": item.get("description", ""),
                            "source": source_name,
                            "region": region,
                            "scraped_at": datetime.now().isoformat()
                        })
            except json.JSONDecodeError:
                pass
        return listings

    # ═══════════════════════════════════════════════════════════════
    #  STAGE 2: PARALLEL JOB EVALUATION
    # ═══════════════════════════════════════════════════════════════

    async def stage_evaluate(self) -> list[dict]:
        """
        Score all discovered jobs against the resume in parallel.
        Filter out jobs below the match threshold.
        """
        print("\n╔══════════════════════════════════════════════════╗")
        print("║  STAGE 2: PARALLEL JOB EVALUATION               ║")
        print("╚══════════════════════════════════════════════════╝")

        if not self.all_jobs:
            print("  No jobs to evaluate.")
            return []

        threshold = self.config.get("match_threshold", 60)

        tasks = [
            AgentTask(
                task_id=f"eval-{i}",
                task_type="evaluate",
                payload={
                    "job": job,
                    "resume_text": self.resume_text,
                    "threshold": threshold
                },
                max_attempts=2
            )
            for i, job in enumerate(self.all_jobs)
        ]

        print(f"  📋 Evaluating {len(tasks)} jobs (threshold: {threshold}/100)")

        pool = WorkerPool(max_workers=5)  # Evaluation is LLM-only, can run more
        reports = await pool.run_all(
            tasks=tasks,
            handler=self._evaluate_handler,
            evaluator=self._evaluate_evaluator
        )

        # Collect qualified jobs
        self.qualified_jobs = []
        for report in reports:
            if report.status == TaskStatus.SUCCESS and report.result:
                job_with_score = report.result
                if job_with_score.get("match_score", 0) >= threshold:
                    self.qualified_jobs.append(job_with_score)

        # Sort by match score descending
        self.qualified_jobs.sort(key=lambda x: x.get("match_score", 0), reverse=True)

        print(f"\n  📊 Stage 2 complete: {len(self.qualified_jobs)}/{len(self.all_jobs)} jobs qualified")

        for i, job in enumerate(self.qualified_jobs[:10]):
            print(f"    {i+1}. [{job['match_score']}] {job['title']} at {job['company']}")

        return self.qualified_jobs

    async def _evaluate_handler(self, payload: dict) -> dict:
        """Score a single job against the resume."""
        job = payload["job"]
        job_text = f"{job.get('title', '')} at {job.get('company', '')}: {job.get('description', '')}"
        score = self.llm.score_match(job_text, payload["resume_text"])
        return {**job, "match_score": score}

    async def _evaluate_evaluator(self, result) -> tuple[bool, str]:
        """Verify evaluation result is valid."""
        if not isinstance(result, dict):
            return False, "Expected dict result"
        if "match_score" not in result:
            return False, "Missing match_score"
        score = result["match_score"]
        if not isinstance(score, int) or score < 0 or score > 100:
            return False, f"Invalid score: {score}"
        return True, "OK"

    # ═══════════════════════════════════════════════════════════════
    #  STAGE 3: SEQUENTIAL APPLICATION (Review-Gated)
    # ═══════════════════════════════════════════════════════════════

    async def stage_apply(self) -> list[dict]:
        """
        For each qualified job: generate cover letter, fill form, pause for review.
        This runs SEQUENTIALLY because each application needs user approval.
        """
        print("\n╔══════════════════════════════════════════════════╗")
        print("║  STAGE 3: APPLICATION (Review-Gated)             ║")
        print("╚══════════════════════════════════════════════════╝")

        if not self.qualified_jobs:
            print("  No qualified jobs to apply to.")
            return []

        from browser_agent import BrowserAgent
        browser = BrowserAgent()

        for i, job in enumerate(self.qualified_jobs):
            print(f"\n  ─── Application {i+1}/{len(self.qualified_jobs)} ───")
            print(f"  Job:     {job['title']} at {job['company']}")
            print(f"  Score:   {job['match_score']}/100")
            print(f"  URL:     {job['url']}")

            # Generate cover letter
            print("  → Generating cover letter...")
            cover_letter = self._generate_cover_letter(job)
            print(f"  → Cover letter ready ({len(cover_letter)} chars)")

            # Fill application form
            print("  → Opening browser and filling form...")
            fill_status = await browser.fill_application(
                job["url"], self.profile, cover_letter
            )

            # Print cover letter preview
            print(f"\n  ── Cover Letter Preview ──")
            print(f"  {cover_letter[:300]}...")

            # User review gate
            print(f"\n  ╔══════════════════════════════════════╗")
            print(f"  ║  ⏸  REVIEW: {job['company'][:26]:<26}  ║")
            print(f"  ╚══════════════════════════════════════╝")

            decision = input("  >> [a]pprove submit / [s]kip / [q]uit pipeline: ").strip().lower()

            if decision == "a":
                print("  → Submitting...")
                submit_result = await browser.submit_form()
                self.application_results.append({
                    **job, "decision": "submitted", "submit_result": submit_result
                })
                print(f"  ✅ Application submitted!")
            elif decision == "q":
                print("  → Stopping pipeline.")
                self.application_results.append({**job, "decision": "quit"})
                break
            else:
                self.application_results.append({**job, "decision": "skipped"})
                print("  → Skipped.")

        return self.application_results

    def _generate_cover_letter(self, job: dict) -> str:
        """Generate a tailored cover letter for a specific job."""
        prompt = f"""Write a concise cover letter for this job:

JOB: {job.get('title', '')} at {job.get('company', '')}
DESCRIPTION: {job.get('description', 'N/A')}

CANDIDATE: {self.profile.get('name', '')}
EXPERIENCE: {self.resume_text[:2000]}

Rules:
- Under 300 words
- Professional but personable
- No markdown. No placeholder brackets like [Your Name]
- Use the candidate's actual name: {self.profile.get('name', '')}
"""
        result = self.llm.generate(prompt,
            system_prompt="Write a concise, tailored cover letter. No markdown. No placeholders.")
        return result.replace("**", "").replace("##", "")

    # ═══════════════════════════════════════════════════════════════
    #  FULL PIPELINE
    # ═══════════════════════════════════════════════════════════════

    async def run(self):
        """Execute the full multi-agent pipeline."""
        print("╔══════════════════════════════════════════════════╗")
        print("║  🤖 MULTI-AGENT JOB APPLICATION SUPERVISOR      ║")
        print("║  Workers: 3 search, 5 evaluate, 1 apply         ║")
        print("╚══════════════════════════════════════════════════╝\n")

        # Stage 1: Find jobs
        jobs = await self.stage_search()
        if not jobs:
            print("\n  ⚠ No jobs found. Try adjusting search_config.json")
            return

        # Stage 2: Evaluate matches
        qualified = await self.stage_evaluate()
        if not qualified:
            print("\n  ⚠ No jobs above threshold. Try lowering match_threshold.")
            return

        # Stage 3: Apply with review
        results = await self.stage_apply()

        # Final Summary
        self._print_summary()

    def _print_summary(self):
        """Print the final summary table."""
        print("\n" + "═" * 65)
        print("  📊 FINAL SUMMARY")
        print("═" * 65)
        print(f"  {'#':<4} {'Company':<18} {'Title':<25} {'Score':<7} {'Result'}")
        print(f"  {'─'*4} {'─'*18} {'─'*25} {'─'*7} {'─'*12}")

        for i, r in enumerate(self.application_results, 1):
            print(f"  {i:<4} {r.get('company','?')[:17]:<18} "
                  f"{r.get('title','?')[:24]:<25} "
                  f"{r.get('match_score','?'):<7} "
                  f"{r.get('decision','?')}")

        total_found = len(self.all_jobs)
        total_qualified = len(self.qualified_jobs)
        total_applied = sum(1 for r in self.application_results if r.get("decision") == "submitted")
        total_skipped = sum(1 for r in self.application_results if r.get("decision") == "skipped")

        print(f"\n  Total jobs found:      {total_found}")
        print(f"  Qualified (≥threshold): {total_qualified}")
        print(f"  Applications sent:     {total_applied}")
        print(f"  Skipped:               {total_skipped}")
        print("═" * 65)
