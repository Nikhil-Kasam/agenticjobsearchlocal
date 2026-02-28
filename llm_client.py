"""
llm_client.py — Async + sync Ollama LLM client.

Uses httpx for async requests (concurrent evaluation via WorkerPool)
and keeps requests for backward-compatible sync methods.
"""
import re
import httpx
import requests
from typing import Optional

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5-coder:32b"


class LocalLLM:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    # ── Synchronous (backward compat) ──

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=300)
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama: {e}")
            return f"Error: {str(e)}"

    def score_match(self, job_description: str, resume_context: str) -> int:
        prompt = self._build_score_prompt(job_description, resume_context)
        result = self.generate(prompt,
            system_prompt="You are a job matching evaluator. Return ONLY a single integer 0-100. No explanation.")
        return self._parse_score(result)

    # ── Async (for WorkerPool concurrency) ──

    async def async_generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(OLLAMA_URL, json=payload)
                response.raise_for_status()
                return response.json().get("response", "")
        except httpx.HTTPError as e:
            print(f"Error connecting to Ollama: {e}")
            return f"Error: {str(e)}"

    async def async_score_match(self, job_description: str, resume_context: str) -> int:
        """Async version of score_match for concurrent evaluation."""
        prompt = self._build_score_prompt(job_description, resume_context)
        result = await self.async_generate(prompt,
            system_prompt="You are a job matching evaluator. Return ONLY a single integer 0-100. No explanation.")
        score = self._parse_score(result)
        # Debug: show raw LLM output and parsed score
        preview = result.strip()[:80] if result else "<EMPTY>"
        print(f"    🔍 LLM raw: '{preview}' → parsed score: {score}")
        return score

    # ── Shared helpers ──

    @staticmethod
    def _build_score_prompt(job_description: str, resume_context: str) -> str:
        return f"""Score how well this resume matches the job description.
Return ONLY a single integer between 0 and 100. Nothing else.

JOB DESCRIPTION:
{job_description[:2000]}

RELEVANT RESUME SKILLS & EXPERIENCE:
{resume_context[:3000]}

Score (0-100):"""

    @staticmethod
    def _parse_score(result: str) -> int:
        if not result or result.startswith("Error:"):
            print(f"    ⚠ LLM returned error/empty: {result[:100] if result else '<NONE>'}")
            return -1  # Mark as failed, don't silently default to 50
        numbers = re.findall(r'\d+', result)
        if numbers:
            score = int(numbers[0])
            return min(max(score, 0), 100)
        print(f"    ⚠ No number found in LLM response: {result[:100]}")
        return -1  # Mark as failed
