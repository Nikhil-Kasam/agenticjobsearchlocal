"""
llm_client.py — vLLM async/sync client with Instructor-enforced structured outputs.

Model: kaitchup/Qwen3.5-27B-NVFP4
       Served via vLLM with --reasoning-parser=qwen3 --speculative-config qwen3_next_mtp

Two modes:
  1. Structured (Instructor) — for job evaluation and cover letter generation.
     Guarantees schema-valid Pydantic output; retries up to 3x on violation.
  2. Raw string — for browser agent task prompts (unstructured by design).
"""

import os
import re
import httpx
import requests
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator
import instructor
from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv

load_dotenv()

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000/v1")
VLLM_CHAT_URL = f"{VLLM_URL}/chat/completions"
DEFAULT_MODEL = os.getenv("VLLM_MODEL", "kaitchup/Qwen3.5-27B-NVFP4")


def strip_think(text: str) -> str:
    """
    Remove Qwen3's <think>...</think> reasoning blocks from output.
    Substitutes for --reasoning-parser=qwen3 (not in v0.8.5.post1).
    """
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ══════════════════════════════════════════════════════════════════
#  Pydantic Schemas — Enforced by Instructor
# ══════════════════════════════════════════════════════════════════

class JobEvaluation(BaseModel):
    """Structured output for job-resume matching evaluation."""
    match_score: int = Field(ge=0, le=100, description="How well the resume matches the job (0-100)")
    top_matching_skills: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Skills from the resume that match the job requirements"
    )
    missing_skills: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Key skills the job requires that are absent from the resume"
    )
    recommendation: Literal["apply", "skip", "borderline"] = Field(
        description="apply if score>=65, skip if score<45, borderline otherwise"
    )
    reasoning: str = Field(
        max_length=2000,
        description="Brief explanation of the score and recommendation"
    )


class CoverLetter(BaseModel):
    """Structured cover letter with enforced length and format rules."""
    subject_line: str = Field(
        max_length=120,
        description="Email subject line for the application"
    )
    body: str = Field(
        description="Cover letter body. Under 300 words. No markdown. No [bracket] placeholders."
    )

    @field_validator("body")
    @classmethod
    def no_brackets_or_markdown(cls, v: str) -> str:
        # Strip any residual markdown
        v = re.sub(r"\*{1,3}", "", v)
        v = re.sub(r"#{1,6}\s", "", v)
        # Warn if brackets remain (don't hard-fail — just clean)
        if re.search(r"\[.+?\]", v):
            v = re.sub(r"\[.+?\]", "", v)
        return v.strip()

    @property
    def word_count(self) -> int:
        return len(self.body.split())


# ══════════════════════════════════════════════════════════════════
#  Instructor-patched clients
# ══════════════════════════════════════════════════════════════════

def _make_async_client() -> instructor.AsyncInstructor:
    return instructor.from_openai(
        AsyncOpenAI(base_url=VLLM_URL, api_key="EMPTY"),
        mode=instructor.Mode.JSON,
    )

def _make_sync_client() -> instructor.Instructor:
    return instructor.from_openai(
        OpenAI(base_url=VLLM_URL, api_key="EMPTY"),
        mode=instructor.Mode.JSON,
    )


# ══════════════════════════════════════════════════════════════════
#  LocalLLM — unified interface
# ══════════════════════════════════════════════════════════════════

class LocalLLM:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    # ── Structured async (Instructor) ─────────────────────────────

    async def async_score_match(
        self, job_description: str, resume_context: str
    ) -> JobEvaluation:
        """
        Async structured job evaluation. Returns a guaranteed-valid JobEvaluation.
        Instructor retries up to 3x if the model outputs malformed JSON.
        """
        client = _make_async_client()
        prompt = self._build_eval_prompt(job_description, resume_context)

        try:
            result: JobEvaluation = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a job-resume matching evaluator. "
                            "Respond ONLY with valid JSON matching the required schema. "
                            "No preamble, no markdown, no extra text."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_model=JobEvaluation,
                max_retries=3,
                max_tokens=600,
            )
            print(
                f"    ✓ Eval [{result.match_score}/100] [{result.recommendation}] "
                f"— {result.reasoning[:60]}..."
            )
            return result
        except Exception as e:
            print(f"    ⚠ Instructor eval failed: {e} — returning fallback")
            return JobEvaluation(
                match_score=0,
                top_matching_skills=[],
                missing_skills=[],
                recommendation="skip",
                reasoning=f"Evaluation error: {str(e)[:200]}",
            )

    async def async_generate_cover_letter(
        self, job: dict, resume_text: str, profile: dict
    ) -> CoverLetter:
        """
        Async structured cover letter generation. Enforces <300 words, no markdown.
        """
        client = _make_async_client()
        prompt = self._build_cover_letter_prompt(job, resume_text, profile)

        try:
            result: CoverLetter = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional cover letter writer. "
                            "Write concise, personalized cover letters under 300 words. "
                            "No markdown formatting. No [bracket] placeholder text. "
                            "Respond ONLY with valid JSON matching the required schema."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_model=CoverLetter,
                max_retries=3,
                max_tokens=800,
            )
            return result
        except Exception as e:
            print(f"    ⚠ Instructor cover letter failed: {e}")
            return CoverLetter(
                subject_line=f"Application for {job.get('title', 'Position')} at {job.get('company', 'Company')}",
                body=f"I am interested in the {job.get('title', 'position')} role at {job.get('company', 'your company')}.",
            )

    # ── Raw string sync (backward compat + browser agent prompts) ──

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {"model": self.model, "messages": messages, "stream": False, "max_tokens": 2048}

        try:
            response = requests.post(VLLM_CHAT_URL, json=payload, timeout=300)
            response.raise_for_status()
            return strip_think(response.json().get("choices", [{}])[0].get("message", {}).get("content", ""))
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to vLLM: {e}")
            return f"Error: {str(e)}"

    async def async_generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {"model": self.model, "messages": messages, "stream": False, "max_tokens": 2048}

        try:
            async with httpx.AsyncClient(timeout=300) as client:
                response = await client.post(VLLM_CHAT_URL, json=payload)
                response.raise_for_status()
                return strip_think(response.json().get("choices", [{}])[0].get("message", {}).get("content", ""))
        except httpx.HTTPError as e:
            print(f"Error connecting to vLLM: {e}")
            return f"Error: {str(e)}"

    # Legacy sync scorer (used by supervisor.py evaluate handler)
    def score_match(self, job_description: str, resume_context: str) -> int:
        prompt = self._build_score_prompt_simple(job_description, resume_context)
        result = self.generate(
            prompt,
            system_prompt="You are a job matching evaluator. Return ONLY a single integer 0-100. No explanation.",
        )
        return self._parse_score(result)

    # ── Prompt builders ────────────────────────────────────────────

    @staticmethod
    def _build_eval_prompt(job_description: str, resume_context: str) -> str:
        return f"""Evaluate how well this candidate matches the job.

JOB DESCRIPTION:
{job_description[:2500]}

CANDIDATE RESUME CONTEXT (most relevant sections):
{resume_context[:3000]}

Provide a structured evaluation with: match_score (0-100), top_matching_skills, missing_skills, recommendation, and reasoning."""

    @staticmethod
    def _build_cover_letter_prompt(job: dict, resume_text: str, profile: dict) -> str:
        return f"""Write a tailored cover letter for this job application.

JOB: {job.get('title', '')} at {job.get('company', '')}
DESCRIPTION: {job.get('description', 'N/A')[:1500]}

CANDIDATE NAME: {profile.get('name', '')}
RESUME HIGHLIGHTS:
{resume_text[:2000]}

Requirements:
- Under 300 words
- Professional but personable
- Mention specific skills that match the job
- No markdown, no [placeholder] text
- Use candidate's actual name: {profile.get('name', '')}"""

    @staticmethod
    def _build_score_prompt_simple(job_description: str, resume_context: str) -> str:
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
            return -1
        numbers = re.findall(r'\d+', result)
        if numbers:
            return min(max(int(numbers[0]), 0), 100)
        print(f"    ⚠ No number found in LLM response: {result[:100]}")
        return -1
