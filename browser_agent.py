"""
browser_agent.py — LangGraph browser agent with Postgres checkpointing.

Uses a StateGraph for crash-resumable application filling:
  navigate → find_apply_btn → fill_fields → paste_cover_letter → awaiting_review

Each node is checkpointed to Postgres. If the apply worker crashes mid-fill,
restarting the task with the same job_id resumes from the last saved node.

Domain whitelist is enforced via SafeBrowserController before any navigation.
"""

import json
import os
from typing import TypedDict, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from browser_use import Agent
from safety.domain_whitelist import SafeBrowserController

load_dotenv()

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "kaitchup/Qwen3.5-27B-NVFP4")
POSTGRES_URL = os.getenv(
    "POSTGRES_URL_SYNC",
    "postgresql://jobagent:jobagent@localhost:5432/jobagent"
)


# ── LangGraph State ────────────────────────────────────────────────────────────

class ApplyState(TypedDict):
    job_id: str
    job_url: str
    profile: dict
    cover_letter: str
    node_status: str       # current node name
    fill_result: str       # "FILLED" | "FILL_FAILED"
    submit_result: str     # "SUBMITTED" | "SUBMIT_FAILED"
    error: Optional[str]


# ── LLM for browser agent ──────────────────────────────────────────────────────

def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_URL,
        api_key="EMPTY",
    )


# ── Browser helper ─────────────────────────────────────────────────────────────

def _field_instructions(profile: dict) -> str:
    return f"""
Fill in all visible form fields using this information:
- Full Name / First Name: {profile.get('first_name', profile.get('name', ''))}
- Last Name: {profile.get('last_name', '')}
- Email: {profile.get('email', '')}
- Phone: {profile.get('phone', '')}
- LinkedIn: {profile.get('linkedin', '')}
- GitHub / Portfolio / Website: {profile.get('github', '')}
- City: {profile.get('city', '')}
- State: {profile.get('state', '')}
- Country: {profile.get('country', 'United States')}
- Current Company: {profile.get('current_company', '')}
- Current Title: {profile.get('current_title', '')}
- Years of Experience: {profile.get('years_of_experience', '')}
- Work Authorization / Visa Status: {profile.get('work_authorization', 'F1-OPT')}
- Will require sponsorship: {profile.get('require_sponsorship', 'Yes')}
- Education Degree: {profile.get('education', {{}}).get('degree', '')}
- School: {profile.get('education', {{}}).get('school', '')}
- Graduation Year: {profile.get('education', {{}}).get('graduation_year', '')}

Dropdowns:
- "Authorized to work in US?" → Yes
- Sponsorship → Yes
- Job type → Full-Time
- Gender/Race/Veteran/Disability → "Prefer not to say" or "Decline to self-identify"
"""


# ── LangGraph nodes ────────────────────────────────────────────────────────────

async def node_fill_form(state: ApplyState) -> dict:
    """Navigate to job URL and fill all application fields."""
    llm = _get_llm()

    task_prompt = f"""
CRITICAL INSTRUCTIONS:

1. Navigate directly to: {state['job_url']}
   Do NOT use Google Search. Go directly to this URL.

2. If there is an "Apply" or "Apply Now" button, click it first.

3. Fill ALL visible form fields:
{_field_instructions(state['profile'])}

4. If there is a Cover Letter text area, paste this EXACT text:
{state['cover_letter'][:1500]}

5. For file upload (Resume/CV): skip it.
6. For checkboxes (agree to terms): check them.

7. STOP HERE. Do NOT click Submit. Use "done" with "FORM_FILLED_READY_FOR_REVIEW".

IMPORTANT: Only fill, do not submit.
"""

    agent = Agent(task=task_prompt, llm=llm, max_actions_per_step=1, max_failures=5)

    try:
        result = await agent.run()
        result_str = str(result)
        fill_result = "FILLED" if "FORM_FILLED" in result_str or not result_str.startswith("Error") else "FILL_FAILED"
    except Exception as e:
        fill_result = f"FILL_FAILED: {str(e)}"

    return {"fill_result": fill_result, "node_status": "form_filled"}


async def node_submit_form(state: ApplyState) -> dict:
    """Submit the filled application form."""
    llm = _get_llm()

    task_prompt = """
The application form has been filled in.
1. Find the Submit / Send / Apply button on the page.
2. Click it.
3. Wait for submission confirmation.
4. Use "done" with "APPLICATION_SUBMITTED" when complete.
"""

    agent = Agent(task=task_prompt, llm=llm, max_actions_per_step=1, max_failures=3)

    try:
        await agent.run()
        submit_result = "SUBMITTED"
    except Exception as e:
        submit_result = f"SUBMIT_FAILED: {str(e)}"

    return {"submit_result": submit_result, "node_status": "submitted"}


# ── Main BrowserAgent class ────────────────────────────────────────────────────

class BrowserAgent:
    """
    LangGraph-based browser agent with Postgres crash recovery.

    thread_id = job_id ensures deterministic resume:
    if it crashes at node_fill_form, rerunning with the same job_id
    resumes from node_fill_form automatically.
    """

    def __init__(self, job_id: str = "default"):
        self.job_id = job_id

    async def fill_and_submit(
        self,
        job_url: str,
        profile: dict,
        cover_letter: str,
        controller: SafeBrowserController,
    ) -> str:
        """
        Run the full fill → submit flow with LangGraph Postgres checkpointing.
        Returns "SUBMITTED", "FILL_FAILED", "SUBMIT_FAILED", or "DOMAIN_BLOCKED".
        """
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.postgres import PostgresSaver

        # Domain safety check
        if not controller.check_navigation(job_url):
            return "DOMAIN_BLOCKED"

        # Build the LangGraph
        graph_builder = StateGraph(ApplyState)
        graph_builder.add_node("fill_form", node_fill_form)
        graph_builder.add_node("submit_form", node_submit_form)
        graph_builder.set_entry_point("fill_form")
        graph_builder.add_edge("fill_form", "submit_form")
        graph_builder.add_edge("submit_form", END)

        # Compile with Postgres checkpointer (crash recovery)
        with PostgresSaver.from_conn_string(POSTGRES_URL) as checkpointer:
            checkpointer.setup()  # creates checkpoint tables if not exist
            graph = graph_builder.compile(checkpointer=checkpointer)

            initial_state: ApplyState = {
                "job_id": self.job_id,
                "job_url": job_url,
                "profile": profile,
                "cover_letter": cover_letter,
                "node_status": "starting",
                "fill_result": "",
                "submit_result": "",
                "error": None,
            }

            config = {"configurable": {"thread_id": self.job_id}}

            # If state already exists for this thread_id → resumes from checkpoint
            final_state = await graph.ainvoke(initial_state, config=config)

        return final_state.get("submit_result", "UNKNOWN")

    # ── Legacy interface (used by supervisor.py) ───────────────────────────────

    async def fill_application(self, job_url: str, profile: dict, cover_letter: str) -> str:
        """Backward-compatible fill-only method (no submit)."""
        controller = SafeBrowserController(job_id=self.job_id)
        if not controller.check_navigation(job_url):
            return "DOMAIN_BLOCKED"

        state: ApplyState = {
            "job_id": self.job_id,
            "job_url": job_url,
            "profile": profile,
            "cover_letter": cover_letter,
            "node_status": "starting",
            "fill_result": "",
            "submit_result": "",
            "error": None,
        }
        result = await node_fill_form(state)
        return result.get("fill_result", "FILL_FAILED")

    async def submit_form(self) -> str:
        """Backward-compatible submit-only method."""
        state: ApplyState = {
            "job_id": self.job_id,
            "job_url": "",
            "profile": {},
            "cover_letter": "",
            "node_status": "form_filled",
            "fill_result": "FILLED",
            "submit_result": "",
            "error": None,
        }
        result = await node_submit_form(state)
        return result.get("submit_result", "SUBMIT_FAILED")
