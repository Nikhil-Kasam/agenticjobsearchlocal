"""
workflow.py — LangGraph state machine for real-time job application automation.

Flow: Job Searcher → Job Evaluator → Cover Letter Generator → Application Filler → Review Gate
      (loops for each job listing)
"""

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from llm_client import LocalLLM
from database import VectorDBClient
from browser_agent import BrowserAgent
from job_scraper import JobScraper


class ApplicationState(TypedDict):
    # Search configuration 
    search_config: dict
    profile: dict
    resume_text: str

    # Job discovery
    job_listings: list
    current_job_index: int
    current_job: dict

    # Evaluation
    match_score: int

    # Generation
    cover_letter: str

    # Application
    fill_status: str
    user_decision: str  # "approve", "skip", "stop"

    # Tracking
    results_summary: list


# Initialize shared clients
llm = LocalLLM()
db_client = VectorDBClient()
browser_agent = BrowserAgent()
job_scraper = JobScraper()


# ─── NODE: Job Searcher ───
async def job_searcher_node(state: ApplicationState):
    print("\n═══════════════════════════════════════")
    print("  🔍 SEARCHING FOR JOBS...")
    print("═══════════════════════════════════════")

    config = state["search_config"]
    listings = await job_scraper.search_jobs(config)

    return {
        "job_listings": listings,
        "current_job_index": 0,
        "results_summary": []
    }


# ─── NODE: Pick Next Job ───
def pick_next_job_node(state: ApplicationState):
    idx = state["current_job_index"]
    listings = state["job_listings"]

    if idx >= len(listings):
        print("\n  [Workflow] No more jobs to process.")
        return {"current_job": {}}

    job = listings[idx]
    print(f"\n═══════════════════════════════════════")
    print(f"  📋 JOB {idx + 1}/{len(listings)}")
    print(f"  Title:   {job.get('title', 'N/A')}")
    print(f"  Company: {job.get('company', 'N/A')}")
    print(f"  URL:     {job.get('url', 'N/A')}")
    print(f"═══════════════════════════════════════")

    return {"current_job": job}


# ─── NODE: Job Evaluator ───
def job_evaluator_node(state: ApplicationState):
    job = state["current_job"]
    if not job:
        return {"match_score": 0}

    print("  → Evaluating job match against resume...")

    job_text = f"{job.get('title', '')} at {job.get('company', '')}: {job.get('description', '')}"
    score = llm.score_match(job_text, state.get("resume_text", ""))

    print(f"  → Match Score: {score}/100")
    threshold = state["search_config"].get("match_threshold", 60)
    if score < threshold:
        print(f"  → Below threshold ({threshold}). Skipping.")
    else:
        print(f"  → Above threshold ({threshold}). Proceeding!")

    return {"match_score": score}


# ─── NODE: Cover Letter Generator ───
def cover_letter_generator_node(state: ApplicationState):
    job = state["current_job"]
    print("  → Generating tailored cover letter...")

    prompt = f"""Write a concise, professional cover letter for this specific job:

JOB TITLE: {job.get('title', 'Software Engineer')}
COMPANY: {job.get('company', 'the company')}
JOB DESCRIPTION: {job.get('description', 'N/A')}

CANDIDATE RESUME HIGHLIGHTS:
{state.get('resume_text', '')[:2000]}

Instructions:
- Keep it under 300 words
- Highlight specific skills matching the job
- Be professional but personable
- Do NOT use markdown formatting
- Do NOT include placeholder brackets like [Your Name]
- Use the candidate's actual name: {state['profile'].get('name', 'the applicant')}
"""

    system_prompt = "You are a professional cover letter writer. Write concise, tailored cover letters. No markdown formatting. No placeholder brackets."
    cover_letter = llm.generate(prompt, system_prompt)

    # Quick cleanup — remove any lingering markdown or brackets
    cover_letter = cover_letter.replace("**", "").replace("##", "").replace("# ", "")

    print(f"  → Cover letter generated ({len(cover_letter)} chars)")
    return {"cover_letter": cover_letter}


# ─── NODE: Application Filler ───
async def application_filler_node(state: ApplicationState):
    job = state["current_job"]
    print("  → Opening browser to fill application form...")

    status = await browser_agent.fill_application(
        job.get("url", ""),
        state["profile"],
        state["cover_letter"]
    )

    return {"fill_status": status}


# ─── NODE: Review Gate (Interactive) ───
def review_gate_node(state: ApplicationState):
    """
    This node pauses execution and asks the user to review.
    In the real flow, main.py handles the interactive input.
    """
    job = state["current_job"]
    print("\n  ╔══════════════════════════════════════╗")
    print("  ║   ⏸  PAUSED FOR YOUR REVIEW          ║")
    print("  ╚══════════════════════════════════════╝")
    print(f"  Job:     {job.get('title', 'N/A')} at {job.get('company', 'N/A')}")
    print(f"  Score:   {state['match_score']}/100")
    print(f"  Status:  {state['fill_status']}")
    print(f"\n  The browser should have the form filled in.")
    print(f"  Check the browser window now.\n")

    # The actual user input is handled in main.py's interactive loop
    # This node just signals readiness
    return {"user_decision": "pending"}


# ─── NODE: Record Result ───
def record_result_node(state: ApplicationState):
    job = state["current_job"]
    decision = state.get("user_decision", "skip")
    summary = state.get("results_summary", [])

    summary.append({
        "title": job.get("title", "N/A"),
        "company": job.get("company", "N/A"),
        "url": job.get("url", ""),
        "score": state["match_score"],
        "decision": decision
    })

    next_idx = state["current_job_index"] + 1
    return {
        "results_summary": summary,
        "current_job_index": next_idx
    }


# ─── ROUTING FUNCTIONS ───
def should_evaluate(state: ApplicationState) -> str:
    """Route after picking a job: evaluate or end."""
    if not state.get("current_job"):
        return "end"
    return "evaluate"


def should_apply(state: ApplicationState) -> str:
    """Route after evaluation: apply if score meets threshold, skip otherwise."""
    threshold = state["search_config"].get("match_threshold", 60)
    if state["match_score"] >= threshold:
        return "apply"
    return "skip"


def has_more_jobs(state: ApplicationState) -> str:
    """Route after recording result: more jobs or finish."""
    idx = state["current_job_index"]
    total = len(state.get("job_listings", []))
    if idx < total:
        return "next"
    return "done"


# ─── BUILD THE GRAPH ───
workflow = StateGraph(ApplicationState)

# Add nodes
workflow.add_node("job_searcher", job_searcher_node)
workflow.add_node("pick_next_job", pick_next_job_node)
workflow.add_node("job_evaluator", job_evaluator_node)
workflow.add_node("cover_letter_generator", cover_letter_generator_node)
workflow.add_node("application_filler", application_filler_node)
workflow.add_node("review_gate", review_gate_node)
workflow.add_node("record_result", record_result_node)

# Set entry
workflow.set_entry_point("job_searcher")

# Edges
workflow.add_edge("job_searcher", "pick_next_job")

workflow.add_conditional_edges("pick_next_job", should_evaluate, {
    "evaluate": "job_evaluator",
    "end": END
})

workflow.add_conditional_edges("job_evaluator", should_apply, {
    "apply": "cover_letter_generator",
    "skip": "record_result"
})

workflow.add_edge("cover_letter_generator", "application_filler")
workflow.add_edge("application_filler", "review_gate")
workflow.add_edge("review_gate", "record_result")

workflow.add_conditional_edges("record_result", has_more_jobs, {
    "next": "pick_next_job",
    "done": END
})

app_workflow = workflow.compile()
