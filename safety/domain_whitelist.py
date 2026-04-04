"""
safety/domain_whitelist.py — Browser action safety guard.

Blocks the LLM from navigating outside of known job application domains.
LinkedIn is restricted to Easy Apply only (no external redirects).

55+ domains covering:
  - ATS platforms (Greenhouse, Lever, Workday, iCIMS, Taleo, ...)
  - Major tech company career portals (Google, Meta, Apple, ...)
  - AI-era companies (OpenAI, Anthropic, Hugging Face, ...)
  - Finance/quant firms hiring MLE (Two Sigma, Citadel, ...)
"""

import re
from fnmatch import fnmatch
from urllib.parse import urlparse


# ── ATS Platforms ─────────────────────────────────────────────────────────────
ATS_DOMAINS = [
    # Greenhouse
    "boards.greenhouse.io", "greenhouse.io",
    # Lever
    "jobs.lever.co", "lever.co",
    # Ashby
    "jobs.ashbyhq.com", "ashbyhq.com",
    # Workday
    "myworkdayjobs.com", "apply.workday.com",
    # SmartRecruiters
    "jobs.smartrecruiters.com", "smartrecruiters.com",
    # Jobvite
    "jobs.jobvite.com", "jobvite.com",
    # iCIMS
    "icims.com",
    # Oracle Taleo
    "taleo.net",
    # Workable
    "apply.workable.com", "workable.com",
    # Recruitee
    "jobs.recruitee.com", "recruitee.com",
    # Teamtailor
    "app.teamtailor.com", "teamtailor.com",
    # Pinpoint
    "pinpointhq.com",
    # BambooHR
    "bamboohr.com",
    # Rippling
    "rippling.com",
    # SAP SuccessFactors
    "successfactors.com", "sap.com",
    # Oracle Recruiting Cloud
    "oraclecloud.com", "oracle.com",
    # ADP
    "careers.adp.com", "adp.com",
    # Wellfound / AngelList Talent
    "wellfound.com", "angel.co",
    
]

# ── Major Tech Company Career Portals ─────────────────────────────────────────
BIG_TECH_DOMAINS = [
    "careers.google.com",
    "amazon.jobs",
    "careers.microsoft.com",
    "metacareers.com",
    "jobs.apple.com", "apple.com",
    "jobs.netflix.com",
    "careers.airbnb.com",
    "uber.com",
    "careers.twitter.com", "x.com",
    "nvidia.com",
    "tesla.com",
    "spacex.com",
    "salesforce.com",
    "adobe.com",
    "databricks.com",
    "stripe.com",
    "cloudflare.com",
    "openai.com",
    "anthropic.com",
    "palantir.com",
    "snowflake.com",
    "figma.com",
    "zoom.us",
    "bytedance.com",
    "tiktok.com",
    "twilio.com",
    "notion.com",
    "hubspot.com",
    "shopify.com",
    "coinbase.com",
    "robinhood.com",
    "duolingo.com",
    "canva.com",
    "asana.com",
    "dropbox.com",
    "box.com",
    "atlassian.com",
]

# ── AI / ML Era Companies ─────────────────────────────────────────────────────
AI_COMPANY_DOMAINS = [
    "scale.com",
    "cohere.com",
    "mistral.ai",
    "huggingface.co",
    "together.ai",
    "anyscale.com",
    "modal.com",
    "runway.ml",
    "stability.ai",
    "inflection.ai",
    "character.ai",
    "perplexity.ai",
    "replit.com",
    "weights.ai",
    "labelbox.com",
    "allenai.org",
    "deepmind.google",
    "research.google",
]

# ── Finance / Quant Firms Hiring MLE ─────────────────────────────────────────
FINANCE_DOMAINS = [
    "careers.jpmorgan.com",
    "goldmansachs.com",
    "careers.bloomberg.com",
    "twosigma.com",
    "citadel.com",
    "jane-street.com",
    "deshaw.com",
    "hudson-trading.com",
    "virtu.com",
    "point72.com",
]

# ── LinkedIn: Easy Apply ONLY ─────────────────────────────────────────────────
# Allowed: /jobs/ paths and Easy Apply modal
# Blocked: linkedin.com/in/*, linkedin.com/feed, etc.
LINKEDIN_ALLOWED_PATTERNS = [
    r"linkedin\.com/jobs/",
    r"linkedin\.com/jobs/view/",
    r"linkedin\.com/jobs/search/",
]

# ── Combined whitelist ─────────────────────────────────────────────────────────
ALLOWED_DOMAINS: list[str] = (
    ATS_DOMAINS + BIG_TECH_DOMAINS + AI_COMPANY_DOMAINS + FINANCE_DOMAINS
)


def _extract_domain(url: str) -> str:
    """Extract the effective domain from a URL."""
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower().lstrip("www.")
    except Exception:
        return ""


def _is_linkedin_allowed(url: str) -> bool:
    """LinkedIn: only allow job listing pages and Easy Apply modal flow."""
    return any(re.search(pattern, url) for pattern in LINKEDIN_ALLOWED_PATTERNS)


def is_allowed(url: str) -> bool:
    """
    Return True if the URL is safe for the browser agent to navigate to.
    
    Rules:
    1. LinkedIn → only /jobs/ paths (Easy Apply)
    2. All others → must match the domain whitelist (exact or wildcard)
    """
    if not url or not url.startswith("http"):
        return False

    url_lower = url.lower()

    # LinkedIn: special rule
    if "linkedin.com" in url_lower:
        return _is_linkedin_allowed(url_lower)

    domain = _extract_domain(url_lower)
    if not domain:
        return False

    for allowed in ALLOWED_DOMAINS:
        allowed = allowed.lower()
        # Exact match
        if domain == allowed:
            return True
        # Subdomain match (e.g. "company.icims.com" matches "icims.com")
        if domain.endswith("." + allowed):
            return True
        # fnmatch wildcard (e.g. "*.greenhouse.io")
        if fnmatch(domain, allowed):
            return True

    return False


class SafeBrowserController:
    """
    Wraps browser-use navigation actions with domain whitelist enforcement.
    
    Usage:
        controller = SafeBrowserController(job_id="eval-0001")
        safe_url = controller.check_navigation("https://boards.greenhouse.io/...")
        if safe_url:
            # proceed
    """

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.blocked_count = 0
        self.allowed_count = 0

    def check_navigation(self, url: str) -> bool:
        """
        Check if a navigation URL is allowed.
        Logs blocked attempts for auditing.
        """
        if is_allowed(url):
            self.allowed_count += 1
            return True

        self.blocked_count += 1
        print(
            f"  🚫 [SafeBrowser] BLOCKED navigation for job {self.job_id}:\n"
            f"     URL: {url}\n"
            f"     Reason: Domain not in whitelist. Marking as domain_blocked."
        )
        return False

    def get_stats(self) -> dict:
        return {
            "job_id": self.job_id,
            "allowed_navigations": self.allowed_count,
            "blocked_navigations": self.blocked_count,
        }
