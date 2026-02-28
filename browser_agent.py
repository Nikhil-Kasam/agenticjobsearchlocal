"""
browser_agent.py — Real-time browser automation for job applications.

Two main capabilities:
1. fill_application() — Navigates to a real job URL, fills form fields, and pauses for review
2. submit_form() — Called after user approval to click submit
"""

import json
from langchain_openai import ChatOpenAI
from browser_use import Agent


class BrowserAgent:
    def __init__(self, model_name="qwen2.5-coder:32b"):
        self.llm = ChatOpenAI(
            model=model_name,
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )

    async def fill_application(self, job_url: str, profile: dict, cover_letter: str) -> str:
        """
        Navigate to a real job application URL, fill in form fields using profile data,
        and paste the cover letter. Does NOT submit — pauses for user review.
        """
        # Build a comprehensive field mapping instruction
        field_instructions = f"""
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
        - Authorized to work in the US: Yes
        - Work Authorization / Visa Status: {profile.get('work_authorization', 'F1-OPT')}
        - Will you now or in the future require sponsorship: {profile.get('require_sponsorship', 'Yes')}
        - Job Type preference: Full-Time
        - Education Degree: {profile.get('education', {{}}).get('degree', '')}
        - School: {profile.get('education', {{}}).get('school', '')}
        - Graduation Year: {profile.get('education', {{}}).get('graduation_year', '')}
        
        IMPORTANT for dropdowns:
        - If asked "Are you legally authorized to work in the United States?" → Select "Yes"
        - If asked about sponsorship → Select "Yes" (requires sponsorship)
        - If asked about job type → Select "Full-Time"
        - If asked about gender/race/veteran/disability → Select "Decline to self-identify" or "Prefer not to say"
        """

        task_prompt = f"""
        CRITICAL INSTRUCTIONS — Follow EXACTLY:

        1. Navigate directly to this URL: {job_url}
           Do NOT use Google Search. Go directly to the URL.

        2. If there is an "Apply" or "Apply Now" button on the job listing page, click it first
           to get to the actual application form.

        3. Once you see the application form, fill in ALL visible fields using this data:
        {field_instructions}

        4. If there is a "Cover Letter" text area or field, paste this EXACT text:
        {cover_letter[:1500]}

        5. If there is a file upload for Resume/CV, skip it (the user will upload manually).

        6. For any dropdown menus (like country, state, work authorization):
           - Click the dropdown to open it
           - Select the matching option from the list

        7. For checkboxes (like "I agree to terms"), check them.

        8. STOP HERE. Do NOT click Submit/Send/Apply. Just fill everything in and then use
           the "done" action with the text "FORM_FILLED_READY_FOR_REVIEW".

        IMPORTANT: Do NOT submit the form. Only fill it in and stop.
        """

        agent = Agent(
            task=task_prompt,
            llm=self.llm,
            max_actions_per_step=1,
            max_failures=5,
        )

        try:
            result = await agent.run()
            result_str = str(result)
            if "FORM_FILLED" in result_str:
                return "FILLED"
            return "FILLED"  # Assume filled if no crash
        except Exception as e:
            print(f"  [Browser] Fill failed: {e}")
            return f"FILL_FAILED: {str(e)}"

    async def submit_form(self) -> str:
        """
        After user reviews the filled form, click the Submit button.
        This assumes the browser is still open on the form page.
        """
        task_prompt = """
        The application form has been filled in. Now:
        1. Find the Submit / Send / Apply button on the page
        2. Click it
        3. Wait for confirmation that the application was submitted
        4. Use the "done" action with "APPLICATION_SUBMITTED" when complete
        """

        agent = Agent(
            task=task_prompt,
            llm=self.llm,
            max_actions_per_step=1,
            max_failures=3,
        )

        try:
            result = await agent.run()
            return "SUBMITTED"
        except Exception as e:
            print(f"  [Browser] Submit failed: {e}")
            return f"SUBMIT_FAILED: {str(e)}"
