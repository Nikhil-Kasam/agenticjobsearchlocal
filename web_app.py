import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

app = FastAPI(title="Job Application Dashboard")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Path to data files
JOBS_FOUND_PATH = "jobs_found.json"
JOBS_EVALUATED_PATH = "jobs_evaluated.json"

def load_json_data(filepath):
    """Load JSON data from a file, returning an empty list if not found or invalid."""
    try:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
    return []

@app.get("/", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    """Render the main dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/jobs/found")
async def get_jobs_found():
    """API endpoint to get the list of found jobs."""
    jobs = load_json_data(JOBS_FOUND_PATH)
    # Sort backwards by timestamp so newest are first
    if jobs and isinstance(jobs, list):
        jobs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return jobs

@app.get("/api/jobs/evaluated")
async def get_jobs_evaluated():
    """API endpoint to get the list of evaluated jobs."""
    jobs = load_json_data(JOBS_EVALUATED_PATH)
    return jobs

if __name__ == "__main__":
    import uvicorn
    # Make sure we're in the right directory structure
    if not os.path.exists("static"):
        os.makedirs("static")
    if not os.path.exists("templates"):
        os.makedirs("templates")
        
    print("Starting Job Application Dashboard server...")
    uvicorn.run("web_app:app", host="127.0.0.1", port=8000, reload=True)
