# AI Job Application Automator

A fully local, privacy-first AI agent that **searches real job boards**, **evaluates listings against your resume**, **writes tailored cover letters**, and **auto-fills application forms** — pausing for your review before submitting. 

Runs entirely on your local hardware using **Qwen 2.5 Coder 32B** (via Ollama) for reasoning and **Qwen3-Embedding-8B** (via vLLM) for ultra-fast, instruction-aware document retrieval.

## Features

- **Automated Scraping**: Aggregates job postings from 50+ ATS portals (Greenhouse, Lever, Workable, etc.) and direct company sites.
- **Intelligent Evaluation**: Uses RAG (Retrieval-Augmented Generation) to score job descriptions against your specific resume experience.
- **Web Dashboard**: Includes a premium, responsive FastAPI + Vanilla JS dashboard (Dark Mode, Glassmorphism) to review found and evaluated jobs.
- **Automated Applying**: Uses `browser-use` to physically open chromium, navigate to the job, and fill the application out for you.
- **100% Local**: No data is sent to OpenAI or Anthropic. Everything runs on your own GPU.

## Architecture

![Pipeline Architecture](agent_history.gif)
*(Note: Replace with standard architecture diagram if preferred)*

```text
┌────────────────────────────────────────────────────────────────────────┐
│                              AI Pipeline                               │
├─────────────┬──────────────────┬───────────────┬───────────────────────┤
│ 1. Search   │ 2. Evaluate      │ 3. Dashboard  │ 4. Apply              │
│ (DuckDuckGo │ (vLLM Embeddings │ (FastAPI Web  │ (Browser-Use +        │
│  Scraping)  │  + Ollama Logic) │  Interface)   │  Ollama Cover Letters)│
├─────────────┴──────────────────┴───────────────┴───────────────────────┤
│                     ⏸ Review Gate (User approves)                      │
└────────────────────────────────────────────────────────────────────────┘
```

## System Requirements
- **OS**: Windows (WSL recommended for vLLM) or Linux.
- **GPU**: An NVIDIA GPU with at least 24GB VRAM (e.g., RTX 3090 / 4090 / 5090) is recommended to run the 32B models and vLLM concurrently.

---

## Setup & Installation

### 1. Install Dependencies
Set up your Python virtual environment and install the required packages:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start the AI Engines
You need two local engines running in the background.

**Terminal 1: Ollama (The Brain)**
```bash
# Download and run the main reasoning model from https://ollama.com
ollama run qwen2.5-coder:32b
```

**Terminal 2: vLLM via WSL (The Memory Engine)**
Because vLLM utilizes `PagedAttention` for continuous batching, it is highly recommended to run this within Windows Subsystem for Linux (WSL) or a native Linux environment.
```bash
# Inside WSL Ubuntu
pip install vllm
vllm serve Qwen/Qwen3-Embedding-8B
```

### 3. Configure Your Profile
Edit the configuration files in the root directory:
- `profile.json`: Your personal details, GitHub links, and QA answers for form filling.
- `search_config.json`: Job titles, target locations, portals to scrape, and minimum match threshold.
- **Resume**: Place your resume in the root folder as `Nikhil_Resume.pdf` (or update `resume_path` in `profile.json`).

---

## Usage Summary

### 1. Run the Pipeline
Execute the main script to start scraping and evaluating jobs:
```bash
python main.py
```
This will generate `jobs_found.json` and `jobs_evaluated.json`.

### 2. View the Web Dashboard
Launch the FastAPI interface to visualize your job matches:
```bash
python web_app.py
```
Open `http://127.0.0.1:8000` in your browser to view the premium job dashboard.

### 3. Automated Application (Review & Submit)
When the agent executes the `browser_agent.py` script to fill a matching application, it **pauses** before submission for your safety:
```text
>> Review the browser. [a]pprove submit / [s]kip / [q]uit:
```
Check the browser window, then type `a` to submit or `s` to skip.

---

## Project Structure
```text
├── main.py              # Primary Entry point
├── web_app.py           # FastAPI Dashboard server
├── workflow.py          # LangGraph state machine orchestration
├── supervisor.py        # Experimental multi-agent orchestration
├── job_scraper.py       # Job discovery via search operators
├── browser_agent.py     # Form filling + submission
├── llm_client.py        # Ollama LLM HTTP interface
├── database.py          # ChromaDB + vLLM OpenAI vector embeddings
├── templates/           # Web dashboard HTML files
├── static/              # Web dashboard CSS/JS assets
├── search_config.json   # Search criteria & portal list
├── profile.json         # Your application profile
└── requirements.txt     # Python dependencies
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
MIT
