# HireAI

**AI-powered recruitment pipeline** using CrewAI, Ollama, and Streamlit. Fully local inference with optional web-based candidate sourcing.

---

## Features

- **Resume Mode** — Score and rank candidates from uploaded or local resumes (`.pdf`, `.txt`)
- **Live Web Scouting** — Search LinkedIn, GitHub, and portfolios for candidates (requires Serper API)
- **Human Approval Layer** — Two-stage flow: review candidates → approve → AI interview
- **Quantitative Scoring** — Match grades (50% Technical + 30% Experience + 20% Interview)
- **Interview Simulation** — 3-turn technical Q&A per candidate
- **Outreach Drafts** — Personalized emails and LinkedIn connection requests
- **PDF Report** — Final recruitment report with executive summary and next steps

---

## Prerequisites

1. **Python 3.10+**
2. **Ollama** — [Install](https://ollama.ai) and run:
   ```bash
   ollama serve
   ollama pull llama3.2
   ```
3. **Serper API Key** (optional) — For Live Web Scouting. Get one at [serper.dev](https://serper.dev)

---

## Installation

```bash
# Clone or navigate to the project
cd HireAI

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Streamlit Dashboard (recommended)

```bash
streamlit run app.py
```

Open the URL shown in the terminal (default: `http://localhost:8501`).

### Command Line

```bash
python main.py
```

Uses `job_description.txt` and resumes from the `resumes/` folder.

---

## Configuration

| File / Folder | Purpose |
|---------------|---------|
| `job_description.txt` | Default job description |
| `resumes/` | Resume files (`.pdf`, `.txt`) |
| `.env` | Optional: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT`, `SERPER_API_KEY` |

### Sidebar Options (Streamlit)

- **Model** — Ollama model (llama3.2, mistral, phi3, etc.)
- **Fast mode** — Limit output for faster runs
- **Live Web Scouting** — Enable web search for candidates
- **Human Approval Layer** — Review before AI interview
- **Safe Mode** — Skip failed Serper calls instead of crashing

---

## Project Structure

```
HireAI/
├── app.py              # Streamlit UI
├── main.py             # CrewAI agents & pipeline
├── job_description.txt # Default JD
├── resumes/            # Resume files
├── outreach_drafts/    # Per-candidate outreach
├── ARCHITECTURE.md     # High-level architecture
├── README.md
└── requirements.txt
```

---

## Outputs

- **Candidate Comparison Table** — Rank, name, fit score, justification
- **LinkedIn Connection Requests** — Drafts under 300 chars (Live Scouting)
- **Interview Simulation** — 3-turn Q&A per candidate
- **Final_Recruitment_Report.pdf** — Executive summary, candidate details, next steps
- **`{name}_outreach.txt`** — Per-candidate outreach drafts

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the high-level system design, agent roles, and pipeline flows.

---

## License

MIT
