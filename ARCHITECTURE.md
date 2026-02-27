# HireAI — High-Level Architecture Overview

> **AI-powered recruitment pipeline** using CrewAI, Ollama, and Streamlit. Fully local inference with optional web-based candidate sourcing.

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              HireAI System                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐     ┌──────────────────┐     ┌─────────────────────────┐  │
│  │  Streamlit  │────▶│  CrewAI Pipeline  │────▶│  Ollama (Local LLM)     │  │
│  │  Dashboard  │     │  (Multi-Agent)    │     │  llama3.2 / mistral     │  │
│  └─────────────┘     └──────────────────┘     └─────────────────────────┘  │
│         │                        │                         │                │
│         │                        │                         │                │
│         ▼                        ▼                         ▼                │
│  ┌─────────────┐     ┌──────────────────┐     ┌─────────────────────────┐  │
│  │  PDF Report │     │  Serper API       │     │  resumes/ folder         │  │
│  │  Outreach   │     │  (optional)       │     │  job_description.txt    │  │
│  └─────────────┘     └──────────────────┘     └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Operating Modes

| Mode | Description | Candidate Source | Agents Used |
|------|-------------|------------------|-------------|
| **Resume Mode** | Process uploaded/local resumes | `resumes/` folder or file upload | Sourcing, Screener, Coordinator, Interview |
| **Live Web Scouting** | Search web for candidates | Serper API (LinkedIn, GitHub, portfolios) | Scout, Researcher, Engagement, Screener, Coordinator, Interview |
| **Human Approval Layer** | Two-stage flow with manual review | Part 1 → Review → Part 2 | Same as above, split into Part 1 & Part 2 |

---

## 3. Agent Architecture

### 3.1 Core Agents (Both Modes)

| Agent | Role | Responsibility |
|-------|------|-----------------|
| **Sourcing Specialist** | Parse job requirements | Extract skills, experience, qualifications from JD |
| **Resume Screener** | Score candidates | Match resumes/profiles to JD; assign 1–10 score + justification |
| **Outreach Coordinator** | Draft outreach | Personalized emails per candidate |
| **Senior Technical Interviewer** | Simulate interviews | 3-turn technical Q&A per candidate |
| **Quantitative Scoring Specialist** | Match grades | Technical (50%) + Experience (30%) + Interview (20%) |

### 3.2 Live Scouting–Only Agents

| Agent | Role | Responsibility |
|-------|------|-----------------|
| **Live Talent Scout** | Web search | Find candidates on LinkedIn, GitHub, portfolios (Serper) |
| **Lead Researcher** | Technical verification | One technical achievement per candidate |
| **Engagement Specialist** | LinkedIn drafts | Connection requests under 300 chars |

---

## 4. Pipeline Flows

### 4.1 Resume Mode (Full Pipeline)

```
Job Description + Resumes
         │
         ▼
┌─────────────────┐
│ Sourcing        │  Parse JD requirements
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Screener        │  Score each resume (1–10)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Coordinator     │  Draft outreach emails
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Interview       │  3-turn interview simulation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Quantitative    │  Match grades
│ Scoring         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Reporting       │  Final PDF + Candidate Comparison Table
└─────────────────┘
```

### 4.2 Live Web Scouting (Full Pipeline)

```
Job Description + Serper API Key
         │
         ▼
┌─────────────────┐
│ Scout           │  Search web for candidates
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Researcher      │  Technical achievements per candidate
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Engagement      │  LinkedIn connection drafts
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Screener        │  Score web-sourced profiles
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Coordinator     │  Outreach emails
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Interview       │  3-turn simulation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Scoring         │  Match grades
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Reporting       │  Final PDF
└─────────────────┘
```

### 4.3 Human Approval Layer (Two-Stage)

```
Part 1: Sourcing & Research
         │
         ▼
┌─────────────────────────────────┐
│ Review Table (st.data_editor)   │
│ ☑ Approve for Interview         │
│ ☐ Reject (with reason)          │
└────────┬────────────────────────┘
         │
         ▼
   [Proceed to AI Interview]
         │
         ▼
Part 2: Interview + Scoring + Reporting
         │
         ▼
   Final PDF (approved candidates only)
```

---

## 5. Key Components

### 5.1 File Structure

```
HireAI/
├── app.py              # Streamlit UI, pipeline orchestration
├── main.py             # CrewAI agents, tasks, pipeline logic
├── job_description.txt # Default JD
├── resumes/            # Resume files (.pdf, .txt)
├── outreach_drafts/    # Per-candidate outreach outputs
├── .hireai_audit/      # Audit logs (Serper searches, task outputs)
├── .hireai_rejections.json  # Rejection reasons (for Scout learning)
├── Final_Recruitment_Report.pdf
└── requirements.txt
```

### 5.2 Outputs

| Output | Description |
|--------|-------------|
| **Candidate Comparison Table** | Rank, name, fit score, justification |
| **LinkedIn Connection Requests** | Name, profile link, drafted message (Live Scouting) |
| **Interview Simulation** | 3-turn Q&A per candidate |
| **Score Justifications** | Technical, Experience, Interview (expandable) |
| **Final PDF Report** | Executive summary, candidate details, next steps |
| **Outreach drafts** | `{candidate}_outreach.txt` per candidate |

---

## 6. Configuration & Features

### 6.1 Sidebar Controls

| Control | Purpose |
|---------|---------|
| **Model** | Ollama model (llama3.2, mistral, phi3, etc.) |
| **Fast mode** | Limit output to 1024 tokens for speed |
| **Temperature** | 0.1–1.0 (lower = stricter) |
| **Live Web Scouting** | Enable Serper-based candidate search |
| **Serper API Key** | Required for Live Scouting |
| **Safe Mode** | Skip failed Serper calls instead of crashing |
| **Human Approval Layer** | Two-stage flow with manual review |
| **Pause for human approval** | Coordinator pauses for human input |

### 6.2 Safeguards

| Safeguard | Description |
|-----------|-------------|
| **Resume count cap** | In resume mode, candidates ≤ number of resumes (prevents LLM hallucination) |
| **Approved-only filter** | Part 2 shows only approved candidates |
| **Rejection learning** | Rejected reasons stored in `.hireai_rejections.json` for future Scout runs |

---

## 7. Technology Stack

| Layer | Technology |
|-------|------------|
| **UI** | Streamlit |
| **Orchestration** | CrewAI (multi-agent) |
| **LLM** | Ollama (local inference) |
| **Web Search** | Serper API (optional) |
| **PDF** | fpdf2 |
| **Resume Parsing** | pdfplumber |

---

## 8. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `llama3.2` | Default model |
| `OLLAMA_TIMEOUT` | `1800` | Request timeout (seconds) |
| `SERPER_API_KEY` | — | Required for Live Scouting |

---

## 9. Event Flow (Real-Time UI)

```
CrewAI TaskStartedEvent  ──▶  task_callback(stage, False)  ──▶  current_activity = stage
CrewAI TaskCompletedEvent ──▶  task_callback(stage, True)  ──▶  lifecycle[stage] = ✓
StreamlitStdout          ──▶  Thought Stream (stdout/stderr)
```

---

## 10. Data Flow Summary

```
Inputs:  job_description.txt, resumes/*.pdf|*.txt, (optional) Serper API key
         │
         ▼
Pipeline: Sourcing → Screener → Coordinator → Interview → Scoring → Reporting
         │
         ▼
Outputs: ranking_table, engagement_data, interview_data, scoring_data
         │
         ▼
UI:      Candidate Comparison Table, LinkedIn table, Interview expanders, PDF download
```

---

*Last updated: February 2025*
