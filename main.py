"""
Local Multi-Agent Recruitment System using CrewAI and LangChain.
Uses Ollama (llama3.2) - no cloud APIs, fully local.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

import pdfplumber
from fpdf import FPDF
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.events import crewai_event_bus, TaskCompletedEvent

try:
    from crewai_tools import SerperDevTool
except ImportError:
    SerperDevTool = None


def create_auditable_serper_tool(n_results: int = 5, audit_log: list | None = None, safe_mode: bool = False):
    """Create SerperDevTool that logs search queries to audit_log for Product Owner review.
    When safe_mode=True, catches API errors and returns fallback instead of crashing."""
    if SerperDevTool is None:
        return None
    if audit_log is None and not safe_mode:
        return SerperDevTool(n_results=n_results)

    class AuditableSerperTool(SerperDevTool):
        """SerperDevTool subclass that logs every search query for audit. Safe mode skips failed searches."""

        def __init__(self, n_results: int = 5, audit_log: list | None = None, safe_mode: bool = False):
            super().__init__(n_results=n_results)
            self._audit_log = audit_log or []
            self._safe_mode = safe_mode

        def _run(self, search_query: str = "", **kwargs):
            self._audit_log.append({
                "timestamp": datetime.now().isoformat(),
                "type": "serper_search",
                "query": search_query,
            })
            try:
                result = super()._run(search_query=search_query, **kwargs)
                preview = str(result)[:500] + ("..." if len(str(result)) > 500 else "")
                self._audit_log[-1]["result_preview"] = preview
                return result
            except Exception as e:
                self._audit_log[-1]["error"] = str(e)
                self._audit_log[-1]["skipped"] = True
                if self._safe_mode:
                    return f"[Search skipped - API or profile unavailable: {str(e)[:100]}]. Continue with next candidate."
                raise

    return AuditableSerperTool(n_results=n_results, audit_log=audit_log, safe_mode=safe_mode)

# Load environment variables
load_dotenv()

# Ollama LLM configuration (local only)
# CrewAI requires "ollama/<model>" prefix to use Ollama provider (avoids OpenAI fallback)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
# Timeout in seconds for Ollama API calls (default 30 min; increase if model is slow)
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "1800"))
# Set LiteLLM request timeout (avoids "Connection timed out after 600 seconds")
os.environ.setdefault("LITELLM_REQUEST_TIMEOUT", str(OLLAMA_TIMEOUT))


def check_ollama_running() -> tuple[bool, str]:
    """Check if Ollama service is running. Returns (success, message)."""
    try:
        base = OLLAMA_BASE_URL.replace("/v1", "").rstrip("/")
        with urlopen(f"{base}/api/tags", timeout=5) as _:
            return True, "Ollama is running."
    except URLError as e:
        return False, "Cannot connect to Ollama. Ensure it is running (e.g. run 'ollama serve')."
    except Exception as e:
        return False, str(e)


llm = LLM(
    model=f"ollama/{OLLAMA_MODEL}",
    base_url=OLLAMA_BASE_URL,
    temperature=0.3,
    timeout=OLLAMA_TIMEOUT,
)


# --- Agents ---

sourcing_agent = Agent(
    role="Sourcing Specialist",
    goal="Parse and extract job requirements from job descriptions accurately and comprehensively",
    backstory="""You are an expert recruiter with years of experience analyzing job descriptions.
    You excel at identifying key requirements: skills, experience levels, qualifications,
    responsibilities, and soft skills. You structure information clearly for downstream use.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

screener_agent = Agent(
    role="Resume Screener",
    goal="Match resumes against job requirements and assign objective scores based on fit",
    backstory="""You are a meticulous recruiter who evaluates candidate resumes against job
    requirements. You score candidates on a scale of 1-10 based on: skill match, experience
    relevance, qualifications, and overall fit. You provide clear, justified scores.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

coordinator_agent = Agent(
    role="Outreach Coordinator",
    goal="Draft personalized email outreach based on candidate scores and job context",
    backstory="""You are an expert at crafting outreach emails that recruit candidates.
    You tailor each email based on the candidate's score and fit. High-scoring candidates
    get more enthusiastic, detailed outreach. You maintain a professional, engaging tone.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

interview_specialist = Agent(
    role="Senior Technical Interviewer",
    goal="Simulate a 3-turn technical interview for each top candidate",
    backstory="""You are an expert technical interviewer who simulates realistic interview conversations.
    You create technical challenges based on the job requirements, then simulate how each specific
    candidate would likely respond based on their skills and background. You follow up with probing
    questions that test depth of knowledge.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

lead_researcher_agent = Agent(
    role="Lead Researcher",
    goal="Verify the technical background of leads by finding concrete technical achievements",
    backstory="""You are a thorough researcher who validates candidate claims. You use web search
    to find evidence of technical work: open-source projects, conference talks, blog posts,
    or notable contributions. You uncover one specific Technical Achievement per candidate.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

engagement_specialist_agent = Agent(
    role="Engagement Specialist",
    goal="Draft high-conversion LinkedIn connection requests using Technical Highlights",
    backstory="""You are a professional recruiter specializing in high-conversion outreach.
    You craft personalized, concise LinkedIn connection requests that reference the candidate's
    technical achievements. Your messages are warm, professional, and under 300 characters.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

scoring_agent = Agent(
    role="Quantitative Scoring Specialist",
    goal="Produce objective, justified scores for each candidate using a weighted scoring model",
    backstory="""You are a data-driven recruiter who assigns quantitative scores with clear justifications.
    You evaluate Technical_Score (skills match), Experience_Score (relevant experience), and Interview_Score
    (quality of simulated interview response). For each score you provide a 1-sentence justification
    (e.g., 'Deducted 10 points for lack of Kubernetes experience'). You calculate Match_Grade using
    weights 50% Technical, 30% Experience, 20% Interview.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)


def create_agents_with_config(
    model: str,
    temperature: float,
    enable_live_scouting: bool = False,
    serper_api_key: str | None = None,
    audit_log: list | None = None,
    enable_safe_mode: bool = False,
):
    """Create agents with custom model and temperature (for Streamlit/configurable runs).
    When enable_live_scouting is True and serper_api_key is provided, the Sourcing agent
    becomes a Live Talent Scout with SerperDevTool for web search."""
    custom_llm = LLM(
        model=f"ollama/{model}",
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        timeout=OLLAMA_TIMEOUT,
    )

    # Sourcing agent: standard or Live Talent Scout (with SerperDevTool)
    if enable_live_scouting and serper_api_key and SerperDevTool is not None:
        os.environ["SERPER_API_KEY"] = serper_api_key
        serper_tool = create_auditable_serper_tool(n_results=5, audit_log=audit_log, safe_mode=enable_safe_mode)
        sourcing_agent_instance = Agent(
            role="Live Talent Scout",
            goal="Search the web for candidates matching job requirements (LinkedIn, GitHub, portfolios)",
            backstory="""You are an expert talent scout who finds candidates online. You use web search
            to discover professionals on LinkedIn, GitHub, personal portfolios, and tech communities.
            You identify relevant candidates based on skills, experience, and public profiles.
            Output each candidate's profile URL clearly so the Lead Researcher can verify their background.""",
            llm=custom_llm,
            verbose=True,
            allow_delegation=False,
            tools=[serper_tool],
        )
        lead_researcher_instance = Agent(
            role="Lead Researcher",
            goal="Verify the technical background of leads by finding concrete technical achievements",
            backstory="""You are a thorough researcher who validates candidate claims. You use web search
            to find evidence of technical work: open-source projects, conference talks, blog posts,
            or notable contributions. You uncover one specific Technical Achievement per candidate.""",
            llm=custom_llm,
            verbose=True,
            allow_delegation=False,
            tools=[serper_tool],
        )
    else:
        sourcing_agent_instance = Agent(
            role="Sourcing Specialist",
            goal="Parse and extract job requirements from job descriptions accurately and comprehensively",
            backstory="""You are an expert recruiter with years of experience analyzing job descriptions.
            You excel at identifying key requirements: skills, experience levels, qualifications,
            responsibilities, and soft skills. You structure information clearly for downstream use.""",
            llm=custom_llm,
            verbose=True,
            allow_delegation=False,
        )

    result = {
        "sourcing": sourcing_agent_instance,
        "screener": Agent(
            role="Resume Screener",
            goal="Match resumes against job requirements and assign objective scores based on fit",
            backstory="""You are a meticulous recruiter who evaluates candidate resumes against job
            requirements. You score candidates on a scale of 1-10 based on: skill match, experience
            relevance, qualifications, and overall fit. You provide clear, justified scores.""",
            llm=custom_llm,
            verbose=True,
            allow_delegation=False,
        ),
        "coordinator": Agent(
            role="Outreach Coordinator",
            goal="Draft personalized email outreach based on candidate scores and job context",
            backstory="""You are an expert at crafting outreach emails that recruit candidates.
            You tailor each email based on the candidate's score and fit. High-scoring candidates
            get more enthusiastic, detailed outreach. You maintain a professional, engaging tone.""",
            llm=custom_llm,
            verbose=True,
            allow_delegation=False,
        ),
        "interview": Agent(
            role="Senior Technical Interviewer",
            goal="Simulate a 3-turn technical interview for each top candidate",
            backstory="""You are an expert technical interviewer who simulates realistic interview conversations.
            You create technical challenges based on the job requirements, then simulate how each specific
            candidate would likely respond based on their skills and background. You follow up with probing
            questions that test depth of knowledge.""",
            llm=custom_llm,
            verbose=True,
            allow_delegation=False,
        ),
        "scoring": Agent(
            role="Quantitative Scoring Specialist",
            goal="Produce objective, justified scores for each candidate using a weighted scoring model",
            backstory="""You are a data-driven recruiter who assigns quantitative scores with clear justifications.
            You evaluate Technical_Score (skills match), Experience_Score (relevant experience), and Interview_Score
            (quality of simulated interview response). For each score you provide a 1-sentence justification.
            You calculate Match_Grade using weights 50% Technical, 30% Experience, 20% Interview.""",
            llm=custom_llm,
            verbose=True,
            allow_delegation=False,
        ),
    }
    if enable_live_scouting and serper_api_key and SerperDevTool is not None:
        result["researcher"] = lead_researcher_instance
        result["engagement"] = Agent(
            role="Engagement Specialist",
            goal="Draft high-conversion LinkedIn connection requests using Technical Highlights",
            backstory="""You are a professional recruiter specializing in high-conversion outreach.
            You craft personalized, concise LinkedIn connection requests that reference the candidate's
            technical achievements. Your messages are warm, professional, and under 300 characters.""",
            llm=custom_llm,
            verbose=True,
            allow_delegation=False,
        )
    return result


# --- Tasks ---

REJECTIONS_FILE = Path(__file__).resolve().parent / ".hireai_rejections.json"


def load_rejection_reasons() -> dict[str, list[str]]:
    """Load persisted rejection reasons: {candidate_name: [reason1, reason2, ...]}."""
    if not REJECTIONS_FILE.exists():
        return {}
    try:
        with open(REJECTIONS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_rejection_reason(candidate_name: str, reason: str) -> None:
    """Persist a rejection reason so the AI can learn to avoid similar profiles in future searches."""
    data = load_rejection_reasons()
    if candidate_name not in data:
        data[candidate_name] = []
    data[candidate_name].append(reason)
    REJECTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REJECTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def create_sourcing_task(
    job_description: str, agent=None, enable_live_scouting: bool = False, rejection_reasons: list[str] | None = None
) -> Task:
    """Create a task to parse JD requirements or search the web for candidates (Live Scouting)."""
    if enable_live_scouting:
        rejection_hint = ""
        if rejection_reasons:
            reasons_text = "; ".join(rejection_reasons[-10:])  # Last 10 to avoid token bloat
            rejection_hint = f"\n\nAVOID profiles similar to these past rejections: {reasons_text}"
        return Task(
            description=f"""Search the web for candidates that match this job description.
            Use your search tool to find:
            - LinkedIn profiles of professionals with relevant skills
            - GitHub repositories and profiles of developers
            - Personal portfolios and professional websites
            Limit to the top 5 most relevant candidates to keep the pipeline efficient.
            For each candidate found, capture: name (or username), profile URL, and a brief summary of their fit.
            Output a structured list of candidate profiles suitable for downstream screening.
            {rejection_hint}

            Job Description:
            {job_description}""",
            expected_output="A structured list of candidate profiles (name/username, profile URL, brief fit summary) found from LinkedIn, GitHub, or portfolios. Include the full LinkedIn/GitHub URL for each candidate so the Lead Researcher can verify their technical background.",
            agent=agent or sourcing_agent,
        )
    return Task(
        description=f"""Parse and extract all requirements from this job description.
        Identify: required skills, preferred skills, experience level, education, responsibilities,
        and any other relevant criteria. Output a structured summary suitable for resume matching.

        Job Description:
        {job_description}""",
        expected_output="A structured summary of job requirements including skills, experience, qualifications, and responsibilities.",
        agent=agent or sourcing_agent,
    )


def create_researcher_task(agent=None) -> Task:
    """Create a task for Lead Researcher to find one Technical Achievement per candidate using SerperDevTool."""
    return Task(
        description="""Using the candidate profiles from the Live Talent Scout (provided in context), extract each candidate's LinkedIn/GitHub/profile URLs.
        For EACH candidate, use your search tool ONCE to find ONE specific Technical Achievement. Be efficient—one targeted search per candidate. Examples:
        - A notable open-source project they contributed to
        - A conference talk or webinar they gave
        - A technical blog post or article they wrote
        - A specific GitHub repo they built or maintain
        Output a list: for each candidate, provide their name and one concrete Technical Achievement (with source/link if found).""",
        expected_output="A list of candidates with one Technical Achievement each, e.g. 'Name: [specific project/talk/contribution]'.",
        agent=agent or lead_researcher_agent,
    )


def create_engagement_task(agent=None) -> Task:
    """Create a task for Engagement Specialist to draft LinkedIn connection requests (under 300 chars each)."""
    return Task(
        description="""Using the candidate profiles (names, profile URLs) from the Live Talent Scout and the Technical Highlights from the Lead Researcher (both provided in context), draft a personalized LinkedIn connection request for EACH candidate.
        Each message MUST be under 300 characters (LinkedIn's limit). Reference their specific technical achievement to personalize the outreach.
        Output EXACTLY in this format, one candidate per line (use pipe | as separator, no line breaks within a row):
        NAME|PROFILE_LINK|DRAFTED_MESSAGE
        Example:
        John Doe|https://linkedin.com/in/johndoe|Hi John! Impressed by your open-source CLI tool. Would love to connect about a role that fits your Python expertise.
        Jane Smith|https://github.com/janesmith|Hi Jane! Your PyCon talk on async Python caught my eye. Would enjoy connecting about opportunities.""",
        expected_output="A table with one row per candidate: NAME|PROFILE_LINK|DRAFTED_MESSAGE. Each message under 300 characters.",
        agent=agent or engagement_specialist_agent,
    )


def create_screener_task(resumes: str, agent=None, enable_live_scouting: bool = False) -> Task:
    """Create a task to score resumes/candidates against JD. Requirements/candidates come from sourcing_task context."""
    if enable_live_scouting:
        return Task(
            description="""Score the candidate profiles provided in context from the Live Talent Scout's web search.
            Each candidate has a name/username, profile URL, and summary. Match them against the job requirements
            (inferred from the search context) and score each on a scale of 1-10.
            For each candidate, provide: name, score, and brief justification for the score.""",
            expected_output="A list of candidates with scores (1-10) and brief justifications for each score.",
            agent=agent or screener_agent,
        )
    return Task(
        description=f"""Match each resume against the job requirements (provided in context from the Sourcing Specialist) and score them on a scale of 1-10.
        For each candidate, provide: name, score, and brief justification for the score.

        Resumes to Score:
        {resumes}""",
        expected_output="A list of candidates with scores (1-10) and brief justifications for each score.",
        agent=agent or screener_agent,
    )


def create_coordinator_task(agent=None, human_input: bool = False) -> Task:
    """Create a task to draft outreach emails. Screening results and job context come from previous tasks.
    When human_input is True, the crew pauses for human review before finalizing."""
    return Task(
        description="""Draft personalized outreach emails for each candidate based on their scores (provided in context from the Screener).
        Higher-scoring candidates should receive more enthusiastic, detailed outreach.
        Lower-scoring candidates can receive shorter, still professional outreach.
        Include: subject line, greeting, and body for each candidate.""",
        expected_output="A set of personalized outreach emails (subject + body) for each candidate, tailored to their score.",
        agent=agent or coordinator_agent,
        human_input=human_input,
    )


def create_generate_questions_task(
    agent=None,
    job_description: str = "",
    approved_names: list[str] | None = None,
    screener_output: str | None = None,
    researcher_output: str | None = None,
) -> Task:
    """Create a task to simulate a 3-turn technical interview for each candidate.
    When screener_output/researcher_output are provided, use them directly (Part 2 standalone).
    When approved_names is provided, ONLY simulate for those candidates (human approval layer)."""
    filter_instruction = ""
    if approved_names:
        names_str = ", ".join(approved_names)
        filter_instruction = f"\n\nIMPORTANT: ONLY simulate interviews for these approved candidates: {names_str}. Do NOT include any other candidates."
    context_section = ""
    if screener_output or researcher_output:
        parts = []
        if screener_output:
            parts.append(f"Screening Results:\n{screener_output[:4000]}")
        if researcher_output:
            parts.append(f"Researcher Technical Highlights:\n{researcher_output[:2000]}")
        context_section = "\n\n".join(parts) + "\n\n"
    return Task(
        description=f"""SIMULATE a 3-turn technical interview for EACH top candidate found by the Screener.
        Use the Screener's assessment and Researcher's Technical Highlights (provided below or in context) to understand each candidate's skills.
        {filter_instruction}
        {context_section}

        For EACH candidate, produce exactly 3 turns in this format:
        --- CANDIDATE: [Name] ---
        Question 1: [A technical challenge based on the Job Description - test a key skill from the role]
        Simulated Answer: [How THIS specific candidate would likely respond, based on their found skills, experience, and technical achievements]
        Follow-up: [A probing question that tests the candidate's depth - goes deeper based on their simulated answer]

        Job Description (use for technical challenges):
        {job_description}

        Output format: Use "--- CANDIDATE: [Name] ---" then "Question 1:", "Simulated Answer:", "Follow-up:" for each candidate.""",
        expected_output="For each candidate: a 3-turn interview simulation (Question 1, Simulated Answer, Follow-up) in the format above.",
        agent=agent or interview_specialist,
    )


def create_reporting_task(
    scout_output: str,
    researcher_output: str,
    interview_output: str,
    screener_output: str,
    scoring_data: list[dict] | None = None,
    engagement_output: str | None = None,
    outreach_output: str | None = None,
    job_description: str = "",
    agent=None,
) -> Task:
    """Create a final Reporting Task for the Coordinator: consolidate all data, executive summary, Markdown format, Next Steps."""
    sections = [
        f"## Scout's Candidate Links & Profiles\n{scout_output}",
        f"## Researcher's Technical Highlights\n{researcher_output}",
        f"## Interview Simulations (3-turn)\n{interview_output}",
        f"## Screener's Scores & Assessments\n{screener_output}",
    ]
    if engagement_output:
        sections.append(f"## LinkedIn Connection Requests\n{engagement_output}")
    if outreach_output:
        sections.append(f"## Outreach Emails\n{outreach_output}")
    if scoring_data:
        scoring_text = "\n".join(
            f"- **{c['Candidate']}**: Match Grade {c['Match_Grade']:.1f} (Tech: {c['Technical_Score']}, Exp: {c['Experience_Score']}, Interview: {c['Interview_Score']})"
            for c in scoring_data
        )
        sections.append(f"## Quantitative Scoring (50/30/20)\n{scoring_text}")

    return Task(
        description=f"""You are producing the FINAL RECRUITMENT REPORT. Aggregate and format ALL the data below into a professional, structured Markdown document.

**Data to consolidate:**
- Scout's candidate links and profiles
- Researcher's technical highlights per candidate
- Interviewer's 3-turn simulation per candidate
- Screener's scores and assessments
{f"- Quantitative Scoring (Match Grades) - include in summary" if scoring_data else ""}

**Required structure (use Markdown headings, lists, tables):**

# Recruitment Report - [Role/Position]

## Executive Summary
Write exactly 2 paragraphs:
1. **Talent Market Overview**: Summarize the quality and availability of candidates for this role. What skills are strong? Any gaps? Overall market assessment.
2. **Key Findings**: Highlight the top 2-3 candidates and why they stand out. Mention any notable technical achievements or concerns.

## Candidate Details
For each candidate, include: Name, Profile Link, Technical Highlight, Screener Score, Interview Summary (brief), Match Grade (if available).

## Next Steps
Provide a clear Call to Action for the recruiter. Use a numbered list, e.g.:
1. Schedule a call with Candidate #1 ([Name]) - top match
2. Send LinkedIn connection request to Candidate #2 ([Name])
3. [Additional actionable next steps]

---

**Raw data to consolidate:**

Job Description (summary):
{job_description[:1500] if job_description else "N/A"}

{chr(10).join(sections)}""",
        expected_output="A complete Markdown report with Executive Summary (2 paragraphs), Candidate Details, and Next Steps. Use ## for sections, - for lists. Format for easy PDF conversion.",
        agent=agent or coordinator_agent,
    )


def create_quantitative_scoring_task(
    screening_results: str,
    researcher_output: str | None = None,
    interview_output: str | None = None,
    agent=None,
) -> Task:
    """Create a task for Quantitative Scoring: JSON with Technical_Score, Experience_Score, Interview_Score,
    Justifications, and Match_Grade (50/30/20 weighted). Output sorted by Match_Grade descending."""
    context_parts = [f"Screening Results:\n{screening_results}"]
    if researcher_output:
        context_parts.append(f"Researcher Technical Achievements:\n{researcher_output}")
    if interview_output:
        context_parts.append(f"Interview Simulations:\n{interview_output}")

    return Task(
        description=f"""For EVERY candidate found in the screening results, generate a JSON object with:
- Technical_Score (0-100): Skill match against job requirements
- Technical_Justification: 1-sentence reason (e.g., "Deducted 10 points for lack of Kubernetes experience")
- Experience_Score (0-100): Relevant experience level
- Experience_Justification: 1-sentence reason
- Interview_Score (0-100): Quality of simulated interview response
- Interview_Justification: 1-sentence reason
- Match_Grade: Weighted average = (Technical_Score * 0.5) + (Experience_Score * 0.3) + (Interview_Score * 0.2)

Output a valid JSON array, one object per candidate, sorted from highest Match_Grade to lowest.
Example format:
[
  {{"Candidate": "John Doe", "Technical_Score": 85, "Technical_Justification": "...", "Experience_Score": 80, "Experience_Justification": "...", "Interview_Score": 75, "Interview_Justification": "...", "Match_Grade": 81.5}},
  ...
]

{chr(10).join(context_parts)}""",
        expected_output="A JSON array of scored candidates, sorted by Match_Grade descending.",
        agent=agent or scoring_agent,
    )


def parse_quantitative_scoring_output(output: str) -> list[dict]:
    """Parse Quantitative Scoring JSON output. Returns list of dicts with Candidate, scores, justifications, Match_Grade.
    Falls back to empty list if parsing fails."""
    import re
    if not output or not output.strip():
        return []
    # Extract JSON array (handle markdown code blocks)
    text = output.strip()
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []
    try:
        data = json.loads(match.group())
        if not isinstance(data, list):
            return []
        # Normalize keys, ensure Match_Grade
        result = []
        for item in data:
            if not isinstance(item, dict):
                continue
            name = item.get("Candidate") or item.get("candidate") or item.get("name") or "Unknown"
            tech = item.get("Technical_Score") or item.get("technical_score") or 0
            exp = item.get("Experience_Score") or item.get("experience_score") or 0
            interv = item.get("Interview_Score") or item.get("interview_score") or 0
            grade = item.get("Match_Grade")
            if grade is None:
                grade = (float(tech) * 0.5) + (float(exp) * 0.3) + (float(interv) * 0.2)
            result.append({
                "Candidate": name,
                "Technical_Score": tech,
                "Technical_Justification": item.get("Technical_Justification") or item.get("technical_justification") or "",
                "Experience_Score": exp,
                "Experience_Justification": item.get("Experience_Justification") or item.get("experience_justification") or "",
                "Interview_Score": interv,
                "Interview_Justification": item.get("Interview_Justification") or item.get("interview_justification") or "",
                "Match_Grade": float(grade),
            })
        return sorted(result, key=lambda x: x["Match_Grade"], reverse=True)
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def create_rank_candidates_task(
    screening_results: str,
    agent=None,
    researcher_output: str | None = None,
) -> Task:
    """Create a task for the Coordinator to rank all candidates by score.
    When researcher_output is provided (Live Scouting), include Technical Highlights in output."""
    if researcher_output:
        return Task(
            description=f"""Review the screening results and researcher output below. Create a ranked list from highest score to lowest.
            For each candidate, extract: name, fit score (1-10), and one Technical Achievement from the Researcher (if available).
            Output EXACTLY in this format, one candidate per line (use pipe | as separator):
            RANK|CANDIDATE_NAME|SCORE|TECHNICAL_HIGHLIGHT
            Example:
            1|John Doe|9|Built open-source CLI tool with 2k stars
            2|Jane Smith|7|Gave PyCon talk on async Python

            Screening Results:
            {screening_results}

            Researcher Technical Achievements:
            {researcher_output}""",
            expected_output="A ranked list in format RANK|CANDIDATE_NAME|SCORE|TECHNICAL_HIGHLIGHT, one line per candidate.",
            agent=agent or coordinator_agent,
        )
    return Task(
        description=f"""Review all screening results below and create a simple ranked list from highest score to lowest.
        For each candidate, extract their name and fit score (1-10).
        Output EXACTLY in this format, one candidate per line (use pipe | as separator):
        RANK|CANDIDATE_NAME|SCORE
        Example:
        1|John Doe|9
        2|Jane Smith|7
        3|Bob Wilson|6

        Screening Results:
        {screening_results}""",
        expected_output="A ranked list in format RANK|CANDIDATE_NAME|SCORE, one line per candidate, sorted highest to lowest score.",
        agent=agent or coordinator_agent,
    )


def parse_interview_simulation_output(
    output: str, candidate_names: list[str] | None = None
) -> dict[str, dict[str, str]]:
    """Parse 3-turn interview simulation into {candidate_name: {question1, simulated_answer, follow_up}}."""
    import re
    result = {}
    if not output or not output.strip():
        return result

    def extract_turns(text: str) -> dict[str, str] | None:
        data = {}
        # Match "Question 1:" or "Question 1 :" etc.
        m1 = re.search(r"Question\s*1\s*:\s*(.+?)(?=Simulated Answer:|$)", text, re.DOTALL | re.IGNORECASE)
        m2 = re.search(r"Simulated\s*Answer\s*:\s*(.+?)(?=Follow-up:|$)", text, re.DOTALL | re.IGNORECASE)
        m3 = re.search(r"Follow-up\s*:\s*(.+?)(?=---|$)", text, re.DOTALL | re.IGNORECASE)
        if m1:
            data["question1"] = m1.group(1).strip()
        if m2:
            data["simulated_answer"] = m2.group(1).strip()
        if m3:
            data["follow_up"] = m3.group(1).strip()
        return data if data else None

    blocks = re.split(r"---\s*CANDIDATE:\s*", output, flags=re.IGNORECASE)
    for block in blocks[1:]:
        parts = block.split("---", 1)
        name = parts[0].strip()
        content = parts[1].strip() if len(parts) > 1 else block
        turns = extract_turns(content)
        if turns:
            result[name] = turns

    if not result and candidate_names:
        turns = extract_turns(output)
        if turns:
            result[candidate_names[0]] = turns

    return result


def parse_interview_output(
    output: str, candidate_names: list[str] | None = None
) -> dict[str, list[tuple[str, str]]] | dict[str, dict[str, str]]:
    """Parse Interview Specialist output. Returns either:
    - 3-turn simulation: {name: {question1, simulated_answer, follow_up}}
    - Legacy Q&A: {name: [(question, expected_answer), ...]}"""
    sim = parse_interview_simulation_output(output, candidate_names)
    if sim:
        return sim

    result = {}
    if not output or not output.strip():
        return result

    def extract_qa(text: str) -> list[tuple[str, str]]:
        qa_list = []
        for m in re.finditer(r"(\d+)\.\s*(.+?)\s+Expected:\s*(.+?)(?=\d+\.|$)", text, re.DOTALL):
            q, exp = m.group(2).strip(), m.group(3).strip()
            qa_list.append((q, exp))
        return qa_list

    blocks = re.split(r"---\s*CANDIDATE:\s*", output, flags=re.IGNORECASE)
    for block in blocks[1:]:
        parts = block.split("---", 1)
        name = parts[0].strip()
        content = parts[1].strip() if len(parts) > 1 else block
        qa_list = extract_qa(content)
        if qa_list:
            result[name] = qa_list

    if not result:
        qa_list = extract_qa(output)
        if qa_list:
            name = candidate_names[0] if candidate_names else "Candidate"
            result[name] = qa_list

    return result


def parse_engagement_output(output: str) -> list[tuple[str, str, str]]:
    """Parse Engagement Specialist output into (name, profile_link, message) rows."""
    rows = []
    for line in output.strip().split("\n"):
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|", 2)]  # max 3 parts: name, link, message
        if len(parts) >= 3:
            name, link, msg = parts[0], parts[1], parts[2]
            if name.upper() != "NAME":  # Skip header if present
                rows.append((name, link, msg))
    return rows


def parse_ranking_output(output: str, include_highlights: bool = False) -> list[tuple[str, str, str]] | list[tuple[str, str, str, str]]:
    """Parse rank_candidates output into table rows. Returns list of (rank, name, score) or (rank, name, score, highlight)."""
    rows = []
    for line in output.strip().split("\n"):
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|", 3)]  # split into max 4 parts for highlight
        if len(parts) >= 3:
            rank, name, score = parts[0], parts[1], parts[2]
            if rank.upper() != "RANK":  # Skip header line if present
                if include_highlights and len(parts) >= 4:
                    rows.append((rank, name, score, parts[3]))
                else:
                    rows.append((rank, name, score))
    return rows


def save_audit_log(audit_log: list, base_dir: Path) -> Path:
    """Save Product Owner audit log to hidden .hireai_audit/ folder. Returns path to saved file."""
    audit_dir = base_dir / ".hireai_audit"
    audit_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = audit_dir / f"audit_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit_log, f, indent=2, ensure_ascii=False)
    return path


def load_resumes_from_folder(folder_path: str = "resumes") -> list[dict[str, str]]:
    """Read all .pdf and .txt files from a folder. Returns list of dicts with 'filename' and 'content'."""
    folder = Path(__file__).resolve().parent / folder_path
    if not folder.is_dir():
        raise FileNotFoundError(f"Resumes folder not found: {folder}")

    resumes = []
    for filepath in sorted(folder.iterdir()):
        if not filepath.is_file():
            continue
        suffix = filepath.suffix.lower()
        if suffix == ".txt":
            content = filepath.read_text(encoding="utf-8").strip()
        elif suffix == ".pdf":
            content_parts = []
            with pdfplumber.open(str(filepath)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        content_parts.append(text)
            content = "\n".join(content_parts).strip()
        else:
            continue

        if content:
            resumes.append({"filename": filepath.name, "content": content})
    return resumes


def run_recruitment_pipeline(
    job_description: str,
    resumes: str,
    model: str | None = None,
    temperature: float | None = None,
    agents: dict | None = None,
    enable_live_scouting: bool = False,
    serper_api_key: str | None = None,
    enable_human_input: bool = False,
    task_callback=None,
):
    """
    Run the full recruitment pipeline: Sourcing -> Screener -> Coordinator -> Interview Specialist.
    All processing is local via Ollama.
    When enable_live_scouting is True, Sourcing searches the web for candidates instead of parsing JD.
    Optional model, temperature, or pre-created agents override the defaults (for Streamlit/configurable runs).
    Returns (CrewOutput, agents_dict) when agents were created (for rank crew); else (CrewOutput, None).
    """
    if agents is not None:
        pass  # use provided agents
    elif model is not None or temperature is not None:
        agents = create_agents_with_config(
            model or OLLAMA_MODEL,
            temperature if temperature is not None else 0.3,
            enable_live_scouting=enable_live_scouting,
            serper_api_key=serper_api_key,
        )
    else:
        agents = {
            "sourcing": sourcing_agent,
            "screener": screener_agent,
            "coordinator": coordinator_agent,
            "interview": interview_specialist,
        }

    # Task 1: Parse JD or search web for candidates (Live Scouting)
    sourcing_task = create_sourcing_task(
        job_description, agent=agents["sourcing"], enable_live_scouting=enable_live_scouting
    )

    if enable_live_scouting and "researcher" in agents:
        # Task 2a: Lead Researcher verifies technical background (Scout passes URLs via context)
        researcher_task = create_researcher_task(agent=agents["researcher"])
        researcher_task.context = [sourcing_task]

        # Task 2b: Engagement Specialist drafts LinkedIn connection requests (under 300 chars)
        engagement_task = create_engagement_task(agent=agents["engagement"])
        engagement_task.context = [sourcing_task, researcher_task]

        # Task 3: Score candidates (depends on sourcing + researcher output)
        screener_task = create_screener_task(
            resumes=resumes, agent=agents["screener"], enable_live_scouting=True
        )
        screener_task.context = [sourcing_task, researcher_task]

        # Task 4: Draft emails (depends on screener + researcher + sourcing)
        coordinator_task = create_coordinator_task(
            agent=agents["coordinator"], human_input=enable_human_input
        )
        coordinator_task.context = [sourcing_task, researcher_task, screener_task]

        # Task 5: Simulate 3-turn technical interview for each candidate
        generate_questions_task = create_generate_questions_task(
            agent=agents["interview"], job_description=job_description
        )
        generate_questions_task.context = [screener_task]

        crew_agents = [agents["sourcing"], agents["researcher"], agents["engagement"], agents["screener"], agents["coordinator"], agents["interview"]]
        crew_tasks = [sourcing_task, researcher_task, engagement_task, screener_task, coordinator_task, generate_questions_task]
    else:
        # Standard flow: no Researcher
        screener_task = create_screener_task(
            resumes=resumes, agent=agents["screener"], enable_live_scouting=False
        )
        screener_task.context = [sourcing_task]

        coordinator_task = create_coordinator_task(
            agent=agents["coordinator"], human_input=enable_human_input
        )
        coordinator_task.context = [sourcing_task, screener_task]

        generate_questions_task = create_generate_questions_task(
            agent=agents["interview"], job_description=job_description
        )
        generate_questions_task.context = [screener_task]

        crew_agents = [agents["sourcing"], agents["screener"], agents["coordinator"], agents["interview"]]
        crew_tasks = [sourcing_task, screener_task, coordinator_task, generate_questions_task]

    # Map agent role -> UI stage name for lifecycle updates
    AGENT_ROLE_TO_STAGE = {
        "Live Talent Scout": "Scout",
        "Lead Researcher": "Researcher",
        "Engagement Specialist": "Engagement",
        "Resume Screener": "Screener",
        "Outreach Coordinator": "Coordinator",
        "Senior Technical Interviewer": "Interview",
        "Sourcing Specialist": "Sourcing",
    }

    def _on_task_completed(source, event):
        """TaskCompletedEvent handler: updates lifecycle when a task completes (more reliable than task_callback)."""
        if task_callback and callable(task_callback) and event.output:
            try:
                agent_role = getattr(event.output, "agent", None) or ""
                stage_name = AGENT_ROLE_TO_STAGE.get(agent_role)
                if stage_name:
                    task_callback(stage_name, True, event.output)
            except Exception:
                pass

    crew_kwargs = {"agents": crew_agents, "tasks": crew_tasks}
    crew = Crew(**crew_kwargs)

    with crewai_event_bus.scoped_handlers():
        crewai_event_bus.on(TaskCompletedEvent)(_on_task_completed)
        return crew.kickoff()


def run_part1_sourcing_research(
    job_description: str,
    resumes: str,
    agents: dict,
    enable_live_scouting: bool = False,
    serper_api_key: str | None = None,
    enable_human_input: bool = False,
    task_callback=None,
) -> "CrewOutput":
    """Run Part 1 only: Sourcing & Research (Scout, Researcher, Engagement, Screener, Coordinator). No Interview."""
    rejection_reasons_flat = []
    for reasons in load_rejection_reasons().values():
        rejection_reasons_flat.extend(reasons)

    sourcing_task = create_sourcing_task(
        job_description,
        agent=agents["sourcing"],
        enable_live_scouting=enable_live_scouting,
        rejection_reasons=rejection_reasons_flat if rejection_reasons_flat else None,
    )

    AGENT_ROLE_TO_STAGE = {
        "Live Talent Scout": "Scout",
        "Lead Researcher": "Researcher",
        "Engagement Specialist": "Engagement",
        "Resume Screener": "Screener",
        "Outreach Coordinator": "Coordinator",
        "Sourcing Specialist": "Sourcing",
    }

    def _on_task_completed(source, event):
        if task_callback and callable(task_callback) and event.output:
            try:
                agent_role = getattr(event.output, "agent", None) or ""
                stage_name = AGENT_ROLE_TO_STAGE.get(agent_role)
                if stage_name:
                    task_callback(stage_name, True, event.output)
            except Exception:
                pass

    if enable_live_scouting and "researcher" in agents:
        researcher_task = create_researcher_task(agent=agents["researcher"])
        researcher_task.context = [sourcing_task]
        engagement_task = create_engagement_task(agent=agents["engagement"])
        engagement_task.context = [sourcing_task, researcher_task]
        screener_task = create_screener_task(
            resumes=resumes, agent=agents["screener"], enable_live_scouting=True
        )
        screener_task.context = [sourcing_task, researcher_task]
        coordinator_task = create_coordinator_task(
            agent=agents["coordinator"], human_input=enable_human_input
        )
        coordinator_task.context = [sourcing_task, researcher_task, screener_task]
        crew_agents = [agents["sourcing"], agents["researcher"], agents["engagement"], agents["screener"], agents["coordinator"]]
        crew_tasks = [sourcing_task, researcher_task, engagement_task, screener_task, coordinator_task]
    else:
        screener_task = create_screener_task(
            resumes=resumes, agent=agents["screener"], enable_live_scouting=False
        )
        screener_task.context = [sourcing_task]
        coordinator_task = create_coordinator_task(
            agent=agents["coordinator"], human_input=enable_human_input
        )
        coordinator_task.context = [sourcing_task, screener_task]
        crew_agents = [agents["sourcing"], agents["screener"], agents["coordinator"]]
        crew_tasks = [sourcing_task, screener_task, coordinator_task]

    crew = Crew(agents=crew_agents, tasks=crew_tasks)
    with crewai_event_bus.scoped_handlers():
        crewai_event_bus.on(TaskCompletedEvent)(_on_task_completed)
        return crew.kickoff()


def run_part2_interview_reporting(
    job_description: str,
    scout_output: str,
    researcher_output: str,
    screener_output: str,
    engagement_output: str,
    outreach_output: str,
    approved_names: list[str],
    agents: dict,
    task_callback=None,
) -> tuple[str, dict]:
    """Run Part 2 only: Interview (for approved candidates) + Scoring + Reporting.
    Returns (interview_output, interview_data_parsed)."""
    if not approved_names:
        return "", {}

    generate_questions_task = create_generate_questions_task(
        agent=agents["interview"],
        job_description=job_description,
        approved_names=approved_names,
        screener_output=screener_output,
        researcher_output=researcher_output,
    )

    interview_crew = Crew(agents=[agents["interview"]], tasks=[generate_questions_task])
    interview_result = interview_crew.kickoff()
    interview_output = str(interview_result)
    if task_callback:
        try:
            task_callback("Interview", True, interview_output)
        except Exception:
            pass

    interview_data = parse_interview_output(interview_output, candidate_names=approved_names)
    return interview_output, interview_data


def save_as_pdf(
    content: str,
    filename: str,
    ranking_table: list[tuple] | None = None,
) -> None:
    """Save the crew's final output as a professional PDF with header and optional ranking table.
    ranking_table may have 3 cols (rank, name, score), 4 (rank, name, score, highlight), or 6 (rank, name, match_grade, tech, exp, interv)."""

    class RecruitmentPDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 14)
            self.cell(0, 10, "AI Recruitment Report - 2026", align="C")
            self.ln(5)

    pdf = RecruitmentPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "", 10)

    # Add candidate ranking table at the top if provided
    if ranking_table:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Candidate Ranking")
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 9)
        has_scoring = len(ranking_table) > 0 and len(ranking_table[0]) >= 6
        has_highlights = not has_scoring and len(ranking_table) > 0 and len(ranking_table[0]) >= 4
        if has_scoring:
            table_data = [("Rank", "Candidate", "Match Grade", "Technical", "Experience", "Interview")] + [
                (str(r), n, str(s), str(t), str(e), str(i)) for r, n, s, t, e, i in ranking_table
            ]
            with pdf.table(table_data, first_row_as_headings=True, col_widths=(0.6, 2, 1, 1, 1, 1)):
                pass
        elif has_highlights:
            table_data = [("Rank", "Candidate Name", "Fit Score", "Technical Highlights")] + [
                (str(r), n, str(s), h) for r, n, s, h in ranking_table
            ]
            with pdf.table(table_data, first_row_as_headings=True, col_widths=(0.8, 2.5, 1, 3)):
                pass
        else:
            table_data = [("Rank", "Candidate Name", "Fit Score")] + [
                (str(r), n, str(s)) for r, n, s in ranking_table
            ]
            with pdf.table(table_data, first_row_as_headings=True, col_widths=(1, 3, 1)):
                pass
        pdf.ln(10)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Detailed Breakdown")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, content)
    pdf.output(filename)


if __name__ == "__main__":
    jd_path = Path(__file__).resolve().parent / "job_description.txt"
    try:
        job_description = jd_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        print(f"Error: Job description file not found: {jd_path}")
        exit(1)
    if not job_description:
        print("Error: job_description.txt is empty.")
        exit(1)

    try:
        resumes = load_resumes_from_folder("resumes")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    if not resumes:
        print("Error: No .pdf or .txt files found in the 'resumes' folder.")
        exit(1)

    output_dir = Path(__file__).resolve().parent / "outreach_drafts"
    output_dir.mkdir(exist_ok=True)

    print(f"Processing {len(resumes)} resume(s)...")
    screening_results = []
    interview_output_combined = ""
    scout_output = ""
    outreach_combined = ""
    for i, resume in enumerate(resumes, 1):
        filename = resume["filename"]
        content = resume["content"]
        base_name = Path(filename).stem
        print("=" * 60)
        print(f"[{i}/{len(resumes)}] Processing: {filename}")
        print("=" * 60)

        result = run_recruitment_pipeline(job_description, content)

        sourcing_output = result.tasks_output[0].raw if len(result.tasks_output) > 0 else ""
        if not scout_output:
            scout_output = sourcing_output
        screener_output = result.tasks_output[1].raw if len(result.tasks_output) > 1 else ""
        outreach = result.tasks_output[2].raw if len(result.tasks_output) > 2 else ""
        interview_questions = result.tasks_output[3].raw if len(result.tasks_output) > 3 else ""

        screening_results.append(f"--- {filename} ---\n{screener_output}")
        if interview_questions:
            interview_output_combined = (interview_output_combined + f"\n\n--- CANDIDATE: {base_name} ---\n{interview_questions}") if interview_output_combined else f"--- CANDIDATE: {base_name} ---\n{interview_questions}"
        if outreach:
            outreach_combined = (outreach_combined + f"\n\n--- {base_name} ---\n{outreach}").strip() if outreach_combined else f"--- {base_name} ---\n{outreach}"

        output_path = output_dir / f"{base_name}_outreach.txt"
        output_path.write_text(outreach, encoding="utf-8")
        print(f"Saved: {output_path}")

    # Run Quantitative Scoring Engine
    print("=" * 60)
    print("Running Quantitative Scoring Engine...")
    print("=" * 60)
    combined_screening = "\n\n".join(screening_results)
    scoring_task = create_quantitative_scoring_task(
        combined_screening,
        interview_output=interview_output_combined or None,
        agent=scoring_agent,
    )
    scoring_crew = Crew(agents=[scoring_agent], tasks=[scoring_task])
    scoring_result = scoring_crew.kickoff()
    scoring_data = parse_quantitative_scoring_output(str(scoring_result))
    if scoring_data:
        ranking_table = [(i + 1, c["Candidate"], f"{c['Match_Grade']:.1f}", c["Technical_Score"], c["Experience_Score"], c["Interview_Score"]) for i, c in enumerate(scoring_data)]
    else:
        rank_task = create_rank_candidates_task(combined_screening)
        rank_crew = Crew(agents=[coordinator_agent], tasks=[rank_task])
        ranking_table = parse_ranking_output(str(rank_crew.kickoff()))

    # Final Reporting Task
    print("=" * 60)
    print("Generating Final Recruitment Report...")
    print("=" * 60)
    reporting_task = create_reporting_task(
        scout_output=scout_output or combined_screening[:3000],
        researcher_output="N/A",
        interview_output=interview_output_combined or "N/A",
        screener_output=combined_screening,
        scoring_data=scoring_data,
        outreach_output=outreach_combined or None,
        job_description=job_description[:2000],
        agent=coordinator_agent,
    )
    reporting_crew = Crew(agents=[coordinator_agent], tasks=[reporting_task])
    final_report = str(reporting_crew.kickoff())
    report_path = Path(__file__).resolve().parent / "Final_Recruitment_Report.pdf"
    save_as_pdf(final_report, str(report_path), ranking_table=ranking_table)
    print(f"Saved: {report_path}")

    print("=" * 60)
    print("Done. All outreach drafts saved to 'outreach_drafts' folder.")
    print("Final report saved as 'Final_Recruitment_Report.pdf'.")
