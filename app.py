"""
Streamlit dashboard to monitor the CrewAI recruitment pipeline.
Live console shows agent output in real-time.
"""
# Disable CrewAI telemetry before any CrewAI imports (avoids signal handler errors in background threads)
import os
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")

import builtins
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

# Activity messages for "Current Activity" display
ACTIVITY_MSGS = {
    "Scout": "Scout is currently searching LinkedIn, GitHub, and portfolios...",
    "Researcher": "Researcher is verifying technical backgrounds...",
    "Engagement": "Engagement is drafting LinkedIn connection requests...",
    "Screener": "Screener is scoring candidates...",
    "Coordinator": "Coordinator is drafting outreach emails...",
    "Interview": "Interview Specialist is simulating 3-turn technical interviews...",
    "Scoring": "Quantitative Scoring Engine is calculating Match Grades...",
    "Reporting": "Coordinator is generating final recruitment report...",
    "Sourcing": "Sourcing is parsing job requirements...",
}

# Import crew logic from main
from main import (
    check_ollama_running,
    create_agents_with_config,
    create_quantitative_scoring_task,
    create_rank_candidates_task,
    create_reporting_task,
    load_rejection_reasons,
    load_resumes_from_folder,
    parse_engagement_output,
    parse_interview_output,
    parse_quantitative_scoring_output,
    parse_ranking_output,
    run_part1_sourcing_research,
    run_part2_interview_reporting,
    run_recruitment_pipeline,
    save_as_pdf,
    save_audit_log,
    save_rejection_reason,
)
from crewai import Crew


def _show_pipeline_output(pdf_path, ranking_table, engagement_data, interview_data, scoring_data=None):
    """Render LinkedIn table, candidate comparison, interview logic, and PDF download."""
    import json
    import streamlit.components.v1 as components

    if engagement_data:
        st.header("📨 LinkedIn Connection Requests")
        st.caption("Personalized drafts under 300 characters. Click **Copy** to copy the message to your clipboard.")
        h1, h2, h3, h4 = st.columns([1, 2, 3, 0.5])
        with h1:
            st.markdown("**Name**")
        with h2:
            st.markdown("**Profile Link**")
        with h3:
            st.markdown("**Drafted Message**")
        with h4:
            st.markdown("**Copy**")
        st.divider()
        for i, (name, profile_link, message) in enumerate(engagement_data):
            col1, col2, col3, col4 = st.columns([1, 2, 3, 0.5])
            with col1:
                st.write(name)
            with col2:
                st.markdown(f"[Open profile]({profile_link})")
            with col3:
                edited_msg = st.text_area(
                    "Drafted Message",
                    value=message,
                    key=f"engagement_msg_{i}_{name}",
                    height=80,
                    label_visibility="collapsed",
                )
            with col4:
                msg_json = json.dumps(edited_msg)
                safe_json = msg_json.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                components.html(
                    f"""<button onclick="var btn=this; var j=document.getElementById('msg-{i}').textContent; navigator.clipboard.writeText(JSON.parse(j)); btn.textContent='✓'; setTimeout(function(){{btn.textContent='Copy'}}, 1500);"
                    style="padding:8px 12px;cursor:pointer;border:1px solid #ccc;border-radius:4px;background:#f0f2f6;font-size:12px;">Copy</button>
                    <script type="application/json" id="msg-{i}">{safe_json}</script>""",
                    height=40,
                )
        st.divider()

    if ranking_table:
        st.header("📊 Candidate Comparison Table")
        has_scoring = scoring_data and len(ranking_table) > 0 and len(ranking_table[0]) >= 6
        has_highlights = not has_scoring and len(ranking_table) > 0 and len(ranking_table[0]) >= 4
        if has_scoring:
            table_data = [["Rank", "Candidate Name", "Match Grade", "Technical", "Experience", "Interview"]] + [
                [str(r), n, str(s), str(t), str(e), str(i)] for r, n, s, t, e, i in ranking_table
            ]
        elif has_highlights:
            col4 = "Technical Highlights" if engagement_data else "Score Justification"
            table_data = [["Rank", "Candidate Name", "Fit Score", col4]] + [
                [str(r), n, str(s), h] for r, n, s, h in ranking_table
            ]
        else:
            table_data = [["Rank", "Candidate Name", "Fit Score"]] + [
                [str(r), n, str(s)] for r, n, s in ranking_table
            ]
        st.table(table_data)
        if has_highlights and not scoring_data:
            col_label = "Technical Highlights" if engagement_data else "Score Justification"
            st.caption("Edit candidate summaries below before export.")
            for idx, row in enumerate(ranking_table):
                name = row[1] if len(row) >= 2 else str(row)
                val = row[3] if len(row) >= 4 else ""
                st.text_area(
                    f"{col_label}: **{name}**",
                    value=val,
                    key=f"highlights_{idx}_{name}",
                    height=60,
                    label_visibility="visible",
                )
        if scoring_data:
            st.caption("Match Grade = 50% Technical + 30% Experience + 20% Interview. Edit summaries below before export.")
            for idx, s in enumerate(scoring_data):
                cand = s.get("Candidate", f"Candidate_{idx}")
                with st.expander(f"📋 Score Justifications: **{cand}**"):
                    tech_edit = st.text_area(
                        f"Technical ({s.get('Technical_Score', '')})",
                        value=s.get("Technical_Justification", ""),
                        key=f"summary_tech_{idx}_{cand}",
                        height=80,
                        label_visibility="visible",
                    )
                    exp_edit = st.text_area(
                        f"Experience ({s.get('Experience_Score', '')})",
                        value=s.get("Experience_Justification", ""),
                        key=f"summary_exp_{idx}_{cand}",
                        height=80,
                        label_visibility="visible",
                    )
                    int_edit = st.text_area(
                        f"Interview ({s.get('Interview_Score', '')})",
                        value=s.get("Interview_Justification", ""),
                        key=f"summary_int_{idx}_{cand}",
                        height=80,
                        label_visibility="visible",
                    )

        # Always show Interview Simulation when we have candidates
        st.subheader("📝 Interview Simulation")
        idata = interview_data or {}
        for row in ranking_table:
            name = row[1] if len(row) >= 2 else str(row)
            data = (
                idata.get(name)
                or idata.get(name.replace(" ", "_").lower())
                or idata.get(name.replace(" ", "_"))
            )
            if not data:
                for k in idata:
                    if name.lower() in k.lower() or k.lower() in name.lower():
                        data = idata[k]
                        break
            with st.expander(f"✨ Interactive Interview Simulation: **{name}**"):
                if data:
                    if isinstance(data, dict) and ("question1" in data or "simulated_answer" in data):
                        # 3-turn simulation format: chat-style transcript
                        st.caption("Simulated 3-turn technical interview")
                        chat = []
                        if data.get("question1"):
                            chat.append(("Interviewer", data["question1"]))
                        if data.get("simulated_answer"):
                            chat.append(("Candidate", data["simulated_answer"]))
                        if data.get("follow_up"):
                            chat.append(("Interviewer", data["follow_up"]))
                        for speaker, text in chat:
                            if speaker == "Interviewer":
                                st.markdown(f"**🎤 Interviewer:** {text}")
                            else:
                                st.markdown(f"**👤 {name}:** {text}")
                            st.divider()
                    else:
                        # Legacy Q&A format
                        qa_list = data if isinstance(data, list) else []
                        for idx, (q, expected) in enumerate(qa_list, 1):
                            st.markdown(f"**Q{idx}:** {q}")
                            st.markdown(f"*Expected:* {expected}")
                            if idx < len(qa_list):
                                st.divider()
                else:
                    st.caption("No interview simulation parsed for this candidate.")

    if pdf_path and pdf_path.exists():
        st.success("Pipeline completed successfully!")
        pdf_bytes = pdf_path.read_bytes()
        st.download_button(
            "📥 Download Final_Recruitment_Report.pdf",
            data=pdf_bytes,
            file_name="Final_Recruitment_Report.pdf",
            mime="application/pdf",
            type="primary",
        )
    else:
        st.warning("Pipeline finished but PDF was not found.")


class StreamlitStdout:
    """
    Redirects sys.stdout and sys.stderr to a Streamlit container for real-time display (Thought Stream).
    Captures trace links, human feedback prompts, and all agent output in the dashboard.
    Restores original stdout/stderr on exit.
    """

    def __init__(self, container, on_write=None):
        self.container = container
        self.buffer = []
        self._lock = threading.Lock()
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self.on_write = on_write  # Callback(text) for immediate streaming

    def write(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            self.buffer.append(text)
            # Don't update Streamlit container here - we may be in a background thread.
            # Streamlit raises NoSessionContext when updating UI from non-main thread.
            # Main thread polls self.buffer and updates the container.

    def flush(self) -> None:
        pass

    def __enter__(self) -> "StreamlitStdout":
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        return False


def run_full_pipeline(
    job_description: str,
    resumes: list[dict],
    selected_model: str,
    temperature: float,
    enable_live_scouting: bool = False,
    serper_api_key: str | None = None,
    enable_safe_mode: bool = False,
    enable_fast_mode: bool = False,
    enable_human_input: bool = False,
    audit_log: list | None = None,
    task_callback=None,
) -> tuple[
    Path | None,
    list[tuple[str, str, str]] | list[tuple[str, str, str, str]] | list,
    list[tuple[str, str, str]] | None,
    dict,
    list,
    list[dict],
]:
    """
    Run the full recruitment pipeline and return (pdf_path, ranking_table, engagement_data, interview_data, audit_log, scoring_data).
    When audit_log is provided, builds Product Owner audit log and saves to .hireai_audit/
    """
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "outreach_drafts"
    output_dir.mkdir(exist_ok=True)

    report_parts = []
    screening_results = []
    researcher_output = ""
    scout_output = ""
    engagement_output = ""
    interview_output = ""
    interview_output_combined = ""
    outreach_combined = ""
    interview_data: dict[str, list[tuple[str, str]]] = {}
    candidate_names_for_interview: list[str] = []

    if audit_log is None:
        audit_log = []
    audit_log.append({
        "timestamp": datetime.now().isoformat(),
        "type": "pipeline_start",
        "mode": "live_scouting" if enable_live_scouting else "standard",
        "model": selected_model,
    })

    agents = create_agents_with_config(
        selected_model,
        temperature,
        enable_live_scouting=enable_live_scouting,
        serper_api_key=serper_api_key,
        audit_log=audit_log,
        enable_safe_mode=enable_safe_mode,
        enable_fast_mode=enable_fast_mode,
    )

    if enable_live_scouting:
        # Live Scouting: Scout -> Researcher -> Screener -> Coordinator -> Interview
        print("=" * 60)
        print("Live Talent Scout: Searching web for candidates...")
        print("=" * 60)
        result = run_recruitment_pipeline(
            job_description,
            "",
            agents=agents,
            enable_live_scouting=True,
            serper_api_key=serper_api_key,
            enable_human_input=enable_human_input,
            task_callback=task_callback,
        )
        # Task indices: 0=Scout, 1=Researcher, 2=Engagement, 3=Screener, 4=Coordinator, 5=Interview
        task_names = ["Scout", "Researcher", "Engagement", "Screener", "Coordinator", "Interview"]
        for i, (name, task_out) in enumerate(zip(task_names, result.tasks_output)):
            audit_log.append({"type": "task_output", "task": name, "output": task_out.raw[:2000] + ("..." if len(task_out.raw) > 2000 else "")})
        scout_output = result.tasks_output[0].raw if len(result.tasks_output) > 0 else ""
        researcher_output = result.tasks_output[1].raw if len(result.tasks_output) > 1 else ""
        engagement_output = result.tasks_output[2].raw if len(result.tasks_output) > 2 else ""
        screener_output = result.tasks_output[3].raw if len(result.tasks_output) > 3 else ""
        outreach = result.tasks_output[4].raw if len(result.tasks_output) > 4 else ""
        interview_output = result.tasks_output[5].raw if len(result.tasks_output) > 5 else ""
        screening_results.append(screener_output)
        interview_output_combined = interview_output
        outreach_combined = outreach
        engagement_section = ""
        if engagement_output:
            engagement_section = f"LINKEDIN CONNECTION REQUESTS (from Engagement Specialist)\n{engagement_output}\n\n"
        report_parts.append(
            f"LIVE SCOUTING RESULTS\n\n"
            f"TECHNICAL HIGHLIGHTS (from Lead Researcher)\n{researcher_output}\n\n"
            f"{engagement_section}"
            f"OUTREACH DRAFT\n{outreach}\n\n"
            f"INTERVIEW SIMULATION (3-turn technical interview)\n{interview_output}"
        )
    else:
        # Standard: per-resume pipeline
        for i, resume in enumerate(resumes, 1):
            filename = resume["filename"]
            content = resume["content"]
            base_name = Path(filename).stem
            print("=" * 60)
            print(f"[{i}/{len(resumes)}] Processing: {filename}")
            print("=" * 60)

            result = run_recruitment_pipeline(
                job_description,
                content,
                agents=agents,
                enable_human_input=enable_human_input,
                task_callback=task_callback,
            )

            task_names_std = ["Sourcing", "Screener", "Coordinator", "Interview"]
            for j, (tname, task_out) in enumerate(zip(task_names_std, result.tasks_output)):
                audit_log.append({"type": "task_output", "task": f"{tname} ({base_name})", "output": task_out.raw[:2000] + ("..." if len(task_out.raw) > 2000 else "")})

            sourcing_output = result.tasks_output[0].raw if len(result.tasks_output) > 0 else ""
            if not scout_output:
                scout_output = sourcing_output
            screener_output = result.tasks_output[1].raw if len(result.tasks_output) > 1 else ""
            outreach = result.tasks_output[2].raw if len(result.tasks_output) > 2 else ""
            iq = result.tasks_output[3].raw if len(result.tasks_output) > 3 else ""

            screening_results.append(f"--- {filename} ---\n{screener_output}")
            outreach_combined = (outreach_combined + f"\n\n--- {base_name} ---\n{outreach}").strip() if outreach else outreach_combined
            output_path = output_dir / f"{base_name}_outreach.txt"
            output_path.write_text(outreach, encoding="utf-8")
            print(f"Saved: {output_path}")

            candidate_section = (
                f"--- {filename} ---\n\n"
                f"OUTREACH DRAFT\n{outreach}\n\n"
                f"INTERVIEW SIMULATION (3-turn technical interview)\n{iq}"
            )
            report_parts.append(candidate_section)
            if iq:
                interview_output_combined = (interview_output_combined + f"\n\n--- CANDIDATE: {base_name} ---\n{iq}") if interview_output_combined else f"--- CANDIDATE: {base_name} ---\n{iq}"
            parsed = parse_interview_output(iq, candidate_names=[base_name])
            interview_data.update(parsed)

    print("=" * 60)
    print("Running Quantitative Scoring Engine...")
    print("=" * 60)
    combined_screening = "\n\n".join(screening_results)
    researcher_output_combined = researcher_output if enable_live_scouting else None
    scoring_task = create_quantitative_scoring_task(
        combined_screening,
        researcher_output=researcher_output_combined,
        interview_output=interview_output_combined or interview_output or None,
        agent=agents["scoring"],
    )
    scoring_crew = Crew(agents=[agents["scoring"]], tasks=[scoring_task])
    scoring_result = scoring_crew.kickoff()
    scoring_output = str(scoring_result)
    scoring_data = parse_quantitative_scoring_output(scoring_output)
    if task_callback and callable(task_callback):
        try:
            task_callback("Scoring", True, scoring_output)
        except Exception:
            pass
    if scoring_data:
        ranking_table = [(i + 1, c["Candidate"], f"{c['Match_Grade']:.1f}", c["Technical_Score"], c["Experience_Score"], c["Interview_Score"]) for i, c in enumerate(scoring_data)]
    else:
        rank_task = create_rank_candidates_task(
            combined_screening,
            agent=agents["coordinator"],
            researcher_output=researcher_output_combined,
        )
        rank_crew = Crew(agents=[agents["coordinator"]], tasks=[rank_task])
        rank_result = rank_crew.kickoff()
        ranking_table = parse_ranking_output(str(rank_result), include_highlights=True)

    # Cap candidates to resume count in resume mode (prevents LLM hallucination)
    if not enable_live_scouting and resumes:
        max_candidates = len(resumes)
        if scoring_data and len(scoring_data) > max_candidates:
            scoring_data = scoring_data[:max_candidates]
            ranking_table = [(i + 1, c["Candidate"], f"{c['Match_Grade']:.1f}", c["Technical_Score"], c["Experience_Score"], c["Interview_Score"]) for i, c in enumerate(scoring_data)]
        elif ranking_table and len(ranking_table) > max_candidates:
            ranking_table = ranking_table[:max_candidates]

    # Final Reporting Task: consolidate all data, executive summary, Markdown, Next Steps
    print("=" * 60)
    print("Generating Final Recruitment Report...")
    print("=" * 60)
    reporting_task = create_reporting_task(
        scout_output=scout_output or combined_screening[:3000],
        researcher_output=researcher_output_combined or "N/A",
        interview_output=interview_output_combined or interview_output or "N/A",
        screener_output=combined_screening,
        scoring_data=scoring_data,
        engagement_output=engagement_output if enable_live_scouting else None,
        outreach_output=outreach_combined or None,
        job_description=job_description[:2000],
        agent=agents["coordinator"],
    )
    reporting_crew = Crew(agents=[agents["coordinator"]], tasks=[reporting_task])
    report_result = reporting_crew.kickoff()
    final_report = str(report_result)
    if task_callback and callable(task_callback):
        try:
            task_callback("Reporting", True, final_report)
        except Exception:
            pass

    report_path = base_dir / "Final_Recruitment_Report.pdf"
    try:
        save_as_pdf(final_report, str(report_path), ranking_table=ranking_table)
    except PermissionError:
        # Fallback: write to timestamped file if original is locked (e.g. open in viewer)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"Final_Recruitment_Report_{timestamp}.pdf"
        save_as_pdf(final_report, str(report_path), ranking_table=ranking_table)
    print(f"Saved: {report_path}")
    print("=" * 60)
    print("Done.")

    engagement_data = parse_engagement_output(engagement_output) if enable_live_scouting and engagement_output else None

    # Parse interview Q&A for Live Scouting (use ranking names)
    if enable_live_scouting and interview_output:
        names = [r[1] for r in ranking_table] if ranking_table else []
        interview_data = parse_interview_output(interview_output, candidate_names=names)

    if audit_log:
        audit_log.append({"type": "pipeline_end", "timestamp": datetime.now().isoformat()})

    return report_path, ranking_table, engagement_data, interview_data, audit_log, scoring_data


def run_part1_pipeline(
    job_description: str,
    resumes: list[dict],
    selected_model: str,
    temperature: float,
    enable_live_scouting: bool = False,
    serper_api_key: str | None = None,
    enable_safe_mode: bool = False,
    enable_fast_mode: bool = False,
    enable_human_input: bool = False,
    audit_log: list | None = None,
    task_callback=None,
) -> tuple[
    list[tuple],
    list[tuple[str, str, str]] | None,
    str,
    str,
    str,
    str,
    str,
    dict,
    list,
]:
    """Run Part 1 only: Sourcing & Research. Returns (ranking_table, engagement_data, scout_output,
    researcher_output, screener_output, engagement_output, outreach_combined, agents, audit_log)."""
    base_dir = Path(__file__).resolve().parent
    screening_results = []
    researcher_output = ""
    scout_output = ""
    engagement_output = ""
    outreach_combined = ""

    if audit_log is None:
        audit_log = []
    audit_log.append({
        "timestamp": datetime.now().isoformat(),
        "type": "pipeline_part1_start",
        "mode": "live_scouting" if enable_live_scouting else "standard",
        "model": selected_model,
    })

    agents = create_agents_with_config(
        selected_model,
        temperature,
        enable_live_scouting=enable_live_scouting,
        serper_api_key=serper_api_key,
        audit_log=audit_log,
        enable_safe_mode=enable_safe_mode,
        enable_fast_mode=enable_fast_mode,
    )

    if enable_live_scouting:
        print("=" * 60)
        print("Part 1: Live Talent Scout → Researcher → Engagement → Screener → Coordinator")
        print("=" * 60)
        result = run_part1_sourcing_research(
            job_description,
            "",
            agents=agents,
            enable_live_scouting=True,
            serper_api_key=serper_api_key,
            enable_human_input=enable_human_input,
            task_callback=task_callback,
        )
        task_names = ["Scout", "Researcher", "Engagement", "Screener", "Coordinator"]
        for i, (name, task_out) in enumerate(zip(task_names, result.tasks_output)):
            audit_log.append({"type": "task_output", "task": name, "output": task_out.raw[:2000] + ("..." if len(task_out.raw) > 2000 else "")})
        scout_output = result.tasks_output[0].raw if len(result.tasks_output) > 0 else ""
        researcher_output = result.tasks_output[1].raw if len(result.tasks_output) > 1 else ""
        engagement_output = result.tasks_output[2].raw if len(result.tasks_output) > 2 else ""
        screener_output = result.tasks_output[3].raw if len(result.tasks_output) > 3 else ""
        outreach_combined = result.tasks_output[4].raw if len(result.tasks_output) > 4 else ""
        screening_results.append(screener_output)
    else:
        # Standard: per-resume Part 1 (no Interview)
        for i, resume in enumerate(resumes, 1):
            filename = resume["filename"]
            content = resume["content"]
            base_name = Path(filename).stem
            print("=" * 60)
            print(f"Part 1 [{i}/{len(resumes)}]: {filename}")
            print("=" * 60)
            result = run_part1_sourcing_research(
                job_description,
                content,
                agents=agents,
                enable_live_scouting=False,
                task_callback=task_callback,
            )
            task_names = ["Sourcing", "Screener", "Coordinator"]
            screener_out = result.tasks_output[1].raw if len(result.tasks_output) > 1 else ""
            outreach = result.tasks_output[2].raw if len(result.tasks_output) > 2 else ""
            screening_results.append(f"--- {filename} ---\n{screener_out}")
            if not scout_output:
                scout_output = result.tasks_output[0].raw if len(result.tasks_output) > 0 else ""
            researcher_output = ""
            engagement_output = ""
            outreach_combined = (outreach_combined + f"\n\n--- {base_name} ---\n{outreach}").strip() if outreach else outreach_combined

    combined_screening = "\n\n".join(screening_results)
    rank_task = create_rank_candidates_task(
        combined_screening,
        agent=agents["coordinator"],
        researcher_output=researcher_output if enable_live_scouting else None,
    )
    rank_crew = Crew(agents=[agents["coordinator"]], tasks=[rank_task])
    ranking_table = parse_ranking_output(str(rank_crew.kickoff()), include_highlights=True)
    # Cap candidates to resume count in resume mode (prevents LLM hallucination)
    if not enable_live_scouting and resumes and ranking_table and len(ranking_table) > len(resumes):
        ranking_table = ranking_table[:len(resumes)]
    engagement_data = parse_engagement_output(engagement_output) if enable_live_scouting and engagement_output else None

    audit_log.append({"type": "pipeline_part1_end", "timestamp": datetime.now().isoformat()})
    return ranking_table, engagement_data, scout_output, researcher_output, combined_screening, engagement_output, outreach_combined, agents, audit_log


def run_part2_pipeline(
    job_description: str,
    ranking_table: list[tuple],
    engagement_data: list[tuple[str, str, str]] | None,
    scout_output: str,
    researcher_output: str,
    screener_output: str,
    engagement_output: str,
    outreach_combined: str,
    approved_names: list[str],
    agents: dict,
    audit_log: list,
    task_callback=None,
) -> tuple[Path | None, list, list | None, dict, list, list[dict]]:
    """Run Part 2: Interview (approved only) + Scoring + Reporting. Returns (pdf_path, ranking_table, engagement_data, interview_data, audit_log, scoring_data)."""
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "outreach_drafts"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Part 2: Interview (approved candidates) → Scoring → Reporting")
    print("=" * 60)

    interview_output, interview_data = run_part2_interview_reporting(
        job_description=job_description,
        scout_output=scout_output,
        researcher_output=researcher_output,
        screener_output=screener_output,
        engagement_output=engagement_output,
        outreach_output=outreach_combined,
        approved_names=approved_names,
        agents=agents,
        task_callback=task_callback,
    )

    # Quantitative Scoring (only for approved candidates)
    print("Running Quantitative Scoring Engine...")
    scoring_task = create_quantitative_scoring_task(
        screener_output,
        researcher_output=researcher_output or None,
        interview_output=interview_output or None,
        agent=agents["scoring"],
    )
    scoring_crew = Crew(agents=[agents["scoring"]], tasks=[scoring_task])
    scoring_output = str(scoring_crew.kickoff())
    scoring_data = parse_quantitative_scoring_output(scoring_output)
    if task_callback:
        try:
            task_callback("Scoring", True, scoring_output)
        except Exception:
            pass

    if scoring_data:
        ranking_table = [(i + 1, c["Candidate"], f"{c['Match_Grade']:.1f}", c["Technical_Score"], c["Experience_Score"], c["Interview_Score"]) for i, c in enumerate(scoring_data)]

    # Cap to approved candidates only (prevents LLM hallucination in Part 2)
    approved_set = set(n.strip().lower() for n in approved_names)
    if scoring_data:
        filtered = [c for c in scoring_data if (c.get("Candidate") or "").strip().lower() in approved_set]
        if filtered:
            scoring_data = filtered
        else:
            scoring_data = scoring_data[:len(approved_names)]  # fallback: truncate if name match fails
        ranking_table = [(i + 1, c["Candidate"], f"{c['Match_Grade']:.1f}", c["Technical_Score"], c["Experience_Score"], c["Interview_Score"]) for i, c in enumerate(scoring_data)]
    elif ranking_table:
        # Filter to approved names only (when no scoring_data)
        filtered = [r for r in ranking_table if (r[1] if len(r) >= 2 else "").strip().lower() in approved_set]
        ranking_table = filtered[:len(approved_names)] if filtered else ranking_table[:len(approved_names)]

    # Reporting
    print("Generating Final Recruitment Report...")
    reporting_task = create_reporting_task(
        scout_output=scout_output,
        researcher_output=researcher_output or "N/A",
        interview_output=interview_output or "N/A",
        screener_output=screener_output,
        scoring_data=scoring_data,
        engagement_output=engagement_output or None,
        outreach_output=outreach_combined or None,
        job_description=job_description[:2000],
        agent=agents["coordinator"],
    )
    reporting_crew = Crew(agents=[agents["coordinator"]], tasks=[reporting_task])
    final_report = str(reporting_crew.kickoff())
    if task_callback:
        try:
            task_callback("Reporting", True, final_report)
        except Exception:
            pass

    report_path = base_dir / "Final_Recruitment_Report.pdf"
    try:
        save_as_pdf(final_report, str(report_path), ranking_table=ranking_table)
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"Final_Recruitment_Report_{timestamp}.pdf"
        save_as_pdf(final_report, str(report_path), ranking_table=ranking_table)
    print(f"Saved: {report_path}")
    audit_log.append({"type": "pipeline_end", "timestamp": datetime.now().isoformat()})
    return report_path, ranking_table, engagement_data, interview_data, audit_log, scoring_data


def extract_resume_content(uploaded_file) -> str:
    """Extract text from uploaded PDF or TXT file using pdfplumber for PDFs."""
    import io
    data = uploaded_file.getvalue()
    if uploaded_file.name.lower().endswith(".txt"):
        return data.decode("utf-8", errors="replace").strip()
    if uploaded_file.name.lower().endswith(".pdf"):
        import pdfplumber
        content_parts = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    content_parts.append(text)
        return "\n".join(content_parts).strip()
    return ""


def process_uploaded_files(
    job_file, resume_files, base_dir: Path, allow_empty_resumes: bool = False
) -> tuple[str, list[dict]]:
    """
    Process uploaded files into job description and candidate data list.
    Extracts text from PDFs using pdfplumber.
    When allow_empty_resumes is True (e.g. Live Scouting), returns [] if no resumes provided.
    """
    if job_file:
        job_description = job_file.getvalue().decode("utf-8", errors="replace").strip()
    else:
        jd_path = base_dir / "job_description.txt"
        if not jd_path.exists():
            raise FileNotFoundError("No job description provided. Upload a file or ensure job_description.txt exists.")
        job_description = jd_path.read_text(encoding="utf-8").strip()

    if resume_files:
        resumes = []
        for f in resume_files:
            content = extract_resume_content(f)
            if content:
                resumes.append({"filename": f.name, "content": content})
    else:
        if allow_empty_resumes:
            resumes = []
        else:
            try:
                resumes = load_resumes_from_folder("resumes")
            except FileNotFoundError:
                raise FileNotFoundError(
                    "No resumes found. Upload .pdf/.txt files or add them to the resumes/ folder."
                )

    return job_description, resumes


st.set_page_config(page_title="Recruitment Pipeline", page_icon="📋", layout="wide")

st.title("📋 CrewAI Recruitment Pipeline Monitor")

# Cost Saved metric (shown when we have pipeline results)
result = st.session_state.get("pipeline_result")
if result and len(result) > 1:
    ranking_table = result[1]
    n_candidates = len(ranking_table) if ranking_table else 0
    if n_candidates > 0:
        hours_saved = max(0.5 * n_candidates, 1.0)
        st.info(f"⏱️ **This run saved ~{hours_saved:.1f} hours** of manual sourcing, screening, and outreach.")

# Sidebar: Model, temperature, file uploads
with st.sidebar:
    st.header("⚙️ Model & Settings")
    model_choice = st.radio(
        "Model",
        options=["Preset", "Custom"],
        index=0,
        horizontal=True,
        key="model_choice",
    )
    if model_choice == "Preset":
        selected_model = st.selectbox(
            "Preset",
            options=["llama3.2", "llama3.2:1b", "llama3.2:3b", "mistral", "phi3", "gemma2:2b", "qwen2.5:0.5b"],
            index=0,
            key="model_preset",
        )
    else:
        custom_model = st.text_input(
            "Model name (from `ollama list`)",
            value="llama3.2",
            placeholder="e.g. llama3.2, mistral, phi3",
            key="model_custom",
        )
        selected_model = custom_model.strip() or "llama3.2"
    st.info("Use `ollama list` to see installed models. Run `ollama pull <model>` if not found.")
    st.caption("💡 For faster runs: use smaller models (e.g. qwen2.5:0.5b, llama3.2:1b) and keep human approval off.")

    enable_fast_mode = st.toggle(
        "Fast mode",
        value=False,
        help="Limits output length (max 1024 tokens) for faster inference. Use when pipeline is slow on CPU.",
        key="fast_mode",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Lower = stricter screening, Higher = more creative",
        key="temp",
    )

    st.divider()
    st.header("🔍 Live Web Scouting")
    enable_live_scouting = st.toggle(
        "Enable Live Web Scouting",
        value=False,
        help="When enabled, the Sourcing Specialist becomes a Live Talent Scout that searches the web for candidates (LinkedIn, GitHub, portfolios) instead of parsing resumes from files.",
        key="live_scouting",
    )
    serper_api_key = None
    if enable_live_scouting:
        serper_api_key = st.text_input(
            "Serper API Key",
            type="password",
            placeholder="Enter your Serper API key (serper.dev)",
            help="Get a free API key at serper.dev. Required for Live Web Scouting.",
            key="serper_key",
        )
        if not serper_api_key or not serper_api_key.strip():
            st.warning("Serper API Key is required when Live Web Scouting is enabled.")
        enable_safe_mode = st.toggle(
            "Safe Mode",
            value=False,
            help="When enabled, skips failed Serper API calls or private LinkedIn profiles instead of crashing. Use for resilient runs.",
            key="safe_mode",
        )
    else:
        enable_safe_mode = False

    st.divider()
    st.header("📋 Recruitment Lifecycle")
    lifecycle_placeholder = st.empty()
    stage_names_sidebar = ["Scout", "Researcher", "Engagement", "Screener", "Coordinator", "Interview", "Scoring", "Reporting"] if enable_live_scouting else ["Sourcing", "Screener", "Coordinator", "Interview", "Scoring", "Reporting"]
    # Initial render of checkboxes (polling loop will update lifecycle_placeholder in real time)
    with lifecycle_placeholder.container():
        lifecycle = st.session_state.get("lifecycle", {})
        for name in stage_names_sidebar:
            done = lifecycle.get(name, False)
            st.markdown(f"{'☑' if done else '☐'} {name}")

    current_activity_placeholder = st.empty()
    with current_activity_placeholder.container():
        current_activity = st.session_state.get("current_activity", "")
        if current_activity:
            st.caption("**Current Activity:**")
            st.caption(current_activity)

    st.divider()
    st.header("⏱️ Speed & Approval")
    enable_human_approval_layer = st.toggle(
        "Human Approval Layer",
        value=False,
        help="When enabled, Part 1 (Sourcing & Research) runs first. You review candidates and approve which ones to send to AI Interview. Rejected candidates can include a reason so the AI learns to avoid similar profiles in future searches.",
        key="human_approval_layer",
    )
    enable_human_input = st.toggle(
        "Pause for human approval (Coordinator)",
        value=False,
        help="When enabled, the pipeline pauses when drafting outreach so you can review before continuing. Disable for faster unattended runs.",
        key="human_approval",
    )

    st.divider()
    st.header("📁 Upload Files")
    st.caption("Upload job description and resumes, or use existing files in the project folder. Resumes are optional when Live Web Scouting is enabled.")

    job_file = st.file_uploader("Job Description (.txt)", type=["txt"], key="jd")
    resume_files = st.file_uploader(
        "Resumes (.pdf, .txt)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="resumes",
    )

    # Preview: show candidate count when files uploaded
    if resume_files:
        st.success(f"✓ {len(resume_files)} file(s) uploaded — text will be extracted on Run")

    st.divider()
    st.caption("If no files are uploaded, the pipeline will use job_description.txt and the resumes/ folder.")
    if not enable_live_scouting:
        st.info("💡 **Resume mode:** Each resume = 1 candidate. To get more candidates, add more .pdf/.txt files to the `resumes/` folder or upload additional files above.")

# Main area: Live Progress & Output
st.header("🖥️ Live Progress")

st.subheader("💭 Thought Stream")
st.caption("Agent logs stream here in real time as each task runs (Scout leads, Researcher findings, etc.).")
thought_placeholder = st.empty()

# Run Pipeline button (disabled while pipeline is running)
pipeline_running = st.session_state.get("pipeline_running", False)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    run_btn = st.button(
        "▶️ Run Pipeline",
        type="primary",
        use_container_width=True,
        disabled=pipeline_running,
    )

# When user clicks Run: set pipeline_running and rerun so button shows disabled immediately
if run_btn and not pipeline_running:
    st.session_state["pipeline_running"] = True
    st.rerun()

# Show pipeline results after rerun (when pipeline_result is stored)
if st.session_state.get("pipeline_result"):
    result = st.session_state.pop("pipeline_result")
    pdf_path, ranking_table, engagement_data, interview_data = result[0], result[1], result[2], result[3]
    scoring_data = result[5] if len(result) > 5 else None
    st.session_state["pipeline_running"] = False
    # Output sections (same as below) - show after rerun so sidebar checkboxes are updated
    _show_pipeline_output(pdf_path, ranking_table, engagement_data, interview_data, scoring_data)

# Human Approval Layer: show Review Table after Part 1, with "Proceed to AI Interview" button
elif st.session_state.get("part1_result") and enable_human_approval_layer:
    part1 = st.session_state["part1_result"]
    ranking_table = part1["ranking_table"]
    engagement_data = part1["engagement_data"]
    job_description = part1["job_description"]

    st.header("✅ Part 1 Complete — Review & Approve Candidates")
    st.caption("Select which candidates to send to the AI Interview. Uncheck to reject; provide a reason so the AI learns to avoid similar profiles in future searches.")

    # Build review dataframe: Approve (checkbox), Rank, Name, Score, Highlights
    if "approve_state" not in st.session_state:
        st.session_state["approve_state"] = {r[1]: True for r in ranking_table}
    approve_state = st.session_state["approve_state"]

    import pandas as pd
    has_highlights = ranking_table and len(ranking_table[0]) >= 4
    rows = []
    for r in ranking_table:
        rank, name, score = r[0], r[1], r[2]
        highlight = r[3] if has_highlights and len(r) >= 4 else ""
        rows.append({
            "Approve for Interview": approve_state.get(name, True),
            "Rank": rank,
            "Name": name,
            "Score": score,
            "Technical Highlights": highlight,
        })
    df = pd.DataFrame(rows)

    edited = st.data_editor(
        df,
        column_config={
            "Approve for Interview": st.column_config.CheckboxColumn("Approve for Interview", default=True),
            "Rank": st.column_config.NumberColumn("Rank", disabled=True),
            "Name": st.column_config.TextColumn("Name", disabled=True),
            "Score": st.column_config.TextColumn("Score", disabled=True),
            "Technical Highlights": st.column_config.TextColumn("Technical Highlights", disabled=True),
        },
        hide_index=True,
        key="review_table_editor",
    )
    # Sync approve_state from edited dataframe
    for _, row in edited.iterrows():
        name = row["Name"]
        approved = bool(row["Approve for Interview"])
        if not approved and approve_state.get(name, True):
            # User just unchecked - need rejection reason
            st.session_state["pending_rejection"] = name
        approve_state[name] = approved

    # Rejection reason prompt (when user unchecked a candidate)
    if st.session_state.get("pending_rejection"):
        reject_name = st.session_state["pending_rejection"]
        st.divider()
        with st.container():
            st.caption(f"**Reason for rejecting {reject_name}?** (e.g., 'Not enough experience', 'Skills mismatch') — helps the AI avoid similar profiles in future searches.")
            reason = st.text_input("Rejection reason", key=f"reject_reason_{reject_name}", placeholder="e.g., Not enough experience")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Save & Continue", key="save_rejection"):
                    if reason and reason.strip():
                        save_rejection_reason(reject_name, reason.strip())
                    st.session_state.pop("pending_rejection", None)
                    st.rerun()
            with col_b:
                if st.button("Cancel (re-approve)", key="cancel_rejection"):
                    approve_state[reject_name] = True
                    st.session_state.pop("pending_rejection", None)
                    st.rerun()

    # Proceed to AI Interview button
    approved_names = [row["Name"] for _, row in edited.iterrows() if row["Approve for Interview"]]
    st.divider()
    proceed_disabled = bool(len(approved_names) == 0 or st.session_state.get("pending_rejection"))
    if st.button(
        "▶️ Proceed to AI Interview",
        type="primary",
        disabled=proceed_disabled,
        key="proceed_part2",
    ) and not proceed_disabled:
        st.session_state["pipeline_running"] = True
        st.session_state["part2_config"] = {
            "approved_names": approved_names,
            "part1": part1,
        }
        st.rerun()

    # Show LinkedIn drafts if available
    if engagement_data:
        st.subheader("📨 LinkedIn Connection Requests (Part 1)")
        for name, profile_link, message in engagement_data:
            if name in approved_names:
                st.markdown(f"**{name}** — [Open profile]({profile_link})")
                st.caption(message[:200] + ("..." if len(message) > 200 else ""))

    # Option to discard Part 1 and start over
    if st.button("🗑️ Discard & Run New Search", key="discard_part1"):
        st.session_state.pop("part1_result", None)
        st.session_state.pop("approve_state", None)
        st.session_state.pop("pending_rejection", None)
        st.rerun()

elif pipeline_running:
    base_dir = Path(__file__).resolve().parent

    # Branch: Part 2 (Human Approval) vs Part 1 vs Full pipeline
    part2_config = st.session_state.pop("part2_config", None)
    if part2_config:
        # Run Part 2 only (Interview + Scoring + Reporting for approved candidates)
        stage_names = ["Interview", "Scoring", "Reporting"]
    else:
        stage_names = ["Scout", "Researcher", "Engagement", "Screener", "Coordinator", "Interview", "Scoring", "Reporting"] if enable_live_scouting else ["Sourcing", "Screener", "Coordinator", "Interview", "Scoring", "Reporting"]
    st.session_state["lifecycle"] = {name: False for name in stage_names}
    st.session_state["current_activity"] = ACTIVITY_MSGS.get(stage_names[0], "Starting...")

    def _on_task_complete(stage_name: str, completed: bool, output):
        """Callback when a task starts or completes: update lifecycle and current activity."""
        if completed:
            st.session_state["lifecycle"][stage_name] = True
            idx = stage_names.index(stage_name) + 1 if stage_name in stage_names else 0
            if idx < len(stage_names):
                st.session_state["current_activity"] = ACTIVITY_MSGS.get(stage_names[idx], "")
            else:
                st.session_state["current_activity"] = "Pipeline complete!"
        else:
            # Task started: show current stage activity (don't mark lifecycle complete)
            st.session_state["current_activity"] = ACTIVITY_MSGS.get(stage_name, f"{stage_name} is running...")

    # 1. Check Ollama is running
    ollama_ok, ollama_msg = check_ollama_running()
    if not ollama_ok:
        st.session_state["pipeline_running"] = False
        st.error(f"**Ollama check failed:** {ollama_msg}")
        st.stop()
    st.toast(ollama_msg, icon="✅")

    # 2. Process uploaded files into candidate data
    try:
        job_description, resumes = process_uploaded_files(
            job_file, resume_files, base_dir, allow_empty_resumes=enable_live_scouting
        )
    except FileNotFoundError as e:
        st.session_state["pipeline_running"] = False
        st.error(str(e))
        st.stop()

    if not job_description:
        st.session_state["pipeline_running"] = False
        st.error("Job description is empty.")
        st.stop()

    if enable_live_scouting:
        if not serper_api_key or not serper_api_key.strip():
            st.session_state["pipeline_running"] = False
            st.error("Serper API Key is required when Live Web Scouting is enabled.")
            st.stop()
        # Live Scouting: resumes optional (Sourcing will search the web)
        if not resumes:
            resumes = []  # Empty list is fine; pipeline will search for candidates
    elif not resumes:
        st.session_state["pipeline_running"] = False
        st.error("No resumes to process. Upload .pdf/.txt files or add them to the resumes/ folder.")
        st.stop()

    # 3. Run pipeline with st.status, Thought Stream, and real-time lifecycle updates
    audit_log = []
    result_holder = [None]
    error_holder = [None]

    with st.status("Running recruitment pipeline...", expanded=True) as status:
        agent_chain = (
            "Live Talent Scout → Lead Researcher → Engagement Specialist → Screener → Coordinator → Interview Specialist"
            if enable_live_scouting
            else "Sourcing → Screener → Coordinator → Interview Specialist"
        )
        st.write(f"**Agents:** {agent_chain}")
        live_console = thought_placeholder
        progress_placeholder = st.empty()
        pdf_path = None
        ranking_table = []
        engagement_data = None
        interview_data = {}

        streamer = StreamlitStdout(live_console)
        feedback_queue = queue.Queue()
        human_feedback_prompt = [None]
        original_input = builtins.input

        def _custom_input(prompt=""):
            human_feedback_prompt[0] = prompt
            return feedback_queue.get()

        def run_pipeline():
            try:
                builtins.input = _custom_input
                with streamer:
                    if part2_config:
                        # Part 2: Interview + Scoring + Reporting
                        p1 = part2_config["part1"]
                        approved_names = part2_config["approved_names"]
                        agents = create_agents_with_config(
                            selected_model,
                            temperature,
                            enable_live_scouting=p1["enable_live_scouting"],
                            serper_api_key=p1.get("serper_api_key"),
                            audit_log=audit_log,
                            enable_safe_mode=p1.get("enable_safe_mode", False),
                            enable_fast_mode=p1.get("enable_fast_mode", False),
                        )
                        result_holder[0] = run_part2_pipeline(
                            job_description=p1["job_description"],
                            ranking_table=p1["ranking_table"],
                            engagement_data=p1["engagement_data"],
                            scout_output=p1["scout_output"],
                            researcher_output=p1["researcher_output"],
                            screener_output=p1["screener_output"],
                            engagement_output=p1["engagement_output"],
                            outreach_combined=p1["outreach_combined"],
                            approved_names=approved_names,
                            agents=agents,
                            audit_log=audit_log,
                            task_callback=_on_task_complete,
                        )
                    elif enable_human_approval_layer:
                        # Part 1 only: Sourcing & Research
                        result_holder[0] = run_part1_pipeline(
                            job_description,
                            resumes,
                            selected_model=selected_model,
                            temperature=temperature,
                            enable_live_scouting=enable_live_scouting,
                            serper_api_key=serper_api_key.strip() if serper_api_key else None,
                            enable_safe_mode=enable_safe_mode,
                            enable_fast_mode=enable_fast_mode,
                            enable_human_input=enable_human_input,
                            audit_log=audit_log,
                            task_callback=_on_task_complete,
                        )
                        # Part 1 returns different shape - store as part1_result
                        result_holder[0] = ("part1", result_holder[0])
                    else:
                        result_holder[0] = run_full_pipeline(
                            job_description,
                            resumes,
                            selected_model=selected_model,
                            temperature=temperature,
                            enable_live_scouting=enable_live_scouting,
                            serper_api_key=serper_api_key.strip() if serper_api_key else None,
                            enable_safe_mode=enable_safe_mode,
                            enable_fast_mode=enable_fast_mode,
                            enable_human_input=enable_human_input,
                            audit_log=audit_log,
                            task_callback=_on_task_complete,
                        )
            except Exception as e:
                error_holder[0] = e
            finally:
                builtins.input = original_input

        pipeline_thread = threading.Thread(target=run_pipeline)
        pipeline_thread.start()

        # Poll and update UI while pipeline runs (real-time lifecycle + thought stream + current activity)
        while pipeline_thread.is_alive():
            # Update thought stream from buffer (main thread only - Streamlit requires this)
            thought_placeholder.empty()
            with thought_placeholder.container():
                full_text = "".join(streamer.buffer)
                if full_text:
                    st.code(full_text[-50000:], language="text")  # Limit size
                else:
                    st.caption("Waiting for agent output...")
            # Update sidebar lifecycle checkboxes in real time
            lifecycle_placeholder.empty()
            with lifecycle_placeholder.container():
                lifecycle = st.session_state.get("lifecycle", {})
                for name in stage_names:
                    done = lifecycle.get(name, False)
                    st.markdown(f"{'☑' if done else '☐'} {name}")
            current_activity_placeholder.empty()
            with current_activity_placeholder.container():
                st.caption("**Current Activity:**")
                st.caption(st.session_state.get("current_activity", "Running..."))
            progress_placeholder.empty()
            with progress_placeholder.container():
                st.caption("**Current Activity:**")
                st.info(st.session_state.get("current_activity", "Running..."))
                lifecycle = st.session_state.get("lifecycle", {})
                completed = [n for n in stage_names if lifecycle.get(n)]
                if completed:
                    st.caption(f"Completed: {', '.join(completed)}")
                # Human Feedback UI: when Coordinator pauses (prompt is set), provide feedback in dashboard
                if enable_human_input and human_feedback_prompt[0] is not None:
                    st.divider()
                    st.markdown("**💬 Human Feedback Required**")
                    st.caption(human_feedback_prompt[0])
                    st.caption("If happy with the result, leave blank and Submit to continue. Otherwise, type improvement requests.")
                    feedback_val = st.text_input("Your feedback", key="human_feedback_input", placeholder="Press Submit to approve, or type improvement requests")
                    if st.button("Submit feedback", key="human_feedback_submit"):
                        feedback_queue.put(feedback_val.strip() if feedback_val else "\n")
                        human_feedback_prompt[0] = None  # Clear so UI hides until next prompt
            time.sleep(1)

        pipeline_thread.join()
        # Final update of sidebar so checkboxes show complete state before rerun
        lifecycle_placeholder.empty()
        with lifecycle_placeholder.container():
            lifecycle = st.session_state.get("lifecycle", {})
            for name in stage_names:
                done = lifecycle.get(name, False)
                st.markdown(f"{'☑' if done else '☐'} {name}")
        current_activity_placeholder.empty()
        with current_activity_placeholder.container():
            st.caption("**Current Activity:**")
            st.caption(st.session_state.get("current_activity", "Pipeline complete!"))
        if error_holder[0]:
            st.session_state["pipeline_running"] = False
            status.update(label="Pipeline failed", state="error")
            st.error(f"Pipeline failed: {error_holder[0]}")
            raise error_holder[0]

        raw_result = result_holder[0]

        # Handle Part 1 result (Human Approval Layer)
        if isinstance(raw_result, tuple) and raw_result[0] == "part1":
            _, part1_data = raw_result
            (ranking_table, engagement_data, scout_output, researcher_output, screener_output,
             engagement_output, outreach_combined, agents, audit_log) = part1_data
            if audit_log:
                thought_text = "".join(streamer.buffer) if streamer.buffer else ""
                audit_log.append({"type": "thought_stream", "content": thought_text[:50000]})
                save_audit_log(audit_log, base_dir)
            st.session_state["current_activity"] = "Part 1 complete — Review candidates"
            for name in stage_names:
                st.session_state["lifecycle"][name] = True
            st.session_state["pipeline_running"] = False
            st.session_state["part1_result"] = {
                "ranking_table": ranking_table,
                "engagement_data": engagement_data,
                "scout_output": scout_output,
                "researcher_output": researcher_output,
                "screener_output": screener_output,
                "engagement_output": engagement_output,
                "outreach_combined": outreach_combined,
                "audit_log": audit_log,
                "job_description": job_description,
                "enable_live_scouting": enable_live_scouting,
                "serper_api_key": serper_api_key.strip() if serper_api_key else None,
                "enable_safe_mode": enable_safe_mode,
                "enable_fast_mode": enable_fast_mode,
            }
            status.update(label="Part 1 complete — Review & approve candidates", state="complete")
            st.rerun()
        else:
            # Full pipeline or Part 2 result
            pdf_path, ranking_table, engagement_data, interview_data, audit_log, scoring_data = raw_result

            # Add thought stream to audit log and save
            if audit_log:
                thought_text = "".join(streamer.buffer) if streamer.buffer else ""
                audit_log.append({"type": "thought_stream", "content": thought_text[:50000]})
                audit_path = save_audit_log(audit_log, base_dir)
                st.caption(f"📁 Audit log saved: `.hireai_audit/{audit_path.name}`")

            st.session_state["current_activity"] = "Pipeline complete!"
            st.session_state.pop("part1_result", None)
            st.session_state.pop("approve_state", None)
            # Mark all lifecycle stages complete (fallback if callbacks didn't fire)
            for name in stage_names:
                st.session_state["lifecycle"][name] = True
            st.session_state["pipeline_running"] = False
            st.session_state["pipeline_result"] = (pdf_path, ranking_table, engagement_data, interview_data, audit_log, scoring_data)
            status.update(label="Pipeline complete!", state="complete")
            st.rerun()  # Refresh to show updated sidebar checkboxes
