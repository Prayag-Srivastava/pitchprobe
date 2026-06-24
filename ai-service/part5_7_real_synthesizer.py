"""
Part 5.7 of PitchProbe — Real LLM Synthesizer
================================================
The FINAL piece of Part 5. The synthesizer becomes a real LLM-powered
reasoner that doesn't just collect specialist reports — it cross-analyzes
them and produces an INVESTMENT RECOMMENDATION.

WHAT'S NEW vs 5.6:
  - FinalInvestmentReport Pydantic schema (the synthesizer's output)
  - Synthesis LLM with structured output
  - Synthesizer node calls LLM instead of just formatting strings
  - New state field: final_report (the structured object)
  - format_reports_for_llm() converts Pydantic reports to LLM-readable text
  - format_final_report() pretty-prints the final recommendation

THE GRAPH STRUCTURE IS UNCHANGED.
"""

from typing import TypedDict, Literal, Annotated, Optional
import operator
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from dotenv import load_dotenv

# The three specialist agents (unchanged)
from pitchprobe_agents.market_agent import (
    create_market_research_agent,
    MarketResearchReport,
)
from pitchprobe_agents.risk_agent import (
    create_risk_agent,
    RiskReport,
)
from pitchprobe_agents.team_agent import (
    create_team_agent,
    TeamReport,
)

load_dotenv()


# ============================================================
# ⭐ NEW SCHEMA: The Final Investment Report
# ============================================================
# This is what a VC partner produces after reviewing analyst reports.
# It's NOT just a summary — it's a REASONED RECOMMENDATION.

class FinalInvestmentReport(BaseModel):
    """The synthesized investment recommendation for a startup."""
    
    startup_name: str = Field(description="The startup being evaluated")
    
    executive_summary: str = Field(
        description="A 2-3 sentence elevator pitch summarizing what this startup is, "
                    "its current state, and the headline recommendation."
    )
    
    investment_recommendation: Literal[
        "strong_invest",
        "invest_with_caution",
        "neutral",
        "weak_pass",
        "strong_pass"
    ] = Field(
        description="The investment recommendation. "
                    "'strong_invest' = compelling across all dimensions, rare. "
                    "'invest_with_caution' = promising with specific reservations. "
                    "'neutral' = mixed signals, more info needed. "
                    "'weak_pass' = mild concerns, would not lead but could follow. "
                    "'strong_pass' = serious concerns make this uninvestable now."
    )
    
    confidence_in_recommendation: Literal["high", "medium", "low"] = Field(
        description="Confidence in the recommendation. "
                    "HIGH only if all 3 specialist reports had 'high' confidence themselves."
    )
    
    key_strengths: list[str] = Field(
        description="3-5 cross-cutting strengths — things that appear strong across "
                    "MULTIPLE specialist reports (not just one)."
    )
    
    key_concerns: list[str] = Field(
        description="3-5 cross-cutting concerns — issues raised across MULTIPLE reports."
    )
    
    red_flags: list[str] = Field(
        default_factory=list,
        description="Critical issues that materially threaten the investment thesis. "
                    "Empty list if none. Be honest — do not invent red flags, but do "
                    "not suppress real ones either."
    )
    
    cross_cutting_insights: list[str] = Field(
        description="ALIGNMENTS (e.g., 'team's hospitality background aligns with market trend') "
                    "or CONTRADICTIONS (e.g., 'risk report says HIGH regulatory exposure but "
                    "team has no compliance leader') discovered by combining the reports. "
                    "These insights are the VALUE-ADD of synthesis — surface them explicitly."
    )
    
    due_diligence_next_steps: list[str] = Field(
        description="Concrete next investigations needed before making a final decision "
                    "(e.g., 'reference-check Brian Chesky's prior co-founder', "
                    "'request 3 years of unit economics', 'speak with 2 enterprise customers')."
    )
    
    overall_reasoning: str = Field(
        description="4-6 sentence narrative that explains the recommendation. "
                    "Must reference SPECIFIC findings from the specialist reports, "
                    "not generic startup analysis."
    )


# ============================================================
# STATE (added one field: final_report)
# ============================================================

class PitchProbeState(TypedDict):
    startup_name: str
    
    is_valid: bool
    validation_message: str
    
    specialists_to_run: list[str]
    supervisor_reasoning: str
    
    specialist_reports: Annotated[list, operator.add]
    
    final_summary: str                          # human-readable formatted text
    final_report: Optional[FinalInvestmentReport]   # ⭐ NEW — the structured object


# ============================================================
# SUPERVISOR'S DECISION SCHEMA (unchanged)
# ============================================================

class SupervisorDecision(BaseModel):
    specialists_to_run: list[Literal["market", "risk", "team"]] = Field(
        description="Which specialists should analyze this startup IN PARALLEL."
    )
    reasoning: str = Field(description="2-3 sentences explaining the choice.")


# ============================================================
# LLMs
# ============================================================

# Supervisor LLM
supervisor_base = init_chat_model(
    "llama-3.3-70b-versatile",
    model_provider="groq",
    temperature=0.2,
    max_tokens=512,
)
supervisor_llm = supervisor_base.with_structured_output(SupervisorDecision)

# ⭐ NEW: Synthesizer LLM (higher temperature for more nuanced reasoning,
# more tokens for longer narrative output)
synthesis_base = init_chat_model(
    "llama-3.3-70b-versatile",
    model_provider="groq",
    temperature=0.4,         # slightly higher for nuanced reasoning
    max_tokens=2048,         # synthesis output is longer than supervisor decision
)
synthesis_llm = synthesis_base.with_structured_output(FinalInvestmentReport)


# ============================================================
# INITIALIZE SPECIALIST AGENTS (once, at module load)
# ============================================================

print("⚙️  Initializing all real agents...")
print("   - Market Research Agent...")
market_agent = create_market_research_agent()
print("   - Risk Analysis Agent...")
risk_agent = create_risk_agent()
print("   - Team Analysis Agent...")
team_agent = create_team_agent()
print("✓ All agents ready.\n")


# ============================================================
# SYNTHESIS SYSTEM PROMPT
# ============================================================

SYNTHESIS_PROMPT = """You are a Senior Partner at a top-tier VC firm. \
Three analysts have just submitted their specialist reports on a startup. \
Your job is to synthesize them into a final investment recommendation.

YOUR JOB IS NOT to summarize the reports — it's to REASON ACROSS them.

WHAT MAKES A GOOD SYNTHESIS:
1. ALIGNMENTS — when multiple reports point to the same insight, that's evidence
2. CONTRADICTIONS — when reports disagree, that's a red flag worth investigating
3. CROSS-CUTTING INSIGHTS — patterns visible only when reports are combined
4. CONCRETE next steps — what would you ask the analysts to investigate further?

RECOMMENDATION CRITERIA:
- 'strong_invest' is RARE. Use only if ALL three reports are highly positive AND \
the cross-cutting insights reinforce each other.
- 'invest_with_caution' is for compelling startups with specific reservations \
that can be addressed in due diligence.
- 'neutral' means mixed signals — you need more info before deciding.
- 'weak_pass' is for startups with mild concerns where you wouldn't lead.
- 'strong_pass' is for serious problems — regulatory bombs, team gaps, dead markets.

QUALITY RULES:
- Reference SPECIFIC findings from the reports. Generic statements are useless.
- If reports have LOW confidence themselves, your recommendation confidence \
should also be capped at MEDIUM.
- Surface contradictions you spot. VCs hate surprises later.
- Be honest about red flags. Suppressing them costs the fund money."""


# ============================================================
# NODES
# ============================================================

def validate_startup(state: PitchProbeState) -> dict:
    print("🔍 [Node] validate_startup")
    name = state["startup_name"]
    if not name or not name.strip():
        return {"is_valid": False, "validation_message": "Invalid: empty name"}
    elif len(name) > 100:
        return {"is_valid": False, "validation_message": "Invalid: name too long"}
    else:
        print(f"   ✓ '{name}' is valid")
        return {"is_valid": True, "validation_message": "Valid"}


def supervisor(state: PitchProbeState) -> dict:
    print("🧠 [Node] supervisor (LLM deciding which specialists...)")
    
    messages = [
        SystemMessage(content=(
            "You are a Supervisor in a startup due diligence system. "
            "Decide which specialists analyze the startup IN PARALLEL. "
            "Available: 'market', 'risk', 'team'. Pick multiple when relevant."
        )),
        HumanMessage(content=f"Startup to analyze: {state['startup_name']}"),
    ]
    
    decision: SupervisorDecision = supervisor_llm.invoke(messages)
    print(f"   → Specialists picked: {decision.specialists_to_run}")
    print(f"   → Reasoning: {decision.reasoning}")
    
    return {
        "specialists_to_run": decision.specialists_to_run,
        "supervisor_reasoning": decision.reasoning,
    }


# ─── Specialists (unchanged from 5.6) ─────────────────────────

def market_specialist(state: PitchProbeState) -> dict:
    print("📊 [Node] market_specialist running — invoking REAL agent...")
    try:
        result = market_agent.invoke({
            "messages": [{"role": "user",
                          "content": f"Analyze the market for the startup '{state['startup_name']}'."}]
        })
        report = result.get("structured_response") or f"[Market agent failed]"
    except Exception as e:
        report = f"[Market agent error: {e}]"
    print("   ✓ market_specialist done")
    return {"specialist_reports": [{"name": "market", "report": report}]}


def risk_specialist(state: PitchProbeState) -> dict:
    print("⚠️  [Node] risk_specialist running — invoking REAL agent...")
    try:
        result = risk_agent.invoke({
            "messages": [{"role": "user",
                          "content": f"Analyze risks for the startup '{state['startup_name']}'."}]
        })
        report = result.get("structured_response") or f"[Risk agent failed]"
    except Exception as e:
        report = f"[Risk agent error: {e}]"
    print("   ✓ risk_specialist done")
    return {"specialist_reports": [{"name": "risk", "report": report}]}


def team_specialist(state: PitchProbeState) -> dict:
    print("👥 [Node] team_specialist running — invoking REAL agent...")
    try:
        result = team_agent.invoke({
            "messages": [{"role": "user",
                          "content": f"Analyze the team of the startup '{state['startup_name']}'."}]
        })
        report = result.get("structured_response") or f"[Team agent failed]"
    except Exception as e:
        report = f"[Team agent error: {e}]"
    print("   ✓ team_specialist done")
    return {"specialist_reports": [{"name": "team", "report": report}]}


# ─── ⭐ NEW SYNTHESIZER — LLM-POWERED REASONER ────────────────

def synthesizer(state: PitchProbeState) -> dict:
    """
    The new synthesizer:
    1. Converts the 3 specialist reports into structured LLM-readable text
    2. Asks an LLM to produce a FinalInvestmentReport
    3. Returns BOTH the structured object AND a formatted human-readable string
    """
    print("📝 [Node] synthesizer — calling LLM for investment recommendation...")
    
    reports = state["specialist_reports"]
    
    # Step 1: Convert reports to text the LLM can read
    context = format_reports_for_llm(state["startup_name"], reports)
    
    # Step 2: Call the synthesis LLM
    try:
        final_report: FinalInvestmentReport = synthesis_llm.invoke([
            SystemMessage(content=SYNTHESIS_PROMPT),
            HumanMessage(content=context),
        ])
        print(f"   → Recommendation: {final_report.investment_recommendation.upper()}")
        print(f"   → Confidence: {final_report.confidence_in_recommendation.upper()}")
    except Exception as e:
        print(f"   ❌ Synthesis LLM error: {e}")
        # Graceful fallback: return a minimal report
        final_report = None
    
    # Step 3: Format the final output
    if final_report:
        formatted = format_final_report(final_report, reports)
    else:
        formatted = (
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"SYNTHESIS FAILED for {state['startup_name']}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"The synthesis LLM did not produce a structured report.\n"
            f"Raw specialist reports collected: {len(reports)}\n"
        )
    
    return {
        "final_summary": formatted,
        "final_report": final_report,
    }


# ============================================================
# REPORT FORMATTING HELPERS
# ============================================================

def format_reports_for_llm(startup_name: str, reports: list[dict]) -> str:
    """
    Convert the 3 Pydantic reports into a clean structured text block
    that the synthesis LLM can read.
    """
    sections = [f"# STARTUP UNDER REVIEW: {startup_name}\n"]
    
    for r in reports:
        name = r["name"]
        payload = r["report"]
        sections.append(f"\n## {name.upper()} SPECIALIST REPORT\n")
        
        if isinstance(payload, MarketResearchReport):
            sections.append(_market_to_text(payload))
        elif isinstance(payload, RiskReport):
            sections.append(_risk_to_text(payload))
        elif isinstance(payload, TeamReport):
            sections.append(_team_to_text(payload))
        else:
            sections.append(str(payload))
    
    return "\n".join(sections)


def _market_to_text(r: MarketResearchReport) -> str:
    return (
        f"Industry: {r.industry}\n"
        f"Market Size: {r.market_size_usd or 'Unknown'}\n"
        f"Deck Available: {r.deck_available} | Confidence: {r.confidence}\n"
        f"Growth Trends: {', '.join(r.growth_trends)}\n"
        f"Target Customers: {', '.join(r.target_customers)}\n"
        f"Main Competitors: {', '.join(r.main_competitors)}\n"
        f"Market Opportunities: {', '.join(r.market_opportunities)}\n"
        f"Market Risks: {', '.join(r.market_risks)}\n"
        f"Sources: {', '.join(r.sources)}"
    )


def _risk_to_text(r: RiskReport) -> str:
    mitigations = ', '.join(r.mitigations) if r.mitigations else 'None mentioned'
    return (
        f"Overall Severity: {r.overall_severity}\n"
        f"Deck Available: {r.deck_available} | Confidence: {r.confidence}\n"
        f"Regulatory Risks: {', '.join(r.regulatory_risks)}\n"
        f"Financial Risks: {', '.join(r.financial_risks)}\n"
        f"Operational Risks: {', '.join(r.operational_risks)}\n"
        f"Market Risks: {', '.join(r.market_risks)}\n"
        f"Mitigations: {mitigations}\n"
        f"Sources: {', '.join(r.sources)}"
    )


def _team_to_text(r: TeamReport) -> str:
    founders_text = "\n".join(
        f"  - {f.name} ({f.role}): {f.background}" for f in r.founders
    )
    execs_text = "\n".join(
        f"  - {e.name} ({e.role}): {e.background}" for e in r.key_executives
    ) if r.key_executives else "  None mentioned"
    gaps = ', '.join(r.notable_gaps) if r.notable_gaps else 'None identified'
    advisors = ', '.join(r.advisors) if r.advisors else 'None mentioned'
    return (
        f"Team Score: {r.overall_team_score}\n"
        f"Team Size: {r.team_size_estimate or 'Unknown'}\n"
        f"Deck Available: {r.deck_available} | Confidence: {r.confidence}\n"
        f"Founders:\n{founders_text}\n"
        f"Key Executives:\n{execs_text}\n"
        f"Notable Strengths: {', '.join(r.notable_strengths)}\n"
        f"Notable Gaps: {gaps}\n"
        f"Advisors: {advisors}\n"
        f"Sources: {', '.join(r.sources)}"
    )


def format_final_report(final: FinalInvestmentReport, raw_reports: list[dict]) -> str:
    """Pretty-print the final investment report for human consumption."""
    
    # Emoji for the recommendation
    rec_emoji = {
        "strong_invest":         "🟢🟢🟢",
        "invest_with_caution":   "🟢⚠️ ",
        "neutral":               "🟡   ",
        "weak_pass":             "🔴   ",
        "strong_pass":           "🔴🔴🔴",
    }.get(final.investment_recommendation, "⚪   ")
    
    lines = [
        "",
        "╔══════════════════════════════════════════════════════════════════════╗",
        f"║   FINAL INVESTMENT REPORT: {final.startup_name:<41}    ║",
        "╚══════════════════════════════════════════════════════════════════════╝",
        "",
        f"{rec_emoji} RECOMMENDATION:  {final.investment_recommendation.upper().replace('_', ' ')}",
        f"     Confidence:      {final.confidence_in_recommendation.upper()}",
        "",
        "── EXECUTIVE SUMMARY ────────────────────────────────────────────────",
        final.executive_summary,
        "",
        "── KEY STRENGTHS ─────────────────────────────────────────────────────",
    ]
    lines += [f"  ✓ {s}" for s in final.key_strengths]
    
    lines += ["", "── KEY CONCERNS ──────────────────────────────────────────────────────"]
    lines += [f"  ✗ {c}" for c in final.key_concerns]
    
    if final.red_flags:
        lines += ["", "── 🚩 RED FLAGS ──────────────────────────────────────────────────────"]
        lines += [f"  ! {r}" for r in final.red_flags]
    
    lines += ["", "── CROSS-CUTTING INSIGHTS ────────────────────────────────────────────"]
    lines += [f"  → {i}" for i in final.cross_cutting_insights]
    
    lines += ["", "── DUE DILIGENCE NEXT STEPS ──────────────────────────────────────────"]
    lines += [f"  • {s}" for s in final.due_diligence_next_steps]
    
    lines += ["", "── OVERALL REASONING ─────────────────────────────────────────────────"]
    lines += [final.overall_reasoning]
    
    lines += ["", "═" * 72]
    lines += ["", f"(Based on {len(raw_reports)} specialist reports)"]
    
    return "\n".join(lines)


# ============================================================
# ROUTING FUNCTIONS (unchanged)
# ============================================================

def route_after_validation(state: PitchProbeState) -> str:
    return "valid" if state["is_valid"] else "invalid"


def route_to_specialists(state: PitchProbeState) -> list[Send]:
    return [Send(f"{name}_specialist", state) for name in state["specialists_to_run"]]


# ============================================================
# BUILD GRAPH (unchanged structure)
# ============================================================

def build_graph():
    workflow = StateGraph(PitchProbeState)
    
    workflow.add_node("validate", validate_startup)
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("market_specialist", market_specialist)
    workflow.add_node("risk_specialist", risk_specialist)
    workflow.add_node("team_specialist", team_specialist)
    workflow.add_node("synthesizer", synthesizer)
    
    workflow.add_edge(START, "validate")
    workflow.add_conditional_edges(
        "validate", route_after_validation,
        {"valid": "supervisor", "invalid": END}
    )
    workflow.add_conditional_edges(
        "supervisor", route_to_specialists,
        ["market_specialist", "risk_specialist", "team_specialist"]
    )
    workflow.add_edge("market_specialist", "synthesizer")
    workflow.add_edge("risk_specialist", "synthesizer")
    workflow.add_edge("team_specialist", "synthesizer")
    workflow.add_edge("synthesizer", END)
    
    return workflow.compile()


# ============================================================
# PRINT & MAIN
# ============================================================

def print_state(state: dict) -> None:
    print("\n" + "=" * 72)
    print("📊 FINAL STATE")
    print("=" * 72)
    print(f"  startup_name:          {state.get('startup_name')}")
    print(f"  is_valid:              {state.get('is_valid')}")
    specs = state.get('specialists_to_run', [])
    print(f"  specialists_to_run:    {', '.join(specs) if specs else '—'}")
    reports = state.get('specialist_reports', [])
    print(f"  # of reports:          {len(reports)}")
    final = state.get('final_report')
    if final:
        print(f"  recommendation:        {final.investment_recommendation.upper()}")
        print(f"  confidence:            {final.confidence_in_recommendation.upper()}")
    if state.get("final_summary"):
        print(state["final_summary"])
    print("=" * 72)


def main():
    print("🚀 PitchProbe — Real LLM Synthesizer (Part 5.7 — FINAL)")
    print("=" * 72)
    
    graph = build_graph()
    
    print("\n\n🧪 TEST 1: Airbnb (deck loaded — full evaluation)")
    print("-" * 72)
    result = graph.invoke({"startup_name": "Airbnb"})
    print_state(result)
    
    print("\n\n🧪 TEST 2: Stripe (no deck — web-only specialist data)")
    print("-" * 72)
    result = graph.invoke({"startup_name": "Stripe"})
    print_state(result)
    
    print("\n\n🧪 TEST 3: Empty name (validation gate)")
    print("-" * 72)
    result = graph.invoke({"startup_name": ""})
    print_state(result)


if __name__ == "__main__":
    main()