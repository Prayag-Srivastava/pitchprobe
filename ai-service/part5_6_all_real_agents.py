"""
Part 5.6 of PitchProbe — All Three Real Agents in Multi-Agent Graph
=====================================================================
The full real-agent integration. All 3 specialists (Market, Risk, Team)
are now production agents with their own tools, RAG, and structured outputs.

WHAT'S NEW vs 5.5:
  - risk_specialist now invokes the REAL Risk Agent (RiskReport)
  - team_specialist now invokes the REAL Team Agent (TeamReport)
  - Synthesizer extended with _format_risk_report() and _format_team_report()
  - Graph structure unchanged — proves the architecture is reusable

WHAT TO WATCH FOR:
  - All 3 agents will run IN PARALLEL when supervisor picks all three
  - Tool prints will interleave: 🔍 (market), ⚠️ (risk), 👥 (team)
  - This is the visual signature of true multi-agent parallelism
  - Total time = time of slowest agent (~30-60s), not sum of all three
"""

from typing import TypedDict, Literal, Annotated
import operator
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from dotenv import load_dotenv

# ⭐ THE THREE REAL AGENT IMPORTS
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
# STATE
# ============================================================

class PitchProbeState(TypedDict):
    startup_name: str
    
    is_valid: bool
    validation_message: str
    
    specialists_to_run: list[str]
    supervisor_reasoning: str
    
    # Merge-list: parallel writes accumulate (real reports + any future toys)
    specialist_reports: Annotated[list, operator.add]
    
    final_summary: str


# ============================================================
# SUPERVISOR'S DECISION SCHEMA
# ============================================================

class SupervisorDecision(BaseModel):
    specialists_to_run: list[Literal["market", "risk", "team"]] = Field(
        description="Which specialists should analyze this startup IN PARALLEL. "
                    "Pick MULTIPLE specialists based on the startup's needs."
    )
    reasoning: str = Field(
        description="2-3 sentences explaining why these specialists were chosen."
    )


# ============================================================
# LLM (for supervisor — agents have their own models internally)
# ============================================================

llm = init_chat_model(
    "llama-3.3-70b-versatile",
    model_provider="groq",
    temperature=0.2,
    max_tokens=512,
)
supervisor_llm = llm.with_structured_output(SupervisorDecision)


# ============================================================
# ⭐ INITIALIZE ALL THREE REAL AGENTS (once, at module load)
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
            "Your job is to decide which specialists should analyze the startup IN PARALLEL. "
            "Available specialists:\n"
            "  - 'market': analyzes market size, trends, opportunity (uses pitch deck + web)\n"
            "  - 'risk': analyzes regulatory, financial, operational, market risks (deck + web)\n"
            "  - 'team': analyzes founders, executives, advisors (deck + web)\n\n"
            "Pick MULTIPLE specialists when relevant. Be decisive."
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


# ─── Market Specialist (real agent — same as 5.5) ─────────────
def market_specialist(state: PitchProbeState) -> dict:
    print("📊 [Node] market_specialist running — invoking REAL agent...")
    
    try:
        result = market_agent.invoke({
            "messages": [{
                "role": "user",
                "content": f"Analyze the market for the startup '{state['startup_name']}'. "
                           f"Produce a comprehensive market research report."
            }]
        })
        report = result.get("structured_response")
        if report is None:
            print("   ⚠️  Market agent did not produce structured output.")
            report = f"[Market agent failed for {state['startup_name']}]"
    except Exception as e:
        print(f"   ❌ Market agent error: {e}")
        report = f"[Market agent error: {e}]"
    
    print("   ✓ market_specialist done")
    return {
        "specialist_reports": [{"name": "market", "report": report}]
    }


# ─── Risk Specialist (REAL agent — NEW in 5.6) ────────────────
def risk_specialist(state: PitchProbeState) -> dict:
    print("⚠️  [Node] risk_specialist running — invoking REAL agent...")
    
    try:
        result = risk_agent.invoke({
            "messages": [{
                "role": "user",
                "content": f"Analyze risks for the startup '{state['startup_name']}'. "
                           f"Cover regulatory, financial, operational, and market risks."
            }]
        })
        report = result.get("structured_response")
        if report is None:
            print("   ⚠️  Risk agent did not produce structured output.")
            report = f"[Risk agent failed for {state['startup_name']}]"
    except Exception as e:
        print(f"   ❌ Risk agent error: {e}")
        report = f"[Risk agent error: {e}]"
    
    print("   ✓ risk_specialist done")
    return {
        "specialist_reports": [{"name": "risk", "report": report}]
    }


# ─── Team Specialist (REAL agent — NEW in 5.6) ────────────────
def team_specialist(state: PitchProbeState) -> dict:
    print("👥 [Node] team_specialist running — invoking REAL agent...")
    
    try:
        result = team_agent.invoke({
            "messages": [{
                "role": "user",
                "content": f"Analyze the team of the startup '{state['startup_name']}'. "
                           f"Evaluate founders, key executives, and team strength."
            }]
        })
        report = result.get("structured_response")
        if report is None:
            print("   ⚠️  Team agent did not produce structured output.")
            report = f"[Team agent failed for {state['startup_name']}]"
    except Exception as e:
        print(f"   ❌ Team agent error: {e}")
        report = f"[Team agent error: {e}]"
    
    print("   ✓ team_specialist done")
    return {
        "specialist_reports": [{"name": "team", "report": report}]
    }


# ─── Synthesizer (polymorphic — handles all 3 report types) ───
def synthesizer(state: PitchProbeState) -> dict:
    print("📝 [Node] synthesizer (combining all reports)")
    reports = state["specialist_reports"]
    
    findings = ""
    for r in reports:
        name = r["name"]
        payload = r["report"]
        findings += f"\n## {name.upper()} SPECIALIST\n"
        
        # Polymorphic formatting — dispatch by type
        if isinstance(payload, MarketResearchReport):
            findings += _format_market_report(payload)
        elif isinstance(payload, RiskReport):
            findings += _format_risk_report(payload)
        elif isinstance(payload, TeamReport):
            findings += _format_team_report(payload)
        elif isinstance(payload, BaseModel):
            # Fallback for any future Pydantic schema
            findings += payload.model_dump_json(indent=2)
        else:
            # String (error message or fallback)
            findings += str(payload)
        
        findings += "\n"
    
    return {
        "final_summary": (
            f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"FINAL ANALYSIS for {state['startup_name']}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Supervisor chose: {', '.join(state['specialists_to_run'])}\n"
            f"Reason: {state['supervisor_reasoning']}\n"
            f"\n--- FINDINGS ({len(reports)} specialists) ---{findings}"
        ),
    }


# ============================================================
# FORMATTING HELPERS (one per report type)
# ============================================================

def _format_market_report(r: MarketResearchReport) -> str:
    lines = [
        f"Industry:        {r.industry}",
        f"Market Size:     {r.market_size_usd or 'Unknown'}",
        f"Deck Available:  {r.deck_available}",
        f"Confidence:      {r.confidence.upper()}",
        "",
        "Growth Trends:",
    ]
    lines += [f"  • {t}" for t in r.growth_trends]
    lines += ["", "Target Customers:"]
    lines += [f"  • {c}" for c in r.target_customers]
    lines += ["", "Competitors:"]
    lines += [f"  • {c}" for c in r.main_competitors]
    lines += ["", "Opportunities:"]
    lines += [f"  • {o}" for o in r.market_opportunities]
    lines += ["", "Risks:"]
    lines += [f"  • {x}" for x in r.market_risks]
    lines += ["", "Sources:"]
    lines += [f"  • {s}" for s in r.sources]
    return "\n".join(lines)


def _format_risk_report(r: RiskReport) -> str:
    lines = [
        f"Overall Severity: {r.overall_severity.upper()}",
        f"Deck Available:   {r.deck_available}",
        f"Confidence:       {r.confidence.upper()}",
        "",
        "Regulatory Risks:",
    ]
    lines += [f"  • {x}" for x in r.regulatory_risks]
    lines += ["", "Financial Risks:"]
    lines += [f"  • {x}" for x in r.financial_risks]
    lines += ["", "Operational Risks:"]
    lines += [f"  • {x}" for x in r.operational_risks]
    lines += ["", "Market Risks:"]
    lines += [f"  • {x}" for x in r.market_risks]
    if r.mitigations:
        lines += ["", "Mitigations:"]
        lines += [f"  • {m}" for m in r.mitigations]
    lines += ["", "Sources:"]
    lines += [f"  • {s}" for s in r.sources]
    return "\n".join(lines)


def _format_team_report(r: TeamReport) -> str:
    lines = [
        f"Team Score:       {r.overall_team_score.upper()}",
        f"Team Size:        {r.team_size_estimate or 'Unknown'}",
        f"Deck Available:   {r.deck_available}",
        f"Confidence:       {r.confidence.upper()}",
        "",
        "Founders:",
    ]
    for f in r.founders:
        lines.append(f"  • {f.name} ({f.role})")
        lines.append(f"      {f.background}")
    if r.key_executives:
        lines += ["", "Key Executives:"]
        for e in r.key_executives:
            lines.append(f"  • {e.name} ({e.role})")
            lines.append(f"      {e.background}")
    lines += ["", "Notable Strengths:"]
    lines += [f"  • {s}" for s in r.notable_strengths]
    if r.notable_gaps:
        lines += ["", "Notable Gaps:"]
        lines += [f"  • {g}" for g in r.notable_gaps]
    if r.advisors:
        lines += ["", "Advisors:"]
        lines += [f"  • {a}" for a in r.advisors]
    lines += ["", "Sources:"]
    lines += [f"  • {s}" for s in r.sources]
    return "\n".join(lines)


# ============================================================
# ROUTING FUNCTIONS (same as 5.5)
# ============================================================

def route_after_validation(state: PitchProbeState) -> str:
    return "valid" if state["is_valid"] else "invalid"


def route_to_specialists(state: PitchProbeState) -> list[Send]:
    return [
        Send(f"{name}_specialist", state)
        for name in state["specialists_to_run"]
    ]


# ============================================================
# BUILD THE GRAPH (same shape as 5.5)
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
        "validate",
        route_after_validation,
        {"valid": "supervisor", "invalid": END}
    )
    
    workflow.add_conditional_edges(
        "supervisor",
        route_to_specialists,
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
    print("\n" + "=" * 70)
    print("📊 FINAL STATE")
    print("=" * 70)
    print(f"  startup_name:          {state.get('startup_name')}")
    print(f"  is_valid:              {state.get('is_valid')}")
    specs = state.get('specialists_to_run', [])
    print(f"  specialists_to_run:    {', '.join(specs) if specs else '—'}")
    reports = state.get('specialist_reports', [])
    print(f"  # of reports:          {len(reports)}")
    if state.get("final_summary"):
        print(state["final_summary"])
    print("=" * 70)


def main():
    print("🚀 PitchProbe — All Real Agents in Multi-Agent Graph (Part 5.6)")
    print("=" * 70)
    
    graph = build_graph()
    
    # ─── Test 1: Airbnb (deck loaded — all 3 agents use deck + web) ───
    print("\n\n🧪 TEST 1: Airbnb (deck loaded)")
    print("-" * 70)
    result = graph.invoke({"startup_name": "Airbnb"})
    print_state(result)
    
    # ─── Test 2: Stripe (no deck — all 3 agents use web only) ─────────
    print("\n\n🧪 TEST 2: Stripe (no deck — web only)")
    print("-" * 70)
    result = graph.invoke({"startup_name": "Stripe"})
    print_state(result)
    
    # ─── Test 3: Validation gate ──────────────────────────────────────
    print("\n\n🧪 TEST 3: Empty name (validation gate)")
    print("-" * 70)
    result = graph.invoke({"startup_name": ""})
    print_state(result)


if __name__ == "__main__":
    main()