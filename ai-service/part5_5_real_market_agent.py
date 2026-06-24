"""
Part 5.5 of PitchProbe — Real Market Agent in the Multi-Agent Graph
=====================================================================
Swaps the toy market_specialist (from 5.4) for the REAL Market
Research Agent (from pitchprobe_agents/market_agent.py).

This is the FIRST integration of a real agent into the LangGraph.
Risk and Team specialists remain toys — they get upgraded in 5.6.

WHAT TO WATCH WHEN YOU RUN IT:
  - Market specialist's "node" now runs a full multi-tool agent INTERNALLY
    (you'll see search_pitch_deck and search_web tool calls)
  - Risk and Team specialists still return placeholder strings
  - The synthesizer correctly handles BOTH real reports AND toy strings
  - This visual contrast = proof that the integration works
"""

from typing import TypedDict, Literal, Annotated, Union
import operator
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from dotenv import load_dotenv

# ⭐ THE KEY IMPORT — pulls in the real agent factory
from pitchprobe_agents.market_agent import (
    create_market_research_agent,
    MarketResearchReport,
)

load_dotenv()


# ============================================================
# STATE
# ============================================================
# specialist_reports is now a list of dicts where each dict's "report"
# field can be EITHER a string (toy specialist) OR a Pydantic object
# (real agent). The synthesizer must handle both.

class PitchProbeState(TypedDict):
    startup_name: str
    
    is_valid: bool
    validation_message: str
    
    specialists_to_run: list[str]
    supervisor_reasoning: str
    
    # Merge-list: parallel writes accumulate
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
# LLM (for supervisor + synthesizer)
# ============================================================

llm = init_chat_model(
    "llama-3.3-70b-versatile",
    model_provider="groq",
    temperature=0.2,
    max_tokens=512,
)
supervisor_llm = llm.with_structured_output(SupervisorDecision)


# ============================================================
# ⭐ THE REAL MARKET AGENT (built once at module load, reused per call)
# ============================================================
# Building the agent involves:
#   - loading the vector store
#   - querying for available decks
#   - constructing the dynamic system prompt
#   - registering tools and middleware
# This is expensive enough that we do it ONCE here, not per invocation.

print("⚙️  Initializing real Market Research Agent...")
market_agent = create_market_research_agent()
print("✓ Market Agent ready.\n")


# ============================================================
# NODES
# ============================================================

# ─── Validation (same as 5.4) ─────────────────────────────────
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


# ─── Supervisor (same as 5.4) ─────────────────────────────────
def supervisor(state: PitchProbeState) -> dict:
    print("🧠 [Node] supervisor (LLM deciding which specialists...)")
    
    messages = [
        SystemMessage(content=(
            "You are a Supervisor in a startup due diligence system. "
            "Your job is to decide which specialists should analyze the startup IN PARALLEL. "
            "Available specialists:\n"
            "  - 'market': analyzes market size, trends, opportunity (uses pitch deck + web)\n"
            "  - 'risk': analyzes regulatory, financial, and operational risks\n"
            "  - 'team': analyzes founders, team strength, hiring\n\n"
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


# ─── ⭐ Market Specialist — NOW USES THE REAL AGENT ──────────────
def market_specialist(state: PitchProbeState) -> dict:
    """
    Wraps the real Market Research Agent as a graph node.
    
    The agent is built ONCE at module load (above). Here we just
    invoke it with the startup name. The agent's internal loop
    (LLM → tool → LLM → tool → ...) all happens inside this call.
    
    The returned MarketResearchReport (a Pydantic object) gets
    stuffed into specialist_reports as the "report" value.
    """
    print("📊 [Node] market_specialist running — invoking REAL agent...")
    
    startup_name = state["startup_name"]
    
    try:
        result = market_agent.invoke({
            "messages": [{
                "role": "user",
                "content": f"Analyze the market for the startup '{startup_name}'. "
                           f"Produce a comprehensive market research report."
            }]
        })
        report = result.get("structured_response")
        
        if report is None:
            # Graceful fallback: agent ran but failed to produce structured output
            # (the known Llama-on-Groq issue documented in v2.2)
            print("   ⚠️  Market agent did not produce structured output.")
            report = (
                f"[Market Agent failed to produce structured output for {startup_name}. "
                f"This is a known Llama+Groq tool-call instability issue.]"
            )
    except Exception as e:
        print(f"   ❌ Market agent error: {e}")
        report = f"[Market agent error: {e}]"
    
    print("   ✓ market_specialist done")
    return {
        "specialist_reports": [{
            "name": "market",
            "report": report,   # Either a MarketResearchReport object or a string fallback
        }]
    }


# ─── Risk Specialist — STILL TOY ──────────────────────────────
def risk_specialist(state: PitchProbeState) -> dict:
    print("⚠️  [Node] risk_specialist running (toy — upgraded in 5.6)")
    return {
        "specialist_reports": [{
            "name": "risk",
            "report": (
                f"[TOY RISK REPORT for {state['startup_name']}] "
                f"Regulatory risks assessed, financial exposure analyzed."
            ),
        }]
    }


# ─── Team Specialist — STILL TOY ──────────────────────────────
def team_specialist(state: PitchProbeState) -> dict:
    print("👥 [Node] team_specialist running (toy — upgraded in 5.6)")
    return {
        "specialist_reports": [{
            "name": "team",
            "report": (
                f"[TOY TEAM REPORT for {state['startup_name']}] "
                f"Founders evaluated, team strength assessed."
            ),
        }]
    }


# ─── Synthesizer — NOW HANDLES BOTH STRINGS AND PYDANTIC OBJECTS ──
def synthesizer(state: PitchProbeState) -> dict:
    """
    Combines all specialist reports.
    
    KEY POINT: specialist_reports contains a MIX of:
      - Pydantic objects (from the real Market Agent)
      - Strings (from toy Risk/Team specialists)
    We must format each appropriately.
    """
    print("📝 [Node] synthesizer (combining all reports)")
    reports = state["specialist_reports"]
    
    findings = ""
    for r in reports:
        name = r["name"]
        payload = r["report"]
        findings += f"\n## {name.upper()} SPECIALIST\n"
        
        # Polymorphic formatting:
        if isinstance(payload, MarketResearchReport):
            # Real structured report — format the rich fields
            findings += _format_market_report(payload)
        elif isinstance(payload, BaseModel):
            # Generic Pydantic object (future-proofing for 5.6's real agents)
            findings += payload.model_dump_json(indent=2)
        else:
            # String (toy report or error message)
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


def _format_market_report(r: MarketResearchReport) -> str:
    """Pretty-format a MarketResearchReport for the final summary."""
    lines = [
        f"Industry:        {r.industry}",
        f"Market Size:     {r.market_size_usd or 'Unknown'}",
        f"Deck Available:  {r.deck_available}",
        f"Confidence:      {r.confidence.upper()}",
        f"",
        f"Growth Trends:",
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


# ============================================================
# ROUTING FUNCTIONS (same as 5.4)
# ============================================================

def route_after_validation(state: PitchProbeState) -> str:
    return "valid" if state["is_valid"] else "invalid"


def route_to_specialists(state: PitchProbeState) -> list[Send]:
    return [
        Send(f"{name}_specialist", state)
        for name in state["specialists_to_run"]
    ]


# ============================================================
# BUILD THE GRAPH (same shape as 5.4)
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
    print("🚀 PitchProbe — Real Market Agent in Multi-Agent Graph (Part 5.5)")
    print("=" * 70)
    
    graph = build_graph()
    
    # ─── Test 1: Airbnb (has a deck → real agent should use it) ───
    print("\n\n🧪 TEST 1: Airbnb (deck loaded — real agent should use RAG + web)")
    print("-" * 70)
    result = graph.invoke({"startup_name": "Airbnb"})
    print_state(result)
    
    # ─── Test 2: Veeba (no deck → real agent should use web only) ──
    print("\n\n🧪 TEST 2: Veeba (no deck loaded — real agent should use web only)")
    print("-" * 70)
    result = graph.invoke({"startup_name": "Veeba"})
    print_state(result)
    
    # ─── Test 3: Validation gate ──────────────────────────────────
    print("\n\n🧪 TEST 3: Empty name (validation gate)")
    print("-" * 70)
    result = graph.invoke({"startup_name": ""})
    print_state(result)


if __name__ == "__main__":
    main()