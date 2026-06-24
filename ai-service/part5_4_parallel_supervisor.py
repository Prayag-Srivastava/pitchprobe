"""
Part 5.4 of PitchProbe — Parallel Specialists with Send()
==========================================================
Upgrade from 5.3: Supervisor now picks MULTIPLE specialists,
and they run IN PARALLEL using the Send() API.

KEY NEW MECHANICS:
  - Send() — spawn N parallel worker nodes at runtime
  - Annotated[list, operator.add] — merge parallel writes
  - Routing function returns list[Send] instead of str
"""

from typing import TypedDict, Literal, Annotated
import operator
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# STATE
# ============================================================
# KEY UPGRADE: specialist_reports uses Annotated[list, operator.add]
# This is what allows parallel writes to MERGE instead of overwrite.

class PitchProbeState(TypedDict):
    startup_name: str
    
    is_valid: bool
    validation_message: str
    
    specialists_to_run: list[str]      # supervisor's plural decision
    supervisor_reasoning: str
    
    # ⭐ MERGE-LIST: parallel writes get concatenated, not overwritten
    specialist_reports: Annotated[list, operator.add]
    
    final_summary: str


# ============================================================
# SUPERVISOR'S DECISION SCHEMA
# ============================================================
# list[Literal[...]] = a list where each element must be one of these 3 values

class SupervisorDecision(BaseModel):
    """The structured output the Supervisor MUST produce."""
    
    specialists_to_run: list[Literal["market", "risk", "team"]] = Field(
        description="Which specialists should analyze this startup IN PARALLEL. "
                    "Pick 'market' for market opportunity analysis, "
                    "'risk' for risk/compliance concerns, "
                    "'team' for founder/team analysis. "
                    "Pick MULTIPLE specialists based on the startup's needs. "
                    "You can pick 1, 2, or all 3."
    )
    reasoning: str = Field(
        description="2-3 sentences explaining why these specialists were chosen."
    )


# ============================================================
# LLM
# ============================================================

llm = init_chat_model(
    "llama-3.3-70b-versatile",
    model_provider="groq",
    temperature=0.2,
    max_tokens=512,
)
supervisor_llm = llm.with_structured_output(SupervisorDecision)


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
    startup_name = state["startup_name"]
    
    messages = [
        SystemMessage(content=(
            "You are a Supervisor in a startup due diligence system. "
            "Your job is to decide which specialists should analyze the startup IN PARALLEL. "
            "Available specialists:\n"
            "  - 'market': analyzes market size, trends, opportunity\n"
            "  - 'risk': analyzes regulatory, financial, and operational risks\n"
            "  - 'team': analyzes founders, team strength, hiring\n\n"
            "Pick MULTIPLE specialists when relevant. Be decisive."
        )),
        HumanMessage(content=f"Startup to analyze: {startup_name}"),
    ]
    
    decision: SupervisorDecision = supervisor_llm.invoke(messages)
    
    print(f"   → Specialists picked: {decision.specialists_to_run}")
    print(f"   → Reasoning: {decision.reasoning}")
    
    return {
        "specialists_to_run": decision.specialists_to_run,
        "supervisor_reasoning": decision.reasoning,
    }


# ─── Specialists: each writes to specialist_reports (the merge-list) ───
# Each returns a SINGLE-ITEM LIST. LangGraph concatenates all such lists
# from parallel workers into one combined list.

def market_specialist(state: PitchProbeState) -> dict:
    print("📊 [Node] market_specialist running (parallel)")
    return {
        "specialist_reports": [{
            "name": "market",
            "report": f"[TOY MARKET REPORT for {state['startup_name']}] "
                      f"Market size estimated, trends identified, opportunities mapped.",
        }]
    }


def risk_specialist(state: PitchProbeState) -> dict:
    print("⚠️  [Node] risk_specialist running (parallel)")
    return {
        "specialist_reports": [{
            "name": "risk",
            "report": f"[TOY RISK REPORT for {state['startup_name']}] "
                      f"Regulatory risks assessed, financial exposure analyzed.",
        }]
    }


def team_specialist(state: PitchProbeState) -> dict:
    print("👥 [Node] team_specialist running (parallel)")
    return {
        "specialist_reports": [{
            "name": "team",
            "report": f"[TOY TEAM REPORT for {state['startup_name']}] "
                      f"Founders evaluated, team strength assessed.",
        }]
    }


def synthesizer(state: PitchProbeState) -> dict:
    print("📝 [Node] synthesizer (combining all reports)")
    reports = state["specialist_reports"]
    
    findings = ""
    for r in reports:
        findings += f"\n## {r['name'].upper()} SPECIALIST\n{r['report']}\n"
    
    return {
        "final_summary": (
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"FINAL ANALYSIS for {state['startup_name']}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Supervisor chose: {', '.join(state['specialists_to_run'])}\n"
            f"Reason: {state['supervisor_reasoning']}\n"
            f"\n--- FINDINGS ({len(reports)} specialists) ---{findings}"
        ),
    }


# ============================================================
# ROUTING FUNCTIONS
# ============================================================

def route_after_validation(state: PitchProbeState) -> str:
    """Validation gate — returns a string."""
    return "valid" if state["is_valid"] else "invalid"


def route_to_specialists(state: PitchProbeState) -> list[Send]:
    """
    ⭐ THE KEY NEW PATTERN ⭐
    Returns a LIST of Send objects — one per specialist to spawn.
    LangGraph runs all of them IN PARALLEL.
    
    Each Send(name, state_dict) creates one parallel worker.
    """
    return [
        Send(f"{name}_specialist", state)
        for name in state["specialists_to_run"]
    ]


# ============================================================
# BUILD THE GRAPH
# ============================================================

def build_graph():
    workflow = StateGraph(PitchProbeState)
    
    # Nodes
    workflow.add_node("validate", validate_startup)
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("market_specialist", market_specialist)
    workflow.add_node("risk_specialist", risk_specialist)
    workflow.add_node("team_specialist", team_specialist)
    workflow.add_node("synthesizer", synthesizer)
    
    # Entry
    workflow.add_edge(START, "validate")
    
    # Validation gate (mapping dict — returning str)
    workflow.add_conditional_edges(
        "validate",
        route_after_validation,
        {"valid": "supervisor", "invalid": END}
    )
    
    # ⭐ Supervisor → parallel specialists via Send()
    # Note: 3rd arg is a LIST of possible target nodes, not a mapping dict
    workflow.add_conditional_edges(
        "supervisor",
        route_to_specialists,
        ["market_specialist", "risk_specialist", "team_specialist"]
    )
    
    # Fan-in: all specialists → synthesizer
    workflow.add_edge("market_specialist", "synthesizer")
    workflow.add_edge("risk_specialist", "synthesizer")
    workflow.add_edge("team_specialist", "synthesizer")
    
    # Exit
    workflow.add_edge("synthesizer", END)
    
    return workflow.compile()


# ============================================================
# PRINT & MAIN
# ============================================================

def print_state(state: dict) -> None:
    print("\n" + "=" * 60)
    print("📊 FINAL STATE")
    print("=" * 60)
    print(f"  startup_name:          {state.get('startup_name')}")
    print(f"  is_valid:              {state.get('is_valid')}")
    print(f"  validation_message:    {state.get('validation_message')}")
    specs = state.get('specialists_to_run', [])
    print(f"  specialists_to_run:    {', '.join(specs) if specs else '—'}")
    print(f"  supervisor_reasoning:  {state.get('supervisor_reasoning', '—')}")
    reports = state.get('specialist_reports', [])
    print(f"  # of reports:          {len(reports)}")
    print()
    if state.get("final_summary"):
        print(state["final_summary"])
    print("=" * 60)


def main():
    print("🚀 PitchProbe — Parallel Specialists with Send() (Part 5.4)")
    print("=" * 60)
    
    graph = build_graph()
    
    print("\n\n🧪 TEST 1: Stripe (likely market + risk)")
    print("-" * 60)
    result = graph.invoke({"startup_name": "Stripe"})
    print_state(result)
    
    print("\n\n🧪 TEST 2: OpenAI (likely all three)")
    print("-" * 60)
    result = graph.invoke({"startup_name": "OpenAI"})
    print_state(result)
    
    print("\n\n🧪 TEST 3: A small local bakery (likely just market)")
    print("-" * 60)
    result = graph.invoke({"startup_name": "a small local bakery in Brooklyn"})
    print_state(result)
    
    print("\n\n🧪 TEST 4: Empty name (validation gate)")
    print("-" * 60)
    result = graph.invoke({"startup_name": ""})
    print_state(result)


if __name__ == "__main__":
    main()