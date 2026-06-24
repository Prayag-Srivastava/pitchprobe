"""
Part 5.3 of PitchProbe — Supervisor Pattern with Conditional Edges
====================================================================
PURPOSE:
  Add the FIRST real intelligence to the graph: an LLM-powered
  Supervisor that decides which specialist to dispatch.

WHAT'S NEW vs Part 5.2:
  1. Real LLM call inside a node (the Supervisor)
  2. with_structured_output() with Literal[] to force a safe choice
  3. add_conditional_edges() — the routing primitive
  4. Multiple destination nodes (3 specialists, but only 1 runs per invocation)
  5. A validation gate (also using conditional edges) that can skip the pipeline

GRAPH STRUCTURE:
                                     ┌→ market_specialist ─┐
  START → validate → [valid?] → YES → supervisor ──────────┼→ synthesizer → END
                       │                                   │
                       │             └→ risk_specialist ───┤
                       │                                   │
                       │             └→ team_specialist ───┘
                       │
                       └─ NO → END

NOTE on routing:
  In this version, the Supervisor picks ONE specialist out of three.
  In Part 5.4+, we'll upgrade to picking MULTIPLE specialists that run
  in PARALLEL. Start simple, escalate complexity later.
"""

# ============================================================
# IMPORTS
# ============================================================

from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# STEP 1: DEFINE THE EXPANDED STATE
# ============================================================
# Notice how state has grown. Each section of the graph contributes
# its own fields. This is the "raw data in state" pattern from the
# context doc — we store the structured decision, not formatted text.

class PitchProbeState(TypedDict):
    """State that flows through the supervisor-routing graph."""
    
    # ─── Input ────────────────────────────────────────────
    startup_name: str
    
    # ─── Validation phase (from Part 5.2) ─────────────────
    is_valid: bool
    validation_message: str
    
    # ─── Supervisor's decision ────────────────────────────
    # The supervisor's structured output gets unpacked into these fields
    next_specialist: str        # one of: "market", "risk", "team"
    supervisor_reasoning: str   # why the supervisor picked this specialist
    
    # ─── Specialist output ────────────────────────────────
    # Whichever specialist runs writes here
    specialist_report: str
    specialist_name: str
    
    # ─── Final synthesis ──────────────────────────────────
    final_summary: str


# ============================================================
# STEP 2: SUPERVISOR'S DECISION SCHEMA
# ============================================================
# Why use Pydantic with Literal[...] here?
#
# Without Literal, the LLM might return "marketing" or "tech" or
# anything it imagines. We need it to pick EXACTLY one of three
# valid options — otherwise our conditional edge won't know where
# to route.
#
# Literal["market", "risk", "team"] makes this a HARD CONSTRAINT
# enforced by structured output parsing. The LLM cannot return
# anything else without the parse failing.

class SupervisorDecision(BaseModel):
    """The structured output the Supervisor MUST produce."""
    
    next_specialist: Literal["market", "risk", "team"] = Field(
        description="Which specialist should analyze this startup first. "
                    "Pick 'market' for market opportunity analysis, "
                    "'risk' for risk/compliance concerns, "
                    "or 'team' for founder/team analysis."
    )
    reasoning: str = Field(
        description="One sentence explaining why this specialist is the best fit "
                    "for analyzing this particular startup first."
    )


# ============================================================
# STEP 3: INITIALIZE THE LLM (shared by all LLM-using nodes)
# ============================================================
# Just the Supervisor uses an LLM in this part. Specialists are
# still toy/fake — real agents come in Part 5.4.

llm = init_chat_model(
    "llama-3.3-70b-versatile",
    model_provider="groq",
    temperature=0.2,           # low — we want consistent routing decisions
    max_tokens=512,            # supervisor's response is small
)

# The supervisor uses structured output. Same pattern as the doc:
# llm.with_structured_output(MySchema) gives back an LLM that
# returns parsed Pydantic objects instead of plain text.
supervisor_llm = llm.with_structured_output(SupervisorDecision)


# ============================================================
# STEP 4: DEFINE THE NODES
# ============================================================

# ─── Node 1: validate_startup (same as 5.2) ─────────────────
def validate_startup(state: PitchProbeState) -> dict:
    """Validates the startup name. Same as Part 5.2."""
    print("🔍 [Node] validate_startup")
    
    name = state["startup_name"]
    
    if not name or not name.strip():
        return {"is_valid": False, "validation_message": "Invalid: empty name"}
    elif len(name) > 100:
        return {"is_valid": False, "validation_message": "Invalid: name too long"}
    else:
        print(f"   ✓ '{name}' is valid")
        return {"is_valid": True, "validation_message": "Valid"}


# ─── Node 2: supervisor (LLM picks specialist) ──────────────
def supervisor(state: PitchProbeState) -> dict:
    """
    The Supervisor LLM looks at the startup name and decides which
    specialist should analyze it first.
    
    Reads:  state["startup_name"]
    Writes: state["next_specialist"], state["supervisor_reasoning"]
    
    KEY POINT: This is where the LLM lives in the graph. The output
    is structured (a Pydantic SupervisorDecision object), so we know
    exactly what shape it'll have.
    """
    print("🧠 [Node] supervisor (LLM is deciding...)")
    
    startup_name = state["startup_name"]
    
    # Build the prompt
    messages = [
        SystemMessage(content=(
            "You are a Supervisor in a startup due diligence system. "
            "Your job is to decide which specialist agent should analyze "
            "the startup FIRST. You have three specialists available:\n"
            "  - 'market': analyzes market size, trends, opportunity\n"
            "  - 'risk': analyzes regulatory, financial, and operational risks\n"
            "  - 'team': analyzes founders, team strength, hiring\n\n"
            "Pick the specialist most critical for THIS startup. "
            "Be decisive and explain your choice briefly."
        )),
        HumanMessage(content=f"Startup to analyze: {startup_name}"),
    ]
    
    # Invoke the structured-output LLM. The return value is a
    # SupervisorDecision instance — NOT a plain string.
    decision: SupervisorDecision = supervisor_llm.invoke(messages)
    
    print(f"   → Supervisor picked: {decision.next_specialist}")
    print(f"   → Reasoning: {decision.reasoning}")
    
    # Unpack the decision into state fields
    return {
        "next_specialist": decision.next_specialist,
        "supervisor_reasoning": decision.reasoning,
    }


# ─── Nodes 3,4,5: The three specialists (TOY — no real LLM yet) ───
# In Part 5.4 these become real create_agent() agents. For now,
# they're placeholders so we can see the routing mechanics clearly.

def market_specialist(state: PitchProbeState) -> dict:
    """TOY specialist — pretends to do market analysis."""
    print("📊 [Node] market_specialist running")
    return {
        "specialist_report": (
            f"[TOY MARKET REPORT for {state['startup_name']}] "
            f"Market size estimated, trends identified, opportunities mapped."
        ),
        "specialist_name": "market",
    }


def risk_specialist(state: PitchProbeState) -> dict:
    """TOY specialist — pretends to do risk analysis."""
    print("⚠️  [Node] risk_specialist running")
    return {
        "specialist_report": (
            f"[TOY RISK REPORT for {state['startup_name']}] "
            f"Regulatory risks assessed, financial exposure analyzed."
        ),
        "specialist_name": "risk",
    }


def team_specialist(state: PitchProbeState) -> dict:
    """TOY specialist — pretends to do team analysis."""
    print("👥 [Node] team_specialist running")
    return {
        "specialist_report": (
            f"[TOY TEAM REPORT for {state['startup_name']}] "
            f"Founders evaluated, team strength assessed."
        ),
        "specialist_name": "team",
    }


# ─── Node 6: synthesizer (combines and finalizes) ───────────
def synthesizer(state: PitchProbeState) -> dict:
    """TOY synthesizer — wraps the specialist's report in a summary."""
    print("📝 [Node] synthesizer")
    
    return {
        "final_summary": (
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"FINAL ANALYSIS for {state['startup_name']}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Supervisor chose: {state['specialist_name']} specialist\n"
            f"Reason: {state['supervisor_reasoning']}\n\n"
            f"Specialist findings:\n{state['specialist_report']}"
        ),
    }


# ============================================================
# STEP 5: ROUTING FUNCTIONS (the "brains" of conditional edges)
# ============================================================
# A routing function is a Python function that:
#   - Takes state
#   - Returns a STRING (which tells LangGraph the next node)
#
# It's just a function. No LLM involved. The LLM already made
# its decision in the supervisor node — these functions just READ
# the result from state and translate it to a destination.

def route_after_validation(state: PitchProbeState) -> str:
    """
    Validation gate: if invalid, skip everything. If valid, proceed.
    
    Return values must match the keys in the mapping dict
    we pass to add_conditional_edges().
    """
    if state["is_valid"]:
        return "valid"
    else:
        return "invalid"


def route_to_specialist(state: PitchProbeState) -> str:
    """
    Reads the Supervisor's decision and returns the specialist name.
    
    Return value MUST match one of the keys in the mapping dict
    we pass to add_conditional_edges() below.
    """
    return state["next_specialist"]  # "market", "risk", or "team"


# ============================================================
# STEP 6: BUILD THE GRAPH
# ============================================================

def build_graph():
    """Build and compile the supervisor-routing graph."""
    
    workflow = StateGraph(PitchProbeState)
    
    # ─── Add all nodes ────────────────────────────────────
    workflow.add_node("validate", validate_startup)
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("market_specialist", market_specialist)
    workflow.add_node("risk_specialist", risk_specialist)
    workflow.add_node("team_specialist", team_specialist)
    workflow.add_node("synthesizer", synthesizer)
    
    # ─── Edges ────────────────────────────────────────────
    
    # Entry: START → validate
    workflow.add_edge(START, "validate")
    
    # CONDITIONAL EDGE #1: Validation gate
    # After "validate", run route_after_validation(state).
    # Use its return value to look up the next node in the dict.
    workflow.add_conditional_edges(
        "validate",                  # source node
        route_after_validation,      # function that returns "valid" or "invalid"
        {
            "valid": "supervisor",   # if function returns "valid" → go to supervisor
            "invalid": END,          # if function returns "invalid" → go to END
        }
    )
    
    # CONDITIONAL EDGE #2: Supervisor's routing
    # After "supervisor", run route_to_specialist(state).
    # Use its return value to pick which specialist to run.
    workflow.add_conditional_edges(
        "supervisor",                # source node
        route_to_specialist,         # function that returns "market"/"risk"/"team"
        {
            "market": "market_specialist",
            "risk": "risk_specialist",
            "team": "team_specialist",
        }
    )
    
    # All three specialists → synthesizer (fan-in)
    # This is a NORMAL edge — no condition. Whichever specialist
    # ran will flow into the synthesizer.
    workflow.add_edge("market_specialist", "synthesizer")
    workflow.add_edge("risk_specialist", "synthesizer")
    workflow.add_edge("team_specialist", "synthesizer")
    
    # Exit: synthesizer → END
    workflow.add_edge("synthesizer", END)
    
    return workflow.compile()


# ============================================================
# STEP 7: PRETTY PRINTING
# ============================================================

def print_state(state: dict) -> None:
    """Display the final state."""
    print("\n" + "=" * 60)
    print("📊 FINAL STATE")
    print("=" * 60)
    print(f"  startup_name:          {state.get('startup_name')}")
    print(f"  is_valid:              {state.get('is_valid')}")
    print(f"  validation_message:    {state.get('validation_message')}")
    print(f"  next_specialist:       {state.get('next_specialist', '—')}")
    print(f"  supervisor_reasoning:  {state.get('supervisor_reasoning', '—')}")
    print(f"  specialist_name:       {state.get('specialist_name', '—')}")
    print(f"  specialist_report:     {state.get('specialist_report', '—')}")
    print()
    if state.get("final_summary"):
        print(state["final_summary"])
    print("=" * 60)


# ============================================================
# STEP 8: MAIN — RUN THE GRAPH
# ============================================================

def main():
    print("🚀 PitchProbe — Supervisor Pattern (Part 5.3)")
    print("=" * 60)
    
    graph = build_graph()
    
    # ─── Test 1: A market-focused startup ─────────────────
    print("\n\n🧪 TEST 1: Stripe (payments/fintech)")
    print("-" * 60)
    result = graph.invoke({"startup_name": "Stripe"})
    print_state(result)
    
    # ─── Test 2: A risk-heavy startup ─────────────────────
    print("\n\n🧪 TEST 2: Anthropic (AI safety, regulatory exposure)")
    print("-" * 60)
    result = graph.invoke({"startup_name": "Anthropic"})
    print_state(result)
    
    # ─── Test 3: Validation gate (empty name) ─────────────
    print("\n\n🧪 TEST 3: Empty name (should hit validation gate, skip pipeline)")
    print("-" * 60)
    result = graph.invoke({"startup_name": ""})
    print_state(result)


if __name__ == "__main__":
    main()