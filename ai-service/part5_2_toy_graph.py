"""
Part 5.2 of PitchProbe — Toy 2-Node LangGraph
==============================================
PURPOSE:
  This is a PRACTICE build, not a PitchProbe feature.
  Goal: internalize the core LangGraph mechanics before building
  the real multi-agent system in Part 5.3+.

WHAT YOU'LL LEARN:
  1. How to define STATE using TypedDict
  2. How to write NODES as plain Python functions
  3. How to wire nodes together with EDGES
  4. How to COMPILE the graph
  5. How to INVOKE the graph and inspect results

WHY NO LLMs HERE:
  By using plain Python in the nodes, we isolate the LangGraph
  mechanics. If something breaks, it's a LangGraph mistake — not
  an LLM mistake. Once you trust the mechanics, we add real LLMs
  in Part 5.3.

THE GRAPH WE'RE BUILDING:
                          
   START → validate_startup → enrich_startup → END

  Two nodes, linear flow. As simple as it gets.
"""

# ============================================================
# IMPORTS
# ============================================================

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
import uuid
from datetime import datetime


# ============================================================
# STEP 1: DEFINE THE STATE
# ============================================================
# State = the shared notebook that ALL nodes can read from and
# write to. It's just a TypedDict (a typed Python dictionary).
#
# Notice:
#   - `startup_name` is the INPUT (provided by user)
#   - `is_valid`, `validation_message` are filled by validate_startup
#   - `analysis_id`, `timestamp`, `status` are filled by enrich_startup
#
# RULE FROM CONTEXT DOC: "Store RAW DATA in state. Format prompts
# INSIDE nodes." That rule doesn't apply heavily here (no LLMs),
# but the structure of having a typed dict with clean fields is
# what we'll carry forward to the real PitchProbe state.

class PitchProbeState(TypedDict):
    """The shared state that flows through every node."""
    
    # INPUT field — provided when we call graph.invoke()
    startup_name: str
    
    # Fields populated by validate_startup node
    is_valid: bool
    validation_message: str
    
    # Fields populated by enrich_startup node
    analysis_id: str
    timestamp: str
    status: str


# ============================================================
# STEP 2: DEFINE THE NODES
# ============================================================
# A node is just a Python function with this signature:
#
#    def node_name(state: StateType) -> dict:
#        # read from state
#        # do work
#        # return DICT of fields to UPDATE in state
#
# CRITICAL: A node returns ONLY the fields it changed, not the
# whole state. LangGraph merges this dict into the global state.
#
# This is called "partial state update" — it's the LangGraph
# convention. Don't return the whole state; just what changed.


def validate_startup(state: PitchProbeState) -> dict:
    """
    Node 1: Validates the startup name.
    
    Reads:  state["startup_name"]
    Writes: state["is_valid"], state["validation_message"]
    """
    print("🔍 [Node 1] validate_startup running...")
    
    # Read from state
    name = state["startup_name"]
    
    # Apply validation rules
    if not name or not name.strip():
        is_valid = False
        message = "Invalid: startup name is empty"
    elif len(name) > 100:
        is_valid = False
        message = f"Invalid: name too long ({len(name)} chars, max 100)"
    else:
        is_valid = True
        message = "Valid"
    
    print(f"   → Result: {message}")
    
    # Return ONLY the fields we want to update
    # LangGraph will merge this into the global state
    return {
        "is_valid": is_valid,
        "validation_message": message,
    }


def enrich_startup(state: PitchProbeState) -> dict:
    """
    Node 2: Enriches the analysis with metadata.
    
    Reads:  (nothing from state, just generates fresh data)
    Writes: state["analysis_id"], state["timestamp"], state["status"]
    
    NOTE: This node runs even if validation failed. In Part 5.3
    we'll add CONDITIONAL EDGES so we can skip enrichment if the
    startup is invalid. For now, keep it simple — linear flow.
    """
    print("✨ [Node 2] enrich_startup running...")
    
    # Generate fresh metadata
    analysis_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    status = "ready_for_analysis"
    
    print(f"   → Generated analysis_id: {analysis_id[:8]}...")
    
    # Return only the new fields
    return {
        "analysis_id": analysis_id,
        "timestamp": timestamp,
        "status": status,
    }


# ============================================================
# STEP 3: BUILD THE GRAPH
# ============================================================
# This is where we WIRE everything together.
#
# The pattern is ALWAYS:
#   1. Create a StateGraph with your state class
#   2. Add nodes (register functions)
#   3. Add edges (connect nodes)
#   4. Compile (validates and freezes the graph)
#
# After compile(), the graph is immutable and ready to run.


def build_graph():
    """Build and compile the toy graph."""
    
    # 1. Create the graph with our state schema
    workflow = StateGraph(PitchProbeState)
    
    # 2. Add nodes
    # Format: workflow.add_node("name_in_graph", function_reference)
    # The "name_in_graph" is how edges refer to this node.
    workflow.add_node("validate_startup", validate_startup)
    workflow.add_node("enrich_startup", enrich_startup)
    
    # 3. Add edges
    # START and END are special built-in nodes provided by LangGraph.
    # START = the entry point. END = the exit point.
    workflow.add_edge(START, "validate_startup")          # Entry: START → validate
    workflow.add_edge("validate_startup", "enrich_startup")  # validate → enrich
    workflow.add_edge("enrich_startup", END)              # Exit: enrich → END
    
    # 4. Compile the graph
    # This validates the structure (e.g., no orphan nodes, edges
    # reference real nodes) and prepares it for execution.
    graph = workflow.compile()
    
    return graph


# ============================================================
# STEP 4: PRETTY-PRINT HELPER
# ============================================================
# Just a utility to display the final state nicely.

def print_state(state: dict) -> None:
    """Print the final state in a readable format."""
    print("\n" + "=" * 60)
    print("📊 FINAL STATE")
    print("=" * 60)
    print(f"  startup_name:        {state.get('startup_name', 'N/A')}")
    print(f"  is_valid:            {state.get('is_valid', 'N/A')}")
    print(f"  validation_message:  {state.get('validation_message', 'N/A')}")
    print(f"  analysis_id:         {state.get('analysis_id', 'N/A')}")
    print(f"  timestamp:           {state.get('timestamp', 'N/A')}")
    print(f"  status:              {state.get('status', 'N/A')}")
    print("=" * 60)


# ============================================================
# STEP 5: MAIN — RUN THE GRAPH
# ============================================================

def main():
    print("🚀 PitchProbe — Toy 2-Node Graph (Part 5.2)")
    print("=" * 60)
    
    # Build the graph once (it's reusable)
    graph = build_graph()
    
    # ─── Test Case 1: Valid startup ───────────────────────
    print("\n\n🧪 TEST 1: Valid startup name")
    print("-" * 60)
    
    # We only need to provide the INPUT field.
    # LangGraph will create the state and let nodes fill in the rest.
    initial_state = {"startup_name": "Airbnb"}
    
    # Invoke the graph. This runs:
    #   START → validate_startup → enrich_startup → END
    # And returns the FINAL state after all nodes have run.
    final_state = graph.invoke(initial_state)
    
    print_state(final_state)
    
    # ─── Test Case 2: Empty startup name ──────────────────
    print("\n\n🧪 TEST 2: Empty startup name (should still run, but flagged invalid)")
    print("-" * 60)
    
    final_state = graph.invoke({"startup_name": ""})
    print_state(final_state)
    
    # ─── Test Case 3: Too long ────────────────────────────
    print("\n\n🧪 TEST 3: Name too long")
    print("-" * 60)
    
    long_name = "A" * 150  # 150 characters
    final_state = graph.invoke({"startup_name": long_name})
    print_state(final_state)


if __name__ == "__main__":
    main()