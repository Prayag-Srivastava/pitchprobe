"""
Part 6.2 of PitchProbe — Streaming Supervisor Graph
======================================================
Practice build: convert part5_3_supervisor_graph.py to use .stream()
instead of .invoke().

GOAL: internalize the .stream() API on a small graph before scaling
to the full 5.7 multi-agent system in 6.6.

WHAT'S NEW vs 5.3:
  - run_with_streaming() function using graph.stream(stream_mode=[...])
  - Two stream modes: 'updates' (node-level) + 'messages' (LLM tokens)
  - v2 format (chunk["type"] + chunk["data"])
  - Side-by-side comparison: .invoke() vs .stream()

WHAT'S THE SAME:
  - The graph itself (imported from 5.3 — no duplication)
  - The supervisor's LLM call
  - All routing and state logic

WHAT TO WATCH:
  - "updates" events fire ONE AT A TIME as each node finishes
  - "messages" events stream the supervisor's LLM tokens
  - Specialist nodes are TOY (no LLM) → only updates, no messages
"""

# ⭐ Reuse the graph from 5.3 — production-style import
from part5_3_supervisor_graph import build_graph


# ============================================================
# STREAMING RUNNER
# ============================================================

def run_with_streaming(graph, startup_name: str) -> None:
    """
    Run the graph using .stream() and print events as they fire.
    
    Stream modes:
      - 'updates': fires after each node completes, payload = {node_name: {state_field: value}}
      - 'messages': fires per LLM token, payload = (token, metadata)
    
    Both use version='v2' which returns dicts with 'type' and 'data' keys.
    """
    print("\n" + "🌊 " * 30)
    print(f"STREAMING RUN: {startup_name}")
    print("🌊 " * 30)
    
    # ─── State tracking for nice display ──────────────────────
    current_llm_node = None   # tracks which node is currently streaming tokens
    
    # ─── The streaming loop ───────────────────────────────────
    for chunk in graph.stream(
        {"startup_name": startup_name},
        stream_mode=["updates", "messages"],
        version="v2",
    ):
        chunk_type = chunk["type"]
        chunk_data = chunk["data"]
        
        # ── Handle "updates" events: a node finished ──
        if chunk_type == "updates":
            # chunk_data is {node_name: {field_name: value, ...}}
            for node_name, node_update in chunk_data.items():
                
                # Print a separator if we were just streaming tokens
                if current_llm_node:
                    print()  # newline to end the token stream
                    current_llm_node = None
                
                # Show what each node did
                print(f"\n📦 [updates] NODE FINISHED: {node_name}")
                for field, value in node_update.items():
                    # Truncate long values for readability
                    value_preview = str(value)
                    if len(value_preview) > 120:
                        value_preview = value_preview[:120] + "..."
                    print(f"     {field}: {value_preview}")
        
        # ── Handle "messages" events: LLM token arrived ──
        elif chunk_type == "messages":
            # chunk_data is (token, metadata)
            token, metadata = chunk_data
            node = metadata.get("langgraph_node", "unknown")
            
            # New LLM node started streaming? Print a header.
            if node != current_llm_node:
                if current_llm_node:
                    print()  # end previous stream's line
                print(f"\n🧠 [messages] LLM STREAMING from node '{node}':")
                print("   ", end="", flush=True)
                current_llm_node = node
            
            # Print the token if it has text
            if token.text:
                print(token.text, end="", flush=True)
    
    # End the final token stream cleanly
    if current_llm_node:
        print()
    
    print("\n" + "🌊 " * 30)
    print("STREAMING RUN COMPLETE")
    print("🌊 " * 30)


# ============================================================
# COMPARISON: BATCH (.invoke) vs STREAMING (.stream)
# ============================================================

def run_with_invoke(graph, startup_name: str) -> None:
    """Run the same graph using .invoke() for comparison."""
    print("\n" + "📦 " * 30)
    print(f"BATCH RUN (using .invoke): {startup_name}")
    print("📦 " * 30)
    print("\n⏳ Running (you'll see SILENCE until it's done)...\n")
    
    result = graph.invoke({"startup_name": startup_name})
    
    print(f"\n✅ DONE — Final state keys: {list(result.keys())}")
    print(f"   specialists_to_run: {result.get('specialists_to_run')}")
    print(f"   final_summary exists: {'final_summary' in result}")
    print("📦 " * 30)


# ============================================================
# MAIN
# ============================================================

def main():
    print("🚀 PitchProbe — Streaming Supervisor (Part 6.2)")
    print("=" * 70)
    print("\nThis script runs the same graph TWICE:")
    print("  1. Using .invoke()  → silent wait, then dump (BATCH)")
    print("  2. Using .stream()  → live events as they happen (STREAMING)")
    print("\nWatch the difference in UX.")
    print("=" * 70)
    
    # Build the graph (imported from 5.3)
    graph = build_graph()
    
    # Same startup, both modes — compare directly
    test_startup = "Stripe"
    
    # ─── Run 1: Batch (.invoke) ────────────────────────────
    run_with_invoke(graph, test_startup)
    
    # ─── Run 2: Streaming (.stream) ────────────────────────
    run_with_streaming(graph, test_startup)
    
    print("\n\n" + "=" * 70)
    print("💡 KEY TAKEAWAYS:")
    print("=" * 70)
    print("• .invoke() blocks until everything finishes — silent then dump")
    print("• .stream() emits events live — 'updates' per node, 'messages' per token")
    print("• 'updates' events tell you WHAT changed in state")
    print("• 'messages' events stream LLM tokens with node metadata")
    print("• In PitchProbe 6.6, we'll use both modes + 'custom' for tool progress")
    print("=" * 70)


if __name__ == "__main__":
    main()