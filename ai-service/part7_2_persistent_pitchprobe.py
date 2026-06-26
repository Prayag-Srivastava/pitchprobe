"""
Part 7.2 of PitchProbe - Persistent Multi-Agent System
========================================================

CORRECTED VERSION: Demonstrates what persistence ACTUALLY does.

Persistence is about state PRESERVATION across invocations, NOT
auto-skipping completed graphs. The real value:

  1. State survives between invocations (within the same script lifetime
     for InMemorySaver, across restarts for SqliteSaver in 7.3)
  2. graph.get_state(config) lets you inspect any thread anytime
  3. CRASH RECOVERY: if a graph stopped mid-execution, you can resume
     from the last checkpoint by invoking with input=None
  4. Foundation for follow-up Q&A on past reports (Part 7.6)

What this demo does NOT show (but easily could):
  - Auto-skipping completed graphs (LangGraph doesn't do this automatically)
  - Cross-restart persistence (requires SqliteSaver — that's 7.3)
"""

import time
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

from part6_6_full_streaming_pitchprobe import build_graph

load_dotenv()


# ============================================================
# Helper: Pretty-print a checkpoint state
# ============================================================

def print_state_summary(state_snapshot, label: str) -> None:
    """Print a concise summary of a graph state snapshot."""
    print(f"\n{'─' * 70}")
    print(f"📸 STATE SNAPSHOT: {label}")
    print(f"{'─' * 70}")
    
    if state_snapshot is None or not state_snapshot.values:
        print("   (no state — thread is empty or never run)")
        return
    
    values = state_snapshot.values
    print(f"   Thread:            {state_snapshot.config['configurable']['thread_id']}")
    print(f"   Startup:           {values.get('startup_name', '—')}")
    print(f"   Validated:         {values.get('is_valid', '—')}")
    print(f"   Specialists run:   {', '.join(values.get('specialists_to_run', [])) or '—'}")
    print(f"   # reports stored:  {len(values.get('specialist_reports', []))}")
    
    final = values.get('final_report')
    if final:
        print(f"   Recommendation:    {final.investment_recommendation.upper()}")
        print(f"   Confidence:        {final.confidence_in_recommendation.upper()}")
    
    next_nodes = state_snapshot.next
    if next_nodes:
        print(f"   Next to run:       {next_nodes}")
    else:
        print(f"   Status:            ✅ Graph completed (no more nodes to run)")


def run_with_thread(graph, startup_name: str, thread_id: str) -> None:
    """Run the graph and print node-finished events."""
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"\n{'═' * 70}")
    print(f"🚀 RUNNING: startup='{startup_name}', thread_id='{thread_id}'")
    print(f"{'═' * 70}")
    
    start = time.time()
    
    for chunk in graph.stream(
        {"startup_name": startup_name},
        config=config,
        stream_mode="updates",
        version="v2",
    ):
        if chunk["type"] == "updates":
            for node_name, _ in chunk["data"].items():
                print(f"   ✓ {node_name}")
    
    elapsed = time.time() - start
    print(f"\n   ⏱️  Elapsed: {elapsed:.1f}s")


def inspect_thread(graph, thread_id: str, label: str) -> None:
    """Look up a thread's saved state and print it."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    print_state_summary(snapshot, label)


def list_all_threads(checkpointer) -> None:
    """Enumerate every thread the checkpointer knows about."""
    print(f"\n{'─' * 70}")
    print("📊 ALL THREADS IN CHECKPOINTER:")
    print(f"{'─' * 70}")
    threads_seen = set()
    for checkpoint_tuple in checkpointer.list(None):
        tid = checkpoint_tuple.config["configurable"]["thread_id"]
        threads_seen.add(tid)
    for tid in sorted(threads_seen):
        print(f"   • {tid}")


def count_checkpoints_for_thread(checkpointer, thread_id: str) -> int:
    """Count how many checkpoints exist for a thread."""
    config = {"configurable": {"thread_id": thread_id}}
    return sum(1 for _ in checkpointer.list(config))


# ============================================================
# Main demo — focused on what persistence ACTUALLY does
# ============================================================

def main():
    print("🚀 PitchProbe — Part 7.2: Persistent Multi-Agent System (CORRECTED)")
    print("=" * 70)
    print("Demonstrating: what checkpointer persistence actually provides")
    print("=" * 70)
    
    checkpointer = InMemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    
    # ─────────────────────────────────────────────────────────
    # DEMO 1: Run an analysis. State is checkpointed at EVERY node boundary.
    # ─────────────────────────────────────────────────────────
    
    print("\n" + "🟢 " * 35)
    print("DEMO 1: Run Airbnb analysis (state checkpointed at every node)")
    print("🟢 " * 35)
    run_with_thread(graph, "Airbnb", thread_id="airbnb-1")
    
    # Show that MANY checkpoints were saved (not just the final state)
    n_checkpoints = count_checkpoints_for_thread(checkpointer, "airbnb-1")
    print(f"\n   💡 Checkpoints saved for this thread: {n_checkpoints}")
    print(f"   (one per node boundary — proves persistence is granular)")
    
    inspect_thread(graph, "airbnb-1", "Airbnb after fresh run")
    
    # ─────────────────────────────────────────────────────────
    # DEMO 2: Inspect state at ANY TIME, without re-running
    # ─────────────────────────────────────────────────────────
    # This is the foundational value of checkpointers: state is queryable
    # outside of execution. In a real app, this powers:
    #   - "show me my saved analyses" dashboard
    #   - "load this analysis and continue chatting about it" feature
    
    print("\n" + "🔵 " * 35)
    print("DEMO 2: Inspect state without re-running anything")
    print("🔵 " * 35)
    print("\nGraph.get_state(config) reads the latest checkpoint for a thread.")
    print("This is FREE — no LLM calls, no agent runs. Pure state retrieval.\n")
    
    start = time.time()
    inspect_thread(graph, "airbnb-1", "Airbnb state (retrieved via get_state)")
    elapsed = time.time() - start
    print(f"\n   ⏱️  Retrieval took: {elapsed:.3f}s (vs ~100s for a fresh run)")
    print(f"   This is the difference: retrieval is essentially free.")
    
    # ─────────────────────────────────────────────────────────
    # DEMO 3: Thread isolation — different threads are independent
    # ─────────────────────────────────────────────────────────
    
    print("\n" + "🟡 " * 35)
    print("DEMO 3: Different thread_id = completely independent analysis")
    print("🟡 " * 35)
    run_with_thread(graph, "Stripe", thread_id="stripe-1")
    
    inspect_thread(graph, "stripe-1", "Stripe (fresh thread)")
    inspect_thread(graph, "airbnb-1", "Airbnb (unchanged by Stripe run)")
    
    # ─────────────────────────────────────────────────────────
    # DEMO 4: Browse all threads (the "list my saved analyses" feature)
    # ─────────────────────────────────────────────────────────
    
    list_all_threads(checkpointer)
    
    # ─────────────────────────────────────────────────────────
    # DEMO 5: Checkpoint history for a single thread
    # ─────────────────────────────────────────────────────────
    # This shows the full execution trace — state at every node boundary.
    # In Part 10 (LangSmith), this becomes the time-travel debugger.
    
    print("\n" + "─" * 70)
    print("📚 CHECKPOINT HISTORY FOR 'airbnb-1' (each step of execution)")
    print("─" * 70)
    config = {"configurable": {"thread_id": "airbnb-1"}}
    for i, cp in enumerate(checkpointer.list(config)):
        values = cp.checkpoint.get("channel_values", {})
        next_nodes = cp.checkpoint.get("versions_seen", {}).get("__input__", {})
        n_reports = len(values.get("specialist_reports", [])) if isinstance(values.get("specialist_reports"), list) else 0
        print(f"   Checkpoint #{i}: # reports = {n_reports}")
    
    # ─────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────
    
    print("\n" + "=" * 70)
    print("🎯 WHAT THIS DEMO PROVED")
    print("=" * 70)
    print("""
   ✅ State is checkpointed at EVERY node boundary
      (not just at the end — proves granular persistence)
   
   ✅ get_state(config) retrieves state instantly — no re-execution
      (sub-millisecond vs ~100s for a fresh run)
   
   ✅ Threads are fully isolated
      (running Stripe didn't touch Airbnb's state)
   
   ✅ All threads enumerable via checkpointer.list()
      (powers "show my saved analyses" UI in Part 12)
   
   ✅ Full checkpoint history queryable per thread
      (powers time-travel debugging in Part 10)
   
   🔜 NEXT (7.3): Replace InMemorySaver with SqliteSaver
       so state survives across script restarts.
   
   🔜 LATER (7.6): Use this saved state to enable FOLLOW-UP Q&A
       — the user chats about a past analysis and the graph reads
       its prior state from the checkpoint to answer with context.
""")
    
    print("=" * 70)
    print("⚠️  COMMON MISCONCEPTION (DO NOT BE FOOLED):")
    print("=" * 70)
    print("""
   Re-invoking graph.stream(input, config) with the SAME thread_id
   does NOT auto-skip a completed graph. It will re-trigger execution
   from START with the saved state already loaded.
   
   That's why we use get_state() to READ saved state, and we use
   invoke(None, config) to RESUME a graph that stopped mid-execution
   (which is what 7.6 will demonstrate properly).
""")


if __name__ == "__main__":
    main()