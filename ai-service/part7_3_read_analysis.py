"""
Part 7.3 of PitchProbe - Read Analysis from Disk (No Re-Running)
==================================================================

This script PROVES persistence works across process restarts.

It opens the SAME .db file that part7_3_write_analysis.py wrote to,
and displays the saved analyses. ZERO LLM calls. ZERO agent runs.
Pure state retrieval from disk.

This is what makes a real product possible:
  - User runs analysis on Monday
  - Closes browser, closes laptop
  - Comes back Wednesday
  - Their analysis is still there

In Part 12 (frontend), this script's logic becomes a REST endpoint:
  GET /analyses           → list all saved analyses
  GET /analyses/{thread_id}  → fetch one analysis's full state
"""

import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# Note: We DO NOT load .env or initialize agents here.
# We're just reading state — no LLM calls needed.
# This makes the script FAST (sub-second) and FREE (no API costs).

DB_FILE = "pitchprobe_checkpoints.db"


def open_existing_checkpointer():
    """
    Open the SAME SqliteSaver that the writer used.
    Must use the same serde (pickle_fallback=True) so it can read
    the data the writer wrote.
    """
    if not os.path.exists(DB_FILE):
        raise FileNotFoundError(
            f"\n❌ {DB_FILE} not found!\n"
            f"   Run 'python part7_3_write_analysis.py' first to create it."
        )
    
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    serde = JsonPlusSerializer(pickle_fallback=True)
    return SqliteSaver(conn, serde=serde)


def list_all_threads(checkpointer) -> list[str]:
    """Find every unique thread_id in the database."""
    threads = set()
    for cp in checkpointer.list(None):
        threads.add(cp.config["configurable"]["thread_id"])
    return sorted(threads)


def get_latest_state_for_thread(checkpointer, thread_id: str):
    """
    Get the LATEST checkpoint for a thread.
    We don't use graph.get_state() here because we don't want to
    build a graph at all — we just want to read raw checkpoint data.
    """
    config = {"configurable": {"thread_id": thread_id}}
    return checkpointer.get_tuple(config)


def print_analysis(thread_id: str, checkpoint_tuple) -> None:
    """Pretty-print a saved analysis from the checkpoint data."""
    if checkpoint_tuple is None:
        print(f"\n   ⚠️  No data found for thread '{thread_id}'")
        return
    
    # The actual state values live in checkpoint['channel_values']
    values = checkpoint_tuple.checkpoint.get("channel_values", {})
    
    print(f"\n{'═' * 70}")
    print(f"📂 ANALYSIS LOADED FROM DISK: thread_id = '{thread_id}'")
    print(f"{'═' * 70}")
    
    print(f"   Startup:        {values.get('startup_name', '—')}")
    print(f"   Validated:      {values.get('is_valid', '—')}")
    print(f"   Specialists:    {', '.join(values.get('specialists_to_run', [])) or '—'}")
    print(f"   # reports:      {len(values.get('specialist_reports', []))}")
    
    final = values.get("final_report")
    if final:
        print(f"\n   🎯 Recommendation:  {final.investment_recommendation.upper().replace('_', ' ')}")
        print(f"   🎯 Confidence:      {final.confidence_in_recommendation.upper()}")
        
        print(f"\n   📋 Executive Summary:")
        print(f"      {final.executive_summary}")
        
        if final.key_strengths:
            print(f"\n   ✓ Key Strengths:")
            for s in final.key_strengths:
                print(f"      • {s}")
        
        if final.key_concerns:
            print(f"\n   ✗ Key Concerns:")
            for c in final.key_concerns:
                print(f"      • {c}")
        
        if final.red_flags:
            print(f"\n   🚩 Red Flags:")
            for r in final.red_flags:
                print(f"      • {r}")
    
    # Metadata about WHEN this was created
    metadata = checkpoint_tuple.metadata
    if metadata:
        print(f"\n   📊 Metadata:")
        print(f"      Source: {metadata.get('source', '—')}")
        print(f"      Step:   {metadata.get('step', '—')}")


def count_checkpoints_for_thread(checkpointer, thread_id: str) -> int:
    """Count how many checkpoints (super-steps) exist for a thread."""
    config = {"configurable": {"thread_id": thread_id}}
    return sum(1 for _ in checkpointer.list(config))


def main():
    print("📂 PitchProbe — Part 7.3 (Reader): Read Saved Analyses from Disk")
    print("=" * 70)
    print(f"Opening database: {DB_FILE}")
    print("(No LLM calls. No agent initialization. Pure disk read.)")
    print("=" * 70)
    
    import time
    start = time.time()
    
    # Open the saved database
    checkpointer = open_existing_checkpointer()
    
    # Discover what threads (analyses) are saved
    threads = list_all_threads(checkpointer)
    
    if not threads:
        print(f"\n⚠️  Database is empty.")
        print(f"   Run 'python part7_3_write_analysis.py' to create analyses.")
        return
    
    print(f"\n📊 Found {len(threads)} saved analyses:")
    for tid in threads:
        n_checkpoints = count_checkpoints_for_thread(checkpointer, tid)
        print(f"   • {tid}  ({n_checkpoints} checkpoints)")
    
    # Print full details for each
    for thread_id in threads:
        cp_tuple = get_latest_state_for_thread(checkpointer, thread_id)
        print_analysis(thread_id, cp_tuple)
    
    elapsed = time.time() - start
    
    print(f"\n{'=' * 70}")
    print(f"⏱️  Total time: {elapsed:.3f}s")
    print(f"{'=' * 70}")
    print(f"""
🎯 WHAT YOU JUST PROVED:

   ✅ State persisted across Python process exit/restart
      (Writer ran in one process, Reader ran in a different process)
   
   ✅ Reading saved state is essentially FREE
      ({elapsed:.3f}s for {len(threads)} analyses vs ~100s per fresh run)
   
   ✅ Zero LLM calls, zero agent initialization
      (We didn't even import the agent modules — just read from disk)
   
   ✅ Foundation for production:
      - "My Saved Analyses" dashboard (Part 12 frontend)
      - GET /analyses REST endpoint (Part 12 backend)
      - Multi-user state isolation (Part 14 PostgreSQL)

🔜 NEXT (7.4): Short-term memory page (LangChain)
   We'll use this saved state to build CHAT-OVER-ANALYSIS:
   User asks "tell me more about Airbnb's regulatory risks?"
   → graph loads saved Airbnb state from disk
   → answers from the saved context, not by re-running everything
""")


if __name__ == "__main__":
    main()