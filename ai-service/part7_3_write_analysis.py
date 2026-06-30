"""
Part 7.3 of PitchProbe - Write Analysis to Disk (SqliteSaver)
================================================================

This script runs a startup analysis and persists ALL checkpoints to a
SQLite .db file on disk. After this script exits, the state is STILL
available — it lives in the file, not in Python's memory.

WHAT'S NEW vs 7.2:
  - SqliteSaver instead of InMemorySaver
  - JsonPlusSerializer(pickle_fallback=True) silences deprecation warnings
    for our custom Pydantic types (MarketResearchReport, etc.)
  - State survives Python process exit — proven by part7_3_read_analysis.py

WORKFLOW:
  1. Run this script:           python part7_3_write_analysis.py
  2. Wait for analysis to complete (~100s)
  3. Close terminal (or just wait)
  4. Run reader script:          python part7_3_read_analysis.py
  5. See the analysis state read FROM THE DISK FILE — no re-running

THE DATABASE FILE:
  Located at: pitchprobe_checkpoints.db (in the same folder as this script)
  You can inspect it with any SQLite browser — it's just a file.
"""

import sqlite3
import time
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from dotenv import load_dotenv

from part6_6_full_streaming_pitchprobe import build_graph

load_dotenv()


# ============================================================
# Database file path
# ============================================================
# This is THE file where state will live. After this script runs,
# this file contains your full analysis. Survives reboots.

DB_FILE = "pitchprobe_checkpoints.db"


def create_persistent_checkpointer():
    """
    Build a SqliteSaver that:
      - Writes to DB_FILE on disk
      - Uses pickle_fallback=True to handle custom Pydantic types
        (silences the "Deserializing unregistered type" warnings)
    
    The check_same_thread=False is needed because LangGraph may access
    the connection from multiple threads when streaming.
    """
    # Open a SQLite connection. check_same_thread=False allows the
    # connection to be shared across threads (LangGraph needs this).
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    
    # Use pickle fallback to handle our custom Pydantic schemas
    # (MarketResearchReport, RiskReport, TeamReport, FinalInvestmentReport).
    # Without this, you get warnings on every checkpoint involving them.
    serde = JsonPlusSerializer(pickle_fallback=True)
    
    return SqliteSaver(conn, serde=serde)


def run_analysis(startup_name: str, thread_id: str):
    """Run a fresh analysis and persist all state to the SQLite file."""
    print(f"\n{'═' * 70}")
    print(f"🚀 RUNNING ANALYSIS")
    print(f"   Startup:   {startup_name}")
    print(f"   Thread ID: {thread_id}")
    print(f"   DB file:   {DB_FILE}")
    print(f"{'═' * 70}")
    
    checkpointer = create_persistent_checkpointer()
    graph = build_graph(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": thread_id}}
    
    start = time.time()
    
    # Stream just node updates (cleaner output than full streaming)
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
    
    # Inspect what was saved
    snapshot = graph.get_state(config)
    if snapshot.values.get("final_report"):
        final = snapshot.values["final_report"]
        print(f"\n   ✅ Saved to {DB_FILE}")
        print(f"      Recommendation: {final.investment_recommendation.upper()}")
        print(f"      Confidence:     {final.confidence_in_recommendation.upper()}")
        print(f"      # reports:      {len(snapshot.values.get('specialist_reports', []))}")


def main():
    print("🚀 PitchProbe — Part 7.3 (Writer): Persist Analysis to Disk")
    print("=" * 70)
    print("This script runs an analysis and SAVES IT TO A FILE.")
    print("Then part7_3_read_analysis.py can read it WITHOUT re-running.")
    print("=" * 70)
    
    # Run two analyses on two different threads.
    # Both get persisted to the same .db file.
    
    run_analysis("Airbnb", thread_id="airbnb-persistent")
    run_analysis("Stripe", thread_id="stripe-persistent")
    
    print(f"\n{'=' * 70}")
    print(f"✅ DONE — All state saved to: {DB_FILE}")
    print(f"{'=' * 70}")
    print("""
NEXT STEP:
   1. Close this terminal (optional — proves persistence is real)
   2. Open a new terminal in the same folder
   3. Run: python part7_3_read_analysis.py
   4. You'll see the saved analyses — no re-running, no LLM calls
""")


if __name__ == "__main__":
    main()