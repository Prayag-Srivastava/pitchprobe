"""
Part 7.4 — Chat Over Saved Analysis
====================================
Interactive multi-turn chat about a saved PitchProbe analysis.

Demonstrates:
- Cross-thread checkpoint access (chat graph reads from analysis graph's state)
- Multi-turn conversation memory via add_messages reducer
- Session persistence (resume chats across process restarts)
- Two persistence use cases coexisting in one SQLite DB
- Streaming responses with node-metadata filtering

Prerequisites:
- Run `part7_3_write_analysis.py` first to populate the DB with Airbnb + Stripe analyses.
- The chat graph writes chat threads to the SAME pitchprobe_checkpoints.db file.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from typing import Annotated, Optional, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# Import the analysis graph so we can query its saved state
# (build_graph accepts optional checkpointer — Part 6.6 modification)
from part6_6_full_streaming_pitchprobe import build_graph as build_analysis_graph

# Import FinalInvestmentReport so pickle can reconstruct it from the DB
# (see Rule #51 — pickle needs the class module imported to deserialize)
# The import happens implicitly via build_analysis_graph, but being explicit is safer.

load_dotenv()

# ============================================================
# Configuration
# ============================================================

DB_PATH = "pitchprobe_checkpoints.db"

# The chat LLM — plain text, no structured output, streams cleanly on Groq
CHAT_LLM = init_chat_model(
    "llama-3.3-70b-versatile",
    model_provider="groq",
    temperature=0.4,
    max_tokens=1024,
)

# ============================================================
# State Schema
# ============================================================

class ChatState(TypedDict):
    """
    State for the chat-over-analysis graph.

    - messages: full conversation history (auto-appended by add_messages reducer)
    - analysis_thread_id: which saved analysis this chat is discussing
    - analysis_context: formatted analysis text (loaded once at startup, cached here)
    """
    messages: Annotated[list, add_messages]
    analysis_thread_id: str
    analysis_context: str

# ============================================================
# Analysis Loading (cross-thread access)
# ============================================================

def format_analysis_for_chat(final_report) -> str:
    """
    Format a FinalInvestmentReport as compact text for the LLM's system prompt.

    We include everything the LLM might need to answer follow-up questions:
    executive summary, recommendation, strengths, concerns, red flags,
    cross-cutting insights, and reasoning.
    """
    lines = [
        f"# Investment Analysis: {final_report.startup_name}",
        "",
        f"## Recommendation: {final_report.investment_recommendation.upper()}",
        f"**Confidence:** {final_report.confidence_in_recommendation}",
        "",
        "## Executive Summary",
        final_report.executive_summary,
        "",
        "## Key Strengths",
    ]
    for i, s in enumerate(final_report.key_strengths, 1):
        lines.append(f"{i}. {s}")

    lines.extend(["", "## Key Concerns"])
    for i, c in enumerate(final_report.key_concerns, 1):
        lines.append(f"{i}. {c}")

    if final_report.red_flags:
        lines.extend(["", "## 🚩 Red Flags"])
        for i, rf in enumerate(final_report.red_flags, 1):
            lines.append(f"{i}. {rf}")

    if final_report.cross_cutting_insights:
        lines.extend(["", "## Cross-Cutting Insights"])
        for i, ci in enumerate(final_report.cross_cutting_insights, 1):
            lines.append(f"{i}. {ci}")

    if final_report.due_diligence_next_steps:
        lines.extend(["", "## Due Diligence Next Steps"])
        for i, dd in enumerate(final_report.due_diligence_next_steps, 1):
            lines.append(f"{i}. {dd}")

    lines.extend(["", "## Overall Reasoning", final_report.overall_reasoning])

    return "\n".join(lines)


def load_analysis_context(analysis_graph, analysis_thread_id: str) -> Optional[str]:
    """
    Cross-thread read: query the analysis graph's saved state for a specific thread.

    Uses get_state() — sub-millisecond, no LLM calls, no graph execution.
    """
    config = {"configurable": {"thread_id": analysis_thread_id}}
    snapshot = analysis_graph.get_state(config)

    if not snapshot.values:
        return None

    final_report = snapshot.values.get("final_report")
    if final_report is None:
        return None

    return format_analysis_for_chat(final_report)

# ============================================================
# Chat Node
# ============================================================

CHAT_SYSTEM_PROMPT_TEMPLATE = """You are a senior investment analyst assistant helping a VC \
review a startup analysis. You have access to a completed multi-agent analysis report below.

Your job:
- Answer questions clearly and concisely, grounded in the report
- Quote specific facts from the report when relevant
- If the user asks something the report doesn't cover, say so explicitly — do NOT make up facts
- If asked for your opinion, reason from the report's evidence
- Be conversational but professional

---

# ANALYSIS REPORT

{analysis_context}

---

Answer the user's questions based on the report above. Do not use outside knowledge unless \
explicitly asked to speculate — and when you do, flag it as speculation.
"""


def chat_node(state: ChatState) -> dict:
    """
    The single node in the chat graph.

    - Builds a system prompt containing the analysis context
    - Prepends it to the accumulated conversation history
    - Calls the LLM, returns the AI response (which add_messages appends to state)
    """
    system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(
        analysis_context=state["analysis_context"]
    )

    # Build the LLM input: SystemMessage (with analysis) + full conversation
    messages_for_llm = [SystemMessage(content=system_prompt)] + state["messages"]

    # Tag this call so we can filter it in the stream (Rule #46 pattern)
    response = CHAT_LLM.with_config({"tags": ["chat-response"]}).invoke(messages_for_llm)

    # add_messages reducer appends this to state["messages"]
    return {"messages": [response]}

# ============================================================
# Build Chat Graph
# ============================================================

def build_chat_graph(checkpointer):
    """Build a simple START → chat_node → END graph with checkpointing."""
    builder = StateGraph(ChatState)
    builder.add_node("chat_node", chat_node)
    builder.add_edge(START, "chat_node")
    builder.add_edge("chat_node", END)
    return builder.compile(checkpointer=checkpointer)

# ============================================================
# CLI: Discovery + Interactive Loop
# ============================================================

def discover_analyses(analysis_graph) -> list[tuple[str, str]]:
    """
    List all analysis threads in the DB that have a final_report.

    Returns list of (thread_id, startup_name) tuples.
    """
    # Collect unique thread_ids from raw checkpoint list
    seen_threads = set()
    for checkpoint_tuple in analysis_graph.checkpointer.list(None):
        thread_id = checkpoint_tuple.config["configurable"]["thread_id"]
        seen_threads.add(thread_id)

    # For each thread, use get_state() to get the clean StateSnapshot
    analyses = []
    for thread_id in seen_threads:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = analysis_graph.get_state(config)

        if not snapshot.values:
            continue

        final_report = snapshot.values.get("final_report")
        if final_report is None:
            continue

        analyses.append((thread_id, final_report.startup_name))

    # Sort for stable ordering
    analyses.sort(key=lambda x: x[1])
    return analyses


def discover_chat_sessions(chat_graph, analysis_thread_id: str) -> list[str]:
    """
    List existing chat sessions that were about a specific analysis.
    """
    seen_threads = set()
    for checkpoint_tuple in chat_graph.checkpointer.list(None):
        thread_id = checkpoint_tuple.config["configurable"]["thread_id"]
        if thread_id.startswith("chat-"):
            seen_threads.add(thread_id)

    sessions = []
    for thread_id in seen_threads:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = chat_graph.get_state(config)

        if not snapshot.values:
            continue

        if snapshot.values.get("analysis_thread_id") != analysis_thread_id:
            continue

        sessions.append(thread_id)

    sessions.sort()
    return sessions


def prompt_choice(prompt: str, options: list[str]) -> int:
    """Simple numeric picker. Returns 0-based index."""
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        raw = input(f"{prompt} ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print(f"⚠️  Please enter a number between 1 and {len(options)}.")


def stream_chat_response(chat_graph, user_input: str, config: dict, analysis_context: str, analysis_thread_id: str):
    """
    Stream the chat node's response token-by-token, filtering by tag.

    Uses stream_mode=["messages"] and filters for our "chat-response" tag.
    """
    input_state = {
        "messages": [HumanMessage(content=user_input)],
        "analysis_context": analysis_context,
        "analysis_thread_id": analysis_thread_id,
    }

    print("\n🤖 ", end="", flush=True)
    token_count = 0

    for chunk_type, chunk_data in chat_graph.stream(
        input_state,
        config=config,
        stream_mode=["messages", "updates"],
    ):
        if chunk_type == "messages":
            token, metadata = chunk_data
            # Only stream tokens tagged as "chat-response"
            if "chat-response" in metadata.get("tags", []):
                if hasattr(token, "text") and token.text:
                    print(token.text, end="", flush=True)
                    token_count += 1
        # We ignore "updates" here — the streamed tokens ARE the update

    print()  # newline after streaming completes
    return token_count


def show_history(chat_graph, config: dict) -> None:
    """Print the full accumulated conversation for the current chat session."""
    snapshot = chat_graph.get_state(config)
    if not snapshot.values or not snapshot.values.get("messages"):
        print("📭 No conversation history yet.")
        return

    messages = snapshot.values["messages"]
    print(f"\n📜 Conversation History ({len(messages)} messages):")
    print("=" * 60)
    for i, msg in enumerate(messages, 1):
        role = "👤 You" if isinstance(msg, HumanMessage) else "🤖 Assistant"
        content = msg.content[:200] + ("..." if len(msg.content) > 200 else "")
        print(f"\n[{i}] {role}:")
        print(f"    {content}")
    print("\n" + "=" * 60)

# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 60)
    print("🚀 PitchProbe — Chat Over Saved Analysis (Part 7.4)")
    print("=" * 60)

    # Check DB exists
    if not os.path.exists(DB_PATH):
        print(f"\n❌ Database not found: {DB_PATH}")
        print("   Run `part7_3_write_analysis.py` first to populate it.")
        sys.exit(1)

    # Open the shared SQLite DB for BOTH graphs
    # check_same_thread=False allows use across the CLI's blocking input calls
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    serde = JsonPlusSerializer(pickle_fallback=True)
    checkpointer = SqliteSaver(conn=conn, serde=serde)

    # Build the analysis graph (read-only for us — we just query its state)
    print("\n🔌 Connecting to analysis graph (this initializes agents once)...")
    analysis_graph = build_analysis_graph(checkpointer=checkpointer)

    # Build the chat graph (this is what we'll actually invoke)
    chat_graph = build_chat_graph(checkpointer=checkpointer)

    # Discover saved analyses
    analyses = discover_analyses(analysis_graph)
    if not analyses:
        print("\n❌ No completed analyses found in the DB.")
        print("   Run `part7_3_write_analysis.py` first.")
        sys.exit(1)

    print(f"\n✅ Found {len(analyses)} saved analysis/analyses:")
    labels = [f"{name}  (thread: {tid})" for tid, name in analyses]
    idx = prompt_choice("\nWhich analysis do you want to chat about?", labels)
    analysis_thread_id, startup_name = analyses[idx]

    # Load the analysis context (cross-thread checkpoint read)
    print(f"\n📖 Loading analysis for '{startup_name}'...")
    import time
    t0 = time.time()
    analysis_context = load_analysis_context(analysis_graph, analysis_thread_id)
    elapsed = time.time() - t0
    print(f"✅ Loaded in {elapsed:.3f}s (cross-thread checkpoint read)")

    if analysis_context is None:
        print("❌ Failed to load analysis context.")
        sys.exit(1)

    # Choose or create a chat session
    existing_sessions = discover_chat_sessions(chat_graph, analysis_thread_id)
    print(f"\n💬 Chat session options for '{startup_name}':")

    session_options = ["✨ Start a new chat session"] + [
        f"↩️  Resume: {sid}" for sid in existing_sessions
    ]
    session_idx = prompt_choice("Choose:", session_options)

    if session_idx == 0:
        # New session — use timestamp-based ID for uniqueness
        session_num = len(existing_sessions) + 1
        chat_thread_id = f"chat-{analysis_thread_id}-session{session_num}"
        print(f"\n🆕 New chat session: {chat_thread_id}")
    else:
        chat_thread_id = existing_sessions[session_idx - 1]
        print(f"\n↩️  Resuming: {chat_thread_id}")

    chat_config = {"configurable": {"thread_id": chat_thread_id}}

    # If resuming, show a hint about history
    if session_idx > 0:
        snapshot = chat_graph.get_state(chat_config)
        if snapshot.values and snapshot.values.get("messages"):
            n = len(snapshot.values["messages"])
            print(f"   💾 {n} messages from previous session(s) loaded.")

    # Interactive loop
    print("\n" + "=" * 60)
    print(f"💬 Chatting about: {startup_name}")
    print("=" * 60)
    print("Commands: /history  /switch  /quit  (or Ctrl+C)")
    print()

    turn = 0
    while True:
        try:
            user_input = input("👤 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye! Your chat is saved.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"/quit", "/exit", "quit", "exit"}:
            print("\n👋 Goodbye! Your chat is saved.")
            break

        if user_input.lower() == "/history":
            show_history(chat_graph, chat_config)
            continue

        if user_input.lower() == "/switch":
            print("\n🔄 Restart the script to switch analyses.")
            print("   Your current session is saved and can be resumed.")
            break

        turn += 1
        try:
            token_count = stream_chat_response(
                chat_graph,
                user_input,
                chat_config,
                analysis_context,
                analysis_thread_id,
            )
            if token_count == 0:
                print("⚠️  No response tokens streamed (possible Groq structured-output issue).")
        except Exception as e:
            print(f"\n❌ Error: {e}")

    # Show final stats
    final_snapshot = chat_graph.get_state(chat_config)
    if final_snapshot.values:
        n_messages = len(final_snapshot.values.get("messages", []))
        print(f"\n📊 Session summary: {n_messages} total messages saved to '{chat_thread_id}'")


if __name__ == "__main__":
    main()