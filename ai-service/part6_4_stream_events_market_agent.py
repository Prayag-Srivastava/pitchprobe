"""
Part 6.4 of PitchProbe - Stream Events on the Market Research Agent
====================================================================

Goal: Use `stream_events(version="v3")` to expose the agent's internal loop
turn-by-turn — every LLM call, every tool call construction, every tool
execution, and the final structured output.

This is a STANDALONE run of the Market Agent (no LangGraph wrapper).
We're learning the stream_events API itself before wiring it into the
full multi-agent graph in 6.6.

Architecture mental model:
  agent.stream_events(input, version="v3") returns a STREAM OBJECT
  which has multiple typed projections:
    - stream.messages       → one ChatModelStream per LLM call
    - stream.tool_calls     → tool execution lifecycle
    - stream.values         → state snapshots
    - stream.output         → final state (terminal value)

  Each `message` from stream.messages has its own sub-streams:
    - message.text          → text deltas
    - message.tool_calls    → tool call args being constructed
    - message.output        → final completed message

We use `stream.interleave(...)` to consume MULTIPLE projections at once
in sync code (the alternative — asyncio.gather — is for async).

Known Groq limitation (discovered in 6.2):
  - message.text is mostly empty for structured-output calls
  - But message.output, stream.tool_calls, and stream.output all work fine
  - So we'll see the TOOL EXECUTION LIFECYCLE clearly even if text streaming
    is sparse
"""

from pyexpat.errors import messages
import sys
from dotenv import load_dotenv

# Import the EXACT same Market Agent factory we built in Part 4 Advanced
# and refactored into a module in Part 5.5 pre-step.
# The agent already has name="market_research_agent" set — that's what
# makes it identifiable in event streams (and in future stream.subagents).
from pitchprobe_agents.market_agent import (
    create_market_research_agent,
    MarketResearchReport,
)

load_dotenv()


# ============================================================
# Helper: Pretty-print the final MarketResearchReport
# ============================================================

def print_final_report(report: MarketResearchReport) -> None:
    """Pretty-print the final structured report.
    Field names match the actual MarketResearchReport schema."""
    print("\n" + "=" * 70)
    print(f"📊 FINAL MARKET RESEARCH REPORT")
    print("=" * 70)
    print(f"Startup:            {report.startup_name}")
    print(f"Industry:           {report.industry}")
    print(f"Market Size:        {report.market_size_usd}")
    print(f"Deck Available:     {report.deck_available}")
    print(f"Confidence:         {report.confidence.upper()}")
    print(f"\nGrowth Trends:")
    for trend in report.growth_trends:
        print(f"  • {trend}")
    print(f"\nTarget Customers:")
    for customer in report.target_customers:
        print(f"  • {customer}")
    print(f"\nCompetitors:")
    for comp in report.main_competitors:
        print(f"  • {comp}")
    print(f"\nMarket Opportunities:")
    for opp in report.market_opportunities:
        print(f"  • {opp}")
    print(f"\nMarket Risks:")
    for risk in report.market_risks:
        print(f"  • {risk}")
    print(f"\nSources:")
    for src in report.sources:
        print(f"  • {src}")
    print("=" * 70)


# ============================================================
# The streaming run function
# ============================================================

def run_with_stream_events(startup_name: str) -> None:
    """
    Run the Market Agent with stream_events and print every event type.

    We consume THREE projections concurrently using stream.interleave():
      1. "messages"    → LLM outputs (one per LLM turn)
      2. "tool_calls"  → tool execution lifecycle (start, output, end)
      3. "values"      → state snapshots after each step

    stream.interleave() is the sync equivalent of asyncio.gather — it
    yields (projection_name, item) tuples in the order events arrive
    across all named projections.
    """
    print("\n" + "🟢 " * 30)
    print(f"📡 STREAMING RUN — startup: {startup_name}")
    print("🟢 " * 30)

    # ---------------------------------------------------------
    # Step 1: Build the agent (loads RAG retriever, embeddings, model)
    # ---------------------------------------------------------
    # Heavy operation — done ONCE per script run, not per stream event.
    print("\n🔧 Initializing Market Research Agent...")
    agent = create_market_research_agent()
    print("✅ Agent ready.\n")

    # ---------------------------------------------------------
    # Step 2: Build the input (same shape as agent.invoke())
    # ---------------------------------------------------------
    user_input = {
    "messages": [
        {
            "role": "user",
            "content": f"Analyze the market for the startup '{startup_name}'.",
        }
     ]
    }

    # ---------------------------------------------------------
    # Step 3: Open the event stream
    # ---------------------------------------------------------
    # version="v3" is REQUIRED for stream_events — v1/v2 are older formats
    # that return raw events without typed projections.
    print("📡 Opening stream_events (version='v3')...\n")
    stream = agent.stream_events(user_input, version="v3")

    # Counters so we can summarize at the end
    llm_call_count = 0
    tool_call_count = 0
    state_snapshot_count = 0

    # Track which message we're currently in (for header changes)
    current_message_idx = -1

    # ---------------------------------------------------------
    # Step 4: Iterate the interleaved stream
    # ---------------------------------------------------------
    # stream.interleave("messages", "tool_calls", "values") yields tuples:
    #   (name, item) where name is one of "messages", "tool_calls", "values"
    #
    # We dispatch on `name` and handle each event type appropriately.
    # ---------------------------------------------------------
    print("─" * 70)
    print("🎬 STREAM BEGINS")
    print("─" * 70)

    try:
        for name, item in stream.interleave("messages", "tool_calls", "values"):

            # -----------------------------------------------
            # CASE 1: "messages" — one LLM call's output stream
            # -----------------------------------------------
            # `item` is a ChatModelStream. It has:
            #   - item.node           → which graph node produced it (e.g. "agent")
            #   - item.text           → iterator of text deltas
            #   - item.tool_calls     → iterator of tool call arg chunks
            #   - item.output         → final completed AIMessage (terminal value)
            #
            # IMPORTANT: We need to drain item.text and item.tool_calls
            # before we can access item.output. They're live iterators.
            # -----------------------------------------------
            if name == "messages":
                llm_call_count += 1
                current_message_idx = llm_call_count

                print(f"\n┌─ 🧠 LLM CALL #{llm_call_count} from node: '{item.node}'")
                print(f"│")

                # Drain text deltas (likely sparse on Groq + structured output)
                text_buffer = ""
                print(f"│  💬 Text deltas: ", end="", flush=True)
                for delta in item.text:
                    # delta is a string fragment
                    if delta:
                        print(delta, end="", flush=True)
                        text_buffer += delta
                if not text_buffer:
                    print("(none — Groq structured output)", end="")
                print()

                # Drain tool_call chunks (the LLM constructing tool calls)
                # On Groq these often arrive as a single chunk, not character-by-character.
                tool_call_chunks_seen = 0
                for chunk in item.tool_calls:
                    tool_call_chunks_seen += 1
                    # chunk is a ToolCallChunk dict: {"name": ..., "args": ..., "id": ..., "index": ...}
                    print(f"│  🔧 Tool call chunk: name={chunk.get('name')!r} args={chunk.get('args')!r}")

                # Get the finalized tool calls (after streaming completes)
                finalized = item.tool_calls.get()
                if finalized:
                    print(f"│  ✅ Finalized {len(finalized)} tool call(s) on this turn:")
                    for tc in finalized:
                        # tc is a ToolCall dict: {"name": ..., "args": {...}, "id": ...}
                        print(f"│      → {tc['name']}({tc['args']})")

                # Get the FINAL completed message (always available after drain)
                final_msg = item.output
                if final_msg:
                    usage = getattr(final_msg, "usage_metadata", None)
                    if usage:
                        print(f"│  📊 Tokens: input={usage.get('input_tokens')} "
                              f"output={usage.get('output_tokens')} "
                              f"total={usage.get('total_tokens')}")

                print(f"└─ END LLM CALL #{llm_call_count}")

            # -----------------------------------------------
            # CASE 2: "tool_calls" — tool EXECUTION lifecycle
            # -----------------------------------------------
            # `item` is a ToolCallStream. It has:
            #   - item.tool_name      → e.g. "search_pitch_deck"
            #   - item.input          → the args the tool was called with
            #   - item.output_deltas  → iterator of output chunks as they arrive
            #   - item.output         → final tool output (terminal value)
            #   - item.error          → error if tool failed
            #
            # This fires AFTER the LLM finalized the tool call and the tool
            # actually starts running. This is the most reliable projection
            # on Groq because tool execution is server-side, not provider-dependent.
            # -----------------------------------------------
            elif name == "tool_calls":
                tool_call_count += 1
                print(f"\n  ▶️  TOOL EXECUTION #{tool_call_count}: {item.tool_name}")
                print(f"     Input: {item.input}")

                # Drain output deltas (most tools return one chunk)
                output_buffer = ""
                for delta in item.output_deltas:
                    if delta:
                        output_buffer += str(delta)

                # Get final output (always available after drain)
                if item.output is not None:
                    output_str = str(item.output)
                    # Truncate for readability
                    if len(output_str) > 200:
                        output_str = output_str[:200] + "... (truncated)"
                    print(f"     ✅ Output: {output_str}")

                if item.error:
                    print(f"     ❌ Error: {item.error}")

            # -----------------------------------------------
            # CASE 3: "values" — state snapshot after each step
            # -----------------------------------------------
            # `item` is a dict containing the agent's current state.
            # For an agent, state primarily contains the messages list.
            # We just count these — printing each would be too verbose.
            # -----------------------------------------------
            elif name == "values":
                state_snapshot_count += 1
                # Uncomment to inspect snapshots:
                # print(f"     📸 State snapshot #{state_snapshot_count}: keys={list(item.keys())}")

    except Exception as e:
        print(f"\n❌ Stream error: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "─" * 70)
    print("🏁 STREAM ENDS")
    print("─" * 70)

    # ---------------------------------------------------------
    # Step 5: Get the final agent output
    # ---------------------------------------------------------
    # stream.output is the FINAL agent state — available only AFTER the
    # stream has been fully consumed (which we just did above).
    # For an agent with response_format=MarketResearchReport, the final
    # state dict contains "structured_response" with our Pydantic object.
    # ---------------------------------------------------------
    final_state = stream.output

        # ---------------------------------------------------------
    # Tool-skip detection (Llama-on-Groq known issue)
    # ---------------------------------------------------------
    if tool_call_count == 0:
        print("\n" + "⚠️ " * 30)
        print("⚠️  WARNING: AGENT SKIPPED ALL TOOLS")
        print("⚠️  The agent produced a report WITHOUT calling any real tools.")
        print("⚠️  Any 'sources' or 'facts' in the output are likely HALLUCINATED.")
        print("⚠️  This is a known Llama-on-Groq behavior — addressed in Parts 9 & 10.")
        print("⚠️ " * 30)

    print(f"\n📈 STREAM SUMMARY:")
    print(f"   LLM calls:        {llm_call_count}")
    print(f"   Tool executions:  {tool_call_count}")
    print(f"   State snapshots:  {state_snapshot_count}")

    # Extract the structured report
    if final_state and isinstance(final_state, dict):
        structured = final_state.get("structured_response")
        if isinstance(structured, MarketResearchReport):
            print_final_report(structured)
        else:
            print(f"\n⚠️  No MarketResearchReport in final state.")
            print(f"   Final state keys: {list(final_state.keys())}")
            # This happens when Llama skips structured output (known issue)
            # Show the last message instead
            messages = final_state.get("messages", [])
            if messages:
                last = messages[-1]
                print(f"\n   Last message content: {getattr(last, 'content', last)[:500]}")
    else:
        print(f"\n⚠️  Unexpected final state type: {type(final_state)}")
        print(f"   Value: {final_state}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("🚀 PitchProbe — Part 6.4: Stream Events on Market Agent")
    print("=" * 70)
    print("This script runs the Market Agent with stream_events(version='v3')")
    print("and exposes every internal event — LLM calls, tool calls, tool")
    print("executions, and the final structured report.")
    print("=" * 70)

    # Test with Airbnb (deck loaded) — should produce rich tool activity
    run_with_stream_events("Airbnb")

    print("\n\n" + "🔵 " * 30)
    print("🧪 DIAGNOSTIC: Same agent, same prompt, but .invoke() instead of .stream_events()")
    print("🔵 " * 30)

    from pitchprobe_agents.market_agent import create_market_research_agent
    test_agent = create_market_research_agent()

    test_result = test_agent.invoke({
    "messages": [
        {"role": "user", "content": "Analyze the market for the startup 'Airbnb'."}
    ]
    })

# Count tool calls in the message history
    messages = test_result.get("messages", [])
    tool_call_count = 0
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_call_count += 1
                print(f"  🔧 {tc['name']}({tc.get('args', {})})")

    print(f"\n📈 .invoke() summary:")
    print(f"   Total messages: {len(messages)}")
    print(f"   Tool calls in history: {tool_call_count}")

    # Uncomment to also test with a startup that has NO deck:
    # run_with_stream_events("Stripe")


if __name__ == "__main__":
    main()