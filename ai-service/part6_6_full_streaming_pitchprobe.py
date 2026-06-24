"""
Part 6.6 of PitchProbe — Full Streaming Multi-Agent System
============================================================

The streaming-enabled version of Part 5.7. This is the closest console
approximation of the WebSocket-driven UI we'll build in Part 12.

WHAT'S NEW vs 5.7:
  1. Dual-LLM synthesizer (Option B):
     - structured_llm → produces FinalInvestmentReport (data layer)
     - narrative_llm  → produces streaming prose (UX layer)
  2. Supervisor LLM tagged with "nostream" to suppress internal noise
  3. Stream 3 modes simultaneously: ["updates", "messages", "custom"]
  4. Custom events from tools (via get_stream_writer in agent modules)
  5. Filter messages by tag — only narrative tokens reach the user
  6. Beautiful live console rendering

PROVIDER-LEVEL ACKNOWLEDGMENTS:
  - Groq + structured output ≠ streamable tokens (documented in v2.5)
  - Llama may still skip tools (documented in v2.4 Section 40)
  - The streaming architecture is correct regardless of those issues
"""

import os
from typing import TypedDict, Literal, Annotated, Optional
import operator

from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from dotenv import load_dotenv

# Specialist agents (same factories as 5.7)
from pitchprobe_agents.market_agent import (
    create_market_research_agent,
    MarketResearchReport,
)
from pitchprobe_agents.risk_agent import (
    create_risk_agent,
    RiskReport,
)
from pitchprobe_agents.team_agent import (
    create_team_agent,
    TeamReport,
)

load_dotenv()


# ============================================================
# SCHEMAS (unchanged from 5.7)
# ============================================================

class FinalInvestmentReport(BaseModel):
    """The synthesized investment recommendation for a startup."""
    startup_name: str = Field(description="The startup being evaluated")
    executive_summary: str = Field(description="2-3 sentence elevator pitch")
    investment_recommendation: Literal[
        "strong_invest", "invest_with_caution", "neutral",
        "weak_pass", "strong_pass"
    ] = Field(description="The investment recommendation")
    confidence_in_recommendation: Literal["high", "medium", "low"] = Field(
        description="Confidence in the recommendation"
    )
    key_strengths: list[str] = Field(description="3-5 cross-cutting strengths")
    key_concerns: list[str] = Field(description="3-5 cross-cutting concerns")
    red_flags: list[str] = Field(default_factory=list, description="Critical issues")
    cross_cutting_insights: list[str] = Field(description="Alignments and contradictions")
    due_diligence_next_steps: list[str] = Field(description="Concrete next investigations")
    overall_reasoning: str = Field(description="4-6 sentence narrative")


class SupervisorDecision(BaseModel):
    specialists_to_run: list[Literal["market", "risk", "team"]] = Field(
        description="Which specialists should analyze this startup in parallel."
    )
    reasoning: str = Field(description="2-3 sentences explaining the choice.")


# ============================================================
# STATE
# ============================================================

class PitchProbeState(TypedDict):
    startup_name: str
    is_valid: bool
    validation_message: str
    specialists_to_run: list[str]
    supervisor_reasoning: str
    specialist_reports: Annotated[list, operator.add]
    final_summary: str
    final_report: Optional[FinalInvestmentReport]
    narrative: str   # ⭐ NEW — the streamed prose


# ============================================================
# LLMs — note the tagging
# ============================================================

# Supervisor: tag with "nostream" so its tokens don't pollute the stream
supervisor_base = init_chat_model(
    "llama-3.3-70b-versatile",
    model_provider="groq",
    temperature=0.2,
    max_tokens=512,
).with_config({"tags": ["nostream"]})
supervisor_llm = supervisor_base.with_structured_output(SupervisorDecision)

# Synthesizer (structured): also tagged nostream — structured output doesn't stream anyway
synthesis_structured_base = init_chat_model(
    "llama-3.3-70b-versatile",
    model_provider="groq",
    temperature=0.4,
    max_tokens=2048,
).with_config({"tags": ["nostream"]})
synthesis_structured_llm = synthesis_structured_base.with_structured_output(FinalInvestmentReport)

# Synthesizer (narrative): tagged "synth-narrative" — this is what the user sees stream
synthesis_narrative_llm = init_chat_model(
    "llama-3.3-70b-versatile",
    model_provider="groq",
    temperature=0.5,
    max_tokens=600,
).with_config({"tags": ["synth-narrative"]})


# ============================================================
# AGENTS (initialized once at module load)
# ============================================================

print("⚙️  Initializing all real agents...")
print("   - Market Research Agent...")
market_agent = create_market_research_agent()
print("   - Risk Analysis Agent...")
risk_agent = create_risk_agent()
print("   - Team Analysis Agent...")
team_agent = create_team_agent()
print("✓ All agents ready.\n")


# ============================================================
# PROMPTS
# ============================================================

SYNTHESIS_STRUCTURED_PROMPT = """You are a Senior Partner at a top-tier VC firm. \
Three analysts have submitted specialist reports on a startup. Synthesize them \
into a structured investment recommendation.

YOUR JOB: REASON ACROSS the reports, don't just summarize them.

GOOD SYNTHESIS LOOKS FOR:
1. ALIGNMENTS — when multiple reports agree, that's evidence
2. CONTRADICTIONS — when they disagree, that's a red flag
3. CROSS-CUTTING INSIGHTS — patterns visible only when reports are combined
4. CONCRETE next steps — specific investigations needed

RECOMMENDATION CRITERIA:
- 'strong_invest' is RARE — only if all reports positive AND insights reinforce
- 'invest_with_caution' is for compelling startups with addressable reservations
- 'neutral' means mixed signals, need more info
- 'weak_pass' is for mild concerns, wouldn't lead
- 'strong_pass' is for serious problems

QUALITY RULES:
- Reference SPECIFIC findings from the reports. Generic statements are useless.
- If reports have LOW confidence, cap your recommendation confidence at MEDIUM.
- Surface contradictions explicitly.
- Be honest about red flags."""


NARRATIVE_PROMPT = """You are a Senior VC Partner writing a narrative briefing \
for the investment committee. Given the structured analysis below, write a \
flowing 3-paragraph prose summary that:

- Paragraph 1: What is this startup, the market opportunity, the headline recommendation
- Paragraph 2: Key strengths and concerns, with specific evidence from the analysis
- Paragraph 3: What due diligence is needed and the final verdict

Write in a confident, partner-level voice. Reference specific facts. Be concise \
but substantive (200-400 words total). Do NOT use bullet points or headers — \
pure prose."""


# ============================================================
# NODES
# ============================================================

def validate_startup(state: PitchProbeState) -> dict:
    name = state["startup_name"]
    if not name or not name.strip():
        return {"is_valid": False, "validation_message": "Invalid: empty name"}
    elif len(name) > 100:
        return {"is_valid": False, "validation_message": "Invalid: name too long"}
    else:
        return {"is_valid": True, "validation_message": "Valid"}


def supervisor(state: PitchProbeState) -> dict:
    messages = [
        SystemMessage(content=(
            "You are a Supervisor in a startup due diligence system. "
            "Decide which specialists analyze the startup IN PARALLEL. "
            "Available: 'market', 'risk', 'team'. Pick multiple when relevant."
        )),
        HumanMessage(content=f"Startup to analyze: {state['startup_name']}"),
    ]
    decision: SupervisorDecision = supervisor_llm.invoke(messages)
    return {
        "specialists_to_run": decision.specialists_to_run,
        "supervisor_reasoning": decision.reasoning,
    }


def market_specialist(state: PitchProbeState) -> dict:
    try:
        result = market_agent.invoke({
            "messages": [{"role": "user",
                          "content": f"Analyze the market for the startup '{state['startup_name']}'."}]
        })
        report = result.get("structured_response") or "[Market agent failed]"
    except Exception as e:
        report = f"[Market agent error: {e}]"
    return {"specialist_reports": [{"name": "market", "report": report}]}


def risk_specialist(state: PitchProbeState) -> dict:
    try:
        result = risk_agent.invoke({
            "messages": [{"role": "user",
                          "content": f"Analyze risks for the startup '{state['startup_name']}'."}]
        })
        report = result.get("structured_response") or "[Risk agent failed]"
    except Exception as e:
        report = f"[Risk agent error: {e}]"
    return {"specialist_reports": [{"name": "risk", "report": report}]}


def team_specialist(state: PitchProbeState) -> dict:
    try:
        result = team_agent.invoke({
            "messages": [{"role": "user",
                          "content": f"Analyze the team of the startup '{state['startup_name']}'."}]
        })
        report = result.get("structured_response") or "[Team agent failed]"
    except Exception as e:
        report = f"[Team agent error: {e}]"
    return {"specialist_reports": [{"name": "team", "report": report}]}


def synthesizer(state: PitchProbeState) -> dict:
    """
    Dual-LLM synthesizer:
    1. Structured call → FinalInvestmentReport (no streaming, data layer)
    2. Narrative call  → streamed prose (UX layer, tagged synth-narrative)
    """
    reports = state["specialist_reports"]
    context = format_reports_for_llm(state["startup_name"], reports)
    
    # Step 1: Structured (does NOT stream — tagged nostream)
    try:
        final_report: FinalInvestmentReport = synthesis_structured_llm.invoke([
            SystemMessage(content=SYNTHESIS_STRUCTURED_PROMPT),
            HumanMessage(content=context),
        ])
    except Exception as e:
        final_report = None
    
    # Step 2: Narrative (DOES stream — tagged synth-narrative)
    narrative_text = ""
    if final_report:
        narrative_input = (
            f"Structured analysis:\n"
            f"Startup: {final_report.startup_name}\n"
            f"Recommendation: {final_report.investment_recommendation}\n"
            f"Confidence: {final_report.confidence_in_recommendation}\n"
            f"Key strengths: {', '.join(final_report.key_strengths)}\n"
            f"Key concerns: {', '.join(final_report.key_concerns)}\n"
            f"Red flags: {', '.join(final_report.red_flags) or 'none'}\n"
            f"Cross-cutting insights: {', '.join(final_report.cross_cutting_insights)}\n"
            f"Due diligence next steps: {', '.join(final_report.due_diligence_next_steps)}\n"
            f"Reasoning: {final_report.overall_reasoning}\n"
        )
        try:
            narrative_msg = synthesis_narrative_llm.invoke([
                SystemMessage(content=NARRATIVE_PROMPT),
                HumanMessage(content=narrative_input),
            ])
            narrative_text = narrative_msg.content
        except Exception as e:
            narrative_text = f"[Narrative generation failed: {e}]"
    
    formatted = (
        format_final_report(final_report, reports)
        if final_report
        else f"[Synthesis failed for {state['startup_name']}]"
    )
    
    return {
        "final_report": final_report,
        "narrative": narrative_text,
        "final_summary": formatted,
    }


# ============================================================
# REPORT FORMATTING (unchanged from 5.7)
# ============================================================

def format_reports_for_llm(startup_name: str, reports: list[dict]) -> str:
    sections = [f"# STARTUP UNDER REVIEW: {startup_name}\n"]
    for r in reports:
        name = r["name"]
        payload = r["report"]
        sections.append(f"\n## {name.upper()} SPECIALIST REPORT\n")
        if isinstance(payload, MarketResearchReport):
            sections.append(_market_to_text(payload))
        elif isinstance(payload, RiskReport):
            sections.append(_risk_to_text(payload))
        elif isinstance(payload, TeamReport):
            sections.append(_team_to_text(payload))
        else:
            sections.append(str(payload))
    return "\n".join(sections)


def _market_to_text(r: MarketResearchReport) -> str:
    return (
        f"Industry: {r.industry}\n"
        f"Market Size: {r.market_size_usd or 'Unknown'}\n"
        f"Deck Available: {r.deck_available} | Confidence: {r.confidence}\n"
        f"Growth Trends: {', '.join(r.growth_trends)}\n"
        f"Target Customers: {', '.join(r.target_customers)}\n"
        f"Main Competitors: {', '.join(r.main_competitors)}\n"
        f"Market Opportunities: {', '.join(r.market_opportunities)}\n"
        f"Market Risks: {', '.join(r.market_risks)}\n"
        f"Sources: {', '.join(r.sources)}"
    )


def _risk_to_text(r: RiskReport) -> str:
    mitigations = ', '.join(r.mitigations) if r.mitigations else 'None mentioned'
    return (
        f"Overall Severity: {r.overall_severity}\n"
        f"Deck Available: {r.deck_available} | Confidence: {r.confidence}\n"
        f"Regulatory Risks: {', '.join(r.regulatory_risks)}\n"
        f"Financial Risks: {', '.join(r.financial_risks)}\n"
        f"Operational Risks: {', '.join(r.operational_risks)}\n"
        f"Market Risks: {', '.join(r.market_risks)}\n"
        f"Mitigations: {mitigations}\n"
        f"Sources: {', '.join(r.sources)}"
    )


def _team_to_text(r: TeamReport) -> str:
    founders_text = "\n".join(
        f"  - {f.name} ({f.role}): {f.background}" for f in r.founders
    )
    execs_text = "\n".join(
        f"  - {e.name} ({e.role}): {e.background}" for e in r.key_executives
    ) if r.key_executives else "  None mentioned"
    gaps = ', '.join(r.notable_gaps) if r.notable_gaps else 'None identified'
    advisors = ', '.join(r.advisors) if r.advisors else 'None mentioned'
    return (
        f"Team Score: {r.overall_team_score}\n"
        f"Team Size: {r.team_size_estimate or 'Unknown'}\n"
        f"Deck Available: {r.deck_available} | Confidence: {r.confidence}\n"
        f"Founders:\n{founders_text}\n"
        f"Key Executives:\n{execs_text}\n"
        f"Notable Strengths: {', '.join(r.notable_strengths)}\n"
        f"Notable Gaps: {gaps}\n"
        f"Advisors: {advisors}\n"
        f"Sources: {', '.join(r.sources)}"
    )


def format_final_report(final, raw_reports: list[dict]) -> str:
    if final is None:
        return "[Final report unavailable]"
    rec_emoji = {
        "strong_invest":       "🟢🟢🟢",
        "invest_with_caution": "🟢⚠️ ",
        "neutral":             "🟡   ",
        "weak_pass":           "🔴   ",
        "strong_pass":         "🔴🔴🔴",
    }.get(final.investment_recommendation, "⚪   ")
    
    lines = [
        "",
        "╔══════════════════════════════════════════════════════════════════════╗",
        f"║   FINAL INVESTMENT REPORT: {final.startup_name:<41}    ║",
        "╚══════════════════════════════════════════════════════════════════════╝",
        "",
        f"{rec_emoji} RECOMMENDATION:  {final.investment_recommendation.upper().replace('_', ' ')}",
        f"     Confidence:      {final.confidence_in_recommendation.upper()}",
        "",
        "── EXECUTIVE SUMMARY ────────────────────────────────────────────────",
        final.executive_summary,
        "",
        "── KEY STRENGTHS ─────────────────────────────────────────────────────",
    ]
    lines += [f"  ✓ {s}" for s in final.key_strengths]
    lines += ["", "── KEY CONCERNS ──────────────────────────────────────────────────────"]
    lines += [f"  ✗ {c}" for c in final.key_concerns]
    if final.red_flags:
        lines += ["", "── 🚩 RED FLAGS ──────────────────────────────────────────────────────"]
        lines += [f"  ! {r}" for r in final.red_flags]
    lines += ["", "── CROSS-CUTTING INSIGHTS ────────────────────────────────────────────"]
    lines += [f"  → {i}" for i in final.cross_cutting_insights]
    lines += ["", "── DUE DILIGENCE NEXT STEPS ──────────────────────────────────────────"]
    lines += [f"  • {s}" for s in final.due_diligence_next_steps]
    lines += ["", "── OVERALL REASONING ─────────────────────────────────────────────────"]
    lines += [final.overall_reasoning]
    lines += ["", "═" * 72]
    lines += ["", f"(Based on {len(raw_reports)} specialist reports)"]
    return "\n".join(lines)


# ============================================================
# ROUTING
# ============================================================

def route_after_validation(state: PitchProbeState) -> str:
    return "valid" if state["is_valid"] else "invalid"


def route_to_specialists(state: PitchProbeState) -> list[Send]:
    return [Send(f"{name}_specialist", state) for name in state["specialists_to_run"]]


# ============================================================
# GRAPH
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
        "validate", route_after_validation,
        {"valid": "supervisor", "invalid": END}
    )
    workflow.add_conditional_edges(
        "supervisor", route_to_specialists,
        ["market_specialist", "risk_specialist", "team_specialist"]
    )
    workflow.add_edge("market_specialist", "synthesizer")
    workflow.add_edge("risk_specialist", "synthesizer")
    workflow.add_edge("team_specialist", "synthesizer")
    workflow.add_edge("synthesizer", END)
    return workflow.compile()


# ============================================================
# STREAMING RENDERERS
# ============================================================

NODE_ICONS = {
    "validate":          "🔍",
    "supervisor":        "🧠",
    "market_specialist": "📊",
    "risk_specialist":   "⚠️ ",
    "team_specialist":   "👥",
    "synthesizer":       "📝",
}

AGENT_ICONS = {"market": "📊", "risk": "⚠️ ", "team": "👥"}


def render_custom_event(event: dict) -> None:
    """Pretty-print a custom event from the tool layer."""
    event_type = event.get("event", "?")
    agent = event.get("agent", "?")
    icon = AGENT_ICONS.get(agent, "🔧")
    
    if event_type == "discovery_started":
        print(f"  {icon} [{agent}] 📚 Checking available pitch decks...")
    elif event_type == "discovery_complete":
        decks = event.get("decks_found", [])
        if decks:
            print(f"     ↳ Found decks: {', '.join(decks)}")
        else:
            print(f"     ↳ No decks loaded")
    elif event_type == "rag_search_started":
        print(f"  {icon} [{agent}] 🔍 Deck search: '{event.get('query', '?')}' "
              f"in '{event.get('deck', '?')}'")
    elif event_type == "rag_search_complete":
        chunks = event.get("chunks_found", 0)
        score = event.get("top_score")
        score_str = f" (top score: {score:.2f})" if score is not None else ""
        print(f"     ↳ Got {chunks} chunks{score_str}")
    elif event_type == "web_search_started":
        print(f"  {icon} [{agent}] 🌐 Web search: '{event.get('query', '?')}'")
    elif event_type == "web_search_complete":
        print(f"     ↳ Got {event.get('results_count', '?')} results")
    else:
        print(f"  {icon} [{agent}] {event_type}: {event}")


def render_node_complete(node_name: str, state_update: dict) -> None:
    """Pretty-print a node-finished event."""
    icon = NODE_ICONS.get(node_name, "•")
    print(f"\n{icon} ✓ Node finished: {node_name}")
    
    if node_name == "validate":
        if state_update.get("is_valid"):
            print(f"     → Startup name is valid")
        else:
            print(f"     → {state_update.get('validation_message')}")
    elif node_name == "supervisor":
        specs = state_update.get("specialists_to_run", [])
        print(f"     → Specialists chosen: {', '.join(specs)}")
    elif node_name in ("market_specialist", "risk_specialist", "team_specialist"):
        reports = state_update.get("specialist_reports", [])
        if reports:
            entry = reports[0]
            r = entry.get("report")
            confidence = getattr(r, "confidence", None) or getattr(r, "confidence_in_recommendation", "?")
            print(f"     → {entry['name']} report delivered (confidence: {confidence})")


# ============================================================
# MAIN STREAMING RUNNER
# ============================================================

def run_streaming(graph, startup_name: str) -> dict:
    """Run the graph with full streaming. Returns the final state."""
    print("\n" + "═" * 72)
    print(f"🚀 LIVE ANALYSIS: {startup_name}")
    print("═" * 72)
    
    final_state = {}
    in_narrative = False
    narrative_tokens_seen = 0
    
    for chunk in graph.stream(
        {"startup_name": startup_name},
        stream_mode=["updates", "messages", "custom"],
        version="v2",
    ):
        ctype = chunk["type"]
        
        if ctype == "custom":
            render_custom_event(chunk["data"])
        
        elif ctype == "updates":
            for node_name, state_update in chunk["data"].items():
                render_node_complete(node_name, state_update)
                # Accumulate into final_state for return
                final_state.update(state_update)
        
        elif ctype == "messages":
            msg, metadata = chunk["data"]
            tags = metadata.get("tags") or []
            # Only stream the narrative tokens — suppress everything else
            if "synth-narrative" in tags and hasattr(msg, "content") and msg.content:
                if not in_narrative:
                    print("\n\n📖 LIVE NARRATIVE BRIEFING:")
                    print("─" * 72)
                    in_narrative = True
                print(msg.content, end="", flush=True)
                narrative_tokens_seen += 1
    
    print()  # newline after narrative
    if in_narrative:
        print("─" * 72)
        print(f"(narrative streamed in {narrative_tokens_seen} chunks)")
    
    return final_state


# ============================================================
# MAIN
# ============================================================

def main():
    print("🚀 PitchProbe — Part 6.6: Full Streaming Multi-Agent System")
    print("=" * 72)
    
    graph = build_graph()
    
    print("\n\n🧪 TEST 1: Airbnb (deck loaded)")
    final = run_streaming(graph, "Airbnb")
    if final.get("final_summary"):
        print(final["final_summary"])
    
    print("\n\n🧪 TEST 2: Stripe (no deck)")
    final = run_streaming(graph, "Stripe")
    if final.get("final_summary"):
        print(final["final_summary"])
    
    print("\n\n🧪 TEST 3: Empty name (validation gate)")
    final = run_streaming(graph, "")
    if not final.get("is_valid"):
        print(f"\n[Validation gate stopped execution: {final.get('validation_message')}]")


if __name__ == "__main__":
    main()