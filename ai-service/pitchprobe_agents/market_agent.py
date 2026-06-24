"""
Market Research Agent for PitchProbe
======================================
Production-hardened single agent that:
  1. Discovers which pitch decks are available
  2. Searches the relevant deck (if any) via document-aware RAG
  3. Searches the web via Tavily
  4. Returns a structured MarketResearchReport

This is a REUSABLE FACTORY. Import create_market_research_agent()
from anywhere — Part 4 demo, Part 5 multi-agent graph, Part 12 backend.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware, ModelRetryMiddleware
from langchain.tools import tool
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

from pitchprobe_agents.rag_retriever import (
    search_deck,
    get_available_decks,
)

load_dotenv()


# ============================================================
# OUTPUT SCHEMA
# ============================================================

class MarketResearchReport(BaseModel):
    """Structured market research output."""
    
    startup_name: str = Field(description="The startup being analyzed")
    industry: str = Field(description="The industry/sector")
    market_size_usd: Optional[str] = Field(
        default=None,
        description="Estimated total addressable market in USD (e.g. '$50B')"
    )
    growth_trends: list[str] = Field(
        description="Key market growth trends and drivers"
    )
    target_customers: list[str] = Field(
        description="Primary customer segments"
    )
    main_competitors: list[str] = Field(
        description="Top competitors in the space"
    )
    market_opportunities: list[str] = Field(
        description="Specific market opportunities for this startup"
    )
    market_risks: list[str] = Field(
        description="Market-specific risks (saturation, regulation, etc.)"
    )
    deck_available: bool = Field(
        description="Whether a pitch deck was available for this startup"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in this analysis. HIGH only if both deck + web sources used."
    )
    sources: list[str] = Field(
        description="URLs or deck names used as evidence"
    )


# ============================================================
# TOOLS  (streaming-aware via get_stream_writer with fallback)
# ============================================================

def _safe_emit(event: dict) -> None:
    """
    Emit a custom event to the LangGraph stream if we're in a graph context.
    Falls back to print() if called standalone (outside a graph).
    This lets the same tool code work in both modes.
    """
    try:
        from langgraph.config import get_stream_writer
        writer = get_stream_writer()
        writer(event)
    except Exception:
        # Not running inside a LangGraph context — fall back to print
        print(f"   [event:{event.get('event','?')}] {event}")


@tool
def list_available_pitch_decks() -> str:
    """
    Discovery tool: list all pitch decks currently loaded in the system.
    
    USE THIS FIRST before attempting to search a deck. Returns the
    exact deck names you must pass to search_pitch_deck.
    """
    _safe_emit({
        "event": "discovery_started",
        "agent": "market",
        "tool": "list_available_pitch_decks",
    })
    
    decks = get_available_decks()
    
    _safe_emit({
        "event": "discovery_complete",
        "agent": "market",
        "tool": "list_available_pitch_decks",
        "decks_found": decks,
    })
    
    if not decks:
        return "No pitch decks are currently loaded in the system."
    return f"Available pitch decks: {', '.join(decks)}"


@tool
def search_pitch_deck(query: str, deck_name: str) -> str:
    """
    Search a SPECIFIC pitch deck for information matching the query.
    
    Args:
        query: A TOPIC question (e.g., "market size", "target customers").
               Do NOT include the company name in the query.
        deck_name: The EXACT deck name from list_available_pitch_decks().
    
    Returns the matching chunks with page numbers, or a message if
    no relevant content was found in this deck.
    """
    _safe_emit({
        "event": "rag_search_started",
        "agent": "market",
        "tool": "search_pitch_deck",
        "query": query,
        "deck": deck_name,
    })
    
    results = search_deck(query=query, deck_name=deck_name, k=5)
    
    _safe_emit({
        "event": "rag_search_complete",
        "agent": "market",
        "tool": "search_pitch_deck",
        "query": query,
        "deck": deck_name,
        "chunks_found": len(results),
        "top_score": results[0][1] if results else None,
    })
    
    if not results:
        return (
            f"No content in deck '{deck_name}' is sufficiently relevant to "
            f"the query '{query}'. The deck does not contain this information."
        )
    
    output = []
    for doc, score in results:
        page = doc.metadata.get("page", "?")
        output.append(
            f"[Page {page + 1 if isinstance(page, int) else page}, "
            f"relevance score: {score:.2f}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(output)


_web_search = TavilySearch(max_results=5, topic="general")

@tool
def search_web(query: str) -> str:
    """
    Search the web for current information about a company, market, or industry.
    Use this for: market size, competitors, recent news, regulatory updates.
    """
    _safe_emit({
        "event": "web_search_started",
        "agent": "market",
        "tool": "search_web",
        "query": query,
    })
    
    results = _web_search.invoke({"query": query})
    
    # Count results defensively (Tavily returns a dict with "results" list)
    try:
        results_count = len(results.get("results", [])) if isinstance(results, dict) else 0
    except Exception:
        results_count = 0
    
    _safe_emit({
        "event": "web_search_complete",
        "agent": "market",
        "tool": "search_web",
        "query": query,
        "results_count": results_count,
    })
    
    return str(results)


# ============================================================
# AGENT FACTORY
# ============================================================

def _build_system_prompt(available_decks: list[str]) -> str:
    """Build the system prompt dynamically based on available decks."""
    
    if available_decks:
        # ⚠️ DO NOT list the deck names inline — that makes Llama think it
        # already has the deck content. Force it to discover them via tool.
        deck_section = (
            "Pitch decks may be loaded in the system. You do NOT know which ones.\n\n"
            "MANDATORY WORKFLOW:\n"
            "1. FIRST call list_available_pitch_decks() to discover loaded decks.\n"
            "   You MUST call this tool — do not assume what is loaded.\n"
            "2. If the target startup matches a loaded deck name:\n"
            "   - Call search_pitch_deck() at LEAST 3 times with different topic queries\n"
            "     (e.g., 'market size', 'target customers', 'competitors')\n"
            "   - Then call search_web() at LEAST 2 times for external validation\n"
            "   - Set deck_available=True\n"
            "3. If no matching deck found:\n"
            "   - Call search_web() at LEAST 3 times for market research\n"
            "   - Set deck_available=False and confidence ≤ 'medium'\n"
            "4. ONLY AFTER calling the above tools, produce the MarketResearchReport.\n"
        )
    else:
        deck_section = (
            "No pitch decks confirmed loaded. Call list_available_pitch_decks() "
            "first to verify, then use web search.\n"
        )
    
    return (
        "You are a Market Research Agent in a VC due diligence system. "
        "Your job is to RESEARCH a startup using tools, then produce a report.\n\n"
        "CRITICAL RULE: You MUST use tools to gather evidence. Never produce a "
        "report from memory alone — your training data is outdated and unreliable. "
        "Every fact in your report MUST come from a tool call.\n\n"
        f"{deck_section}\n"
        "QUERY GUIDELINES:\n"
        "- Use TOPIC-only queries for the deck (e.g., 'market size', not "
        "  'Airbnb market size'). The deck filter handles the company.\n"
        "- Use FULL queries for the web (e.g., 'Airbnb market size 2024').\n\n"
        "QUALITY RULES:\n"
        "- Ground every claim in either deck content or web search results.\n"
        "- Cite sources (deck name + page, or URL) — only REAL sources from tool outputs.\n"
        "- Be honest about confidence: HIGH only with both deck + web evidence."
    )

def create_market_research_agent(
    model_name: str = "llama-3.3-70b-versatile",
    model_provider: str = "groq",
    temperature: float = 0.2,
):
    """
    Factory: builds and returns a ready-to-invoke Market Research Agent.
    
    Returns a LangChain agent that, when invoked with a startup name,
    will produce a structured MarketResearchReport.
    
    Usage:
        agent = create_market_research_agent()
        result = agent.invoke({
            "messages": [{"role": "user", "content": "Analyze Airbnb"}]
        })
        report: MarketResearchReport = result["structured_response"]
    """
    
    # Build prompt with current deck inventory at agent-creation time
    available_decks = get_available_decks()
    system_prompt = _build_system_prompt(available_decks)
    
    model = init_chat_model(
        model_name,
        model_provider=model_provider,
        temperature=temperature,
        max_tokens=2048,
    )
    
    agent = create_agent(
        model=model,
        tools=[list_available_pitch_decks, search_pitch_deck, search_web],
        system_prompt=system_prompt,
        response_format=MarketResearchReport,
        name="market_research_agent",   # ← critical for LangGraph in Part 5
        middleware=[
            ToolRetryMiddleware(max_retries=2),
            ModelRetryMiddleware(max_retries=3),
        ],
    )
    
    return agent