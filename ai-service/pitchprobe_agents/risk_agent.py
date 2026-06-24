"""
Risk Agent for PitchProbe
==========================
Production-hardened single agent that:
  1. Discovers which pitch decks are available
  2. Searches the relevant deck for risk-related content
  3. Searches the web for regulatory news, lawsuits, market shifts
  4. Returns a structured RiskReport

This is a REUSABLE FACTORY. Same architectural pattern as market_agent.py.
The DIFFERENCES from market_agent.py are:
  - RiskReport schema (different output fields)
  - System prompt focuses on risk discovery
  - Tool docstrings mention risk-relevant content
  - Tool prints prefixed with ⚠️ for traceability in parallel runs
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware, ModelRetryMiddleware
from langchain.tools import tool
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

# Same shared retriever — all agents read from the same vector store
from pitchprobe_agents.rag_retriever import (
    search_deck,
    get_available_decks,
)

load_dotenv()

# Add this helper near the top (after imports)
def _safe_emit(event: dict) -> None:
    """Emit a custom event to the LangGraph stream if we're in a graph context."""
    try:
        from langgraph.config import get_stream_writer
        writer = get_stream_writer()
        writer(event)
    except Exception:
        print(f"   [event:{event.get('event','?')}] {event}")

# ============================================================
# OUTPUT SCHEMA
# ============================================================
# Notice how this is RISK-specific (not market-specific).
# Different fields capture different aspects of due diligence.

class RiskReport(BaseModel):
    """Structured risk analysis output."""
    
    startup_name: str = Field(description="The startup being analyzed")
    
    regulatory_risks: list[str] = Field(
        description="Legal, compliance, licensing, or regulatory exposure "
                    "(e.g., GDPR, FDA approvals, AI regulation, sanctions)"
    )
    financial_risks: list[str] = Field(
        description="Funding, burn rate, unit economics, profitability concerns, "
                    "currency exposure, dependence on a single investor"
    )
    operational_risks: list[str] = Field(
        description="Supply chain, key personnel dependencies, single-vendor lock-in, "
                    "infrastructure fragility, technical debt"
    )
    market_risks: list[str] = Field(
        description="Competition, market saturation, customer concentration, "
                    "shifting consumer behavior, macro downturns"
    )
    mitigations: list[str] = Field(
        default_factory=list,
        description="How the startup is addressing or hedging against the identified risks. "
                    "Empty if no mitigations are mentioned anywhere."
    )
    overall_severity: Literal["low", "medium", "high", "critical"] = Field(
        description="Overall risk severity rating. "
                    "'low' = manageable risks, well-mitigated. "
                    "'medium' = standard startup risks. "
                    "'high' = serious concerns that could threaten viability. "
                    "'critical' = existential threats present."
    )
    deck_available: bool = Field(
        description="Whether a pitch deck was available for this startup"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in this analysis. HIGH only if both deck + web sources used."
    )
    sources: list[str] = Field(
        description="URLs, deck pages, or document names used as evidence"
    )


# ============================================================
# TOOLS
# ============================================================
# Each tool prints with the ⚠️ prefix so we can SEE which agent is
# calling tools during parallel execution. This is essential for
# debugging multi-agent runs.
#
# The tool DOCSTRINGS are tuned for risk discovery — they tell the
# LLM "use this for finding risk-relevant content", which biases
# what queries the LLM crafts.

@tool
def list_available_pitch_decks() -> str:
    """
    Discovery tool: list all pitch decks currently loaded in the system.
    
    USE THIS FIRST before attempting to search a deck. Returns the
    exact deck names you must pass to search_pitch_deck.
    """
    decks = get_available_decks()
    if not decks:
        return "No pitch decks are currently loaded in the system."
    return f"Available pitch decks: {', '.join(decks)}"


@tool
def search_pitch_deck(query: str, deck_name: str) -> str:
    """
    Search a SPECIFIC pitch deck for RISK-RELATED information matching the query.
    
    Args:
        query: A TOPIC query relevant to risk analysis. Examples:
               "regulatory compliance", "competition", "financial projections",
               "key personnel", "legal", "supply chain". 
               Do NOT include the company name in the query.
        deck_name: The EXACT deck name from list_available_pitch_decks().
    
    Returns the matching chunks with page numbers, or a message if
    no relevant content was found in this deck.
    """
    print(f"   ⚠️  [Risk-Tool] search_pitch_deck(query='{query}', deck='{deck_name}')")
    
    results = search_deck(query=query, deck_name=deck_name, k=5)
    
    if not results:
        return (
            f"No content in deck '{deck_name}' is sufficiently relevant to "
            f"the query '{query}'. The deck does not contain this information."
        )
    
    print(f"      → {len(results)} relevant chunks found (scores: "
          f"{[f'{s:.2f}' for _, s in results]})")
    
    output = []
    for doc, score in results:
        page = doc.metadata.get("page", "?")
        output.append(
            f"[Page {page + 1 if isinstance(page, int) else page}, "
            f"relevance score: {score:.2f}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(output)

"""
Risk Agent for PitchProbe
==========================
Production-hardened single agent that:
  1. Discovers which pitch decks are available
  2. Searches the relevant deck for risk-related content
  3. Searches the web for regulatory news, lawsuits, market shifts
  4. Returns a structured RiskReport

VERSION HISTORY:
  - 5.6: Initial version with print() and inline deck names
  - 6.6: Switched to get_stream_writer() events + removed inline deck names
         to force tool-based discovery (Rule #39)
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
# STREAMING-AWARE EVENT EMITTER (with fallback to print)
# ============================================================

def _safe_emit(event: dict) -> None:
    """
    Emit a custom event to the LangGraph stream if we're in a graph context.
    Falls back to print() if called standalone (outside a graph).
    """
    try:
        from langgraph.config import get_stream_writer
        writer = get_stream_writer()
        writer(event)
    except Exception:
        print(f"   [event:{event.get('event','?')}] {event}")


# ============================================================
# OUTPUT SCHEMA
# ============================================================

class RiskReport(BaseModel):
    """Structured risk analysis output."""
    
    startup_name: str = Field(description="The startup being analyzed")
    
    regulatory_risks: list[str] = Field(
        description="Legal, compliance, licensing, or regulatory exposure "
                    "(e.g., GDPR, FDA approvals, AI regulation, sanctions)"
    )
    financial_risks: list[str] = Field(
        description="Funding, burn rate, unit economics, profitability concerns, "
                    "currency exposure, dependence on a single investor"
    )
    operational_risks: list[str] = Field(
        description="Supply chain, key personnel dependencies, single-vendor lock-in, "
                    "infrastructure fragility, technical debt"
    )
    market_risks: list[str] = Field(
        description="Competition, market saturation, customer concentration, "
                    "shifting consumer behavior, macro downturns"
    )
    mitigations: list[str] = Field(
        default_factory=list,
        description="How the startup is addressing or hedging against the identified risks. "
                    "Empty if no mitigations are mentioned anywhere."
    )
    overall_severity: Literal["low", "medium", "high", "critical"] = Field(
        description="Overall risk severity rating. "
                    "'low' = manageable risks, well-mitigated. "
                    "'medium' = standard startup risks. "
                    "'high' = serious concerns that could threaten viability. "
                    "'critical' = existential threats present."
    )
    deck_available: bool = Field(
        description="Whether a pitch deck was available for this startup"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in this analysis. HIGH only if both deck + web sources used."
    )
    sources: list[str] = Field(
        description="URLs, deck pages, or document names used as evidence"
    )


# ============================================================
# TOOLS (with custom event emission for streaming)
# ============================================================

@tool
def list_available_pitch_decks() -> str:
    """
    Discovery tool: list all pitch decks currently loaded in the system.
    
    USE THIS FIRST before attempting to search a deck. Returns the
    exact deck names you must pass to search_pitch_deck.
    """
    _safe_emit({
        "event": "discovery_started",
        "agent": "risk",
        "tool": "list_available_pitch_decks",
    })
    
    decks = get_available_decks()
    
    _safe_emit({
        "event": "discovery_complete",
        "agent": "risk",
        "tool": "list_available_pitch_decks",
        "decks_found": decks,
    })
    
    if not decks:
        return "No pitch decks are currently loaded in the system."
    return f"Available pitch decks: {', '.join(decks)}"


@tool
def search_pitch_deck(query: str, deck_name: str) -> str:
    """
    Search a SPECIFIC pitch deck for RISK-RELATED information matching the query.
    
    Args:
        query: A TOPIC query relevant to risk analysis. Examples:
               "regulatory compliance", "competition", "financial projections",
               "key personnel", "legal", "supply chain". 
               Do NOT include the company name in the query.
        deck_name: The EXACT deck name from list_available_pitch_decks().
    
    Returns the matching chunks with page numbers, or a message if
    no relevant content was found in this deck.
    """
    _safe_emit({
        "event": "rag_search_started",
        "agent": "risk",
        "tool": "search_pitch_deck",
        "query": query,
        "deck": deck_name,
    })
    
    results = search_deck(query=query, deck_name=deck_name, k=5)
    
    _safe_emit({
        "event": "rag_search_complete",
        "agent": "risk",
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
    Search the web for current RISK-RELATED information.
    
    Use this for: regulatory updates, recent lawsuits or fines, industry
    compliance news, competitor moves, market shifts, sector downturns,
    macro risks affecting the startup's industry.
    
    Args:
        query: A search query. Include the company/industry name AND the risk
               aspect (e.g., "fintech regulation 2024", "Stripe lawsuit",
               "AI compliance EU AI Act").
    """
    _safe_emit({
        "event": "web_search_started",
        "agent": "risk",
        "tool": "search_web",
        "query": query,
    })
    
    results = _web_search.invoke({"query": query})
    
    try:
        results_count = len(results.get("results", [])) if isinstance(results, dict) else 0
    except Exception:
        results_count = 0
    
    _safe_emit({
        "event": "web_search_complete",
        "agent": "risk",
        "tool": "search_web",
        "query": query,
        "results_count": results_count,
    })
    
    return str(results)


# ============================================================
# AGENT FACTORY
# ============================================================

def _build_system_prompt(available_decks: list[str]) -> str:
    """
    Build the system prompt. CRITICAL: do NOT inline the deck names —
    that makes Llama think it already has the deck content (Rule #39).
    Force discovery via the tool.
    """
    
    if available_decks:
        deck_section = (
            "Pitch decks may be loaded in the system. You do NOT know which ones.\n\n"
            "MANDATORY WORKFLOW:\n"
            "1. FIRST call list_available_pitch_decks() to discover loaded decks.\n"
            "   You MUST call this tool — do not assume what is loaded.\n"
            "2. If the target startup matches a loaded deck name:\n"
            "   - Call search_pitch_deck() at LEAST 3 times with different risk-themed\n"
            "     queries (e.g., 'regulatory', 'competition', 'financial projections')\n"
            "   - Then call search_web() at LEAST 2 times for external risk news\n"
            "   - Set deck_available=True\n"
            "3. If no matching deck found:\n"
            "   - Call search_web() at LEAST 3 times for risk research\n"
            "   - Set deck_available=False and confidence ≤ 'medium'\n"
            "4. ONLY AFTER calling the above tools, produce the RiskReport.\n"
        )
    else:
        deck_section = (
            "No pitch decks confirmed loaded. Call list_available_pitch_decks() "
            "first to verify, then use web search.\n"
        )
    
    return (
        "You are a Risk Analysis Agent in a VC due diligence system. "
        "Your job is to RESEARCH risks for a startup using tools, then produce a report.\n\n"
        "CRITICAL RULE: You MUST use tools to gather evidence. Never produce a "
        "risk report from memory alone — your training data is outdated and unreliable. "
        "Every risk in your report MUST come from a tool call.\n\n"
        f"{deck_section}\n"
        "RISK CATEGORIES TO INVESTIGATE:\n"
        "- REGULATORY: licenses, compliance, AI/data/privacy regulations, sanctions\n"
        "- FINANCIAL: burn rate, runway, unit economics, customer concentration\n"
        "- OPERATIONAL: supply chain, key person dependency, infrastructure fragility\n"
        "- MARKET: competition, saturation, customer churn, macro environment\n\n"
        "QUERY GUIDELINES:\n"
        "- Deck queries should be TOPIC-only (e.g., 'regulatory', 'competition'). "
        "  The deck filter handles the company.\n"
        "- Web queries should include the company/industry AND the risk aspect "
        "  (e.g., 'EU AI Act enforcement 2024', 'fintech licensing requirements').\n\n"
        "QUALITY RULES:\n"
        "- Search the deck for risk disclosures founders themselves mention.\n"
        "- Search the web for risks founders may NOT mention (lawsuits, regulatory shifts).\n"
        "- Cite sources for every claim — only REAL sources from tool outputs.\n"
        "- Assess severity HONESTLY. Do not soften critical risks."
    )


def create_risk_agent(
    model_name: str = "llama-3.3-70b-versatile",
    model_provider: str = "groq",
    temperature: float = 0.2,
):
    """
    Factory: builds and returns a ready-to-invoke Risk Analysis Agent.
    """
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
        response_format=RiskReport,
        name="risk_analysis_agent",
        middleware=[
            ToolRetryMiddleware(max_retries=2),
            ModelRetryMiddleware(max_retries=3),
        ],
    )
    
    return agent
_web_search = TavilySearch(max_results=5, topic="general")

@tool
def search_web(query: str) -> str:
    """
    Search the web for current RISK-RELATED information.
    
    Use this for: regulatory updates, recent lawsuits or fines, industry
    compliance news, competitor moves, market shifts, sector downturns,
    macro risks affecting the startup's industry.
    
    Args:
        query: A search query. Include the company/industry name AND the risk
               aspect (e.g., "fintech regulation 2024", "Stripe lawsuit",
               "AI compliance EU AI Act").
    """
    print(f"   ⚠️  [Risk-Tool] search_web(query='{query}')")
    results = _web_search.invoke({"query": query})
    return str(results)


# ============================================================
# AGENT FACTORY
# ============================================================

def _build_system_prompt(available_decks: list[str]) -> str:
    """Build the system prompt dynamically based on available decks."""
    
    if available_decks:
        deck_list = ", ".join(f"'{d}'" for d in available_decks)
        deck_section = (
            f"Currently loaded pitch decks: {deck_list}\n\n"
            f"WORKFLOW:\n"
            f"1. ALWAYS call list_available_pitch_decks() FIRST to confirm what's loaded.\n"
            f"2. If the startup matches a loaded deck → search the deck "
            f"   (multiple risk-themed queries) AND search the web for regulatory news. "
            f"   Set deck_available=True.\n"
            f"3. If the startup does NOT match any loaded deck → web search only. "
            f"   Set deck_available=False and confidence ≤ 'medium'.\n"
        )
    else:
        deck_section = (
            "No pitch decks are currently loaded. Use web search only. "
            "Set deck_available=False and confidence ≤ 'medium'.\n"
        )
    
    return (
        "You are a Risk Analysis Agent in a VC due diligence system. "
        "Your job is to produce a comprehensive RISK assessment of a startup.\n\n"
        f"{deck_section}\n"
        "RISK CATEGORIES TO INVESTIGATE:\n"
        "- REGULATORY: licenses, compliance, AI/data/privacy regulations, sanctions\n"
        "- FINANCIAL: burn rate, runway, unit economics, customer concentration\n"
        "- OPERATIONAL: supply chain, key person dependency, infrastructure fragility\n"
        "- MARKET: competition, saturation, customer churn, macro environment\n\n"
        "QUERY GUIDELINES:\n"
        "- Deck queries should be TOPIC-only (e.g., 'regulatory', 'competition', "
        "  'financial projections'). The deck filter handles the company.\n"
        "- Web queries should include the company/industry AND the risk aspect "
        "  (e.g., 'EU AI Act enforcement 2024', 'fintech licensing requirements').\n\n"
        "QUALITY RULES:\n"
        "- Search the deck for risk disclosures the founders themselves mention.\n"
        "- Search the web for risks the founders may NOT mention (lawsuits, regulatory shifts).\n"
        "- Ground every claim in either deck content or web search results.\n"
        "- Cite sources (deck name + page, or URL).\n"
        "- Assess severity HONESTLY. Do not soften critical risks."
    )


def create_risk_agent(
    model_name: str = "llama-3.3-70b-versatile",
    model_provider: str = "groq",
    temperature: float = 0.2,
):
    """
    Factory: builds and returns a ready-to-invoke Risk Analysis Agent.
    
    Returns a LangChain agent that, when invoked with a startup name,
    will produce a structured RiskReport.
    
    Usage:
        agent = create_risk_agent()
        result = agent.invoke({
            "messages": [{"role": "user", "content": "Analyze risks for Stripe"}]
        })
        report: RiskReport = result["structured_response"]
    """
    
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
        response_format=RiskReport,
        name="risk_analysis_agent",   # critical for LangGraph
        middleware=[
            ToolRetryMiddleware(max_retries=2),
            ModelRetryMiddleware(max_retries=3),
        ],
    )
    
    return agent