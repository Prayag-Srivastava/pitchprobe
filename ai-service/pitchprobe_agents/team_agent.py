"""
Team Agent for PitchProbe
==========================
Production-hardened single agent that:
  1. Discovers which pitch decks are available
  2. Searches the relevant deck for team/founder content
  3. Searches the web for founder backgrounds, prior experience, advisors
  4. Returns a structured TeamReport

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

class FounderProfile(BaseModel):
    """Profile of a single founder or key executive."""
    name: str = Field(description="Full name")
    role: str = Field(description="Role/title (e.g., CEO, CTO, Co-founder)")
    background: str = Field(
        description="Brief summary of prior experience, education, notable achievements"
    )


class TeamReport(BaseModel):
    """Structured team analysis output."""
    
    startup_name: str = Field(description="The startup being analyzed")
    
    founders: list[FounderProfile] = Field(
        description="The founding team members with their backgrounds"
    )
    key_executives: list[FounderProfile] = Field(
        default_factory=list,
        description="Non-founder C-suite or key leadership (CFO, CTO if not founder, etc.)"
    )
    team_size_estimate: Optional[str] = Field(
        default=None,
        description="Estimated total team size (e.g., '50-100 employees', 'under 10')"
    )
    notable_strengths: list[str] = Field(
        description="Concrete strengths of the team "
                    "(e.g., 'CEO previously led $1B exit at Acme Corp', "
                    "'CTO has 15 years at Google on relevant tech')"
    )
    notable_gaps: list[str] = Field(
        default_factory=list,
        description="Concrete gaps or weaknesses "
                    "(e.g., 'No technical co-founder', 'Missing GTM leadership', "
                    "'All founders are first-time entrepreneurs')"
    )
    advisors: list[str] = Field(
        default_factory=list,
        description="Notable advisors, investors, or board members mentioned"
    )
    overall_team_score: Literal["weak", "average", "strong", "exceptional"] = Field(
        description="Overall team quality rating. "
                    "'weak' = serious concerns about team's ability to execute. "
                    "'average' = standard early-stage team. "
                    "'strong' = relevant domain expertise, good track record. "
                    "'exceptional' = world-class founders, proven track record, "
                    "complementary skills."
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
        "agent": "team",
        "tool": "list_available_pitch_decks",
    })
    
    decks = get_available_decks()
    
    _safe_emit({
        "event": "discovery_complete",
        "agent": "team",
        "tool": "list_available_pitch_decks",
        "decks_found": decks,
    })
    
    if not decks:
        return "No pitch decks are currently loaded in the system."
    return f"Available pitch decks: {', '.join(decks)}"


@tool
def search_pitch_deck(query: str, deck_name: str) -> str:
    """
    Search a SPECIFIC pitch deck for TEAM-RELATED information matching the query.
    
    Args:
        query: A TOPIC query relevant to team analysis. Examples:
               "founders", "team", "leadership", "advisors", "key personnel",
               "experience", "backgrounds". 
               Do NOT include the company name in the query.
        deck_name: The EXACT deck name from list_available_pitch_decks().
    
    Returns the matching chunks with page numbers, or a message if
    no relevant content was found in this deck.
    """
    _safe_emit({
        "event": "rag_search_started",
        "agent": "team",
        "tool": "search_pitch_deck",
        "query": query,
        "deck": deck_name,
    })
    
    results = search_deck(query=query, deck_name=deck_name, k=5)
    
    _safe_emit({
        "event": "rag_search_complete",
        "agent": "team",
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
    Search the web for FOUNDER and TEAM information.
    
    Use this for: founder backgrounds, prior companies, education, exits,
    notable executive hires, board members, advisors, public interviews,
    GitHub/LinkedIn presence.
    
    Args:
        query: A search query. Include the founder/executive name AND the
               context (e.g., "Brian Chesky Airbnb founder background",
               "Stripe Collison brothers prior experience",
               "OpenAI Sam Altman Y Combinator").
    """
    _safe_emit({
        "event": "web_search_started",
        "agent": "team",
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
        "agent": "team",
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
            "   - Call search_pitch_deck() at LEAST 3 times with different team-themed\n"
            "     queries (e.g., 'founders', 'team', 'leadership')\n"
            "   - Then call search_web() at LEAST 2 times for founder background research\n"
            "   - Set deck_available=True\n"
            "3. If no matching deck found:\n"
            "   - Call search_web() at LEAST 3 times for founder research\n"
            "   - Set deck_available=False and confidence ≤ 'medium'\n"
            "4. ONLY AFTER calling the above tools, produce the TeamReport.\n"
        )
    else:
        deck_section = (
            "No pitch decks confirmed loaded. Call list_available_pitch_decks() "
            "first to verify, then use web search.\n"
        )
    
    return (
        "You are a Team Analysis Agent in a VC due diligence system. "
        "Your job is to RESEARCH the team using tools, then produce a report.\n\n"
        "CRITICAL RULE: You MUST use tools to gather evidence. Never produce a "
        "team report from memory alone — your training data is outdated and unreliable. "
        "Every founder claim in your report MUST come from a tool call.\n\n"
        f"{deck_section}\n"
        "WHAT TO INVESTIGATE:\n"
        "- FOUNDERS: names, roles, prior experience, education, prior exits/failures\n"
        "- KEY EXECUTIVES: non-founder leadership and their relevant track records\n"
        "- TEAM SIZE: rough headcount if mentioned\n"
        "- STRENGTHS: domain expertise, complementary skills, prior successes\n"
        "- GAPS: missing roles (e.g., no CTO), inexperience, single-person dependencies\n"
        "- ADVISORS: notable advisors, board members, investors who lend credibility\n\n"
        "QUERY GUIDELINES:\n"
        "- Deck queries should be TOPIC-only (e.g., 'founders', 'team', 'leadership'). "
        "  The deck filter handles the company.\n"
        "- Web queries should include FOUNDER NAMES if you know them, plus context "
        "  (e.g., 'Brian Chesky background', 'Collison brothers prior startups').\n\n"
        "QUALITY RULES:\n"
        "- Search the deck for team slides — most decks have a dedicated team page.\n"
        "- Search the web for founder backgrounds — decks often oversell teams.\n"
        "- Be CONCRETE: 'Founder X previously sold Y to Z for $1B' is useful. "
        "  'Team has industry experience' is not.\n"
        "- Cite sources for every founder claim — only REAL sources from tool outputs.\n"
        "- Score teams honestly. Most teams are 'average' or 'strong' — 'exceptional' "
        "  is rare and requires evidence."
    )


def create_team_agent(
    model_name: str = "llama-3.3-70b-versatile",
    model_provider: str = "groq",
    temperature: float = 0.2,
):
    """
    Factory: builds and returns a ready-to-invoke Team Analysis Agent.
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
        response_format=TeamReport,
        name="team_analysis_agent",
        middleware=[
            ToolRetryMiddleware(max_retries=2),
            ModelRetryMiddleware(max_retries=3),
        ],
    )
    
    return agent