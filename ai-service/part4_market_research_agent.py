"""
Part 4 of PitchProbe — Market Research Agent
=============================================
The FIRST real agent of the PitchProbe multi-agent system.

This agent combines:
  - RAG over the uploaded pitch deck (Part 3 retriever)
  - Live web search via Tavily (Part 2)
  - Structured output (Part 1)
  - Carefully engineered system prompt (Part 4 — context engineering)

The agent autonomously decides:
  - When to search the pitch deck
  - When to search the web
  - How many searches to make
  - When it has enough info to write the report

Output: a structured MarketResearchReport that downstream agents (in Part 5)
will consume as input.
"""

import os
import warnings
from typing import Optional, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


# ============================================================
# Configuration
# ============================================================

PERSIST_DIR = "./chroma_db_advanced"   # reusing Part 3 Advanced vector store
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "llama-3.3-70b-versatile"
TOP_K = 4
TAVILY_MAX_RESULTS = 4


# ============================================================
# Structured Output Schema
# ============================================================

class MarketResearchReport(BaseModel):
    """Comprehensive market research report on a startup."""

    company_name: str = Field(description="The name of the startup being analyzed")
    industry: str = Field(description="The primary industry/sector of the startup")
    market_description: str = Field(
        description="A 2-3 sentence description of the market this startup operates in"
    )
    market_size_estimate: str = Field(
        description="TAM/SAM estimate with source if available, or 'Unknown' if no data found"
    )
    key_trends: list[str] = Field(
        description="3-5 major trends shaping this market right now"
    )
    target_customers: list[str] = Field(
        description="The customer segments this startup serves"
    )
    main_competitors: list[str] = Field(
        description="3-5 main competitors in this market"
    )
    market_opportunities: list[str] = Field(
        description="2-3 specific opportunities this startup could capture"
    )
    market_risks: list[str] = Field(
        description="2-3 specific market risks/headwinds the startup faces"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description=(
            "How confident you are in the analysis. "
            "'high' = strong data from both deck and web. "
            "'medium' = partial data. "
            "'low' = limited data or significant gaps."
        )
    )
    sources_used: list[str] = Field(
        description="List of sources: 'pitch deck' and any web URLs cited"
    )
    reasoning: str = Field(
        description="A 2-3 sentence summary of your overall market assessment"
    )


# ============================================================
# Setup: Load retriever and Tavily ONCE (module level)
# ============================================================
# Why module level? So they're loaded ONE TIME when the script starts,
# not every time a tool is called. Tools capture them via closure.

print("⚙️  Initializing retriever and search tools...")

embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)
_retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

_tavily = TavilySearch(
    max_results=TAVILY_MAX_RESULTS,
    topic="general",
    search_depth="advanced",
)

print("   ✅ Ready.\n")


# ============================================================
# Tool 1: Search the pitch deck (wraps Part 3 RAG)
# ============================================================

@tool
def search_pitch_deck(query: str) -> str:
    """Search the uploaded pitch deck for information about the startup.

    Use this tool FIRST when researching a startup to find what the startup
    says about itself: its market, target customers, competitors, business
    model, growth strategy, and traction.

    Args:
        query: A focused search query. Examples:
               - "target market and customers"
               - "business model and revenue"
               - "competitors and market position"
               - "market size and opportunity"
    """
    docs = _retriever.invoke(query)
    if not docs:
        return "No relevant content found in the pitch deck for this query."

    formatted = []
    for d in docs:
        page = d.metadata.get("page", "?")
        page_label = (page + 1) if isinstance(page, int) else page
        formatted.append(f"[Pitch Deck - Page {page_label}]\n{d.page_content}")

    return "\n\n---\n\n".join(formatted)


# ============================================================
# Tool 2: Search the web (wraps Tavily)
# ============================================================

@tool
def search_web(query: str) -> str:
    """Search the live internet for current market information.

    Use this tool AFTER consulting the pitch deck to:
      - Verify or challenge claims the startup makes about its market
      - Find external data on market size, growth rates, and trends
      - Discover competitors not mentioned in the deck
      - Get recent news about the industry

    Args:
        query: A focused search query. Examples:
               - "short-term rental market size 2024"
               - "vacation rental industry trends"
               - "Airbnb competitors VRBO Booking"
               - "hospitality industry growth forecast"
    """
    results = _tavily.invoke({"query": query})

    # Tavily returns a dict with a 'results' list — format it cleanly
    if isinstance(results, dict) and "results" in results:
        formatted = []
        for r in results["results"]:
            title = r.get("title", "No title")
            url = r.get("url", "No URL")
            content = r.get("content", "")
            formatted.append(f"[{title}]\nURL: {url}\n{content}")
        return "\n\n---\n\n".join(formatted) if formatted else "No web results found."

    return str(results)


# ============================================================
# System Prompt — the heart of context engineering
# ============================================================

SYSTEM_PROMPT = """You are an expert Market Research Analyst working at a top venture capital firm.

Your task is to produce a thorough, evidence-based market research report on a startup.

# AVAILABLE TOOLS

You have two tools at your disposal:
1. `search_pitch_deck(query)` — search the startup's pitch deck for what THEY say.
2. `search_web(query)` — search the live internet for EXTERNAL market data.

# YOUR PROCESS — FOLLOW THIS STRICTLY

Step 1: Search the pitch deck FIRST.
   - Make 2-3 targeted searches to understand what the startup claims about its market,
     customers, competitors, and business model.
   - Examples: "target market", "competitors", "business model", "market opportunity"

Step 2: Search the web SECOND.
   - Make 2-4 targeted web searches to:
     * Verify the startup's market size claims
     * Find data on industry growth, trends, and competition
     * Discover competitors the deck didn't mention
     * Identify external risks (regulation, macro trends)
   - Examples: "[industry] market size 2024", "[company name] competitors",
     "[industry] trends and growth forecast"

Step 3: Synthesize a structured report.
   - Combine internal claims (from deck) with external data (from web)
   - Flag any contradictions between what the startup says vs. reality
   - Be honest about confidence: 'high' only if both sources agree and data is solid

# QUALITY RULES

- Make MULTIPLE searches. A single search is rarely enough.
- If the pitch deck doesn't mention something, search the web for it.
- Always cite sources in `sources_used` (use "pitch deck" + web URLs).
- If data is missing or weak, set confidence to "medium" or "low" — be honest.
- Do NOT use outside knowledge. Only use what your tools return.
"""


# ============================================================
# Agent Factory
# ============================================================

def create_market_research_agent():
    """Create the Market Research Agent with both tools and structured output."""
    model = init_chat_model(
        LLM_MODEL,
        model_provider="groq",
        temperature=0.3,
        max_tokens=2048,
    )

    return create_agent(
        model=model,
        tools=[search_pitch_deck, search_web],
        system_prompt=SYSTEM_PROMPT,
        response_format=MarketResearchReport,
        name="market_research_agent",   # ← important for Part 5 LangGraph integration
    )


# ============================================================
# Helper: Inspect tool calls (debugging gold)
# ============================================================

def print_tool_calls(result) -> None:
    """Print every tool call made by the agent, in order."""
    print("\n🔧 Tool Calls Made by Agent:")
    print("-" * 60)
    count = 0
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                count += 1
                args_preview = str(tc["args"])[:100]
                print(f"  {count}. 🔍 {tc['name']}({args_preview}{'...' if len(str(tc['args'])) > 100 else ''})")
    if count == 0:
        print("  ⚠️  NO TOOLS were called — agent answered from memory!")
    print("-" * 60)
    print(f"   Total tool calls: {count}\n")


# ============================================================
# Core Logic
# ============================================================

def analyze_market(agent, startup_name: str) -> Optional[MarketResearchReport]:
    """Run the Market Research Agent on a startup name."""
    user_message = (
        f"Produce a comprehensive market research report on the startup '{startup_name}'. "
        f"Follow your process: search the pitch deck first, then the web, then synthesize."
    )

    try:
        result = agent.invoke({
            "messages": [{"role": "user", "content": user_message}]
        })
        print_tool_calls(result)
        return result["structured_response"]
    except Exception as e:
        print(f"❌ Agent error: {e}")
        return None


# ============================================================
# Pretty Printing
# ============================================================

def print_report(report: MarketResearchReport) -> None:
    """Nicely format and print the market research report."""
    print("\n" + "=" * 60)
    print(f"📊 MARKET RESEARCH REPORT: {report.company_name}")
    print("=" * 60)

    print(f"\n🏢 Industry:           {report.industry}")
    print(f"\n📝 Market Description:\n   {report.market_description}")
    print(f"\n📏 Market Size:        {report.market_size_estimate}")

    print(f"\n📈 Key Trends:")
    for t in report.key_trends:
        print(f"   • {t}")

    print(f"\n👥 Target Customers:")
    for c in report.target_customers:
        print(f"   • {c}")

    print(f"\n⚔️  Main Competitors:")
    for c in report.main_competitors:
        print(f"   • {c}")

    print(f"\n🌟 Opportunities:")
    for o in report.market_opportunities:
        print(f"   • {o}")

    print(f"\n⚠️  Risks:")
    for r in report.market_risks:
        print(f"   • {r}")

    print(f"\n🎯 Confidence:         {report.confidence.upper()}")

    print(f"\n📚 Sources Used:")
    for s in report.sources_used:
        print(f"   • {s}")

    print(f"\n🤔 Reasoning:\n   {report.reasoning}")
    print("=" * 60)


# ============================================================
# Main: Interactive Loop
# ============================================================

def main() -> None:
    print("🚀 PitchProbe - Market Research Agent (Part 4)")
    print("=" * 60)

    agent = create_market_research_agent()

    print("\n✅ Agent ready. Enter a startup name (whose pitch deck is loaded).")
    print("   Type 'quit' to exit.\n")

    while True:
        startup_name = input("💬 Startup name: ").strip()

        if startup_name.lower() in {"quit", "exit", "q", ""}:
            print("\n👋 Goodbye!")
            break

        print(f"\n⏳ Analyzing market for '{startup_name}'...")
        print("   (Agent will search the pitch deck and the web autonomously...)\n")

        report = analyze_market(agent, startup_name)

        if report:
            print_report(report)
        else:
            print("⚠️  Analysis failed. Try again.")


if __name__ == "__main__":
    main()