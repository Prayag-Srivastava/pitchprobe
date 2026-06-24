"""
Startup Summarizer - Part 2 of PitchProbe (Tavily Direct + LLM)
Architecture:
  1. Tavily search runs OUTSIDE the LLM (no token limit issues)
  2. Gemini synthesizes the search results into structured output
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# Schema Definition
# ============================================================

class StartupInfo(BaseModel):
    """Structured analysis of a startup for investment purposes."""

    name: str = Field(description="The name of the startup")

    industry: Optional[str] = Field(
        default=None,
        description="Industry of the startup or null"
    )

    one_line_description: Optional[str] = Field(
        default=None,
        description="One line startup description or null"
    )

    founded_year: Optional[int] = Field(
        default=None,
        description="Year founded as integer or null. Never return strings."
    )

    headquarters: Optional[str] = Field(
        default=None,
        description="Headquarters location or null if unknown"
    )

    key_products: list[str] = Field(
        default_factory=list,
        description="Main products or services"
    )

    target_market: Optional[
        Literal["B2B", "B2C", "B2B2C", "Government"]
    ] = Field(
        default=None,
        description=(
            "Target market. Must be one of "
            "B2B, B2C, B2B2C, Government or null."
        )
    )

    estimated_valuation_usd_millions: Optional[float] = Field(
        default=None,
        description=(
            "Valuation in MILLIONS USD. "
            "Example: $18B => 18000. "
            "Must be number or null."
        )
    )

    main_competitors: list[str] = Field(
        default_factory=list,
        description="Top competitors"
    )

    risk_level: Literal["low", "medium", "high"] = Field(
        description="Investment risk level"
    )

    investment_potential: Literal["strong", "moderate", "weak"] = Field(
        description="Investment potential"
    )

    reasoning: str = Field(
        description="Reasoning behind assessment"
    )

    information_sources: list[str] = Field(
        default_factory=list,
        description="URLs used during research"
    )


# ============================================================
# Direct Tavily Search (No LLM involved)
# ============================================================

search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced",
    include_raw_content=False,
)


def search_web_directly(startup_name: str) -> tuple[str, list[str]]:
    """
    Run Tavily search directly (not via LLM).
    Returns: (compact research text, list of source URLs)
    """
    print(f"\n🔍 Searching web for: '{startup_name}'")

    # Run the search
    results = search_tool.invoke({
        "query": f"{startup_name} startup funding products competitors"
    })

    # Extract results (Tavily returns dict with 'results' key)
    if isinstance(results, dict):
        search_results = results.get("results", [])
    elif isinstance(results, list):
        search_results = results
    else:
        search_results = []

    print(f"✅ Got {len(search_results)} results from Tavily")

    # Build compact text (only title + snippet, no raw content)
    research_lines = [f"Search results for '{startup_name}':\n"]
    sources = []

    for i, res in enumerate(search_results, 1):
        title = res.get("title", "No title")
        url = res.get("url", "")
        # Truncate snippet to 300 chars to save tokens
        snippet = res.get("content", "")[:300]

        research_lines.append(f"\n[Result {i}]")
        research_lines.append(f"Title: {title}")
        research_lines.append(f"URL: {url}")
        research_lines.append(f"Snippet: {snippet}")

        if url:
            sources.append(url)

    research_text = "\n".join(research_lines)
    return research_text, sources


# ============================================================
# Single Agent (Just for Synthesis)
# ============================================================

SYNTHESIS_PROMPT = """
You are a venture capital analyst.

You will receive search results about a startup.

Convert them into the StartupInfo schema.

Rules:
- Use ONLY facts found in the search results.
- If information is unavailable, return null.
- Never return the string "Unknown".
- Numeric fields must contain numbers or null.
- founded_year must be an integer or null.
- estimated_valuation_usd_millions must be a number or null.
- headquarters must be a string or null.
- target_market must be one of:
  B2B, B2C, B2B2C, Government, or null.
- Be objective when assigning risk_level.
- Be objective when assigning investment_potential.
- Include all source URLs in information_sources.
"""


def create_synthesis_agent():
    """Synthesis Agent — Groq for reliable structured output."""
    model = init_chat_model(
        "llama-3.3-70b-versatile",
        model_provider="groq",
        temperature=0.2,
    )

    return create_agent(
        model=model,
        tools=[],
        system_prompt=SYNTHESIS_PROMPT,
        response_format=StartupInfo,
    )


# ============================================================
# Core Logic
# ============================================================

def analyze_startup(
    synthesis_agent,
    startup_name: str,
    verbose: bool = True,
) -> Optional[StartupInfo]:
    """
    Pipeline:
      1. Search web with Tavily DIRECTLY (no LLM, no token issue)
      2. Send search results to Groq for structured analysis
    """
    try:
        # ============================================
        # STEP 1: Direct Tavily Search
        # ============================================
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔍 STEP 1: Web Search (Tavily Direct)")
            print(f"{'='*60}")

        research_text, sources = search_web_directly(startup_name)

        if verbose:
            print(f"\n📋 Search Results Preview ({len(research_text)} chars):")
            print("-" * 60)
            print(research_text[:600] + ("..." if len(research_text) > 600 else ""))
            print("-" * 60)
            print(f"\n📚 Source URLs found: {len(sources)}")

        # ============================================
        # STEP 2: Groq Synthesis
        # ============================================
        if verbose:
            print(f"\n{'='*60}")
            print(f"🧠 STEP 2: Synthesis (Groq + StartupInfo Schema)")
            print(f"{'='*60}")

        synthesis_result = synthesis_agent.invoke({
            "messages": [{
                "role": "user",
                "content": (
                    f"Analyze startup '{startup_name}' based on these search results. "
                    f"Include these URLs in information_sources: {sources}\n\n"
                    f"=== SEARCH RESULTS ===\n{research_text}\n=== END ==="
                )
            }]
        })

        return synthesis_result["structured_response"]

    except Exception as e:
        print(f"\n❌ Error analyzing {startup_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# Pretty Print
# ============================================================

def print_analysis(summary: StartupInfo) -> None:
    """Pretty-print the startup analysis."""
    print("\n" + "=" * 60)
    print(f"📊 FINAL ANALYSIS: {summary.name}")
    print("=" * 60)
    print(f"🏢 Industry:        {summary.industry}")
    print(f"📝 Description:     {summary.one_line_description}")
    print(f"📅 Founded:         {summary.founded_year or 'Unknown'}")
    print(f"📍 HQ:              {summary.headquarters}")
    print(f"🎯 Market:          {summary.target_market}")

    valuation = (
        f"${summary.estimated_valuation_usd_millions:,.1f}M"
        if summary.estimated_valuation_usd_millions
        else "Unknown"
    )
    print(f"💰 Valuation:       {valuation}")

    print(f"🛍️  Key Products:    {', '.join(summary.key_products) if summary.key_products else 'Unknown'}")
    print(f"⚔️  Competitors:     {', '.join(summary.main_competitors) if summary.main_competitors else 'Unknown'}")
    print(f"⚠️  Risk Level:      {summary.risk_level.upper()}")
    print(f"💎 Potential:       {summary.investment_potential.upper()}")
    print(f"\n🤔 Reasoning:\n   {summary.reasoning}")

    if summary.information_sources:
        print(f"\n📚 Sources ({len(summary.information_sources)}):")
        for i, source in enumerate(summary.information_sources, 1):
            print(f"   {i}. {source}")
    else:
        print("\n📚 Sources: None cited")

    print("=" * 60)


# ============================================================
# Main
# ============================================================

def main():
    """Run the startup analyzer in interactive mode."""
    print("=" * 60)
    print("🚀 PitchProbe - Startup Analyzer (Part 2 Final)")
    print("   Tavily (direct) + Groq (synthesis)")
    print("=" * 60)

    print("\n⚙️  Initializing agent...")
    synthesis_agent = create_synthesis_agent()
    print("✅ Agent ready!\n")

    while True:
        startup_name = input("💬 Enter startup name (or 'quit' to exit): ").strip()

        if startup_name.lower() in ['quit', 'exit', 'q', '']:
            print("\n👋 Goodbye!")
            break

        print(f"\n⏳ Analyzing '{startup_name}'...")

        summary = analyze_startup(
            synthesis_agent,
            startup_name,
            verbose=True
        )

        if summary:
            print_analysis(summary)
        else:
            print("⚠️  Analysis failed. Please try another startup.")


if __name__ == "__main__":
    main()