"""
Startup Summarizer - Part 1 of PitchProbe
Analyzes startups using LLM and returns structured investment analysis.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# Schema Definition
# ============================================================

class StartupInfo(BaseModel):
    """Structured analysis of a startup for investment purposes."""
    
    name: str = Field(description="The name of the startup")
    industry: str = Field(description="The industry the startup operates in")
    one_line_description: str = Field(description="A brief one-line description")
    founded_year: Optional[int] = Field(default=None, description="Year founded")
    headquarters: str = Field(description="Location of headquarters")
    key_products: list[str] = Field(description="Main products or services")
    target_market: Literal["B2B", "B2C", "B2B2C", "Government"] = Field(
        description="Primary target market segment"
    )
    estimated_valuation_usd_millions: Optional[float] = Field(
        default=None,
        description="Estimated valuation in millions USD"
    )
    main_competitors: list[str] = Field(description="Top 3-5 main competitors")
    risk_level: Literal["low", "medium", "high"] = Field(
        description="Investment risk level"
    )
    investment_potential: Literal["strong", "moderate", "weak"] = Field(
        description="Overall investment potential"
    )
    reasoning: str = Field(description="Detailed reasoning for the verdict")


# ============================================================
# Agent Factory
# ============================================================

SYSTEM_PROMPT = """You are an expert startup analyst and venture capital researcher.

Your task is to analyze startups and provide structured investment analysis.

Guidelines:
- Be objective and evidence-based
- If data is uncertain, mark it appropriately (use None for unknown values)
- Provide clear reasoning for risk and investment verdicts
- Consider market position, competition, growth potential, and team
- Base analysis on publicly known information
"""


def create_analysis_agent():
    """Create and return the startup analysis agent."""
    model = init_chat_model(
        "llama-3.3-70b-versatile",
        model_provider="groq",
        temperature=0.3,
        max_tokens=2048,
    )
    
    return create_agent(
        model=model,
        tools=[],
        system_prompt=SYSTEM_PROMPT,
        response_format=StartupInfo,
    )


# ============================================================
# Core Logic
# ============================================================

def analyze_startup(agent, startup_name: str) -> Optional[StartupInfo]:
    """Analyze a startup and return structured information."""
    try:
        result = agent.invoke({
            "messages": [{
                "role": "user",
                "content": f"Provide a detailed investment analysis of the startup '{startup_name}'."
            }]
        })
        return result["structured_response"]
    except Exception as e:
        print(f"❌ Error analyzing {startup_name}: {e}")
        return None


def print_analysis(summary: StartupInfo) -> None:
    """Pretty-print the startup analysis."""
    print("\n" + "=" * 60)
    print(f"📊 ANALYSIS: {summary.name}")
    print("=" * 60)
    print(f"🏢 Industry:        {summary.industry}")
    print(f"📝 Description:     {summary.one_line_description}")
    print(f"📅 Founded:         {summary.founded_year or 'Unknown'}")
    print(f"📍 HQ:              {summary.headquarters}")
    print(f"🎯 Market:          {summary.target_market}")
    
    valuation = f"${summary.estimated_valuation_usd_millions}M" if summary.estimated_valuation_usd_millions else "Unknown"
    print(f"💰 Valuation:       {valuation}")
    
    print(f"🛍️  Key Products:    {', '.join(summary.key_products)}")
    print(f"⚔️  Competitors:     {', '.join(summary.main_competitors)}")
    print(f"⚠️  Risk Level:      {summary.risk_level.upper()}")
    print(f"💎 Potential:       {summary.investment_potential.upper()}")
    print(f"\n🤔 Reasoning:\n   {summary.reasoning}")
    print("=" * 60)


# ============================================================
# Main
# ============================================================

def main():
    """Run the startup analyzer in interactive mode."""
    print("🚀 PitchProbe - Startup Analyzer (Part 1)")
    print("=" * 60)
    
    agent = create_analysis_agent()
    
    while True:
        startup_name = input("\n💬 Enter startup name (or 'quit' to exit): ").strip()
        
        if startup_name.lower() in ['quit', 'exit', 'q', '']:
            print("\n👋 Goodbye!")
            break
        
        print(f"\n⏳ Analyzing {startup_name}...")
        summary = analyze_startup(agent, startup_name)
        
        if summary:
            print_analysis(summary)
        else:
            print("⚠️  Analysis failed. Please try another startup.")


if __name__ == "__main__":
    main()