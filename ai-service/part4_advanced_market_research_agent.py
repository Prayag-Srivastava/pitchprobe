"""
Part 4 Advanced — Market Research Agent DEMO
=============================================
Interactive demo that uses the reusable Market Research Agent
from pitchprobe_agents/market_agent.py.

The agent itself lives in the module. This file is just a runner.
"""

from pitchprobe_agents.market_agent import (
    create_market_research_agent,
    MarketResearchReport,
)


def print_tool_calls(result):
    print("\n🔧 Tool Calls Made:")
    print("-" * 60)
    count = 0
    for msg in result.get("messages", []):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                count += 1
                print(f"  {count}. {tc['name']}({tc['args']})")
    if count == 0:
        print("  ⚠️  No tools called.")
    print("-" * 60)


def print_report(report: MarketResearchReport):
    print("\n" + "=" * 60)
    print(f"📊 MARKET RESEARCH: {report.startup_name}")
    print("=" * 60)
    print(f"Industry:        {report.industry}")
    print(f"Market Size:     {report.market_size_usd or 'Unknown'}")
    print(f"Deck Available:  {report.deck_available}")
    print(f"Confidence:      {report.confidence.upper()}")
    print(f"\nGrowth Trends:")
    for t in report.growth_trends:
        print(f"  • {t}")
    print(f"\nTarget Customers:")
    for c in report.target_customers:
        print(f"  • {c}")
    print(f"\nCompetitors:")
    for c in report.main_competitors:
        print(f"  • {c}")
    print(f"\nOpportunities:")
    for o in report.market_opportunities:
        print(f"  • {o}")
    print(f"\nRisks:")
    for r in report.market_risks:
        print(f"  • {r}")
    print(f"\nSources:")
    for s in report.sources:
        print(f"  • {s}")
    print("=" * 60)


def main():
    print("🚀 PitchProbe — Market Research Agent (Part 4 Advanced Demo)")
    print("=" * 60)
    
    agent = create_market_research_agent()
    
    while True:
        startup = input("\n💬 Enter startup name (or 'quit'): ").strip()
        if startup.lower() in {"quit", "exit", "q", ""}:
            print("👋 Goodbye!")
            break
        
        print(f"\n⏳ Researching {startup}...\n")
        try:
            result = agent.invoke({
                "messages": [{"role": "user", "content": f"Analyze the market for {startup}"}]
            })
            print_tool_calls(result)
            report = result.get("structured_response")
            if report:
                print_report(report)
            else:
                print("⚠️  Agent did not produce a structured report.")
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()