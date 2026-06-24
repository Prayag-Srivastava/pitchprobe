# 🚀 PitchProbe

> Production-grade AI-powered startup due diligence platform built with LangChain, LangGraph, and modern agentic AI architecture.

## What It Is

PitchProbe is a multi-agent AI system that analyzes startups for venture capital due diligence. It coordinates specialized AI agents (Market Research, Risk Analysis, Team Analysis) to produce structured investment recommendations grounded in pitch deck content and real-time web research.

## Architecture

- **Multi-Agent Orchestration:** LangGraph supervisor pattern with parallel specialist agents
- **Document-Aware RAG:** ChromaDB + Gemini embeddings with metadata filtering
- **Live Streaming:** Real-time progress events and token-level narrative streaming
- **Dual-LLM Synthesis:** Structured data layer + streamed narrative layer
- **Production Patterns:** Middleware, retry logic, graceful degradation, score thresholding

## Tech Stack

**AI/ML:**
- LangChain (agents, tools, RAG)
- LangGraph (orchestration, state management, streaming)
- Groq Llama 3.3 70B (reasoning/generation)
- Google Gemini (embeddings)
- ChromaDB (vector store)
- Tavily (web search)

**Infrastructure (in progress):**
- FastAPI (AI service)
- Express.js + Socket.io (backend)
- Next.js (frontend)
- PostgreSQL + Redis
- Docker

## Project Status

🚧 **In active development.** Following a 15-part learning roadmap.

| Part | Topic | Status |
|------|-------|--------|
| 1 | Foundation (models, structured output) | ✅ Complete |
| 2 | Tools & tool calling | ✅ Complete |
| 3 | RAG pipeline | ✅ Complete |
| 4 | Single-agent deep dive | ✅ Complete |
| 5 | Multi-agent with LangGraph | ✅ Complete |
| 6 | Streaming | ✅ Complete |
| 7 | Memory & persistence | 🚧 Next |
| 8 | Human-in-the-loop | ⏳ Planned |
| 9 | Production features | ⏳ Planned |
| 10 | Observability (LangSmith) | ⏳ Planned |
| 11-15 | Backend, frontend, deployment | ⏳ Planned |

## Setup

```bash
# Clone
git clone https://github.com/Prayag-Srivastava/pitchprobe.git
cd pitchprobe/ai-service

# Virtual environment
python -m venv venv
.\venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install langchain langchain-core langchain-community langchain-chroma chromadb \
            langchain-text-splitters langchain-google-genai langchain-tavily \
            langchain-groq langgraph pypdf python-dotenv pydantic

# Configure API keys
cp .env.example .env
# Edit .env and add your GROQ_API_KEY, GOOGLE_API_KEY, TAVILY_API_KEY

# Build the vector store (one-time, from a pitch deck PDF)
python part3_advanced_pitch_deck_analyzer.py

# Run the full multi-agent streaming system
python part6_6_full_streaming_pitchprobe.py