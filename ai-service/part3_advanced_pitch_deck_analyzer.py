"""
Part 3 Advanced of PitchProbe - Enhanced Pitch Deck Analyzer
=============================================================
Builds on the base Part 3 RAG pipeline with production enhancements:

1. Retrieval debugging (see which chunks were retrieved)
2. Similarity scores (confidence measurement)
3. MMR retrieval option (diverse results)
4. Structured RAG output (Pydantic schema for answers)
5. Multi-document support (load multiple PDFs, filter by source)

Pipeline:
  PDF(s) -> Loader -> Splitter -> Embeddings -> Chroma -> Retriever -> LLM -> Structured Answer
"""

import os
import warnings
from typing import Optional, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.documents import Document

load_dotenv()


# ============================================================
# Configuration
# ============================================================

# You can add multiple PDFs here — each will be tagged with a label
PDF_SOURCES = {
    "Airbnb Pitch Deck": r"C:\Users\sriva\Downloads\Pitch-Example-Air-BnB-PDF_rotated.pdf",
    # Add more PDFs here in the future:
    # "Uber Pitch Deck": r"C:\path\to\uber_pitch.pdf",
    # "Company X Financial Report": r"C:\path\to\report.pdf",
}

PERSIST_DIR = "./chroma_db_advanced"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 4

EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "llama-3.3-70b-versatile"


# ============================================================
# Structured Output Schema for RAG Answers
# ============================================================

class PitchDeckAnswer(BaseModel):
    """Structured answer from pitch deck analysis."""

    answer: str = Field(
        description="The detailed answer to the user's question, grounded in the pitch deck context"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description=(
            "How confident you are in this answer based on the available context. "
            "'high' = context directly addresses the question, "
            "'medium' = context partially addresses it, "
            "'low' = context barely relates to the question"
        )
    )
    key_facts: list[str] = Field(
        description="Specific facts, numbers, or quotes extracted from the context that support the answer"
    )
    source_summary: str = Field(
        description="Brief summary of which parts of the pitch deck the answer came from"
    )


# ============================================================
# Prompt Template (Enhanced for Structured Output)
# ============================================================

SYSTEM_PROMPT = """You are an expert startup analyst helping a VC analyze pitch decks.

Your task is to answer questions about a startup's pitch deck using ONLY the provided context.

Rules:
- Answer ONLY from the provided context. Do not use outside knowledge.
- If the answer is not in the context, say so clearly and set confidence to "low".
- Quote specific numbers, names, or facts when relevant.
- Extract key facts as a list.
- Assess your confidence honestly based on how well the context addresses the question.
- Summarize which parts of the pitch deck your answer draws from.
"""


# ============================================================
# Step 1: Load and Split PDFs (Multi-Document Support)
# ============================================================

def load_and_split_pdfs(pdf_sources: dict[str, str]) -> list[Document]:
    """
    Load multiple PDFs and split them into chunks.
    Each chunk gets a 'document_name' metadata field so we can filter later.

    Args:
        pdf_sources: dict mapping document labels to file paths
                     e.g., {"Airbnb Pitch Deck": "/path/to/airbnb.pdf"}

    Returns:
        list[Document] — all chunks from all PDFs, with metadata
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_chunks = []

    for doc_name, path in pdf_sources.items():
        print(f"\n📄 Loading: {doc_name}")
        print(f"   Path: {path}")

        # --- Load ---
        loader = PyPDFLoader(path)
        pages = loader.load()
        print(f"   Loaded {len(pages)} pages.")

        # --- Split ---
        chunks = splitter.split_documents(pages)
        print(f"   Created {len(chunks)} chunks.")

        # --- Safety check ---
        if len(chunks) == 0:
            total_text = sum(len(p.page_content.strip()) for p in pages)
            print(f"   ⚠️  WARNING: 0 chunks created! Total text: {total_text} chars.")
            print(f"   This PDF is likely image-based. Skipping it.")
            continue

        # --- Add custom metadata to every chunk ---
        # This is how we tag chunks so we can filter by document later
        for chunk in chunks:
            chunk.metadata["document_name"] = doc_name

        all_chunks.extend(chunks)

    print(f"\n📊 Total chunks across all documents: {len(all_chunks)}")

    if len(all_chunks) == 0:
        raise ValueError("No chunks were created from any PDF. Cannot build vector store.")

    return all_chunks


# ============================================================
# Step 2: Build or Load the Vector Store
# ============================================================

def get_vector_store() -> Chroma:
    """
    Return a persistent Chroma vector store.
    Builds from PDFs on first run, loads from disk on subsequent runs.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        print(f"\n💾 Loading existing vector store from: {PERSIST_DIR}")
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )
        print("   Vector store loaded successfully.")
    else:
        print("\n🔨 No existing vector store found. Building a new one...")
        chunks = load_and_split_pdfs(PDF_SOURCES)

        print(f"\n🧠 Embedding {len(chunks)} chunks with {EMBEDDING_MODEL}...")
        print("   (This happens only once — subsequent runs will load from disk.)")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
        )
        print(f"✅ Vector store built and saved to: {PERSIST_DIR}")

    return vector_store


# ============================================================
# Step 3: Retrieve with Debugging & Scores
# ============================================================

def retrieve_with_debug(
    question: str,
    vector_store: Chroma,
    search_type: str = "similarity",
    k: int = TOP_K,
    show_debug: bool = True,
) -> list[tuple[Document, float]]:
    """
    Retrieve relevant chunks and optionally print debug information.

    Uses similarity_search_with_score() instead of a retriever so we
    can access the similarity scores for transparency.

    Args:
        question: the user's question
        vector_store: the Chroma vector store
        search_type: "similarity" or "mmr"
        k: number of chunks to retrieve
        show_debug: whether to print the retrieved chunks

    Returns:
        list of (Document, score) tuples, sorted by relevance
    """
    if search_type == "mmr":
        # MMR doesn't return scores directly, so we use the retriever
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k * 4, "lambda_mult": 0.5},
        )
        docs = retriever.invoke(question)
        # Pair with placeholder scores (MMR doesn't provide scores)
        results = [(doc, None) for doc in docs]
    else:
        # Similarity search WITH scores — this is the key method
        # Lower score = more similar (it's distance, not similarity)
        results = vector_store.similarity_search_with_score(question, k=k)

    if show_debug:
        print("\n" + "=" * 60)
        print(f"🔍 RETRIEVAL DEBUG ({search_type}, k={k})")
        print("=" * 60)

        for i, (doc, score) in enumerate(results):
            page = doc.metadata.get("page", "?")
            doc_name = doc.metadata.get("document_name", "Unknown")
            content_preview = doc.page_content[:150].replace("\n", " ")

            print(f"\n  Chunk {i + 1}:")
            if score is not None:
                print(f"    📏 Distance Score: {score:.4f} (lower = more relevant)")
            else:
                print(f"    📏 Score: N/A (MMR mode)")
            print(f"    📄 Source: {doc_name}, Page {page + 1 if isinstance(page, int) else page}")
            print(f"    📝 Preview: {content_preview}...")

        print("\n" + "-" * 60)

    return results


# ============================================================
# Step 4: Build Context from Retrieved Chunks
# ============================================================

def build_context(results: list[tuple[Document, float]]) -> str:
    """
    Combine retrieved chunks into a single context string.
    Each chunk is labeled with its source for the LLM's reference.
    """
    context_parts = []

    for i, (doc, score) in enumerate(results):
        page = doc.metadata.get("page", "?")
        doc_name = doc.metadata.get("document_name", "Unknown")

        # Label each chunk so the LLM knows where it came from
        label = f"[Source: {doc_name}, Page {page + 1 if isinstance(page, int) else page}]"
        context_parts.append(f"{label}\n{doc.page_content}")

    return "\n\n---\n\n".join(context_parts)


# ============================================================
# Step 5: Create the Analysis Agent (Structured Output)
# ============================================================

def create_rag_agent():
    """
    Create an agent that produces structured PitchDeckAnswer responses.
    This combines Part 1's structured output with Part 3's RAG.
    """
    model = init_chat_model(
        LLM_MODEL,
        model_provider="groq",
        temperature=0.2,
        max_tokens=1024,
    )

    return create_agent(
        model=model,
        tools=[],  # No tools — this agent only synthesizes retrieved context
        system_prompt=SYSTEM_PROMPT,
        response_format=PitchDeckAnswer,
    )


# ============================================================
# Step 6: The Full Enhanced RAG Flow
# ============================================================

def answer_question(
    question: str,
    vector_store: Chroma,
    agent,
    search_type: str = "similarity",
    show_debug: bool = True,
) -> Optional[PitchDeckAnswer]:
    """
    Execute one full enhanced RAG cycle:
      1. Retrieve top-K chunks (with debug output)
      2. Build labeled context
      3. Send to agent for structured answer
      4. Return structured PitchDeckAnswer

    Returns:
        PitchDeckAnswer or None if an error occurs
    """
    # --- Step 1: Retrieve with debugging ---
    results = retrieve_with_debug(
        question=question,
        vector_store=vector_store,
        search_type=search_type,
        show_debug=show_debug,
    )

    if not results:
        print("⚠️  No chunks retrieved. The pitch deck may not contain relevant info.")
        return None

    # --- Step 2: Build labeled context ---
    context = build_context(results)

    # --- Step 3: Build the user message with context ---
    user_message = f"""Here is the context from the pitch deck:

{context}

Question: {question}"""

    # --- Step 4: Get structured answer from agent ---
    try:
        result = agent.invoke({
            "messages": [{"role": "user", "content": user_message}]
        })
        return result["structured_response"]
    except Exception as e:
        print(f"❌ Agent error: {e}")
        return None


# ============================================================
# Step 7: Pretty Printing (Enhanced)
# ============================================================

def print_answer(answer: PitchDeckAnswer) -> None:
    """Nicely format the structured answer."""
    print("\n" + "=" * 60)
    print("💡 ANSWER")
    print("=" * 60)
    print(answer.answer)

    print(f"\n🎯 Confidence: {answer.confidence.upper()}")

    if answer.key_facts:
        print(f"\n📌 Key Facts:")
        for fact in answer.key_facts:
            print(f"   • {fact}")

    print(f"\n📚 Sources: {answer.source_summary}")
    print("=" * 60)


# ============================================================
# Step 8: Settings Menu
# ============================================================

def get_search_settings() -> dict:
    """
    Let the user configure search settings at startup.
    Returns a dict with search_type and show_debug.
    """
    print("\n⚙️  SEARCH SETTINGS")
    print("-" * 40)
    print("1. Similarity search (default — finds most relevant chunks)")
    print("2. MMR search (diverse results — avoids redundant chunks)")

    choice = input("Choose search type [1/2] (default: 1): ").strip()
    search_type = "mmr" if choice == "2" else "similarity"

    debug_choice = input("Show retrieval debug info? [y/n] (default: y): ").strip().lower()
    show_debug = debug_choice != "n"

    print(f"\n   Search type: {search_type}")
    print(f"   Debug output: {'ON' if show_debug else 'OFF'}")
    print("-" * 40)

    return {"search_type": search_type, "show_debug": show_debug}


# ============================================================
# Main: Interactive Loop
# ============================================================

def main() -> None:
    print("🚀 PitchProbe - Advanced Pitch Deck Analyzer (Part 3)")
    print("=" * 60)

    # Build or load vector store
    vector_store = get_vector_store()

    # Create the structured-output agent
    agent = create_rag_agent()

    # Let the user configure search settings
    settings = get_search_settings()

    print("\n✅ Ready! Ask any question about the pitch deck(s).")
    print("   Type 'quit' to exit.")
    print("   Type 'settings' to change search settings.\n")

    while True:
        question = input("💬 Your question: ").strip()

        if question.lower() in {"quit", "exit", "q", ""}:
            print("\n👋 Goodbye!")
            break

        if question.lower() == "settings":
            settings = get_search_settings()
            continue

        print("\n⏳ Analyzing pitch deck...")
        answer = answer_question(
            question=question,
            vector_store=vector_store,
            agent=agent,
            search_type=settings["search_type"],
            show_debug=settings["show_debug"],
        )

        if answer:
            print_answer(answer)
        else:
            print("⚠️  Could not generate an answer. Try rephrasing your question.")


if __name__ == "__main__":
    main()