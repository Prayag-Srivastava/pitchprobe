"""
Part 3 of PitchProbe - Pitch Deck Analyzer (2-Step RAG)
========================================================
Loads a pitch deck PDF, indexes it into a persistent ChromaDB vector store,
and lets the user ask questions about it using Retrieval-Augmented Generation.

Pipeline:
  PDF -> Loader -> Splitter -> Embeddings -> Chroma -> Retriever -> LLM -> Answer
"""

import os
from typing import Optional

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document

load_dotenv()


# ============================================================
# Configuration
# ============================================================

PDF_PATH = r"C:\Users\sriva\Downloads\Pitch-Example-Air-BnB-PDF_rotated.pdf"
PERSIST_DIR = "./chroma_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 4  # number of chunks to retrieve per query

EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "llama-3.3-70b-versatile"


# ============================================================
# Prompt Template
# ============================================================

RAG_PROMPT_TEMPLATE = """You are an expert startup analyst helping a VC analyze a pitch deck.

Answer the question based ONLY on the following context from the pitch deck.

Rules:
- If the answer is not in the context, say "This information is not available in the pitch deck."
- Quote specific numbers, names, or facts from the context when relevant.
- Be concise but thorough.
- Do not use outside knowledge.

Context from pitch deck:
{context}

Question: {question}

Answer:"""


# ============================================================
# Step 1: Load and Split the PDF
# ============================================================

def load_and_split_pdf(path: str) -> list[Document]:
    """
    Load a PDF file and split it into small overlapping chunks.

    Returns a list of Document objects ready to be embedded and stored.
    """
    print(f"📄 Loading PDF from: {path}")
    loader = PyPDFLoader(path)
    pages = loader.load()  # list[Document], one Document per page
    print(f"   Loaded {len(pages)} pages.")

    print(f"✂️  Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)
    print(f"   Created {len(chunks)} chunks.")

    # Safety check: detect image-based PDFs
    if len(chunks) == 0:
        total_text = sum(len(p.page_content.strip()) for p in pages)
        raise ValueError(
            f"\n❌ PDF loaded {len(pages)} pages but produced 0 chunks.\n"
            f"   Total extractable text: {total_text} characters.\n"
            f"   This PDF is likely image-based (scanned or vector graphics).\n"
            f"   PyPDFLoader cannot read text inside images.\n"
            f"   Fix: use a text-based PDF, or use OCR (covered in a later phase).\n"
        )

    return chunks


# ============================================================
# Step 2: Build or Load the Vector Store (smart "build once" pattern)
# ============================================================

def get_vector_store() -> Chroma:
    """
    Return a Chroma vector store.

    If a persisted store exists on disk, load it (fast).
    Otherwise, build it from the PDF for the first time (slow, one-time cost).
    """
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    # Check if Chroma has already been built and persisted
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        print(f"💾 Loading existing vector store from: {PERSIST_DIR}")
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )
        print("   Vector store loaded.")
    else:
        print("🔨 No existing vector store found. Building a new one...")
        chunks = load_and_split_pdf(PDF_PATH)

        print(f"🧠 Embedding {len(chunks)} chunks with {EMBEDDING_MODEL}...")
        print("   (This will take a few seconds — happens only once.)")
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
        )
        print(f"✅ Vector store built and saved to: {PERSIST_DIR}")

    return vector_store


# ============================================================
# Step 3: Build the Grounded Prompt
# ============================================================

def build_grounded_prompt(context: str, question: str) -> str:
    """Insert retrieved context + user question into the prompt template."""
    return RAG_PROMPT_TEMPLATE.format(context=context, question=question)


# ============================================================
# Step 4: The Full RAG Flow (retrieve + generate)
# ============================================================

def answer_question(question: str, retriever, llm) -> dict:
    """
    Execute one full 2-Step RAG cycle:
      1. Retrieve top-K relevant chunks
      2. Build a grounded prompt
      3. Send to the LLM
      4. Return the answer + source page numbers

    Returns:
        {
            "answer": str,
            "source_pages": list[int],
            "num_chunks_used": int
        }
    """
    # --- Step 1: Retrieve ---
    retrieved_docs: list[Document] = retriever.invoke(question)

    if not retrieved_docs:
        return {
            "answer": "No relevant context found in the pitch deck.",
            "source_pages": [],
            "num_chunks_used": 0,
        }

    # --- Step 2: Combine chunks into a single context string ---
    context = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)

    # --- Step 3: Build the grounded prompt ---
    prompt = build_grounded_prompt(context, question)

    # --- Step 4: Generate the answer ---
    response = llm.invoke(prompt)

    # --- Step 5: Extract source page numbers from metadata ---
    source_pages = sorted({
        doc.metadata.get("page", "?") for doc in retrieved_docs
    })

    return {
        "answer": response.content,
        "source_pages": source_pages,
        "num_chunks_used": len(retrieved_docs),
    }


# ============================================================
# Step 5: Pretty Printing
# ============================================================

def print_answer(result: dict) -> None:
    """Nicely format the answer + sources for the user."""
    print("\n" + "=" * 60)
    print("💡 ANSWER")
    print("=" * 60)
    print(result["answer"])
    print()
    print("-" * 60)
    pages_str = ", ".join(str(p + 1) for p in result["source_pages"])  # +1 to make 1-indexed for humans
    print(f"📚 Sources: page(s) {pages_str}  |  {result['num_chunks_used']} chunks used")
    print("=" * 60)


# ============================================================
# Main: Interactive Loop
# ============================================================

def main() -> None:
    print("🚀 PitchProbe - Pitch Deck Analyzer (Part 3)")
    print("=" * 60)

    # Build or load the vector store
    vector_store = get_vector_store()

    # Wrap the vector store in a retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

    # Initialize the answer-generating LLM (Groq, same as Part 1 & 2)
    llm = init_chat_model(
        LLM_MODEL,
        model_provider="groq",
        temperature=0.2,   # low temperature => factual, grounded answers
        max_tokens=1024,
    )

    print("\n✅ Ready! Ask any question about the pitch deck.")
    print("   Type 'quit' to exit.\n")

    while True:
        question = input("💬 Your question: ").strip()

        if question.lower() in {"quit", "exit", "q", ""}:
            print("\n👋 Goodbye!")
            break

        print("\n⏳ Searching pitch deck and generating answer...")
        try:
            result = answer_question(question, retriever, llm)
            print_answer(result)
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()