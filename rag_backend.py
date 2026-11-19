import hashlib
import os
from datetime import date
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path(__file__).resolve().parent
CHROMA_DIR = PROJECT_DIR / "data" / "chroma_db"
COLLECTION_NAME = "pdf_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RESULT_COUNT = 5

load_dotenv(PROJECT_DIR / ".env")


@lru_cache(maxsize=1)
def _get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _split_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks


def ingest_pdfs(pdfs: list[tuple[str, bytes]]) -> int:
    documents = []
    metadatas = []
    ids = []

    for filename, pdf_bytes in pdfs:
        pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
        reader = PdfReader(BytesIO(pdf_bytes))
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            for chunk_number, chunk in enumerate(_split_text(page_text)):
                chunk_id = hashlib.sha256(
                    f"{pdf_hash}:{page_number}:{chunk_number}".encode("utf-8")
                ).hexdigest()
                documents.append(chunk)
                metadatas.append({"source": filename, "page": page_number})
                ids.append(chunk_id)

    if not documents:
        raise ValueError("No readable text was found in the uploaded PDF files.")

    embeddings = _get_embedding_model().encode(documents).tolist()
    _get_collection().upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    return len(documents)


def get_indexed_chunk_count() -> int:
    return _get_collection().count()


def _extract_response_text(content: object) -> str:
    if isinstance(content, str):
        return content

    blocks = [content] if isinstance(content, dict) else content
    if isinstance(blocks, list):
        text_parts = []
        for block in blocks:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    text_parts.append(text)

        text = "\n".join(part for part in text_parts if part.strip()).strip()
        if text:
            return text

    raise TypeError("Gemini returned no textual content.")


def answer_question(question: str) -> tuple[str, list[dict[str, str | int]]]:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_actual_gemini_api_key":
        raise ValueError(
            "Set GOOGLE_API_KEY to your Gemini API key in the project's .env file."
        )

    collection = _get_collection()
    if collection.count() == 0:
        raise ValueError("Upload and process at least one PDF before asking a question.")

    query_embedding = _get_embedding_model().encode([question])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(RESULT_COUNT, collection.count()),
        include=["documents", "metadatas"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    sources = []
    context_parts = []
    seen_sources = set()

    for document, metadata in zip(documents, metadatas):
        source = metadata["source"]
        page = metadata["page"]
        context_parts.append(f"[{source}, page {page}]\n{document}")
        source_key = (source, page)
        if source_key not in seen_sources:
            sources.append({"source": source, "page": page})
            seen_sources.add(source_key)

    prompt = (
        "Answer the question using only the supplied PDF excerpts. "
        "Cite factual statements with the source and page in the format "
        "[filename, page N]. If the excerpts do not contain the answer, say "
        "that you could not find it in the uploaded PDFs. For age questions, "
        "if a date of birth is provided, calculate the age as of the current "
        "date below instead of repeating an age printed in the PDF. Give the "
        "result in years, months, and days, and show the date of birth and "
        "calculation date. If there is no date of birth, report the age stated "
        "in the PDF and explain that a current age cannot be calculated without "
        "a birth date. Never guess a birth date.\n"
        f"Current date for age calculations: {date.today().isoformat()}.\n\n"
        f"PDF excerpts:\n{chr(10).join(context_parts)}\n\n"
        f"Question: {question}"
    )
    model = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=api_key,
        temperature=0,
    )
    response = model.invoke(prompt)
    return _extract_response_text(response.content), sources