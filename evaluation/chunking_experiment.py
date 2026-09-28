"""Compare three chunkers on the fixed synthetic corpus without changing the database."""

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
from math import isfinite, sqrt
from pathlib import Path
from datetime import datetime, timezone

from app.config import get_settings
from app.schemas import DocumentChunk, ExtractedDocument
from app.services.chunking_service import chunk_document
from app.services.document_service import extract_pdf
from app.services.embedding_service import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL
from app.services.voyage_client import VoyageRequestError, embed_texts
from evaluation.runner import CASE_FILE, RECALL_CUTOFFS


PDF_DIR = Path(__file__).resolve().parent / "fixtures" / "pdfs"
DEFAULT_OUTPUT = Path(__file__).with_name("chunking_results.json")
SECTION_HEADING = re.compile(r"^\d+\.\s+\S")
MAX_PARAGRAPH_CHARS = 1000
BATCH_SIZE = 4
CACHE_FILE = Path(__file__).with_name(".chunking_embedding_cache.sqlite")


def paragraph_chunks(document: ExtractedDocument) -> list[DocumentChunk]:
    """Keep numbered policy sections together; split long sections at sentence boundaries."""
    chunks: list[DocumentChunk] = []
    for page in document.pages:
        sections: list[str] = []
        current: list[str] = []
        for line in page.text.splitlines():
            line = line.strip()
            if not line:
                continue
            if SECTION_HEADING.match(line) and current:
                sections.append(" ".join(current))
                current = []
            current.append(line)
        if current:
            sections.append(" ".join(current))

        for section in sections:
            # A section can exceed the size cap; prefer a sentence break, then a word break.
            remaining = section
            while remaining:
                if len(remaining) <= MAX_PARAGRAPH_CHARS:
                    part, remaining = remaining, ""
                else:
                    prefix = remaining[:MAX_PARAGRAPH_CHARS]
                    sentence_end = max(prefix.rfind(". "), prefix.rfind("; "))
                    word_end = prefix.rfind(" ")
                    cut = sentence_end + 1 if sentence_end >= MAX_PARAGRAPH_CHARS // 2 else word_end
                    if cut <= 0:
                        cut = MAX_PARAGRAPH_CHARS
                    part, remaining = remaining[:cut], remaining[cut:].lstrip()
                chunks.append(DocumentChunk(
                    text=part.strip(), chunk_index=len(chunks), page_number=page.page_number
                ))
    return chunks


def embed_batches(texts: list[str], api_key: str, input_type: str, cache: sqlite3.Connection) -> list[list[float]]:
    keys = [hashlib.sha256(f"{EMBEDDING_MODEL}\0{input_type}\0{text}".encode()).hexdigest() for text in texts]
    found = {}
    for key in set(keys):
        row = cache.execute("SELECT vector FROM embeddings WHERE key = ?", (key,)).fetchone()
        if row:
            found[key] = json.loads(row[0])
    missing = [(key, text) for key, text in dict(zip(keys, texts)).items() if key not in found]
    for start in range(0, len(missing), BATCH_SIZE):
        batch = missing[start:start + BATCH_SIZE]
        for attempt in range(5):
            try:
                vectors = embed_texts([text for _, text in batch], api_key, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS, input_type=input_type)
                break
            except VoyageRequestError as error:
                if "HTTP 429" not in str(error) or attempt == 4:
                    raise
                time.sleep(5 * 2 ** attempt)
        if len(vectors) != len(batch):
            raise VoyageRequestError("Invalid experiment embedding count")
        cache.executemany("INSERT INTO embeddings (key, vector) VALUES (?, ?)",
                          [(key, json.dumps(vector)) for (key, _), vector in zip(batch, vectors)])
        cache.commit()
        found.update((key, vector) for (key, _), vector in zip(batch, vectors))
        print(f"Embedded {min(start + BATCH_SIZE, len(missing))}/{len(missing)} missing {input_type} texts", flush=True)
    vectors = [found[key] for key in keys]
    if len(vectors) != len(texts) or any(
        len(vector) != EMBEDDING_DIMENSIONS or not all(isfinite(value) for value in vector)
        for vector in vectors
    ):
        raise VoyageRequestError("Invalid experiment embeddings")
    return vectors


def cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))
    return 1 - dot / (norm_a * norm_b)


def score_strategy(cases: list[dict], chunks: list[dict], chunk_vectors: list[list[float]], query_vectors: list[list[float]]) -> dict:
    per_question = []
    for case, vector in zip(cases, query_vectors):
        ranked = sorted(
            range(len(chunks)),
            key=lambda index: (cosine_distance(vector, chunk_vectors[index]), index),
        )
        top = [chunks[index] for index in ranked[:20]]
        evidence_ranks = [
            next((rank for rank, hit in enumerate(top, 1) if (
                hit["source_file"] == evidence["source_file"]
                and evidence["anchor"].casefold() in hit["text"].casefold()
            )), None)
            for evidence in case["evidence"]
        ]
        per_question.append({
            "id": case["id"],
            "category": case["category"],
            "evidence_ranks": evidence_ranks,
            "recall_at": {
                str(k): round(sum(rank is not None and rank <= k for rank in evidence_ranks) / len(evidence_ranks), 3)
                for k in RECALL_CUTOFFS
            },
            "all_evidence_at": {
                str(k): all(rank is not None and rank <= k for rank in evidence_ranks)
                for k in RECALL_CUTOFFS
            },
        })
    return {
        "chunk_count": len(chunks),
        "mean_chunk_chars": round(sum(len(chunk["text"]) for chunk in chunks) / len(chunks), 1),
        "mean_recall_at": {
            str(k): round(sum(item["recall_at"][str(k)] for item in per_question) / len(per_question), 3)
            for k in RECALL_CUTOFFS
        },
        "all_evidence_at": {
            str(k): round(sum(item["all_evidence_at"][str(k)] for item in per_question) / len(per_question), 3)
            for k in RECALL_CUTOFFS
        },
        "questions": per_question,
    }


def run_experiment() -> dict:
    cases_file = json.loads(CASE_FILE.read_text(encoding="utf-8"))
    cases = cases_file["qa"]
    documents = [extract_pdf(PDF_DIR / filename) for filename in cases_file["corpus"]["source_files"]]
    settings = get_settings()
    api_key = settings.voyage_api_key.get_secret_value() if settings.voyage_api_key else ""
    if not api_key:
        raise ValueError("VOYAGE_API_KEY is required for the experiment")
    cache = sqlite3.connect(CACHE_FILE)
    cache.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector TEXT NOT NULL)")
    query_vectors = embed_batches([case["question"] for case in cases], api_key, "query", cache)
    strategies = {
        "A_500_100": lambda doc: chunk_document(doc, 500, 100),
        "B_1000_200": lambda doc: chunk_document(doc, 1000, 200),
        "C_paragraph": paragraph_chunks,
    }
    results = {}
    for name, chunker in strategies.items():
        chunks = [
            {"source_file": document.source_file, "text": chunk.text, "page_number": chunk.page_number}
            for document in documents for chunk in chunker(document)
        ]
        vectors = embed_batches([chunk["text"] for chunk in chunks], api_key, "document", cache)
        results[name] = score_strategy(cases, chunks, vectors, query_vectors)
    cache.close()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": EMBEDDING_MODEL,
        "corpus": cases_file["corpus"]["source_files"],
        "question_count": len(cases),
        "strategy_definitions": {
            "A_500_100": "500 characters, 100 character overlap, page bounded",
            "B_1000_200": "1000 characters, 200 character overlap, page bounded",
            "C_paragraph": "numbered policy sections, joined lines, max 1000 characters, no overlap, page bounded",
        },
        "metric_note": "Anchor and source-file match; cosine distance; all strategies use the same queries and PDFs.",
        "strategies": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare chunking strategies on the synthetic corpus")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_experiment()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps({name: {k: value[k] for k in ("chunk_count", "mean_recall_at", "all_evidence_at")}
                      for name, value in result["strategies"].items()}))


if __name__ == "__main__":
    main()
