"""Compare retrieval on a fixed challenge set; no LLM calls or ticket writes."""

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter

from app.database import get_connection
from app.repositories import embedding_repository
from app.schemas import DocumentPage, ExtractedDocument, SearchHit
from app.services.chunking_service import chunk_document
from app.services.document_service import extract_pdf, save_extracted_document
from app.services.embedding_service import EMBEDDING_MODEL, EMBEDDING_PROVIDER
from evaluation.chunking_experiment import CACHE_FILE, PDF_DIR, embed_batches
from evaluation.runner import CASE_FILE, RECALL_CUTOFFS, retrieval_metrics


CHALLENGE_FILE = Path(__file__).with_name("retrieval_cases.json")
RUNBOOK_FILE = Path(__file__).with_name("fixtures") / "runbooks.json"


def load_suite() -> tuple[list[dict], list[ExtractedDocument]]:
    original = json.loads(CASE_FILE.read_text(encoding="utf-8"))
    challenge = json.loads(CHALLENGE_FILE.read_text(encoding="utf-8"))
    documents = [extract_pdf(PDF_DIR / name) for name in original["corpus"]["source_files"]]
    documents += [ExtractedDocument(source_file=item["source_file"], pages=[
        DocumentPage(page_number=1, text=item["text"])
    ]) for item in json.loads(RUNBOOK_FILE.read_text(encoding="utf-8"))]
    cases = [{**case, "category": "original_policy"} for case in original["qa"]] + challenge["qa"]
    validate_suite(cases, documents)
    return cases, documents


def validate_suite(cases: list[dict], documents: list[ExtractedDocument]) -> None:
    """Reject labels that cannot be found in an actual application chunk."""
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    chunks = {document.source_file: chunk_document(document) for document in documents}
    for case in cases:
        if not case["question"].strip() or not case["evidence"]:
            raise ValueError(f"Missing question or evidence: {case['id']}")
        for evidence in case["evidence"]:
            anchor = evidence["anchor"].casefold()
            if not anchor or not any(anchor in chunk.text.casefold() for chunk in chunks.get(evidence["source_file"], [])):
                raise ValueError(f"Evidence missing from application chunks: {case['id']}")


def prepare_corpus(documents: list[ExtractedDocument]) -> list[int]:
    """Reuse exact content or ingest fixtures; never delete other documents."""
    document_ids = []
    for document in documents:
        expected_pages = [page.text for page in document.pages]
        expected_chunks = [chunk.text for chunk in chunk_document(document)]
        with get_connection() as connection:
            rows = connection.execute("""
                SELECT d.id,
                    ARRAY(SELECT p.text FROM document_pages p WHERE p.document_id = d.id ORDER BY p.page_number),
                    ARRAY(SELECT c.text FROM document_chunks c WHERE c.document_id = d.id ORDER BY c.chunk_index)
                FROM documents d WHERE d.source_file = %s ORDER BY d.id DESC
            """, (document.source_file,)).fetchall()
        existing = next((row[0] for row in rows if row[1] == expected_pages and row[2] == expected_chunks), None)
        document_ids.append(existing if existing is not None else save_extracted_document(document).id)
    return document_ids


def prepare_vectors(cases: list[dict], document_ids: list[int]) -> tuple[list[list[float]], dict]:
    from app.config import get_settings
    settings = get_settings()
    if not settings.voyage_api_key:
        raise ValueError("VOYAGE_API_KEY is required")
    with get_connection() as connection:
        chunks = connection.execute("SELECT id, text FROM document_chunks WHERE document_id = ANY(%s) ORDER BY id", (document_ids,)).fetchall()
    started = perf_counter()
    with sqlite3.connect(CACHE_FILE) as cache:
        cache.execute("CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector TEXT NOT NULL)")
        vectors = embed_batches([row[1] for row in chunks], settings.voyage_api_key.get_secret_value(), "document", cache)
        embedding_repository.insert_embeddings(EMBEDDING_PROVIDER, EMBEDDING_MODEL, [(row[0], vector) for row, vector in zip(chunks, vectors)])
        queries = embed_batches([case["question"] for case in cases], settings.voyage_api_key.get_secret_value(), "query", cache)
    return queries, {"chunk_count": len(chunks), "embedding_setup_ms": round((perf_counter() - started) * 1000, 2)}


def summarize(items: list[dict]) -> dict:
    """Keep denominators explicit; partial evidence is not full question coverage."""
    count = len(items)
    return {
        "questions": count,
        "mean_recall_at": {str(k): round(sum(item["recall_at"][str(k)] for item in items) / count, 4) if count else None for k in RECALL_CUTOFFS},
        "all_evidence_counts": {str(k): sum(item["all_evidence_at"][str(k)] for item in items) for k in RECALL_CUTOFFS},
        "median_database_ms": round(median(item["database_ms"] for item in items), 2) if count else None,
        "failures_at_5": [item["id"] for item in items if not item["all_evidence_at"]["5"]],
    }


def run(strategies: list[str]) -> dict:
    cases, documents = load_suite()
    ids = prepare_corpus(documents)
    vectors, setup = prepare_vectors(cases, ids)
    fingerprint = hashlib.sha256(json.dumps({"cases": cases, "documents": [d.model_dump() for d in documents]}, sort_keys=True).encode()).hexdigest()
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "dataset_sha256": fingerprint,
        "embedding_model": EMBEDDING_MODEL, "document_ids": ids, **setup,
        "retrieval_settings": {"text_config": "english", "lexical_operator": "OR",
                               "candidates_per_branch": embedding_repository.CANDIDATE_LIMIT,
                               "rrf_constant": embedding_repository.RRF_CONSTANT,
                               "chunk_size_chars": 1000, "overlap_chars": 200},
        "timing_note": "Database retrieval only, including connection setup. Query embeddings cached and shared; not end-to-end request latency.",
        "strategies": {},
    }
    results = {strategy: [] for strategy in strategies}
    for index, (case, vector) in enumerate(zip(cases, vectors)):
        # Rotate execution order to reduce systematic warm-cache advantage.
        order = strategies[index % len(strategies):] + strategies[:index % len(strategies)]
        for strategy in order:
            started = perf_counter()
            rows = retrieve(strategy, case["question"], vector, ids)
            elapsed = (perf_counter() - started) * 1000
            hits = [SearchHit.model_validate(row) for row in rows]
            results[strategy].append({
                "id": case["id"], "category": case["category"], "question": case["question"],
                "evidence": case["evidence"], **retrieval_metrics(case, hits), "database_ms": round(elapsed, 2),
                "candidates": [hit.model_dump() for hit in hits],
            })
    for strategy, items in results.items():
        report["strategies"][strategy] = {"summary": summarize(items), "by_category": {
            category: summarize([item for item in items if item["category"] == category])
            for category in sorted({case["category"] for case in cases})
        }, "cases": items}
    return report


def retrieve(strategy: str, question: str, vector: list[float], document_ids: list[int]) -> list[dict]:
    if strategy == "vector":
        return embedding_repository.search_chunks_for_documents(vector, EMBEDDING_PROVIDER, EMBEDDING_MODEL, 20, document_ids)
    return embedding_repository.search_ranked_chunks(question, vector, EMBEDDING_PROVIDER, EMBEDDING_MODEL, 20, strategy, document_ids)


def render_report(report: dict) -> str:
    """Explain the recorded comparison; all comparison scores come from its data."""
    strategies = report["strategies"]
    vector, hybrid = strategies["vector"], strategies["hybrid"]
    total = vector["summary"]["questions"]
    vector_counts = vector["summary"]["all_evidence_counts"]
    hybrid_counts = hybrid["summary"]["all_evidence_counts"]
    hybrid_cases = {case["id"]: case for case in hybrid["cases"]}
    vector_cases = {case["id"]: case for case in vector["cases"]}
    top5_lesson = (
        "The current run does not show an overall Top-5 improvement from equal-weight hybrid fusion."
        if hybrid_counts["5"] <= vector_counts["5"] else
        "Hybrid improved Top-5 coverage in this run; review the default using a separate held-out comparison."
    )
    lines = [
        "# Evaluation Report", "",
        "## Summary", "",
        f"**At Top-5, vector search found all required evidence for {vector_counts['5']}/{total} questions; "
        f"hybrid search did so for {hybrid_counts['5']}/{total}.** "
        "The question groups and changed ranks below explain the gains and regressions.", "",
        "**Current decision:** keep vector search as the default and hybrid as an explicit option. "
        "Reranking remains deferred; it was not tested in this comparison.", "",
        f"Results below are from **{report['generated_at'][:10]}**, using **{total} answerable questions** "
        f"over five synthetic policy PDFs and 13 synthetic runbooks (**{report['chunk_count']} chunks**). "
        "This is a retrieval comparison, not a measurement of overall answer accuracy.", "",
        "## 1. Overall retrieval results", "",
        "**Vector** matches meaning using embeddings. **Lexical** ranks matching words. "
        "**Hybrid** combines the two ranked result lists.", "",
        "**Top-k coverage:** a question passes only when the first k chunks contain every expected source "
        "and its labeled text anchor. For a question requiring two sources, finding only one does not pass. "
        "Such a question cannot achieve full coverage at Top-1.", "",
        "| Strategy | Top-1 | Top-3 | Top-5 | Top-20 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, result in strategies.items():
        counts = result["summary"]["all_evidence_counts"]
        count = result["summary"]["questions"]
        lines.append(f"| {name.title()} | " + " | ".join(f"{counts[str(k)]}/{count}" for k in RECALL_CUTOFFS) + " |")
    lines += ["", "## 2. Where hybrid helped and hurt", ""]
    groups = vector["by_category"]
    if groups:
        lines += ["| Question group | Questions | Vector Top-1 | Hybrid Top-1 | Vector Top-5 | Hybrid Top-5 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        labels = {"identifier": "Error codes / KB identifiers", "paraphrase": "Rephrased questions",
                  "near_match": "Similar terms or deadlines", "exact_term": "Exact names / terms",
                  "original_policy": "Original policy questions", "cross_document": "Additional multi-source questions"}
        for category, group in groups.items():
            v = group["all_evidence_counts"]
            h = hybrid["by_category"][category]["all_evidence_counts"]
            n = group["questions"]
            lines.append(f"| {labels.get(category, category)} | {n} | {v['1']}/{n} | {h['1']}/{n} | {v['5']}/{n} | {h['5']}/{n} |")
        lines += ["", "The original-policy group includes two multi-source questions; the additional multi-source group contains two more.", ""]
    lines += ["**Examples that explain the trade-off:**", ""]
    for case_id, label in (("gp1001", "GlobalProtect GP1001"), ("kb432", "KB-432"),
                           ("random_mfa_prompt", "Unsolicited MFA login prompt"),
                           ("screen_lock", "Screen-lock timeout"), ("contractor_vpn_mfa", "Contractor VPN access and MFA")):
        if case_id in vector_cases and case_id in hybrid_cases:
            v = vector_cases[case_id]["evidence_ranks"]
            h = hybrid_cases[case_id]["evidence_ranks"]
            if len(v) == len(h) == 1:
                lines.append(f"- **{label}:** vector rank {v[0]} → hybrid rank {h[0]} (`{case_id}`).")
            else:
                lines.append(f"- **{label}:** required evidence ranks {v} → {h} (`{case_id}`).")
    lines += ["", "## 3. Lessons and decisions", "",
              "1. **Choose the default from measured coverage.** "
              f"Vector covers {vector_counts['5']}/{total} questions at Top-5, compared with {hybrid_counts['5']}/{total} for hybrid. "
              + top5_lesson,
              "2. **Hybrid is useful for a specific kind of query.** Exact error codes and KB identifiers benefited at Top-1; "
              "that does not establish that hybrid should handle every query. Changes to fusion weights or query routing "
              "need a separate held-out comparison before becoming the default.",
              "3. **Equal totals can conceal different failures.** Inspect the misses below, not only the headline score. "
              "The agent's search tool defaults to Top-3, so Top-5 success does not guarantee that its default retrieval has all evidence.", "",
              "| Strategy | Questions missing required evidence at Top-3 |", "| --- | --- |"]
    for name in ("vector", "hybrid"):
        misses = [f"`{case['id']}`" for case in strategies[name]["cases"] if not case["all_evidence_at"]["3"]]
        lines.append(f"| {name.title()} | {', '.join(misses) or 'None'} |")
    lines += ["",
              "4. **Do not infer a reranking benefit without testing it.** "
              f"Top-20 coverage is {vector_counts['20']}/{total} for vector and {hybrid_counts['20']}/{total} for hybrid. "
              "When the evidence is present at Top-20 but missing at Top-5, the problem is ranking within the candidate set. "
              "This supports investigating fusion quality; it does not prove that reranking would or would not help.", "",
              "## 4. Retrieval time", "",
              "| Strategy | Median database retrieval |", "| --- | ---: |"]
    for name, result in strategies.items():
        lines.append(f"| {name.title()} | {result['summary']['median_database_ms']:.2f} ms |")
    difference = hybrid["summary"]["median_database_ms"] - vector["summary"]["median_database_ms"]
    lines += ["", f"Hybrid's median was **{abs(difference):.2f} ms {'higher' if difference >= 0 else 'lower'}** than vector's in this run. "
              "These are one-run local measurements including connection setup, excluding query embedding and LLM latency. "
              "They do not establish production performance.", "",
              "## 5. Earlier checks — September 28, 2026", "",
              "These are separate historical checks on the original five-PDF, 11-question dataset. "
              "They do not measure hybrid answer quality or validate later chat changes.", "",
              "| Check | Recorded result | Lesson |", "| --- | --- | --- |",
              "| Chunking: 500/100, 1000/200, section-aware | All reached 11/11 at Top-3. Chunk counts: 47, 24, 50; Top-1: 7/11, 8/11, 7/11. | Keep 1000/200: fewer chunks and the best Top-1 result in this small test. |",
              "| RAG answer checks | 11/11 contained expected terms and relevant citations. | Phrase/citation checks passed; they do not verify every claim. |",
              "| Unanswerable question | 1/1 triggered abstention. | One successful example is insufficient to estimate general abstention quality. |",
              "| Agent scenarios | 4/4 passed; ticket writes were mocked. | Confirms these scenarios, not production write reliability. |",
              "| Retrieved prompt injection | Malicious text reached Claude; 0 `create_ticket` calls, with the tool available. | This attack was resisted; one fixture does not establish general attack resistance. |", "",
              "## Limits", "",
              "- The corpus and questions are small, synthetic and English. The development set is not an independent held-out or production dataset.",
              "- Retrieval coverage checks source names and text anchors. It does not measure whether the final answer is correct.",
              "- The 41-question comparison uses no LLM or reranker and contains only answerable questions. It does not test hallucinations, abstention or prompt-injection resistance.",
              "- The historical answer, agent and attack checks are separate from the retrieval comparison. Their pass counts must not be combined into a single accuracy score.", "",
              "<details>", "<summary>Metric details and all changed evidence ranks</summary>", "",
              "**Mean Recall@k** averages the fraction of expected evidence found for each question. "
              "Finding one of two required sources contributes 50%, whereas the full-coverage table above counts that question as a miss.", "",
              "| Strategy | Recall@1 | Recall@3 | Recall@5 | Recall@20 |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, result in strategies.items():
        recalls = result["summary"]["mean_recall_at"]
        lines.append(f"| {name.title()} | " + " | ".join(f"{100 * recalls[str(k)]:.1f}%" for k in RECALL_CUTOFFS) + " |")
    lines += ["", "| Case ID | Vector ranks | Hybrid ranks |", "| --- | --- | --- |"]
    for case in vector["cases"]:
        other = hybrid_cases[case["id"]]
        if case["evidence_ranks"] != other["evidence_ranks"]:
            lines.append(f"| `{case['id']}` | {case['evidence_ranks']} | {other['evidence_ranks']} |")
    lines += ["", "Each rank list follows the labeled evidence order. `None` means the evidence was absent from Top-20.", "",
              "</details>", "", "<details>", "<summary>Method, source data and reproduction</summary>", "",
              "All strategies use the same selected corpus. Expected source/anchor labels are validated against application chunks "
              "before embedding. Chunking is 1000 characters with 200-character overlap; embeddings use Voyage `voyage-4`.", "",
              "Lexical search is PostgreSQL English full-text search with OR terms and `ts_rank_cd`; it is not BM25. "
              "Hybrid fuses up to 20 vector and 20 lexical candidates using equal-weight Reciprocal Rank Fusion (constant 60) "
              "and deduplicates chunks. Unrelated documents are excluded. Query embeddings are shared/cached and strategy order rotates per question.", "",
              "Runbooks contain fictional procedures for evaluation, not vendor troubleshooting guidance.", "",
              "Recorded comparison: `evaluation/retrieval_comparison.json` (local generated output, excluded from Git). "
              "Labels: [`cases.json`](cases.json), [`retrieval_cases.json`](retrieval_cases.json). "
              "Runbooks: [`fixtures/runbooks.json`](fixtures/runbooks.json).", "",
              f"Dataset SHA-256: `{report['dataset_sha256']}`", "",
              "To repeat the retrieval comparison, configure `.env`, start PostgreSQL, then run from the repository root:", "",
              "```powershell", ".\\.venv\\Scripts\\python.exe -m app.init_db",
              ".\\.venv\\Scripts\\python.exe -m evaluation.retrieval_benchmark --output evaluation/retrieval_comparison.json --report evaluation/REPORT.md", "```", "",
              "Repeating the benchmark ingests or reuses the exact fixtures and can call Voyage for missing embeddings. "
              "It does not call Claude or create tickets. Generated JSON and embedding caches stay local; the report, labels and fixtures are public.", "",
              "</details>"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategies", nargs="+", choices=["vector", "lexical", "hybrid"], default=["vector", "lexical", "hybrid"])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("retrieval_comparison.json"))
    parser.add_argument("--report", type=Path, help="Write the readable Markdown report for all three strategies")
    args = parser.parse_args()
    if args.report and set(args.strategies) != {"vector", "lexical", "hybrid"}:
        parser.error("--report requires vector, lexical and hybrid")
    report = run(list(dict.fromkeys(args.strategies)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report(report), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps({strategy: result["summary"] for strategy, result in report["strategies"].items()}))


if __name__ == "__main__":
    main()
