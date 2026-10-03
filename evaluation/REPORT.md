# Evaluation Report

## Summary

**At Top-5, vector search found all required evidence for 41/41 questions; hybrid search did so for 40/41.** The question groups and changed ranks below explain the gains and regressions.

**Current decision:** keep vector search as the default and hybrid as an explicit option. Reranking remains deferred; it was not tested in this comparison.

Results below are from **2026-09-30**, using **41 answerable questions** over five synthetic policy PDFs and 13 synthetic runbooks (**37 chunks**). This is a retrieval comparison, not a measurement of overall answer accuracy.

## 1. Overall retrieval results

**Vector** matches meaning using embeddings. **Lexical** ranks matching words. **Hybrid** combines the two ranked result lists.

**Top-k coverage:** a question passes only when the first k chunks contain every expected source and its labeled text anchor. For a question requiring two sources, finding only one does not pass. Such a question cannot achieve full coverage at Top-1.

| Strategy | Top-1 | Top-3 | Top-5 | Top-20 |
| --- | ---: | ---: | ---: | ---: |
| Vector | 31/41 | 39/41 | 41/41 | 41/41 |
| Lexical | 28/41 | 35/41 | 37/41 | 40/41 |
| Hybrid | 31/41 | 39/41 | 40/41 | 41/41 |

## 2. Where hybrid helped and hurt

| Question group | Questions | Vector Top-1 | Hybrid Top-1 | Vector Top-5 | Hybrid Top-5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Additional multi-source questions | 2 | 0/2 | 0/2 | 2/2 | 2/2 |
| Exact names / terms | 5 | 4/5 | 4/5 | 5/5 | 5/5 |
| Error codes / KB identifiers | 13 | 11/13 | 13/13 | 13/13 | 13/13 |
| Similar terms or deadlines | 4 | 3/4 | 3/4 | 4/4 | 4/4 |
| Original policy questions | 11 | 8/11 | 8/11 | 11/11 | 11/11 |
| Rephrased questions | 6 | 5/6 | 3/6 | 6/6 | 5/6 |

The original-policy group includes two multi-source questions; the additional multi-source group contains two more.

**Examples that explain the trade-off:**

- **GlobalProtect GP1001:** vector rank 2 → hybrid rank 1 (`gp1001`).
- **KB-432:** vector rank 2 → hybrid rank 1 (`kb432`).
- **Unsolicited MFA login prompt:** vector rank 1 → hybrid rank 10 (`random_mfa_prompt`).
- **Screen-lock timeout:** vector rank 4 → hybrid rank 2 (`screen_lock`).
- **Contractor VPN access and MFA:** required evidence ranks [1, 4] → [1, 3] (`contractor_vpn_mfa`).

## 3. Lessons and decisions

1. **Choose the default from measured coverage.** Vector covers 41/41 questions at Top-5, compared with 40/41 for hybrid. The current run does not show an overall Top-5 improvement from equal-weight hybrid fusion.
2. **Hybrid is useful for a specific kind of query.** Exact error codes and KB identifiers benefited at Top-1; that does not establish that hybrid should handle every query. Changes to fusion weights or query routing need a separate held-out comparison before becoming the default.
3. **Equal totals can conceal different failures.** Inspect the misses below, not only the headline score. The agent's search tool defaults to Top-3, so Top-5 success does not guarantee that its default retrieval has all evidence.

| Strategy | Questions missing required evidence at Top-3 |
| --- | --- |
| Vector | `screen_lock`, `contractor_vpn_mfa` |
| Hybrid | `random_mfa_prompt`, `vpn_admin_rights` |

4. **Do not infer a reranking benefit without testing it.** Top-20 coverage is 41/41 for vector and 41/41 for hybrid. When the evidence is present at Top-20 but missing at Top-5, the problem is ranking within the candidate set. This supports investigating fusion quality; it does not prove that reranking would or would not help.

## 4. Retrieval time

| Strategy | Median database retrieval |
| --- | ---: |
| Vector | 49.44 ms |
| Lexical | 54.40 ms |
| Hybrid | 59.68 ms |

Hybrid's median was **10.24 ms higher** than vector's in this run. These are one-run local measurements including connection setup, excluding query embedding and LLM latency. They do not establish production performance.

## 5. Earlier checks — September 28, 2026

These are separate historical checks on the original five-PDF, 11-question dataset. They do not measure hybrid answer quality or validate later chat changes.

| Check | Recorded result | Lesson |
| --- | --- | --- |
| Chunking: 500/100, 1000/200, section-aware | All reached 11/11 at Top-3. Chunk counts: 47, 24, 50; Top-1: 7/11, 8/11, 7/11. | Keep 1000/200: fewer chunks and the best Top-1 result in this small test. |
| RAG answer checks | 11/11 contained expected terms and relevant citations. | Phrase/citation checks passed; they do not verify every claim. |
| Unanswerable question | 1/1 triggered abstention. | One successful example is insufficient to estimate general abstention quality. |
| Agent scenarios | 4/4 passed; ticket writes were mocked. | Confirms these scenarios, not production write reliability. |
| Retrieved prompt injection | Malicious text reached Claude; 0 `create_ticket` calls, with the tool available. | This attack was resisted; one fixture does not establish general attack resistance. |

## Limits

- The corpus and questions are small, synthetic and English. The development set is not an independent held-out or production dataset.
- Retrieval coverage checks source names and text anchors. It does not measure whether the final answer is correct.
- The 41-question comparison uses no LLM or reranker and contains only answerable questions. It does not test hallucinations, abstention or prompt-injection resistance.
- The historical answer, agent and attack checks are separate from the retrieval comparison. Their pass counts must not be combined into a single accuracy score.

<details>
<summary>Metric details and all changed evidence ranks</summary>

**Mean Recall@k** averages the fraction of expected evidence found for each question. Finding one of two required sources contributes 50%, whereas the full-coverage table above counts that question as a miss.

| Strategy | Recall@1 | Recall@3 | Recall@5 | Recall@20 |
| --- | ---: | ---: | ---: | ---: |
| Vector | 78.0% | 96.3% | 100.0% | 100.0% |
| Lexical | 70.7% | 89.0% | 92.7% | 97.6% |
| Hybrid | 79.3% | 96.3% | 97.6% | 100.0% |

| Case ID | Vector ranks | Hybrid ranks |
| --- | --- | --- |
| `lost_laptop_reporting` | [2, 3] | [2, 1] |
| `random_mfa_prompt` | [1] | [10] |
| `ransomware_power` | [2] | [3] |
| `work_backup` | [1] | [2] |
| `screen_lock` | [4] | [2] |
| `vpn_hostname` | [3] | [2] |
| `contractor_vpn_mfa` | [1, 4] | [1, 3] |
| `vpn_admin_rights` | [3, 2] | [5, 2] |
| `gp1001` | [2] | [1] |
| `kb432` | [2] | [1] |

Each rank list follows the labeled evidence order. `None` means the evidence was absent from Top-20.

</details>

<details>
<summary>Method, source data and reproduction</summary>

All strategies use the same selected corpus. Expected source/anchor labels are validated against application chunks before embedding. Chunking is 1000 characters with 200-character overlap; embeddings use Voyage `voyage-4`.

Lexical search is PostgreSQL English full-text search with OR terms and `ts_rank_cd`; it is not BM25. Hybrid fuses up to 20 vector and 20 lexical candidates using equal-weight Reciprocal Rank Fusion (constant 60) and deduplicates chunks. Unrelated documents are excluded. Query embeddings are shared/cached and strategy order rotates per question.

Runbooks contain fictional procedures for evaluation, not vendor troubleshooting guidance.

Recorded comparison: `evaluation/retrieval_comparison.json` (local generated output, excluded from Git). Labels: [`cases.json`](cases.json), [`retrieval_cases.json`](retrieval_cases.json). Runbooks: [`fixtures/runbooks.json`](fixtures/runbooks.json).

Dataset SHA-256: `7e2b895a5322ddec4f61b493006c209bc71503ca2fd6a5eeaba58d8922da041e`

To repeat the retrieval comparison, configure `.env`, start PostgreSQL, then run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m app.init_db
.\.venv\Scripts\python.exe -m evaluation.retrieval_benchmark --output evaluation/retrieval_comparison.json --report evaluation/REPORT.md
```

Repeating the benchmark ingests or reuses the exact fixtures and can call Voyage for missing embeddings. It does not call Claude or create tickets. Generated JSON and embedding caches stay local; the report, labels and fixtures are public.

</details>
