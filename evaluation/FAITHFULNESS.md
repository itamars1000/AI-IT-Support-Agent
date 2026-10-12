# Answer Faithfulness Evaluation

## Summary

**On a held-out set built by a separate model and scored once, the judge flagged 15/15 answers containing an injected error and raised 0 false alarms on 15 unchanged answers.** All 26 real RAG answers it judged were faithful to their sources.

Faithfulness asks whether every claim in an answer is stated by the sources the answer was given. It is not correctness: a claim can be true in the world and still unsupported by the retrieved text. The existing answer checks look for expected terms and a relevant citation, so an answer that is mostly right but adds one invented sentence still passes them.

Results are from **2026-10-11**. Answers: `claude-haiku-4-5-20251001` (the application's default at the time). Judge: `claude-haiku-5-5`. Held-out errors were written by `claude-sonnet-5-5`.

## 1. How an answer is scored

1. The judge splits the answer into atomic claims and labels each one **supported**, **unsupported** (no source states it, including broader or narrower scope) or **contradicted** (a source states the opposite or a different value). Omissions are not faithfulness errors.
2. For supported and contradicted claims, the judge must quote the evidence verbatim and name its source. Structured outputs enforce the response schema.
3. Code checks that each quote exists in the named source, ignoring case, punctuation and PDF line breaks. A missing or invented quote is a **judge error**, and the claim does not count as supported.
4. An answer is faithful only when every claim is verified as supported. Claims without a citation (`uncited`) or with a citation to a different source (`miscited`) are reported separately.

Generation and judging are separate steps, so the judge can be rerun on frozen answers.

## 2. Is the judge reliable?

A schema guarantees the shape of the judge's output, not its correctness. Hand labels alone could not answer the question: all 31 claims hand-labeled from real answers were supported, so a judge that always answers "supported" would also have agreed 31/31. The judge was therefore tested on answers with one known error each.

| Set | Built by | Errors caught | False alarms |
| --- | --- | ---: | ---: |
| Development: 11 questions | Hand-written edits | 16/16 | 0/11 |
| **Held-out A: 15 new questions** | **Separate model, scored once** | **15/15** | **0/15** |
| Hand labels: 31 real claims | Blind labeling | Agreement 31/31 | — |

95% Wilson intervals on held-out A: recall [0.80, 1.0], false-alarm rate [0.0, 0.20].

Each set rotates four error types: changed number, negation, invented fact and broader scope. Every type was caught in full in both sets. The judge's specific label (unsupported or contradicted) matched the expected one for 15/16 development errors; on held-out A this secondary measure is noisy, because some model-written edits did not match their requested type.

## 3. How the held-out set was kept honest

- **The development set was tuned on.** One false alarm there came from a quote that differed from its source only by a comma, so the quote check was changed to ignore punctuation. Its scores are therefore optimistic.
- **The held-out questions were split before any answer existed** and committed to Git first: 30 retrieval-challenge questions never used for faithfulness, balanced by category, with near-identical questions (for example GP1001 and GP1002) kept in the same set.
- **A different model wrote the held-out errors** from the sources and the answer alone, without the judge's prompt. Code rejected edits whose new text already appeared in a source. Only held-out B was reviewed by hand; held-out A was not inspected.
- **Held-out A was judged once**, and its report shows totals only. Held-out B remains sealed for the next change to the judge.

## Limits

- The errors were injected and are fairly clear; some edited answers even contradict their own quotes. These results do not show how the judge handles subtle, natural hallucinations, such as those that appear when retrieval misses the evidence.
- The samples are small and synthetic. 15/15 establishes recall of at least about 80% for these error types, not 100%.
- The 26 real answers come from questions whose evidence retrieval found. Faithfulness when retrieval fails was not measured.
- The application now answers with Claude Haiku 5.5, the judge's model. Judging answers written by the same model risks self-preference; these results were measured on Claude Haiku 4.5 answers, so re-run the held-out B set with a different judge model before relying on them for the new default.
- Claim-level comparisons depend on how the judge splits an answer, which can change between runs. Hand labels apply to one frozen judge run.

<details>
<summary>Reproduction and files</summary>

Requires PostgreSQL with the embedded fixtures (see the [evaluation report](REPORT.md)) and `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY` in `.env`. Install `requirements-dev.txt`, which adds the Anthropic SDK for evaluation only.

```powershell
# Development answers, judge, hand labels
.\.venv\Scripts\python.exe -m evaluation.faithfulness_judge generate
.\.venv\Scripts\python.exe -m evaluation.faithfulness_judge judge
.\.venv\Scripts\python.exe -m evaluation.label_faithfulness
.\.venv\Scripts\python.exe -m evaluation.faithfulness_calibration human

# Judge calibration on known errors
.\.venv\Scripts\python.exe -m evaluation.faithfulness_calibration run
.\.venv\Scripts\python.exe -m evaluation.faithfulness_calibration run --set heldout_a
```

Committed data: [`faithfulness_mutations.json`](faithfulness_mutations.json) (development), [`faithfulness_split.json`](faithfulness_split.json), [`faithfulness_heldout_mutations.json`](faithfulness_heldout_mutations.json). Generated answers, judge results and hand labels stay local. A held-out set refuses a second scoring run; regenerating its answers or errors also refuses to overwrite the frozen files.

Recorded cost of all runs on 2026-10-11 was about US$0.25, mostly answer generation and error writing; the judge accounted for about US$0.04.

</details>
