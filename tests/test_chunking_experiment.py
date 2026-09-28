import unittest

from app.schemas import DocumentPage, ExtractedDocument
from evaluation.chunking_experiment import paragraph_chunks, score_strategy


class ChunkingExperimentTests(unittest.TestCase):
    def test_paragraph_chunker_keeps_sections_and_page_numbers(self):
        document = ExtractedDocument(source_file="policy.pdf", pages=[
            DocumentPage(page_number=1, text="Header\n1. First\nSentence one.\nSentence two.\n2. Second\nDifferent rule."),
            DocumentPage(page_number=2, text="3. Third\nFinal rule."),
        ])
        chunks = paragraph_chunks(document)
        self.assertEqual([chunk.page_number for chunk in chunks], [1, 1, 1, 2])
        self.assertEqual(chunks[1].text, "1. First Sentence one. Sentence two.")
        self.assertEqual([chunk.chunk_index for chunk in chunks], list(range(4)))

    def test_scoring_requires_both_cross_document_anchors(self):
        cases = [{"id": "cross", "category": "cross_document", "evidence": [
            {"source_file": "a.pdf", "anchor": "first rule"},
            {"source_file": "b.pdf", "anchor": "second rule"},
        ]}]
        chunks = [
            {"source_file": "a.pdf", "text": "first rule"},
            {"source_file": "other.pdf", "text": "irrelevant"},
            {"source_file": "b.pdf", "text": "second rule"},
        ]
        result = score_strategy(cases, chunks, [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], [[1.0, 0.0]])
        self.assertEqual(result["questions"][0]["evidence_ranks"], [1, 3])
        self.assertEqual(result["mean_recall_at"]["1"], 0.5)
        self.assertEqual(result["all_evidence_at"]["1"], 0.0)
        self.assertEqual(result["all_evidence_at"]["3"], 1.0)


if __name__ == "__main__":
    unittest.main()
