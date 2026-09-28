import unittest

from app.schemas import DocumentPage, ExtractedDocument
from app.services.chunking_service import chunk_document


class ChunkingTests(unittest.TestCase):
    def test_overlap_preserves_all_text_without_redundant_tail(self):
        text = "abcdefghijklmnopqrstuvwxyz"
        document = ExtractedDocument(
            source_file="example.pdf", pages=[DocumentPage(page_number=1, text=text)]
        )
        chunks = chunk_document(document, chunk_size=10, overlap=2)
        self.assertEqual([chunk.text for chunk in chunks], [text[0:10], text[8:18], text[16:26]])
        reconstructed = chunks[0].text + "".join(chunk.text[2:] for chunk in chunks[1:])
        self.assertEqual(reconstructed, text)

    def test_page_numbers_and_global_indices_with_blank_page(self):
        document = ExtractedDocument(
            source_file="example.pdf",
            pages=[
                DocumentPage(page_number=1, text="abc"),
                DocumentPage(page_number=2, text=""),
                DocumentPage(page_number=3, text="def"),
            ],
        )
        chunks = chunk_document(document)
        self.assertEqual([(chunk.chunk_index, chunk.page_number) for chunk in chunks], [(0, 1), (1, 3)])

    def test_invalid_sizes_do_not_enter_chunking_loop(self):
        document = ExtractedDocument(source_file="example.pdf", pages=[])
        for size, overlap in [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11)]:
            with self.subTest(size=size, overlap=overlap):
                with self.assertRaises(ValueError):
                    chunk_document(document, chunk_size=size, overlap=overlap)

    def test_whitespace_only_page_has_no_chunks(self):
        document = ExtractedDocument(
            source_file="example.pdf", pages=[DocumentPage(page_number=1, text="   \n  ")]
        )
        self.assertEqual(chunk_document(document), [])


if __name__ == "__main__":
    unittest.main()
