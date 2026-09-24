"""
Unit and End-to-End Test Suite for OSTutorLLM RAG Pipeline (Phase 2).
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np

from config import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS
from rag.chunk import chunk_document_record, load_chunks_jsonl, save_chunks_jsonl, split_text_into_chunks
from rag.clean import clean_text, remove_headers_and_footers, remove_hyphenated_breaks
from rag.embed import embed_query, embed_texts
from rag.ingest import discover_documents, extract_txt, ingest_document, load_and_validate_metadata
from rag.retrieve import VectorStoreIndex, build_index, load_index, retrieve, save_index


class TestRAGIngestion(unittest.TestCase):
    """Test document discovery and text extraction components."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.sample_txt = self.temp_dir / "test_doc.txt"
        self.sample_txt.write_text("Operating Systems manage CPU scheduling and memory.", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_extract_txt(self):
        pages = extract_txt(self.sample_txt)
        self.assertEqual(len(pages), 1)
        self.assertEqual(pages[0][0], 1)
        self.assertIn("Operating Systems", pages[0][1])

    def test_discover_documents(self):
        docs = discover_documents(self.temp_dir)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].name, "test_doc.txt")

    def test_ingest_document(self):
        catalog = {
            "test_doc.txt": {
                "document_id": "doc_test_01",
                "filename": "test_doc.txt",
                "unit": 2,
                "topic": "System Calls",
                "subtopic": "Traps",
                "source_type": "faculty_material",
                "source": "Test Source",
                "license_or_access": "Open",
                "description": "Test doc",
            }
        }
        recs = ingest_document(self.sample_txt, catalog)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["document_id"], "doc_test_01")
        self.assertEqual(recs[0]["unit"], 2)


class TestRAGCleaning(unittest.TestCase):
    """Test text cleaning functions."""

    def test_remove_hyphenated_breaks(self):
        text = "schedul-\ning algorithm"
        cleaned = remove_hyphenated_breaks(text)
        self.assertEqual(cleaned, "scheduling algorithm")

    def test_remove_headers_and_footers(self):
        text = "Page 12 of 30\nOperating Systems\nConfidential - Internal Use Only\nCore Concept"
        cleaned = remove_headers_and_footers(text)
        self.assertNotIn("Page 12", cleaned)
        self.assertNotIn("Confidential", cleaned)
        self.assertIn("Core Concept", cleaned)

    def test_clean_text_full(self):
        raw = "CPU    scheduling\n\n\n\nis the process of schedul-\ning..."
        cleaned = clean_text(raw)
        self.assertNotIn("  ", cleaned)
        self.assertEqual(cleaned, "CPU scheduling\n\nis the process of scheduling...")


class TestRAGChunking(unittest.TestCase):
    """Test chunking logic, metadata preservation, and deterministic IDs."""

    def test_split_text_into_chunks(self):
        text = "word " * 1500
        chunks = split_text_into_chunks(text, target_tokens=300, overlap_tokens=50)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertTrue(len(c.split()) > 0)

    def test_chunk_document_record(self):
        record = {
            "document_id": "unit3_rr",
            "filename": "rr.pdf",
            "unit": 3,
            "topic": "CPU Scheduling",
            "subtopic": "Round Robin",
            "source": "rr.pdf",
            "page": 5,
            "cleaned_text": "Round Robin scheduling uses a time quantum. " * 40,
        }
        chunks = chunk_document_record(record, target_tokens=100, overlap_tokens=20)
        self.assertGreater(len(chunks), 0)
        first_chunk = chunks[0]
        self.assertTrue(first_chunk["chunk_id"].startswith("unit3_rr_p005_chunk01"))
        self.assertEqual(first_chunk["unit"], 3)
        self.assertEqual(first_chunk["topic"], "CPU Scheduling")

    def test_jsonl_persistence(self):
        temp_file = Path(tempfile.mktemp(suffix=".jsonl"))
        try:
            chunks = [{"chunk_id": "c1", "text": "hello"}, {"chunk_id": "c2", "text": "world"}]
            save_chunks_jsonl(chunks, temp_file)
            loaded = load_chunks_jsonl(temp_file)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0]["chunk_id"], "c1")
        finally:
            if temp_file.exists():
                temp_file.unlink()


class TestRAGEmbeddings(unittest.TestCase):
    """Test embedding generation and shape guarantees."""

    def test_embed_texts_and_query(self):
        texts = ["Operating system kernel", "Process scheduling algorithm"]
        vecs = embed_texts(texts)
        self.assertEqual(vecs.shape[0], 2)
        self.assertGreater(vecs.shape[1], 0)
        self.assertEqual(vecs.dtype, np.float32)

        q_vec = embed_query("What is process scheduling?")
        self.assertEqual(q_vec.shape[0], vecs.shape[1])
        # Vector should be L2-normalized
        norm = np.linalg.norm(q_vec)
        self.assertAlmostEqual(norm, 1.0, places=3)


class TestRAGRetrievalAndFAISS(unittest.TestCase):
    """Test FAISS index construction, persistence, retrieval ranking, and filtering."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.chunks = [
            {
                "chunk_id": "u3_c1",
                "unit": 3,
                "topic": "CPU Scheduling",
                "subtopic": "Round Robin",
                "filename": "sched.pdf",
                "page": 1,
                "text": "Round Robin is a preemptive CPU scheduling algorithm using time quantum.",
            },
            {
                "chunk_id": "u4_c1",
                "unit": 4,
                "topic": "Semaphores",
                "subtopic": "Binary Semaphore",
                "filename": "concurrency.pdf",
                "page": 2,
                "text": "A binary semaphore acts as a mutex lock enforcing mutual exclusion.",
            },
            {
                "chunk_id": "u5_c1",
                "unit": 5,
                "topic": "Paging",
                "subtopic": "Page Table",
                "filename": "memory.pdf",
                "page": 3,
                "text": "Virtual memory paging translates logical page numbers into physical frames.",
            },
        ]
        chunk_texts = [c["text"] for c in self.chunks]
        self.embeddings = embed_texts(chunk_texts)
        self.vstore = build_index(self.chunks, self.embeddings)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_faiss_save_and_load(self):
        fpath, mpath = save_index(self.vstore, index_dir=self.temp_dir)
        self.assertTrue(fpath.exists())
        self.assertTrue(mpath.exists())

        loaded_vstore = load_index(index_dir=self.temp_dir)
        self.assertEqual(loaded_vstore.total_vectors, 3)

    def test_retrieval_query(self):
        save_index(self.vstore, index_dir=self.temp_dir)
        results = retrieve(
            query="Tell me about time quantum in Round Robin scheduling",
            top_k=2,
            index_dir=self.temp_dir,
            vector_store=self.vstore,
        )
        self.assertGreater(len(results), 0)
        top1 = results[0]
        self.assertEqual(top1["chunk_id"], "u3_c1")
        self.assertEqual(top1["unit"], 3)
        self.assertGreater(top1["score"], 0.3)

    def test_retrieval_unit_filtering(self):
        save_index(self.vstore, index_dir=self.temp_dir)
        # Query unit 4 specifically
        results = retrieve(
            query="memory page tables",
            top_k=5,
            unit=4,
            index_dir=self.temp_dir,
            vector_store=self.vstore,
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["unit"], 4)
        self.assertEqual(results[0]["chunk_id"], "u4_c1")


if __name__ == "__main__":
    unittest.main()
