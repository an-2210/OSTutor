"""
Document Chunking Module for OSTutorLLM RAG Pipeline.

Splits cleaned OS educational texts into overlapping, semantic-aware chunks
while attaching document metadata (unit, topic, subtopic, source, page/slide number)
and generating deterministic chunk IDs.
"""

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS, CHUNKS_JSONL_PATH

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a text string using word and character heuristics (~1.3 tokens per word).

    Args:
        text: Input text string.

    Returns:
        Estimated token count.
    """
    words = text.split()
    if not words:
        return 0
    # Average token length approximation
    return max(1, int(len(words) * 1.3))


def split_text_into_chunks(
    text: str,
    target_tokens: int = CHUNK_SIZE_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> List[str]:
    """
    Split a long text string into overlapping text blocks based on word boundaries and target token count.

    Args:
        text: Text string to chunk.
        target_tokens: Target token count per chunk (~700).
        overlap_tokens: Overlap token count (~100).

    Returns:
        List of chunk text strings.
    """
    words = text.split()
    if not words:
        return []

    # Convert token counts to target word counts (~1 word = 0.75 tokens)
    target_words = max(20, int(target_tokens / 1.3))
    overlap_words = max(5, int(overlap_tokens / 1.3))

    if overlap_words >= target_words:
        overlap_words = target_words // 4

    chunks: List[str] = []
    start_idx = 0
    total_words = len(words)

    while start_idx < total_words:
        end_idx = min(start_idx + target_words, total_words)

        # Ensure we do not leave an awkward tiny tail chunk (< 20 words) if we are near the end
        if total_words - end_idx < 20 and end_idx < total_words:
            end_idx = total_words

        chunk_str = " ".join(words[start_idx:end_idx])
        if chunk_str.strip():
            chunks.append(chunk_str.strip())

        if end_idx == total_words:
            break

        start_idx = end_idx - overlap_words

    return chunks


def chunk_document_record(
    record: Dict[str, Any],
    target_tokens: int = CHUNK_SIZE_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    chunk_counter: int = 0,
) -> List[Dict[str, Any]]:
    """
    Chunk a single document page/slide record and attach all document metadata.

    Args:
        record: Document page/slide record dictionary.
        target_tokens: Target chunk size in tokens.
        overlap_tokens: Target overlap size in tokens.
        chunk_counter: Global or document-level chunk index offset.

    Returns:
        List of generated chunk record dictionaries.
    """
    text = record.get("cleaned_text", record.get("raw_text", ""))
    if not text.strip():
        return []

    text_chunks = split_text_into_chunks(text, target_tokens, overlap_tokens)
    doc_id = record.get("document_id", "doc_unknown")
    page_num = record.get("page")
    slide_num = record.get("slide")

    # Build deterministic chunk ID prefix
    if page_num is not None:
        prefix = f"{doc_id}_p{page_num:03d}"
    elif slide_num is not None:
        prefix = f"{doc_id}_s{slide_num:03d}"
    else:
        prefix = f"{doc_id}"

    chunk_records: List[Dict[str, Any]] = []

    for idx, c_text in enumerate(text_chunks, start=1):
        c_id = f"{prefix}_chunk{idx:02d}"
        chunk_item = {
            "chunk_id": c_id,
            "document_id": doc_id,
            "filename": record.get("filename", ""),
            "unit": record.get("unit", 1),
            "topic": record.get("topic", "General OS"),
            "subtopic": record.get("subtopic", "General"),
            "source_type": record.get("source_type", "faculty_material"),
            "source": record.get("source", ""),
            "license_or_access": record.get("license_or_access", ""),
            "page": page_num,
            "slide": slide_num,
            "estimated_tokens": estimate_tokens(c_text),
            "text": c_text,
        }
        chunk_records.append(chunk_item)

    return chunk_records


def process_and_chunk_all(
    page_records: List[Dict[str, Any]],
    target_tokens: int = CHUNK_SIZE_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> List[Dict[str, Any]]:
    """
    Process all document page/slide records into a flattened list of metadata-rich chunks.

    Args:
        page_records: List of document page/slide records.
        target_tokens: Target size in tokens.
        overlap_tokens: Overlap in tokens.

    Returns:
        List of chunk record dictionaries.
    """
    all_chunks: List[Dict[str, Any]] = []

    for rec in page_records:
        chunks = chunk_document_record(
            record=rec,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
        )
        all_chunks.extend(chunks)

    logger.info(f"Generated {len(all_chunks)} total chunks from {len(page_records)} document page/slide records.")
    return all_chunks


def save_chunks_jsonl(chunks: List[Dict[str, Any]], output_path: Path = CHUNKS_JSONL_PATH) -> None:
    """
    Save chunk records to a JSONL file.

    Args:
        chunks: List of chunk record dictionaries.
        output_path: Path to output JSONL file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(chunks)} chunks to {output_path}")


def load_chunks_jsonl(input_path: Path = CHUNKS_JSONL_PATH) -> List[Dict[str, Any]]:
    """
    Load chunk records from a JSONL file.

    Args:
        input_path: Path to input JSONL file.

    Returns:
        List of chunk record dictionaries.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Chunks JSONL file not found at: {input_path}")

    chunks: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_rec = {
        "document_id": "unit3_cpu_scheduling",
        "filename": "cpu_scheduling.pdf",
        "unit": 3,
        "topic": "CPU Scheduling",
        "subtopic": "Round Robin",
        "source": "cpu_scheduling.pdf",
        "page": 12,
        "cleaned_text": "Round Robin scheduling is a preemptive scheduling algorithm. " * 30,
    }
    res = chunk_document_record(sample_rec)
    print(f"Generated {len(res)} chunks for sample record.")
    print("Chunk #1 ID:", res[0]["chunk_id"])
    print("Chunk #1 Token Count:", res[0]["estimated_tokens"])
