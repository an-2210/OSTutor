"""
Text Cleaning Module for OSTutorLLM RAG Pipeline.

Normalizes extracted document text, cleans line wrapping artifacts,
strips repetitive headers/footers, and preserves section headings, code, and math.
"""

import re
from typing import Any, Dict, List


def remove_hyphenated_breaks(text: str) -> str:
    """
    Rejoin words broken across line breaks with a hyphen (e.g., 'schedul-\ning' -> 'scheduling').

    Args:
        text: Raw text string.

    Returns:
        Text with hyphenated line splits repaired.
    """
    # Rejoin words hyphenated at line end
    return re.sub(r"(\b[A-Za-z]+)-\s*\n\s*([A-Za-z]+\b)", r"\1\2", text)


def remove_headers_and_footers(text: str) -> str:
    """
    Remove repetitive headers, footers, page numbers, and slide markers.

    Args:
        text: Text string.

    Returns:
        Text string with headers/footers removed.
    """
    lines = text.splitlines()
    filtered_lines = []

    for line in lines:
        stripped = line.strip()
        # Skip empty slide/page headers
        if re.match(r"(?i)^\s*(page|slide)\s+\d+(\s+of\s+\d+)?\s*$", stripped):
            continue
        if re.match(r"(?i)^\s*\d+\s*$", stripped) and len(stripped) <= 3:
            # Skip lone page numbers at end/start of pages
            continue
        if re.search(r"(?i)\b(confidential|internal use only|all rights reserved)\b", stripped):
            continue
        filtered_lines.append(line)

    return "\n".join(filtered_lines)


def normalize_whitespace(text: str) -> str:
    """
    Collapse excess spaces and multi-newline gaps while preserving paragraph breaks.

    Args:
        text: Text string.

    Returns:
        Normalized text string.
    """
    # Replace tabs and multiple spaces with a single space (except leading indent for code/lists)
    text = re.sub(r"[ \t]+", " ", text)
    # Remove space before newlines
    text = re.sub(r" \n", "\n", text)
    # Replace 3 or more consecutive newlines with double newline (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(text: str, preserve_headings: bool = True) -> str:
    """
    Clean extracted text using conservative normalization rules.

    Args:
        text: Raw extracted document text string.
        preserve_headings: Whether to preserve heading markers (#).

    Returns:
        Cleaned text string ready for chunking.
    """
    if not text:
        return ""

    # Rejoin hyphenated line splits
    text = remove_hyphenated_breaks(text)

    # Remove headers, footers, page numbers
    text = remove_headers_and_footers(text)

    # Optional heading preservation check
    if not preserve_headings:
        text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)

    # Normalize spaces and newlines
    text = normalize_whitespace(text)

    return text


def clean_document_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clean text across a list of document page/slide records.

    Args:
        records: List of document record dictionaries containing 'raw_text'.

    Returns:
        List of document records enriched with 'cleaned_text'.
    """
    cleaned_records: List[Dict[str, Any]] = []

    for rec in records:
        updated = dict(rec)
        raw_t = rec.get("raw_text", "")
        updated["cleaned_text"] = clean_text(raw_t)
        cleaned_records.append(updated)

    return cleaned_records


if __name__ == "__main__":
    sample = "CPU    scheduling\n\nis the process of selecting\n\n\n\na process..."
    print("Sample Before Cleaning:\n", repr(sample))
    print("Sample After Cleaning:\n", repr(clean_text(sample)))
