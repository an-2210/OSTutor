"""
Retrieval CLI Testing Utility for OSTutorLLM RAG Pipeline.

Allows testing semantic vector search over built FAISS index using command line arguments.
Supports filtering by OS unit and topic.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import DEFAULT_TOP_K, INDEX_DIR
from rag.retrieve import load_index, retrieve


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OSTutorLLM Vector Retrieval CLI Tester",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "query",
        type=str,
        help="Natural language student query or question (e.g., 'What is a process?')",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of top relevant chunks to retrieve",
    )
    parser.add_argument(
        "--unit",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7],
        default=None,
        help="Optional OS Unit filter (1-7)",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Optional OS Topic substring filter",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=INDEX_DIR,
        help="Path to directory containing FAISS index and metadata",
    )

    args = parser.parse_args()

    print("\nOSTutorLLM Vector Retrieval CLI")
    print("===============================\n")
    print(f"Query : '{args.query}'")
    if args.unit:
        print(f"Filter: Unit {args.unit}")
    if args.topic:
        print(f"Filter: Topic '{args.topic}'")
    print(f"Top-K : {args.top_k}\n")

    try:
        results = retrieve(
            query=args.query,
            top_k=args.top_k,
            unit=args.unit,
            topic=args.topic,
            index_dir=args.index_dir,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] Index not found: {e}")
        print("Please build the index first using `python3 rag/build_index.py`.")
        sys.exit(1)

    if not results:
        print("No matching relevant chunks found.")
        return

    print(f"Top {len(results)} Retrieved Chunks")
    print("=========================")

    for item in results:
        page_str = f"Page: {item['page']}" if item.get("page") else (f"Slide: {item['slide']}" if item.get("slide") else "Page: N/A")
        print(f"\n[{item['rank']}] Score: {item['score']:.4f}")
        print(f"    Unit    : Unit {item['unit']}")
        print(f"    Topic   : {item['topic']}")
        print(f"    Subtopic: {item['subtopic']}")
        print(f"    Source  : {item['source']} ({page_str})")
        print(f"    Chunk ID: {item['chunk_id']}")
        print("    Text    :")
        for line in item["text"].splitlines():
            print(f"      {line}")
        print("-" * 50)


if __name__ == "__main__":
    main()
