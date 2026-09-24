"""
RAG Pipeline Index Builder for OSTutorLLM.

Executes the end-to-end RAG pipeline:
Document Discovery -> Text Extraction -> Cleaning -> Chunking ->
Embedding Generation -> FAISS Index Construction -> Artifact Persistence.
"""

import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import (
    CHUNKS_JSONL_PATH,
    EMBEDDING_MODEL_NAME,
    INDEX_DIR,
    METADATA_CSV_PATH,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TAXONOMY_JSON_PATH,
    ensure_directories_exist,
)
from rag.chunk import process_and_chunk_all, save_chunks_jsonl
from rag.clean import clean_document_records
from rag.embed import embed_texts, load_embedding_model
from rag.ingest import discover_documents, ingest_all_documents
from rag.retrieve import build_index, save_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("rag.build_index")


def build_rag_pipeline(
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    index_dir: Path = INDEX_DIR,
    metadata_csv: Path = METADATA_CSV_PATH,
    taxonomy_json: Path = TAXONOMY_JSON_PATH,
    model_name: str = EMBEDDING_MODEL_NAME,
) -> bool:
    """
    Execute the complete RAG index build pipeline.

    Returns:
        True if build completed successfully, False otherwise.
    """
    start_time = time.time()
    print("\nOSTutorLLM RAG Index Builder")
    print("============================\n")

    ensure_directories_exist()

    # Step 1: Discover Raw Documents
    discovered = discover_documents(raw_dir)
    print(f"Step 1: Discovered {len(discovered)} document files in '{raw_dir}'")
    for doc_path in discovered:
        print(f"  - {doc_path.relative_to(raw_dir.parent if raw_dir.parent.exists() else raw_dir)}")

    if not discovered:
        print("\n[WARNING] No raw documents (.pdf, .pptx, .txt, .docx) found in 'data/rag/raw/'.")
        print("To build an index from sample files, place documents in 'data/rag/raw/' or run fixture generator.")
        return False

    # Step 2: Ingest Raw Documents
    print("\nStep 2: Extracting document text and attaching metadata...")
    page_records = ingest_all_documents(
        raw_dir=raw_dir, metadata_csv_path=metadata_csv, taxonomy_json_path=taxonomy_json
    )
    print(f"  -> Extracted {len(page_records)} total page/slide records.")

    # Step 3: Clean Document Text
    print("\nStep 3: Cleaning text and removing formatting noise...")
    cleaned_records = clean_document_records(page_records)

    # Step 4: Chunk Document Records
    print("\nStep 4: Chunking document records into overlapping token blocks...")
    chunks = process_and_chunk_all(cleaned_records)
    print(f"  -> Generated {len(chunks)} metadata-rich chunks.")

    if not chunks:
        print("\n[ERROR] No non-empty text chunks generated. Aborting index build.")
        return False

    # Step 5: Save Processed Chunks
    print(f"\nStep 5: Saving processed chunks to '{CHUNKS_JSONL_PATH}'...")
    save_chunks_jsonl(chunks, output_path=CHUNKS_JSONL_PATH)

    # Step 6: Generate Embeddings
    print(f"\nStep 6: Loading embedding model '{model_name}' and generating embeddings...")
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embed_texts(chunk_texts, model_name=model_name)
    print(f"  -> Generated embeddings tensor of shape: {embeddings.shape}")

    # Step 7: Build FAISS Vector Index
    print("\nStep 7: Constructing FAISS IndexFlatIP vector index...")
    vstore = build_index(chunks, embeddings)

    # Step 8: Save Index Artifacts
    print(f"\nStep 8: Persisting vector index artifacts to '{index_dir}'...")
    faiss_path, meta_path = save_index(vstore, index_dir=index_dir)

    elapsed = time.time() - start_time
    print("\n==================================================")
    print(" RAG Build Completed Successfully!")
    print("==================================================")
    print(f"Documents processed  : {len(discovered)}")
    print(f"Pages/slides total   : {len(page_records)}")
    print(f"Chunks generated     : {len(chunks)}")
    print(f"Embedding Model      : {model_name}")
    print(f"Vector Dimension     : {vstore.vector_dimension}")
    print(f"FAISS Index Path     : {faiss_path}")
    print(f"Metadata JSON Path   : {meta_path}")
    print(f"Chunks JSONL Path    : {CHUNKS_JSONL_PATH}")
    print(f"Total Time Elapsed   : {elapsed:.2f} seconds")
    print("==================================================\n")

    return True


if __name__ == "__main__":
    success = build_rag_pipeline()
    if not success:
        sys.exit(1)
