"""
ingest.py - build the index. Run once, and again when you add PDFs.

Flow: load PDFs -> split into overlapping chunks -> tag with source metadata
-> embed and persist to Chroma. We ALSO pickle the raw chunks, because the
sparse (BM25) retriever is keyword based and has to be rebuilt in memory at
serve time from the original text. Chroma stores vectors, the pickle stores
the text BM25 needs.

Re-running this script is safe — the existing collection is cleared first so
you never get duplicate chunks from repeat runs.
"""

import pickle
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from embeddings import Qwen3Embeddings

_ROOT = Path(__file__).parent.parent
PDF_DIR = _ROOT / "pdfs"
PERSIST_DIR = str(_ROOT / "chroma_db")
DOCSTORE = str(_ROOT / "chunks.pkl")

CHUNK_SIZE = 800        # characters, ~180 tokens
CHUNK_OVERLAP = 120     # ~15%, keeps boundary sentences intact
BATCH = 256


def load_and_chunk():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs in {PDF_DIR.resolve()}")

    chunks = []
    for i, pdf in enumerate(pdfs, 1):
        pages = PyPDFLoader(str(pdf)).load()
        if not any(p.page_content.strip() for p in pages):
            print(f"  skip {pdf.name}: no extractable text (scanned?)")
            continue
        cs = splitter.split_documents(pages)
        for c in cs:
            c.metadata["source"] = pdf.name      # used for citing and filtering
        chunks.extend(cs)
        print(f"  [{i}/{len(pdfs)}] {pdf.name}: {len(cs)} chunks")
    return chunks


def main():
    chunks = load_and_chunk()
    print(f"\nEmbedding {len(chunks)} chunks (slow step, runs once)...")

    emb = Qwen3Embeddings()

    # Clear any existing vectors so re-running never creates duplicate chunks
    store = Chroma(
        collection_name="pdfs",
        embedding_function=emb,
        persist_directory=PERSIST_DIR,
    )
    existing = store.get()
    if existing["ids"]:
        print(f"  Resetting {len(existing['ids'])} existing vectors...")
        store.delete_collection()
        store = Chroma(
            collection_name="pdfs",
            embedding_function=emb,
            persist_directory=PERSIST_DIR,
        )

    for i in range(0, len(chunks), BATCH):
        store.add_documents(chunks[i:i + BATCH])
        print(f"  indexed {min(i + BATCH, len(chunks))}/{len(chunks)}")

    with open(DOCSTORE, "wb") as f:
        pickle.dump(chunks, f)

    print(f"\nDone. Vectors in ./{PERSIST_DIR}, chunks in ./{DOCSTORE}")


if __name__ == "__main__":
    main()
