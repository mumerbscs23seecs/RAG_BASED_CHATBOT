"""
retrieval.py - the full retrieval stack, each stage toggleable.

Order of operations for one question:
  1. (optional) multi-query: LLM rewrites the question into several variants
  2. dense retrieval (Qwen3 vectors via Chroma)  -> top 15
     sparse retrieval (BM25 keywords)            -> top 15
  3. merge the two ranked lists with Reciprocal Rank Fusion (EnsembleRetriever)
  4. (optional) metadata filter on the dense side
  5. rerank the merged candidates with a cross-encoder, keep top 5

Why each stage exists is explained in the guide. Dense finds meaning, sparse
finds exact terms (codes, names, acronyms), RRF merges rankings without needing
comparable scores, and the cross-encoder reorders by reading query+passage
together instead of separately.
"""

import pickle

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

from embeddings import Qwen3Embeddings

from pathlib import Path
_ROOT = Path(__file__).parent.parent
PERSIST_DIR = str(_ROOT / "chroma_db")
DOCSTORE = str(_ROOT / "chunks.pkl")
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # not BAAI, tiny, fast, free

FETCH_K = 15   # each retriever over-fetches this many
TOP_K = 5      # kept after the reranker

# Load once at import. These are expensive to build.
_emb = Qwen3Embeddings()
_store = Chroma(collection_name="pdfs", embedding_function=_emb, persist_directory=PERSIST_DIR)

try:
    with open(DOCSTORE, "rb") as f:
        _chunks = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError(
        f"Chunk store not found at {DOCSTORE}. "
        "Run ingest first: python src/ingest.py"
    )

_reranker = CrossEncoderReranker(
    model=HuggingFaceCrossEncoder(model_name=RERANK_MODEL),
    top_n=TOP_K,
)


def build_retriever(llm=None, metadata_filter=None, use_multiquery=False):
    # --- dense side ---
    search_kwargs = {"k": FETCH_K}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter   # e.g. {"source": "report.pdf"}
    dense = _store.as_retriever(search_kwargs=search_kwargs)

    # --- sparse side ---
    # BM25 can't read Chroma's metadata filter, so if filtering we pre-filter
    # the chunk list before building it.
    pool = _chunks
    if metadata_filter and "source" in metadata_filter:
        pool = [c for c in _chunks if c.metadata.get("source") == metadata_filter["source"]]
    sparse = BM25Retriever.from_documents(pool)
    sparse.k = FETCH_K

    # --- merge with Reciprocal Rank Fusion ---
    hybrid = EnsembleRetriever(retrievers=[dense, sparse], weights=[0.5, 0.5])

    base = hybrid
    if use_multiquery and llm is not None:
        # Wraps the hybrid retriever: the LLM generates query variants, each is
        # run through the hybrid retriever, and results are unioned.
        base = MultiQueryRetriever.from_llm(retriever=hybrid, llm=llm)

    # --- rerank: up to 30 merged (15×2) -> top 5 ---
    return ContextualCompressionRetriever(base_compressor=_reranker, base_retriever=base)
