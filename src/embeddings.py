"""
embeddings.py - multi-qa-MiniLM-L6-cos-v1 wrapped in LangChain's Embeddings interface.

Trained specifically on question-answer pairs, making it well suited for RAG retrieval.
No instruction prefix needed — query and document are embedded the same way.
"""

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"


class DocumentEmbeddings(Embeddings):
    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True, batch_size=64)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()
