"""
embeddings.py - Qwen3-Embedding-0.6B wrapped in LangChain's Embeddings interface.
 
Qwen3 is INSTRUCTION-AWARE: it expects a short task instruction prepended to
the QUERY, while documents are embedded as plain text. Query and document are
embedded differently on purpose, which is what lifts retrieval quality. We
implement the two sides separately so the rest of the app just calls a normal
embeddings object.
"""
 
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
 
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
 
# The instruction tells the model what "relevant" means for this task.
QUERY_INSTRUCTION = (
    "Instruct: Given a search query, retrieve relevant passages that answer it\nQuery: "
)
 
 
class Qwen3Embeddings(Embeddings):
    def __init__(self, model_name: str = MODEL_NAME):
        # Loaded once and reused. On CPU this is the heaviest object in the app.
        self.model = SentenceTransformer(model_name)
 
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Documents: no instruction. normalize so cosine == dot product.
        vecs = self.model.encode(texts, normalize_embeddings=True, batch_size=64)
        return vecs.tolist()
 
    def embed_query(self, text: str) -> list[float]:
        # Query: prepend the instruction so it lands near matching passages.
        vec = self.model.encode(QUERY_INSTRUCTION + text, normalize_embeddings=True)
        return vec.tolist()
