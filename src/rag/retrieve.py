from typing import List, Dict, Tuple, Optional
import numpy as np
import faiss

from openai import OpenAI
from src.config import OPENAI_API_KEY, OPENAI_EMBED_MODEL
from src.rag.vector_store import load_index

client = OpenAI(api_key=OPENAI_API_KEY)

# Cache index + chunks to avoid reloading from disk on every request
_INDEX_CACHE: Optional[Tuple[faiss.Index, List[Dict]]] = None

def _get_index():
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        _INDEX_CACHE = load_index()
    return _INDEX_CACHE

def embed_query(q: str) -> np.ndarray:
    q = (q or "").strip()
    if not q:
        # return a dummy vector (won't retrieve anything)
        return np.zeros((1, 1), dtype="float32")

    resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=[q])
    vec = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec

def retrieve(query: str, k: int = 4, min_score: float = 0.25) -> List[Dict]:
    query = (query or "").strip()
    if len(query) < 3:
        return []

    index, chunks = _get_index()

    qv = embed_query(query)

    # If embed_query returned dummy vector, return empty
    if qv.shape[1] != index.d:
        return []

    scores, idxs = index.search(qv, k)

    results: List[Dict] = []
    for score, i in zip(scores[0], idxs[0]):
        if i == -1:
            continue
        if float(score) < float(min_score):
            continue

        item = chunks[int(i)]
        results.append({**item, "score": float(score)})

    return results
