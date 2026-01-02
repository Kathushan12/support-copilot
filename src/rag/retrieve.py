from typing import List, Dict
import numpy as np
import faiss

from openai import OpenAI
from src.config import OPENAI_API_KEY, OPENAI_EMBED_MODEL
from src.rag.vector_store import load_index

client = OpenAI(api_key=OPENAI_API_KEY)

def embed_query(q: str) -> np.ndarray:
    resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=[q])
    vec = np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec

def retrieve(query: str, k: int = 4) -> List[Dict]:
    index, chunks = load_index()
    qv = embed_query(query)
    scores, idxs = index.search(qv, k)
    results = []
    for score, i in zip(scores[0], idxs[0]):
        if i == -1:
            continue
        item = chunks[int(i)]
        item = {**item, "score": float(score)}
        results.append(item)
    return results
