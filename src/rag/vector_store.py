import json
import os
from typing import List, Dict, Tuple
import numpy as np
import faiss

from openai import OpenAI
from src.config import OPENAI_API_KEY, OPENAI_EMBED_MODEL, INDEX_DIR

client = OpenAI(api_key=OPENAI_API_KEY)

def embed_texts(texts: List[str]) -> np.ndarray:
    resp = client.embeddings.create(
        model=OPENAI_EMBED_MODEL,
        input=texts
    )
    vectors = [d.embedding for d in resp.data]
    return np.array(vectors, dtype="float32")

def build_faiss_index(chunks: List[Dict]) -> Tuple[faiss.IndexFlatIP, List[Dict]]:
    texts = [c["chunk"] for c in chunks]
    vecs = embed_texts(texts)

    # cosine similarity = inner product after normalization
    faiss.normalize_L2(vecs)

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index, chunks

def save_index(index, chunks: List[Dict]):
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(INDEX_DIR, "kb.faiss"))
    with open(os.path.join(INDEX_DIR, "kb_chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

def load_index():
    index = faiss.read_index(os.path.join(INDEX_DIR, "kb.faiss"))
    with open(os.path.join(INDEX_DIR, "kb_chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks
