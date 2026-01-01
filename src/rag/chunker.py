from typing import List, Dict
import os

def read_kb_files(kb_dir: str) -> List[Dict]:
    docs = []
    for fn in os.listdir(kb_dir):
        if fn.endswith(".md") or fn.endswith(".txt"):
            path = os.path.join(kb_dir, fn)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            docs.append({"doc_id": fn, "title": fn.replace("_", " ").replace(".md",""), "text": text})
    return docs

def chunk_text(doc: Dict, chunk_size: int = 800, overlap: int = 120) -> List[Dict]:
    text = doc["text"]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        chunks.append({
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "chunk": chunk,
        })
        start += max(1, chunk_size - overlap)
    return chunks
