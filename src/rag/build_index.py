from src.config import KB_DIR
from src.rag.chunker import read_kb_files, chunk_text
from src.rag.vector_store import build_faiss_index, save_index

def main():
    docs = read_kb_files(KB_DIR)
    all_chunks = []
    for d in docs:
        all_chunks.extend(chunk_text(d))
    index, chunks = build_faiss_index(all_chunks)
    save_index(index, chunks)
    print(f"Indexed {len(chunks)} chunks from {len(docs)} docs.")

if __name__ == "__main__":
    main()
