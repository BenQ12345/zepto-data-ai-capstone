from __future__ import annotations

import shutil
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
DB_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "zepto_policy"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks() -> tuple[list[str], list[str], list[dict]]:
    ids, texts, metadatas = [], [], []
    for path in sorted(DOCS_DIR.glob("doc_*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        doc_id = path.stem
        ids.append(doc_id)
        texts.append(text)
        metadatas.append({"document_id": doc_id, "source_file": path.name})
    if len(texts) != 8:
        raise RuntimeError(f"Expected exactly 8 policy documents, found {len(texts)}")
    return ids, texts, metadatas


def build_index() -> None:
    ids, texts, metadatas = load_chunks()
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()

    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    print(f"Indexed {collection.count()} chunks into {DB_DIR}")


if __name__ == "__main__":
    build_index()
