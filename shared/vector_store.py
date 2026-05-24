import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import shared.config as config
from shared.llm_backend import get_llm


class ChromaStore:
    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or config.CHROMA_PERSIST_DIR
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.llm = get_llm()

    def add_collection(self, name: str, documents: List[str], metadatas: List[Dict], ids: List[str]):
        collection = self.client.get_or_create_collection(name=name)
        embeddings = self.llm.embed(documents)
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings,
        )
        return collection

    def query(self, collection_name: str, query_text: str, n_results: int = 5, where: Optional[Dict] = None):
        collection = self.client.get_or_create_collection(name=collection_name)
        embedding = self.llm.embed([query_text])[0]
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        return results

    def get(self, collection_name: str, ids: List[str]):
        collection = self.client.get_or_create_collection(name=collection_name)
        return collection.get(ids=ids, include=["documents", "metadatas"])

    def list_collections(self):
        return self.client.list_collections()

    def delete_collection(self, name: str):
        try:
            self.client.delete_collection(name=name)
        except Exception:
            pass


# Singleton instance
_chroma_store = None

def get_chroma() -> ChromaStore:
    global _chroma_store
    if _chroma_store is None:
        _chroma_store = ChromaStore()
    return _chroma_store
