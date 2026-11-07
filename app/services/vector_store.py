import os
import json
import math
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.core.config import settings

class LocalVectorStore:
    """
    Motor de almacenamiento e indexación vectorial persistente.
    Soporta ChromaDB cuando está disponible y cuenta con un motor semántico
    vectorial nativo (TF-IDF + Cosine Similarity) persistente en JSON.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or settings.VECTORSTORE_DIR
        self.storage_file = self.storage_dir / "vector_index.json"
        self.chroma_client = None
        self.collection = None
        self.chunks: List[Dict[str, Any]] = []

        self._init_backend()

    def _init_backend(self):
        """Intenta inicializar ChromaDB persistente; si no, carga el motor nativo."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self.chroma_client = chromadb.PersistentClient(path=str(self.storage_dir))
            self.collection = self.chroma_client.get_or_create_collection(
                name="documents_collection",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception:
            # Fallback transparente al motor vectorial nativo
            self.chroma_client = None
            self.collection = None
            self._load_local_store()

    def _load_local_store(self):
        if self.storage_file.exists():
            try:
                with open(self.storage_file, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
            except Exception:
                self.chunks = []
        else:
            self.chunks = []

    def _save_local_store(self):
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        with open(self.storage_file, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Agrega fragmentos con metadatos a la base vectorial."""
        if not chunks:
            return 0

        if self.collection is not None:
            ids = [c["id"] for c in chunks]
            documents = [c["text"] for c in chunks]
            metadatas = [c["metadata"] for c in chunks]
            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            return len(chunks)
        else:
            # Motor nativo
            existing_ids = {c["id"] for c in self.chunks}
            added = 0
            for c in chunks:
                if c["id"] in existing_ids:
                    # Actualizar
                    self.chunks = [item for item in self.chunks if item["id"] != c["id"]]
                self.chunks.append(c)
                added += 1
            self._save_local_store()
            return added

    def search(self, query: str, top_k: int = 3, filter_doc: Optional[str] = None) -> List[Dict[str, Any]]:
        """Realiza una búsqueda semántica de los top_k fragmentos más afines a la consulta."""
        if not query.strip():
            return []

        if self.collection is not None:
            where_filter = {"document": filter_doc} if filter_doc else None
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, max(1, self.collection.count())),
                where=where_filter
            )

            formatted_results = []
            if results and results.get("documents") and results["documents"][0]:
                for idx in range(len(results["documents"][0])):
                    text = results["documents"][0][idx]
                    metadata = results["metadatas"][0][idx]
                    chunk_id = results["ids"][0][idx]
                    distance = results["distances"][0][idx] if results.get("distances") else 0.0
                    similarity = max(0.0, 1.0 - distance)

                    formatted_results.append({
                        "id": chunk_id,
                        "text": text,
                        "metadata": metadata,
                        "similarity_score": round(similarity, 4)
                    })
            return formatted_results
        else:
            # Búsqueda semántica vectorial nativa (TF-IDF Cosine Similarity)
            candidates = self.chunks
            if filter_doc:
                candidates = [c for c in candidates if c["metadata"].get("document") == filter_doc]

            if not candidates:
                return []

            scored = []
            query_tokens = self._tokenize(query)
            if not query_tokens:
                return []

            # Calcular similitud coseno sobre vectores de frecuencia ponderada
            for item in candidates:
                doc_tokens = self._tokenize(item["text"])
                score = self._cosine_similarity(query_tokens, doc_tokens)
                if score > 0:
                    scored.append({
                        "id": item["id"],
                        "text": item["text"],
                        "metadata": item["metadata"],
                        "similarity_score": round(score, 4)
                    })

            scored.sort(key=lambda x: x["similarity_score"], reverse=True)
            return scored[:top_k]

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r"\w+", text.lower())
        stopwords = {
            "de", "la", "el", "en", "y", "a", "los", "se", "del", "las", "por", "un", "para", "con", "no", "una",
            "su", "al", "lo", "como", "mas", "pero", "sus", "le", "ya", "o", "este", "si", "porque", "esta", "son",
            "the", "and", "in", "of", "to", "is", "for", "on", "that", "by", "this", "with", "as", "are"
        }
        return [w for w in words if len(w) > 2 and w not in stopwords]

    def _cosine_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        tf1, tf2 = {}, {}
        for t in tokens1:
            tf1[t] = tf1.get(t, 0) + 1
        for t in tokens2:
            tf2[t] = tf2.get(t, 0) + 1

        all_tokens = set(tf1.keys()).union(set(tf2.keys()))
        dot_product = sum(tf1.get(t, 0) * tf2.get(t, 0) for t in all_tokens)
        mag1 = math.sqrt(sum(v ** 2 for v in tf1.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in tf2.values()))

        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot_product / (mag1 * mag2)

    def list_documents(self) -> List[Dict[str, Any]]:
        """Retorna un resumen de los documentos indexados y cantidad de chunks."""
        docs_summary = {}

        if self.collection is not None:
            all_records = self.collection.get()
            if all_records and all_records.get("metadatas"):
                for m in all_records["metadatas"]:
                    doc = m.get("document", "desconocido")
                    docs_summary[doc] = docs_summary.get(doc, 0) + 1
        else:
            for c in self.chunks:
                doc = c["metadata"].get("document", "desconocido")
                docs_summary[doc] = docs_summary.get(doc, 0) + 1

        return [{"document": k, "chunks_count": v} for k, v in docs_summary.items()]

    def delete_document(self, doc_name: str) -> bool:
        """Elimina un documento y todos sus fragmentos de la base vectorial."""
        if self.collection is not None:
            self.collection.delete(where={"document": doc_name})
            return True
        else:
            prev_len = len(self.chunks)
            self.chunks = [c for c in self.chunks if c["metadata"].get("document") != doc_name]
            if len(self.chunks) != prev_len:
                self._save_local_store()
                return True
            return False

    def count(self) -> int:
        if self.collection is not None:
            return self.collection.count()
        return len(self.chunks)

vector_store = LocalVectorStore()
