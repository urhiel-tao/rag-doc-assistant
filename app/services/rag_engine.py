import json
import requests
from typing import Dict, Any, List, Optional
from app.core.config import settings
from app.services.vector_store import vector_store

class RAGEngine:
    """
    Motor central de búsqueda semántica y generación contextual.
    Recupera los fragmentos más afines del documento y consulta al modelo.
    """

    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.model = settings.GEMINI_MODEL

    def answer_query(self, query: str, top_k: Optional[int] = None, filter_doc: Optional[str] = None) -> Dict[str, Any]:
        k = top_k or settings.DEFAULT_TOP_K
        retrieved_chunks = vector_store.search(query=query, top_k=k, filter_doc=filter_doc)

        if not retrieved_chunks:
            return {
                "query": query,
                "answer": "No encontré información relacionada en los documentos cargados. Probá reformular la pregunta o subir el archivo correspondiente.",
                "sources": [],
                "model_used": "none"
            }

        context_blocks = []
        sources = []
        for idx, chunk in enumerate(retrieved_chunks):
            doc = chunk["metadata"].get("document", "Documento")
            page = chunk["metadata"].get("page", 1)
            score = chunk.get("similarity_score", 0.0)

            context_blocks.append(f"[Fragmento {idx + 1} | Archivo: {doc} | Pág. {page}]\n{chunk['text']}")
            sources.append({
                "document": doc,
                "page": page,
                "chunk_id": chunk["id"],
                "similarity_score": score,
                "excerpt": chunk["text"][:220] + "..." if len(chunk["text"]) > 220 else chunk["text"]
            })

        context_str = "\n\n".join(context_blocks)

        system_prompt = (
            "Respondé la pregunta del usuario utilizando exclusivamente los fragmentos del documento provistos abajo.\n"
            "- Sé claro, directo y conciso.\n"
            "- Indicá el número de página de donde sale la información.\n"
            "- Si el dato exacto no figura en el texto, decí simplemente que no está especificado en el documento.\n"
            "- No inventes suposiciones ni agregues datos externos."
        )

        user_content = f"--- FRAGMENTOS DEL DOCUMENTO ---\n{context_str}\n\n--- PREGUNTA ---\n{query}"

        if self.api_key:
            answer = self._call_gemini_api(system_prompt, user_content)
            model_used = self.model
        else:
            answer = (
                f"Encontré este párrafo relevante en **{sources[0]['document']}** (Pág. {sources[0]['page']}):\n\n"
                f"> \"{sources[0]['excerpt']}\"\n\n"
                "*(Podés sumar tu clave de Gemini en el archivo `.env` para redactar respuestas completas automáticamente)*."
            )
            model_used = "local-retrieval"

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "model_used": model_used
        }

    def _call_gemini_api(self, system_instruction: str, content: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "system_instruction": {
                "parts": [{"text": system_instruction}]
            },
            "contents": [
                {
                    "parts": [{"text": content}]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.8,
                "maxOutputTokens": 1024
            }
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=20)
            if response.status_code == 200:
                data = response.json()
                candidates = data.get("candidates", [])
                if candidates and candidates[0].get("content", {}).get("parts"):
                    return candidates[0]["content"]["parts"][0]["text"].strip()
                return "No se pudo leer la respuesta del servidor."
            else:
                return f"Error al consultar el modelo (Código {response.status_code})."
        except Exception as e:
            return f"Error de conexión: {str(e)}"

rag_engine = RAGEngine()
