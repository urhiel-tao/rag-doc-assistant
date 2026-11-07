import json
import requests
from typing import Dict, Any, List, Optional
from app.core.config import settings
from app.services.vector_store import vector_store

class RAGEngine:
    """
    Motor central de RAG (Retrieval-Augmented Generation).
    1. Recupera los fragmentos semánticamente relevantes desde la base vectorial.
    2. Construye el prompt con contexto delimitado.
    3. Consulta al LLM (Google Gemini) con salvaguardas contra alucinaciones.
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
                "answer": "No se encontraron fragmentos relevantes en los documentos indexados para responder a esta consulta.",
                "sources": [],
                "model_used": "none"
            }

        # Formatear contexto estructurado
        context_blocks = []
        sources = []
        for idx, chunk in enumerate(retrieved_chunks):
            doc = chunk["metadata"].get("document", "desconocido")
            page = chunk["metadata"].get("page", 1)
            score = chunk.get("similarity_score", 0.0)

            context_blocks.append(f"[Fragmento {idx + 1} | Documento: {doc} | Página: {page}]\n{chunk['text']}")
            sources.append({
                "document": doc,
                "page": page,
                "chunk_id": chunk["id"],
                "similarity_score": score,
                "excerpt": chunk["text"][:220] + "..." if len(chunk["text"]) > 220 else chunk["text"]
            })

        context_str = "\n\n".join(context_blocks)

        system_prompt = (
            "Eres un analista experto en lectura y análisis de documentación técnica y corporativa.\n"
            "Tu misión es responder con precisión a la pregunta del usuario utilizando ÚNICAMENTE los siguientes fragmentos oficiales.\n\n"
            "REGLAS OBLIGATORIAS:\n"
            "1. Responde de forma clara, profesional y estructurada (usa viñetas si ayuda a la claridad).\n"
            "2. Cita la página o el documento que respalda tu respuesta.\n"
            "3. Si la respuesta NO está en los fragmentos, di textualmente: 'La información solicitada no figura en los documentos provistos.'\n"
            "4. NO agregues suposiciones ni inventes datos que no figuren en el contexto."
        )

        user_content = f"--- FRAGMENTOS DE CONTEXTO ---\n{context_str}\n\n--- PREGUNTA DEL USUARIO ---\n{query}"

        # Llamada a Google Gemini si hay API Key disponible
        if self.api_key:
            answer = self._call_gemini_api(system_prompt, user_content)
            model_used = self.model
        else:
            answer = (
                "ℹ️ **Modo Demostración Local (Sin API Key configurada):**\n\n"
                f"El motor vectorial recuperó exitosamente **{len(sources)} fragmentos clave** con alta similitud semántica.\n\n"
                "**Extracto más relevante encontrado:**\n"
                f"> \"{sources[0]['excerpt']}\" *(Documento: {sources[0]['document']}, Página {sources[0]['page']})*\n\n"
                "Para obtener síntesis generativa fluida en lenguaje natural, agrega tu `GEMINI_API_KEY` en el archivo `.env`."
            )
            model_used = "local-retrieval-demo"

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "model_used": model_used
        }

    def _call_gemini_api(self, system_instruction: str, content: str) -> str:
        """Invoca la API oficial REST de Google Gemini."""
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
                return "No se pudo extraer una respuesta válida del modelo."
            else:
                return f"Error en API Gemini (Status {response.status_code}): {response.text}"
        except Exception as e:
            return f"Error de conexión con el proveedor de IA: {str(e)}"

rag_engine = RAGEngine()
