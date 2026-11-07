"""
Script de verificación integral de la aplicación DocDesk (rag-doc-assistant).
Valida:
1. Carga de módulos y configuraciones
2. Respuesta de la API y Frontend
3. Ingestión y troceado de texto
4. Búsqueda y similitud vectorial
5. Generación de respuesta con citas
6. Limpieza posterior
"""

import sys
import os
from pathlib import Path

# Configurar UTF-8 en salida de consola
sys.stdout.reconfigure(encoding='utf-8')

print("==================================================")
print("🔍 INICIANDO CHEQUEO INTEGRAL DE LA APLICACIÓN")
print("==================================================")

# 1. Imports
try:
    from app.main import app
    from app.core.config import settings
    from app.services.pdf_loader import chunk_text
    from app.services.vector_store import vector_store
    from app.services.rag_engine import rag_engine
    from fastapi.testclient import TestClient
    print("[OK] Modulos principales importados sin errores.")
except Exception as e:
    print(f"[FAIL] Error en imports: {e}")
    sys.exit(1)

client = TestClient(app)

# 2. Verificar Frontend
try:
    res = client.get("/")
    assert res.status_code == 200
    assert "DocDesk" in res.text
    assert "Búsqueda y extracción contextual" in res.text
    print("[OK] Frontend UI (index.html) responde 200 OK y contiene el nuevo diseño DocDesk.")
except Exception as e:
    print(f"[FAIL] Error en Frontend UI: {e}")
    sys.exit(1)

# 3. Verificar Health Check
try:
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    print(f"[OK] Health check API responde 200 OK (Version {data['version']}, Status: {data['status']}).")
except Exception as e:
    print(f"[FAIL] Error en Health check: {e}")
    sys.exit(1)

# 4. Probar Ingestión y Chunking
try:
    sample_pages = [
        {
            "page": 1,
            "text": "Artículo 1: El presente reglamento establece las normas de convivencia académica y regularidad. Los alumnos deberán registrar un mínimo del 75 por ciento de asistencia a clases presenciales."
        },
        {
            "page": 2,
            "text": "Artículo 12: Las licencias extraordinarias por razones laborales o médicas deberán solicitarse con un plazo mínimo de 48 horas de anticipación ante la Secretaría del Instituto."
        }
    ]
    chunks = chunk_text(sample_pages, doc_name="reglamento_academico_test.pdf", chunk_size=20, chunk_overlap=5)
    assert len(chunks) >= 2
    assert chunks[0]["metadata"]["document"] == "reglamento_academico_test.pdf"
    print(f"[OK] Algoritmo de chunking ejecutado con exito ({len(chunks)} fragmentos generados preservando metadatos).")
except Exception as e:
    print(f"[FAIL] Error en chunking: {e}")
    sys.exit(1)

# 5. Probar Base Vectorial (Indexación y Búsqueda)
try:
    added = vector_store.add_chunks(chunks)
    assert added > 0
    print(f"[OK] Fragmentos indexados correctamente en la base vectorial ({added} chunks).")

    # Búsqueda semántica por palabra clave / concepto
    results = vector_store.search("plazo de licencias médicas secretaría", top_k=2)
    assert len(results) > 0
    top_hit = results[0]
    assert "48 horas" in top_hit["text"]
    assert top_hit["metadata"]["page"] == 2
    print(f"[OK] Busqueda vectorial semantica exitosa: top hit encontrado en Pag. {top_hit['metadata']['page']} con {round(top_hit['similarity_score']*100, 1)}% de afinidad.")
except Exception as e:
    print(f"[FAIL] Error en base vectorial: {e}")
    sys.exit(1)

# 6. Probar Consulta RAG mediante la API
try:
    query_payload = {
        "query": "¿Con cuánta anticipación deben pedirse las licencias médicas?",
        "top_k": 2
    }
    res_query = client.post("/api/query", json=query_payload)
    assert res_query.status_code == 200
    q_data = res_query.json()
    assert len(q_data["sources"]) > 0
    assert q_data["sources"][0]["page"] == 2
    print(f"[OK] Endpoint /api/query responde correctamente con citas respaldadas.")
    print(f"     -> Cita recuperada: Pag. {q_data['sources'][0]['page']} ({round(q_data['sources'][0]['similarity_score']*100, 1)}% afinidad)")
except Exception as e:
    print(f"[FAIL] Error en endpoint /api/query: {e}")
    sys.exit(1)

# 7. Limpieza del documento de prueba
try:
    vector_store.delete_document("reglamento_academico_test.pdf")
    docs = vector_store.list_documents()
    test_docs = [d for d in docs if d["document"] == "reglamento_academico_test.pdf"]
    assert len(test_docs) == 0
    print("[OK] Limpieza del documento de prueba confirmada (Base vectorial restablecida).")
except Exception as e:
    print(f"[FAIL] Error en limpieza: {e}")
    sys.exit(1)

print("==================================================")
print("🎉 CHEQUEO COMPLETADO: 100% DE PRUEBAS SUPERADAS")
print("==================================================")
