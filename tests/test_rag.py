import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.pdf_loader import chunk_text
from app.services.vector_store import LocalVectorStore

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data

def test_chunking_logic():
    sample_pages = [
        {"page": 1, "text": "Este es un parrafo de prueba sobre el reglamento de cursadas y asistencias de alumnos."},
        {"page": 2, "text": "El regimen de aprobacion requiere un porcentaje minimo del 80 por ciento de asistencia a clases practicas."}
    ]
    chunks = chunk_text(sample_pages, doc_name="reglamento.pdf", chunk_size=10, chunk_overlap=2)
    assert len(chunks) >= 2
    assert chunks[0]["metadata"]["document"] == "reglamento.pdf"
    assert chunks[0]["metadata"]["page"] in [1, 2]

def test_vector_store_operations(tmp_path):
    store = LocalVectorStore(storage_dir=tmp_path)
    sample_chunks = [
        {
            "id": "doc1_p1_c0",
            "text": "Las licencias por examen deben solicitarse con 48 horas de anticipacion.",
            "metadata": {"document": "estatuto.pdf", "page": 1, "chunk_index": 0}
        },
        {
            "id": "doc1_p1_c1",
            "text": "Los docentes titulares deben presentar el programa de la materia antes del inicio del ciclo lectivo.",
            "metadata": {"document": "estatuto.pdf", "page": 1, "chunk_index": 1}
        }
    ]

    added = store.add_chunks(sample_chunks)
    assert added == 2
    assert store.count() == 2

    # Búsqueda semántica
    results = store.search("licencias por examen plazo horas", top_k=1)
    assert len(results) == 1
    assert "48 horas" in results[0]["text"]
    assert results[0]["metadata"]["page"] == 1

    # Eliminación
    deleted = store.delete_document("estatuto.pdf")
    assert deleted is True
    assert store.count() == 0

def test_query_api_without_documents():
    payload = {"query": "¿Cual es el plazo para rendir finales?", "top_k": 3}
    response = client.post("/api/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
