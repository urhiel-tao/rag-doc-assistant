from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.pdf_loader import extract_text_from_pdf, chunk_text
from app.services.vector_store import vector_store
from app.services.rag_engine import rag_engine

router = APIRouter(tags=["RAG Operations"])

class QueryRequest(BaseModel):
    query: str = Field(..., description="Pregunta del usuario sobre la documentación.")
    top_k: Optional[int] = Field(default=3, ge=1, le=10, description="Cantidad de fragmentos a recuperar.")
    filter_doc: Optional[str] = Field(default=None, description="Filtrar opcionalmente por un documento específico.")

class SourceCitation(BaseModel):
    document: str
    page: int
    chunk_id: str
    similarity_score: float
    excerpt: str

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceCitation]
    model_used: str

class UploadResponse(BaseModel):
    filename: str
    pages_processed: int
    chunks_indexed: int
    status: str

@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """Sube un archivo PDF, lo procesa, genera fragmentos y los indexa en la base vectorial."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Solo se admiten documentos en formato PDF (.pdf)."
        )

    file_path = settings.UPLOADS_DIR / file.filename

    # Guardar archivo en disco
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al guardar el archivo: {str(e)}"
        )

    # Procesar y trocear el PDF
    try:
        pages = extract_text_from_pdf(file_path)
        if not pages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El PDF no contiene texto legible (puede ser un escaneo o estar vacío)."
            )

        chunks = chunk_text(
            pages_data=pages,
            doc_name=file.filename,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

        indexed_count = vector_store.add_chunks(chunks)

        return UploadResponse(
            filename=file.filename,
            pages_processed=len(pages),
            chunks_indexed=indexed_count,
            status="Indexado exitosamente en ChromaDB"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante el procesamiento e indexación: {str(e)}"
        )

@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Realiza una consulta en lenguaje natural sobre los documentos indexados."""
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La consulta no puede estar vacía."
        )

    result = rag_engine.answer_query(
        query=request.query,
        top_k=request.top_k,
        filter_doc=request.filter_doc
    )
    return result

@router.get("/documents")
async def list_documents():
    """Retorna la lista de documentos actualmente indexados."""
    docs = vector_store.list_documents()
    return {
        "total_documents": len(docs),
        "total_chunks": vector_store.count(),
        "documents": docs
    }

@router.delete("/documents/{doc_name}")
async def delete_document(doc_name: str):
    """Elimina un documento de la base vectorial y su archivo asociado."""
    deleted = vector_store.delete_document(doc_name)
    file_path = settings.UPLOADS_DIR / doc_name
    if file_path.exists():
        try:
            file_path.unlink()
        except Exception:
            pass

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Documento '{doc_name}' no encontrado."
        )

    return {"message": f"Documento '{doc_name}' eliminado correctamente."}

@router.get("/health")
async def health_check():
    """Chequeo de salud del servicio y de la base vectorial."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "indexed_chunks": vector_store.count()
    }
