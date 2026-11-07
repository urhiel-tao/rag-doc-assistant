from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extrae el texto de un PDF manteniendo la referencia al número de página.
    Retorna una lista de diccionarios: [{'page': 1, 'text': '...'}]
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages_data = []

    for idx, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        clean_text = " ".join(raw_text.split())  # Limpieza de espacios y saltos de línea irregulares
        if clean_text:
            pages_data.append({
                "page": idx + 1,
                "text": clean_text
            })

    return pages_data


def chunk_text(pages_data: List[Dict[str, Any]], doc_name: str, chunk_size: int = 500, chunk_overlap: int = 60) -> List[Dict[str, Any]]:
    """
    Divide el texto de las páginas en fragmentos (chunks) más pequeños con solapamiento (overlap).
    Preserva los metadatos (nombre de documento, número de página y ID de fragmento).
    """
    chunks = []
    chunk_counter = 0

    for item in pages_data:
        page_num = item["page"]
        text = item["text"]
        words = text.split()

        if not words:
            continue

        step = max(1, chunk_size - chunk_overlap)
        for i in range(0, len(words), step):
            chunk_words = words[i:i + chunk_size]
            chunk_content = " ".join(chunk_words)

            # Evitar fragmentos extremadamente cortos
            if len(chunk_words) < 10 and chunks:
                continue

            chunk_id = f"{doc_name}_p{page_num}_c{chunk_counter}"
            chunks.append({
                "id": chunk_id,
                "text": chunk_content,
                "metadata": {
                    "document": doc_name,
                    "page": page_num,
                    "chunk_index": chunk_counter,
                    "word_count": len(chunk_words)
                }
            })
            chunk_counter += 1

    return chunks
