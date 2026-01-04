# ingest_client.py
from vector_store import VectorStore
from chunker import chunk_text  # tu función existente de chunking

# usa los mismos paths por defecto que VectorStore
vs = VectorStore()  

def ingest_document(doc_id: str, text: str, chunk_size: int = 500, overlap: int = 50):
    if not text or len(text.strip()) == 0:
        raise ValueError("El texto a ingestar está vacío.")

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if len(chunks) == 0:
        raise RuntimeError("No se pudieron generar chunks del documento.")

    added = vs.add(doc_id, chunks)
    vs.save()  # guarda índice + metadata en disco

    return {
        "status": "ok",
        "chunks_added": added,
        "doc_id": doc_id,
        "index_info": vs.info()
    }
