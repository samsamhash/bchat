# ingest_client.py
from vector_store import VectorStore
from chunker import chunk_text  


vs = VectorStore()  

def ingest_document(doc_id: str, text: str, chunk_size: int = 500, overlap: int = 50):
    if not text or len(text.strip()) == 0:
        raise ValueError("empty text.")

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if len(chunks) == 0:
        raise RuntimeError("could not generate the chunks for the doc.")

    added = vs.add(doc_id, chunks)
    vs.save()

    return {
        "status": "ok",
        "chunks_added": added,
        "doc_id": doc_id,
        "index_info": vs.info()
    }

