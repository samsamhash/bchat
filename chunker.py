import re
import nltk

# Descargar punkt si no está disponible
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


def clean_text(text: str) -> str:
    """
    Limpia texto básico: espacios, saltos de línea, etc.
    """
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sentence_tokenize(text: str):
    """
    Divide en oraciones usando NLTK.
    """
    return nltk.sent_tokenize(text)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Trocea el documento en chunks solapados.
    
    Estrategia:
    - Primero tokenizamos por oraciones para evitar cortar frases
    - Vamos construyendo chunks que aproximan `chunk_size` tokens (palabras)
    - Añadimos solapamiento de `overlap` palabras para mejorar coherencia RAG
    """
    text = clean_text(text)
    sentences = sentence_tokenize(text)

    chunks = []
    current_chunk = []

    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        length = len(words)

        # Si añadir esta frase excede el tamaño, cerramos chunk
        if current_len + length > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            # Comenzar un nuevo chunk con overlap
            if overlap > 0:
                overlap_words = " ".join(current_chunk).split()[-overlap:]
                current_chunk = overlap_words.copy()
                current_len = len(current_chunk)
            else:
                current_chunk = []
                current_len = 0

        # Añadimos la frase al chunk actual
        current_chunk.extend(words)
        current_len += length

    # último chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
