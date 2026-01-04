import re
import nltk


try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


def clean_text(text: str) -> str:
    
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sentence_tokenize(text: str):
    
    return nltk.sent_tokenize(text)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
  
    text = clean_text(text)
    sentences = sentence_tokenize(text)

    chunks = []
    current_chunk = []

    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        length = len(words)

       
        if current_len + length > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))

           
            if overlap > 0:
                overlap_words = " ".join(current_chunk).split()[-overlap:]
                current_chunk = overlap_words.copy()
                current_len = len(current_chunk)
            else:
                current_chunk = []
                current_len = 0

        
        current_chunk.extend(words)
        current_len += length

   
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
