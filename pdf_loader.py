"""
pdf_loader.py
Carga y extrae texto de archivos PDF en Google Colab.

Requiere:
    pip install pdfplumber
"""

import pdfplumber

def load_pdf_text(file_path: str) -> str:
    """
    Carga un PDF desde la ruta `file_path` y devuelve todo el texto extraído.
    
    Args:
        file_path (str): Ruta al archivo PDF.
    
    Returns:
        str: Todo el texto concatenado del PDF.
    """
    text = []

    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text.append(extracted)
    except Exception as e:
        raise RuntimeError(f"Error leyendo PDF: {e}")

    if not text:
        return ""

    return "\n".join(text)
