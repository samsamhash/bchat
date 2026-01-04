"""
pdf_loader.py
Load pdf and get text
"""

import pdfplumber

def load_pdf_text(file_path: str) -> str:
   
    text = []

    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text.append(extracted)
    except Exception as e:
        raise RuntimeError(f"Error reading PDF: {e}")

    if not text:
        return ""

    return "\n".join(text)

