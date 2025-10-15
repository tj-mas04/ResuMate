"""
PDF text extraction service.
"""
import PyPDF2


def extract_text_from_pdf(file):
    """
    Extract text from a PDF file.
    
    Args:
        file: PDF file object
        
    Returns:
        str: Extracted text or error message
    """
    try:
        reader = PyPDF2.PdfReader(file)
        text = "".join(
            page.extract_text() or "" for page in reader.pages
        ).strip()
        return text if text else "No readable text found in PDF."
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"
