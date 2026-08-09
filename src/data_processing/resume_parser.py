"""Utilities for parsing resumes (PDF or DOCX) into plain text.

The function `parse_resume` accepts a file-like object (as returned by Streamlit's
`st.file_uploader`) and returns a cleaned string containing the extracted text.
"""
import io
import pdfplumber
import docx

def _extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file given its raw bytes.

    Args:
        file_bytes: The binary content of the PDF.
    Returns:
        A string with the concatenated text of all pages.
    """
    text = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)

def _extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX file given its raw bytes.
    """
    doc = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(paragraphs)

def parse_resume(uploaded_file) -> str:
    """Parse an uploaded resume file (PDF or DOCX) into plain text.

    Parameters
    ----------
    uploaded_file: UploadedFile
        The file object provided by Streamlit. It must have `name` and `read()`.

    Returns
    -------
    str
        Cleaned textual representation of the resume.
    """
    # Read the bytes once
    file_bytes = uploaded_file.read()
    filename = uploaded_file.name.lower()
    if filename.endswith('.pdf'):
        raw_text = _extract_text_from_pdf(file_bytes)
    elif filename.endswith('.docx'):
        raw_text = _extract_text_from_docx(file_bytes)
    else:
        raise ValueError('Unsupported file type. Please upload a PDF or DOCX.')

    # Basic cleanup: normalize whitespace
    cleaned = "\n".join(line.strip() for line in raw_text.splitlines() if line.strip())
    return cleaned
