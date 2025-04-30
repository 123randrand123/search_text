import io

import streamlit as st
from PyPDF2 import PdfReader, errors as pdf_errors
from docx import Document
from docx.opc.exceptions import PackageNotFoundError

from utils import logger


@st.cache_data(max_entries=10)
def pdf_to_text(file_content_bytes):
    """Extracts text from PDF file content bytes."""
    try:
        pdf_reader = PdfReader(io.BytesIO(file_content_bytes))
        if pdf_reader.is_encrypted:
            logger.warning("Uploaded PDF is encrypted.")
            st.warning("🔒 PDF is encrypted and cannot be processed.")
            return ""
        text = "\n\n".join(
            p.extract_text() or "" for p in pdf_reader.pages if p.extract_text()
        )
        logger.info(f"Extracted text from PDF ({len(text)} chars)")
        return text
    except pdf_errors.PdfReadError as e:
        logger.error(f"Error reading PDF (PdfReadError): {e}")
        st.error(f"Error reading PDF: Invalid or corrupted. Details: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading PDF: {e}", exc_info=True)
        st.error(f"Unexpected error reading PDF: {e}")
        return None


@st.cache_data(max_entries=10)
def docx_to_text(file_content_bytes):
    """Extracts text from DOCX file content bytes."""
    try:
        doc = Document(io.BytesIO(file_content_bytes))
        text = "\n\n".join(p.text for p in doc.paragraphs if p.text)
        logger.info(f"Extracted text from DOCX ({len(text)} chars)")
        return text
    except PackageNotFoundError as e:
        logger.error(f"Error reading DOCX (PackageNotFoundError): {e}")
        st.error(f"Error reading DOCX: Invalid or corrupted. Details: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading DOCX: {e}", exc_info=True)
        st.error(f"Unexpected error reading DOCX: {e}")
        return None
