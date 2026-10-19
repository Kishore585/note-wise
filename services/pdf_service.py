"""
PDF Service for NoteWise
Handles reading, text extraction, page parsing, and validation using PyMuPDF.
"""

import io
import os
from typing import List, Dict, Any, Union
import pymupdf
from utils.text_processing import clean_text, compute_document_stats


class PDFExtractionResult:
    """Container for PDF extraction outputs."""
    def __init__(
        self,
        full_text: str,
        pages_data: List[Dict[str, Any]],
        stats: Dict[str, Any],
        is_scanned_or_empty: bool = False,
        warning_message: str = "",
        metadata: Dict[str, Any] = None
    ):
        self.full_text = full_text
        self.pages_data = pages_data
        self.stats = stats
        self.is_scanned_or_empty = is_scanned_or_empty
        self.warning_message = warning_message
        self.metadata = metadata or {}


def extract_text_from_pdf(pdf_source: Union[bytes, io.BytesIO, str]) -> PDFExtractionResult:
    """
    Extracts text page-by-page from a PDF document using PyMuPDF.
    
    Args:
        pdf_source: Can be file path (str), raw bytes (bytes), or io.BytesIO buffer.
        
    Returns:
        PDFExtractionResult object with full text, per-page data, and statistics.
    """
    doc = None
    try:
        # Handle empty raw input immediately
        if isinstance(pdf_source, (bytes, bytearray)) and len(pdf_source) == 0:
            return PDFExtractionResult(
                full_text="",
                pages_data=[],
                stats=compute_document_stats("", 0),
                is_scanned_or_empty=True,
                warning_message="The uploaded PDF file is completely empty (0 bytes)."
            )

        if isinstance(pdf_source, (bytes, bytearray)):
            doc = pymupdf.open(stream=pdf_source, filetype="pdf")
        elif isinstance(pdf_source, io.BytesIO):
            val = pdf_source.getvalue()
            if len(val) == 0:
                return PDFExtractionResult(
                    full_text="",
                    pages_data=[],
                    stats=compute_document_stats("", 0),
                    is_scanned_or_empty=True,
                    warning_message="The uploaded PDF buffer is empty (0 bytes)."
                )
            doc = pymupdf.open(stream=val, filetype="pdf")
        elif hasattr(pdf_source, "read"):
            data = pdf_source.read()
            if not data or len(data) == 0:
                return PDFExtractionResult(
                    full_text="",
                    pages_data=[],
                    stats=compute_document_stats("", 0),
                    is_scanned_or_empty=True,
                    warning_message="The uploaded document contains 0 bytes."
                )
            doc = pymupdf.open(stream=data, filetype="pdf")
        elif isinstance(pdf_source, str):
            if not os.path.exists(pdf_source) or os.path.getsize(pdf_source) == 0:
                return PDFExtractionResult(
                    full_text="",
                    pages_data=[],
                    stats=compute_document_stats("", 0),
                    is_scanned_or_empty=True,
                    warning_message="Specified PDF file path does not exist or is 0 bytes."
                )
            doc = pymupdf.open(pdf_source)
        else:
            raise ValueError(f"Unsupported PDF source type: {type(pdf_source)}")

        # Check for encrypted or password-protected PDFs
        if getattr(doc, "is_encrypted", False) and getattr(doc, "needs_pass", False):
            return PDFExtractionResult(
                full_text="",
                pages_data=[],
                stats=compute_document_stats("", 0),
                is_scanned_or_empty=True,
                warning_message="The uploaded PDF is password-protected/encrypted. Please upload an unprotected document."
            )

        num_pages = len(doc)
        if num_pages == 0:
            return PDFExtractionResult(
                full_text="",
                pages_data=[],
                stats=compute_document_stats("", 0),
                is_scanned_or_empty=True,
                warning_message="The uploaded PDF contains no pages."
            )

        pages_data = []
        full_text_parts = []
        total_images_found = 0

        for page_idx in range(num_pages):
            page = doc[page_idx]
            raw_text = page.get_text()
            cleaned = clean_text(raw_text)
            
            # Check for embedded images
            image_list = page.get_images()
            total_images_found += len(image_list)

            pages_data.append({
                "page_number": page_idx + 1,
                "text": cleaned,
                "word_count": len(cleaned.split()),
                "has_images": len(image_list) > 0,
                "image_count": len(image_list)
            })

            if cleaned:
                full_text_parts.append(cleaned)

        full_text = "\n\n".join(full_text_parts).strip()
        stats = compute_document_stats(full_text, num_pages)
        stats["image_count"] = total_images_found

        doc_meta = doc.metadata or {}

        # Graceful check for scanned or empty documents
        is_scanned_or_empty = False
        warning = ""

        if stats["word_count"] == 0:
            is_scanned_or_empty = True
            if total_images_found > 0:
                warning = (
                    "Warning: No extractable text was found in this PDF, but images were detected. "
                    "This appears to be a scanned document or photograph of notes. "
                    "Please ensure notes are text-based or provide searchable digital notes."
                )
            else:
                warning = "Warning: The document appears completely empty."
        elif stats["word_count"] < 30 and num_pages > 1 and total_images_found > 0:
            is_scanned_or_empty = True
            warning = (
                "Notice: Very little text extracted relative to page count. "
                "Some pages may be scanned images."
            )

        return PDFExtractionResult(
            full_text=full_text,
            pages_data=pages_data,
            stats=stats,
            is_scanned_or_empty=is_scanned_or_empty,
            warning_message=warning,
            metadata=doc_meta
        )

    except Exception as err:
        return PDFExtractionResult(
            full_text="",
            pages_data=[],
            stats=compute_document_stats("", 0),
            is_scanned_or_empty=True,
            warning_message=f"Could not read or parse the PDF document: {str(err)}"
        )

    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass
