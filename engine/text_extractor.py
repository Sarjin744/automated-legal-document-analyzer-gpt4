"""
Text Extraction and Preprocessing Module for Legal Documents
Course: 24CS2506 – Building Applications using GPT-4
Experiment 5: Automated Legal Document Analyzer using GPT-4
"""

import io
import re
from typing import Dict, Any, Tuple, Optional


class LegalTextExtractor:
    """
    Extracts and preprocesses raw text from various document formats (PDF, TXT, MD, strings)
    for downstream GPT-4 legal analysis.
    """

    @staticmethod
    def clean_text(raw_text: str) -> str:
        """
        Preprocesses and normalizes legal text:
        - Harmonizes diverse quotation marks and dashes
        - Replaces excessive consecutive blank lines while preserving paragraph boundaries
        - Normalizes non-breaking spaces and formatting artifacts
        """
        if not raw_text:
            return ""

        text = raw_text
        # Normalize smart quotes and apostrophes
        text = text.replace("“", "\"").replace("”", "\"")
        text = text.replace("‘", "'").replace("’", "'")
        text = text.replace("—", " - ").replace("–", " - ")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        
        # Remove null bytes or control chars except standard whitespace
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        
        # Replace multiple horizontal spaces/tabs with single space
        text = re.sub(r"[ \t]+", " ", text)
        
        # Collapse 3 or more consecutive newlines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        return text.strip()

    @classmethod
    def extract_from_pdf_bytes(cls, pdf_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
        """Extract text and metadata from PDF bytes using pypdf."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(pdf_bytes))
            page_count = len(reader.pages)
            extracted_pages = []
            
            for idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                extracted_pages.append(f"--- Page {idx + 1} ---\n" + page_text)
                
            full_text = "\n\n".join(extracted_pages)
            cleaned = cls.clean_text(full_text)
            
            metadata = {
                "format": "PDF",
                "page_count": page_count,
                "char_count": len(cleaned),
                "word_count": len(cleaned.split()),
                "encrypted": reader.is_encrypted
            }
            return cleaned, metadata
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from PDF: {str(e)}")

    @classmethod
    def extract_from_file_path(cls, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract text from local file path (.pdf, .txt, .md)."""
        import os
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            with open(file_path, "rb") as f:
                return cls.extract_from_pdf_bytes(f.read())
        else:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read()
            cleaned = cls.clean_text(raw_text)
            metadata = {
                "format": ext.replace(".", "").upper() or "TXT",
                "page_count": 1,
                "char_count": len(cleaned),
                "word_count": len(cleaned.split()),
                "file_path": file_path
            }
            return cleaned, metadata

    @classmethod
    def extract_from_text(cls, raw_text: str, name: str = "Pasted Contract") -> Tuple[str, Dict[str, Any]]:
        """Extract and clean raw user-pasted text."""
        cleaned = cls.clean_text(raw_text)
        metadata = {
            "format": "TEXT",
            "page_count": 1,
            "char_count": len(cleaned),
            "word_count": len(cleaned.split()),
            "source_name": name
        }
        return cleaned, metadata
