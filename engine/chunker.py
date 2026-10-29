"""
Intelligent Document Chunking Module for Long Legal Agreements
Course: 24CS2506 – Building Applications using GPT-4
Experiment 5: Automated Legal Document Analyzer using GPT-4
"""

import re
from typing import List
from .models import DocumentChunk


class LegalDocumentChunker:
    """
    Partitions long legal contracts into cohesive, semantic chunks
    adhering to Section 9 (Chunking Long Documents).
    """

    DEFAULT_CHUNK_SIZE = 2500       # Target characters per chunk (~400-500 words)
    DEFAULT_CHUNK_OVERLAP = 300     # Overlap characters to prevent boundary severance

    @classmethod
    def chunk_document(
        cls,
        text: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        force_chunking: bool = False
    ) -> List[DocumentChunk]:
        """
        Splits legal text into manageable chunks. If text is shorter than chunk_size and
        force_chunking is False, returns a single chunk.
        """
        if not text or not text.strip():
            return []

        text = text.strip()
        
        # If document is small enough and not explicitly forced, return single chunk
        if len(text) <= chunk_size and not force_chunking:
            return [
                DocumentChunk(
                    chunk_id=1,
                    total_chunks=1,
                    text=text,
                    char_count=len(text),
                    word_count=len(text.split()),
                    section_title="Entire Document"
                )
            ]

        # First attempt: Split by clause / section headers
        # E.g., "Section 1.", "Clause 2", "Article IV", "1. Parties", "2. Obligations"
        section_pattern = re.compile(
            r"(?:\n|^)(?=(?:SECTION\s+\d+|ARTICLE\s+[IVXLCDM\d]+|CLAUSE\s+\d+|\d+\.\s+[A-Z]))",
            re.IGNORECASE
        )
        
        raw_sections = [s.strip() for s in section_pattern.split(text) if s.strip()]
        
        chunks_text: List[str] = []
        current_chunk = ""
        
        if len(raw_sections) > 1:
            for sec in raw_sections:
                if len(current_chunk) + len(sec) + 2 <= chunk_size:
                    current_chunk = f"{current_chunk}\n\n{sec}" if current_chunk else sec
                else:
                    if current_chunk:
                        chunks_text.append(current_chunk.strip())
                    if len(sec) > chunk_size:
                        # Split very long single section using rolling window
                        sub_chunks = cls._rolling_window_split(sec, chunk_size, chunk_overlap)
                        chunks_text.extend(sub_chunks)
                        current_chunk = ""
                    else:
                        current_chunk = sec
            if current_chunk:
                chunks_text.append(current_chunk.strip())
        else:
            # Fallback to rolling window split
            chunks_text = cls._rolling_window_split(text, chunk_size, chunk_overlap)

        total = len(chunks_text)
        result: List[DocumentChunk] = []
        for i, c_text in enumerate(chunks_text, start=1):
            # Extract possible title
            first_line = c_text.splitlines()[0][:60] if c_text.splitlines() else f"Chunk {i}"
            result.append(
                DocumentChunk(
                    chunk_id=i,
                    total_chunks=total,
                    text=c_text,
                    char_count=len(c_text),
                    word_count=len(c_text.split()),
                    section_title=first_line
                )
            )
            
        return result

    @classmethod
    def _rolling_window_split(cls, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Splits long continuous text into overlapping chunks along paragraph or sentence boundaries."""
        paragraphs = text.split("\n\n")
        chunks = []
        current = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current) + len(para) + 2 <= chunk_size:
                current = f"{current}\n\n{para}" if current else para
            else:
                if current:
                    chunks.append(current)
                    # Keep tail overlap
                    overlap_text = current[-chunk_overlap:] if len(current) > chunk_overlap else ""
                    current = f"{overlap_text}\n\n{para}" if overlap_text else para
                else:
                    # Paragraph itself exceeds chunk_size, split by sentences
                    sentences = re.split(r"(?<=[.!?]) +", para)
                    sub_cur = ""
                    for s in sentences:
                        if len(sub_cur) + len(s) + 1 <= chunk_size:
                            sub_cur = f"{sub_cur} {s}" if sub_cur else s
                        else:
                            if sub_cur:
                                chunks.append(sub_cur.strip())
                            sub_cur = s
                    if sub_cur:
                        current = sub_cur
                        
        if current and (not chunks or chunks[-1] != current):
            chunks.append(current.strip())
            
        return chunks if chunks else [text]
