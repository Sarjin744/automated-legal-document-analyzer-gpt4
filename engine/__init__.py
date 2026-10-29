"""
Engine package initialization for Experiment 5: Automated Legal Document Analyzer
Course: 24CS2506 – Building Applications using GPT-4
"""

from .models import LegalAnalysisResult, KeyClause, DocumentChunk
from .text_extractor import LegalTextExtractor
from .chunker import LegalDocumentChunker
from .prompt_builder import LegalPromptBuilder
from .llm_client import LegalLLMClient, DeterministicLegalEngine
from .analyzer import LegalDocumentAnalyzer
from .analytics import LegalAnalytics

__all__ = [
    "LegalAnalysisResult",
    "KeyClause",
    "DocumentChunk",
    "LegalTextExtractor",
    "LegalDocumentChunker",
    "LegalPromptBuilder",
    "LegalLLMClient",
    "DeterministicLegalEngine",
    "LegalDocumentAnalyzer",
    "LegalAnalytics"
]
