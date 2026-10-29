"""
Data Models for Experiment 5: Automated Legal Document Analyzer using GPT-4
Course: 24CS2506 – Building Applications using GPT-4
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class DocumentChunk:
    """Represents a discrete segment of a long legal document."""
    chunk_id: int
    total_chunks: int
    text: str
    char_count: int
    word_count: int
    section_title: Optional[str] = None


@dataclass
class KeyClause:
    """Represents an extracted legal clause with contextual analysis."""
    clause_type: str  # Confidentiality, Payment, Termination, Liability, Intellectual Property, Dispute Resolution, etc.
    title: str
    summary: str
    snippet: str
    risk_level: str = "Low"  # Low, Medium, High


@dataclass
class LegalAnalysisResult:
    """
    Standardized analysis result strictly matching Section 10 (Recommended Output Format)
    and supporting rich analytical introspection.
    """
    document_type: str
    parties: str
    purpose: str
    duration: str
    key_obligations: List[str]
    payment: str
    termination: str
    important_dates: Dict[str, str]
    potentially_important_clauses: List[str]
    summary: str
    
    # Extended structural attributes
    governing_law: str = "Not specified"
    liability: str = "Not specified"
    dispute_resolution: str = "Not specified"
    extractive_summary: List[str] = field(default_factory=list)
    detailed_clauses: List[KeyClause] = field(default_factory=list)
    raw_response: Optional[str] = None
    engine_used: str = "GPT-4"
    execution_time: float = 0.0
    chunk_count: int = 1
    
    def to_standard_format(self) -> str:
        """
        Formats the output exactly as specified in Experiment 5 PDF (Section 10 Recommended Output Format).
        """
        lines = [
            "LEGAL DOCUMENT ANALYSIS",
            "-----------------------",
            "Document Type:",
            self.document_type,
            "",
            "Parties:",
            self.parties,
            "",
            "Purpose:",
            self.purpose,
            "",
            "Duration:",
            self.duration,
            "",
            "Key Obligations:"
        ]
        for ob in self.key_obligations:
            prefix = "• " if not ob.strip().startswith("•") else ""
            lines.append(f"{prefix}{ob.strip()}")
            
        lines.extend([
            "",
            "Payment:",
            self.payment,
            "",
            "Termination:",
            self.termination,
            "",
            "Important Dates:"
        ])
        
        if self.important_dates:
            for k, v in self.important_dates.items():
                lines.append(f"{k}: {v}")
        else:
            lines.append("None explicitly stated")
            
        lines.extend([
            "",
            "Potentially Important Clauses:"
        ])
        for cl in self.potentially_important_clauses:
            prefix = "• " if not cl.strip().startswith("•") else ""
            lines.append(f"{prefix}{cl.strip()}")
            
        lines.extend([
            "",
            "Summary:",
            self.summary
        ])
        
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON export."""
        return {
            "document_type": self.document_type,
            "parties": self.parties,
            "purpose": self.purpose,
            "duration": self.duration,
            "key_obligations": self.key_obligations,
            "payment": self.payment,
            "termination": self.termination,
            "important_dates": self.important_dates,
            "potentially_important_clauses": self.potentially_important_clauses,
            "summary": self.summary,
            "governing_law": self.governing_law,
            "liability": self.liability,
            "dispute_resolution": self.dispute_resolution,
            "extractive_summary": self.extractive_summary,
            "detailed_clauses": [
                {
                    "clause_type": c.clause_type,
                    "title": c.title,
                    "summary": c.summary,
                    "snippet": c.snippet,
                    "risk_level": c.risk_level
                }
                for c in self.detailed_clauses
            ],
            "metadata": {
                "engine_used": self.engine_used,
                "execution_time_seconds": round(self.execution_time, 3),
                "chunks_analyzed": self.chunk_count
            }
        }
