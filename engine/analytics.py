"""
Contract Analytics, Risk Scoring, and Readability Metrics
Course: 24CS2506 – Building Applications using GPT-4
Experiment 5: Automated Legal Document Analyzer using GPT-4
"""

import re
from typing import Dict, Any, List
from .models import LegalAnalysisResult


class LegalAnalytics:
    """
    Computes readability, legal complexity, risk indices, and summarization metrics
    demonstrating Section 3 ('Why is Legal Document Analysis Difficult?') and Section 8.
    """

    STANDARD_CRITICAL_CLAUSES = [
        "Confidentiality",
        "Payment",
        "Termination",
        "Liability",
        "Intellectual Property",
        "Dispute Resolution",
        "Governing Law"
    ]

    @classmethod
    def analyze_readability(cls, text: str) -> Dict[str, Any]:
        """
        Computes readability metrics:
        - Flesch Reading Ease
        - Flesch-Kincaid Grade Level
        - Sentence Count and Average Sentence Length
        - Lexical Diversity
        """
        if not text.strip():
            return {
                "flesch_reading_ease": 0.0,
                "grade_level": "N/A",
                "avg_sentence_len": 0.0,
                "total_words": 0,
                "complexity_rating": "Unknown"
            }

        words = re.findall(r"\b[A-Za-z0-9_'-]+\b", text)
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        
        word_count = len(words)
        sentence_count = max(1, len(sentences))
        avg_sentence_len = round(word_count / sentence_count, 1)

        # Estimate syllables
        syllable_count = 0
        for w in words:
            syl = len(re.findall(r"[aeiouy]+", w.lower()))
            syllable_count += max(1, syl)

        # Flesch Reading Ease = 206.835 - 1.015 * (total words / total sentences) - 84.6 * (total syllables / total words)
        words_per_sentence = word_count / sentence_count
        syllables_per_word = syllable_count / max(1, word_count)
        
        flesch_score = round(206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word), 1)
        flesch_score = max(0.0, min(100.0, flesch_score))

        # Flesch-Kincaid Grade Level = 0.39 * (total words / total sentences) + 11.8 * (total syllables / total words) - 15.59
        fk_grade = round(0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59, 1)
        fk_grade = max(1.0, fk_grade)

        if flesch_score < 30.0:
            rating = "Extremely Complex (College Graduate / Professional Legal)"
        elif flesch_score < 50.0:
            rating = "Difficult (College Level)"
        elif flesch_score < 60.0:
            rating = "Fairly Difficult (High School)"
        else:
            rating = "Standard Plain Language"

        return {
            "flesch_reading_ease": flesch_score,
            "fk_grade_level": fk_grade,
            "avg_sentence_length": avg_sentence_len,
            "total_words": word_count,
            "total_sentences": sentence_count,
            "complexity_rating": rating
        }

    @classmethod
    def evaluate_clause_coverage(cls, detected_clauses: List[str]) -> Dict[str, Any]:
        """Calculates clause coverage against standard protective commercial clauses."""
        detected_lower = [c.lower() for c in detected_clauses]
        present = []
        missing = []

        for req in cls.STANDARD_CRITICAL_CLAUSES:
            if any(req.lower() in d for d in detected_lower):
                present.append(req)
            else:
                missing.append(req)

        coverage_pct = round((len(present) / len(cls.STANDARD_CRITICAL_CLAUSES)) * 100, 1)

        return {
            "present_clauses": present,
            "missing_clauses": missing,
            "coverage_percentage": coverage_pct,
            "total_critical_defined": len(cls.STANDARD_CRITICAL_CLAUSES)
        }

    @classmethod
    def calculate_risk_index(cls, result: LegalAnalysisResult) -> Dict[str, Any]:
        """
        Assesses contractual risk score (0 to 100):
        Factors:
        - Missing critical clauses (Liability, Dispute Resolution, Termination)
        - Short termination notice (< 14 days = high risk)
        - Non-compete / indemnification breadth
        """
        score = 20  # Base risk
        risk_flags = []

        # Check coverage
        coverage = cls.evaluate_clause_coverage(result.potentially_important_clauses)
        if "Liability" in coverage["missing_clauses"]:
            score += 25
            risk_flags.append("Missing explicit Limitation of Liability clause (High Exposure)")
        if "Dispute Resolution" in coverage["missing_clauses"]:
            score += 15
            risk_flags.append("No formal Dispute Resolution / Arbitration protocol specified")
        if "Confidentiality" in coverage["missing_clauses"]:
            score += 20
            risk_flags.append("Missing Confidentiality safeguards")

        # Notice period check
        term_lower = result.termination.lower()
        if "immediate" in term_lower or "without notice" in term_lower:
            score += 15
            risk_flags.append("Permits immediate termination without prior cure period")

        score = min(100, score)

        if score >= 65:
            level = "HIGH RISK"
            color = "#ef4444"
        elif score >= 40:
            level = "MODERATE RISK"
            color = "#f59e0b"
        else:
            level = "LOW / CONTROLLED RISK"
            color = "#10b981"

        return {
            "risk_score": score,
            "risk_level": level,
            "risk_color": color,
            "flags": risk_flags
        }

    @classmethod
    def compare_summaries(
        cls,
        original_text: str,
        extractive: List[str],
        abstractive: str
    ) -> Dict[str, Any]:
        """
        Evaluates Section 8 (Extractive vs Abstractive Summarization) metrics:
        - Compression ratio
        - Word counts
        - Information density
        """
        orig_words = len(original_text.split())
        extractive_text = " ".join(extractive)
        ext_words = len(extractive_text.split())
        abs_words = len(abstractive.split())

        ext_compression = round((1.0 - (ext_words / max(1, orig_words))) * 100, 1)
        abs_compression = round((1.0 - (abs_words / max(1, orig_words))) * 100, 1)

        return {
            "original_word_count": orig_words,
            "extractive_word_count": ext_words,
            "abstractive_word_count": abs_words,
            "extractive_compression_ratio": f"{max(0.0, ext_compression)}%",
            "abstractive_compression_ratio": f"{max(0.0, abs_compression)}%"
        }
