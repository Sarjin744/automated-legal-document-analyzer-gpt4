"""
Automated Test Suite for Legal Document Analyzer (Experiment 5)
Verifies all modules, data integrity, pipeline execution, PDF extraction,
chunking, analytics, and export formats.
"""

import os
import sys
import json
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from engine.models import LegalAnalysisResult, KeyClause, DocumentChunk
from engine.text_extractor import LegalTextExtractor
from engine.chunker import LegalDocumentChunker
from engine.prompt_builder import LegalPromptBuilder
from engine.llm_client import LegalLLMClient, DeterministicLegalEngine
from engine.analyzer import LegalDocumentAnalyzer
from engine.analytics import LegalAnalytics


def test_text_extractor():
    print("[TEST 1/8] Testing LegalTextExtractor...")
    # 1. Clean text
    dirty = "This  is   a “contract”\r\nwith smart—quotes and   multiple   spaces.\n\n\n\nNext clause."
    cleaned = LegalTextExtractor.clean_text(dirty)
    assert "“" not in cleaned and "”" not in cleaned, "Smart quotes not cleaned"
    assert "—" not in cleaned, "Em-dash not cleaned"
    assert "\r" not in cleaned, "Carriage returns not removed"
    assert "\n\n\n" not in cleaned, "Excessive newlines not collapsed"
    assert "multiple spaces" in cleaned, "Spacing not normalized"
    
    # 2. Extract from TXT
    sample_file = "data/employment_agreement.txt"
    txt, meta = LegalTextExtractor.extract_from_file_path(sample_file)
    assert len(txt) > 500, "Text extraction returned too few characters"
    assert meta["format"] == "TXT", f"Unexpected format: {meta.get('format')}"
    assert meta["word_count"] > 100, "Word count too low"

    # 3. Test PDF generation and extraction with pypdf
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    pdf_path = "test_sample.pdf"
    with open(pdf_path, "wb") as f:
        writer.write(f)
    try:
        pdf_txt, pdf_meta = LegalTextExtractor.extract_from_file_path(pdf_path)
        assert pdf_meta["format"] == "PDF", "PDF format not recognized"
        assert pdf_meta["page_count"] == 1, "Page count mismatch"
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)

    print("   [PASS] LegalTextExtractor working correctly.")


def test_chunker():
    print("[TEST 2/8] Testing LegalDocumentChunker...")
    sample_file = "data/long_commercial_contract.txt"
    with open(sample_file, "r", encoding="utf-8") as f:
        long_text = f.read()

    # Test single-chunk mode on short text
    short_text = "This is a short contract text."
    c_single = LegalDocumentChunker.chunk_document(short_text, chunk_size=1000)
    assert len(c_single) == 1, "Short text should produce exactly 1 chunk"

    # Test long contract chunking
    chunks = LegalDocumentChunker.chunk_document(long_text, chunk_size=1800, chunk_overlap=200, force_chunking=True)
    assert len(chunks) >= 3, f"Long contract should produce >=3 chunks, got {len(chunks)}"
    
    for c in chunks:
        assert c.chunk_id > 0, "Invalid chunk_id"
        assert c.total_chunks == len(chunks), "Total chunks mismatch"
        assert len(c.text.strip()) > 0, "Empty chunk text"
        assert c.word_count > 0, "Invalid word count"

    # Test edge case: empty string
    empty_c = LegalDocumentChunker.chunk_document("")
    assert len(empty_c) == 0, "Empty text should produce 0 chunks"

    print("   [PASS] LegalDocumentChunker working correctly.")


def test_prompt_builder():
    print("[TEST 3/8] Testing LegalPromptBuilder...")
    sys_prompt = LegalPromptBuilder.build_system_prompt()
    assert "legal document analysis" in sys_prompt.lower(), "System prompt missing role keyword"

    analysis_prompt = LegalPromptBuilder.build_analysis_prompt("Sample Agreement Text Here")
    assert "### ROLE" in analysis_prompt, "Missing ROLE section"
    assert "### TASK" in analysis_prompt, "Missing TASK section"
    assert "### INFORMATION TO EXTRACT" in analysis_prompt, "Missing extraction section"
    assert "### CONSTRAINTS" in analysis_prompt, "Missing constraints section"
    assert "do not invent facts" in analysis_prompt.lower(), "Missing negative constraint"
    assert "Sample Agreement Text Here" in analysis_prompt, "Text not injected"

    chunk_prompt = LegalPromptBuilder.build_chunk_analysis_prompt("Chunk text", 1, 3)
    assert "section 1 of 3" in chunk_prompt.lower(), "Chunk index missing"

    combine_prompt = LegalPromptBuilder.build_combine_chunks_prompt([{"chunk_id": 1, "summary": "Test"}])
    assert "INTERMEDIATE SECTION FINDINGS" in combine_prompt, "Combine prompt missing intermediate data"

    print("   [PASS] LegalPromptBuilder working correctly.")


def test_deterministic_engine():
    print("[TEST 4/8] Testing DeterministicLegalEngine...")
    with open("data/employment_agreement.txt", "r", encoding="utf-8") as f:
        text = f.read()

    res = DeterministicLegalEngine.analyze(text)
    assert res.document_type == "Employment Agreement", f"Doc type wrong: {res.document_type}"
    assert "Apex Global Technologies" in res.parties, f"Party extraction failed: {res.parties}"
    assert "Rahul Sharma" in res.parties, f"Employee extraction failed: {res.parties}"
    assert "12 months" in res.duration, f"Duration extraction failed: {res.duration}"
    assert "₹50,000" in res.payment, f"Payment extraction failed: {res.payment}"
    assert "30 days" in res.termination, f"Termination extraction failed: {res.termination}"
    assert len(res.key_obligations) > 0, "Key obligations list is empty"
    assert len(res.potentially_important_clauses) >= 3, "Insufficient clauses extracted"
    assert len(res.summary) > 50, "Abstractive summary is too short"
    assert len(res.extractive_summary) > 0, "Extractive summary is empty"

    print("   [PASS] DeterministicLegalEngine working correctly.")


def test_analytics():
    print("[TEST 5/8] Testing LegalAnalytics...")
    sample_text = (
        "The employee shall maintain strict confidentiality regarding all proprietary information "
        "obtained during the period of employment and shall not disclose such information to any third party "
        "without prior written authorization from the Employer. This Agreement shall be governed by the laws of India."
    )
    
    # Readability
    read = LegalAnalytics.analyze_readability(sample_text)
    assert "flesch_reading_ease" in read, "Missing reading ease"
    assert "fk_grade_level" in read, "Missing grade level"
    assert read["total_words"] > 10, "Word count mismatch"
    assert read["flesch_reading_ease"] >= 0, "Invalid reading ease score"

    # Clause coverage
    detected = ["Confidentiality", "Termination", "Governing Law"]
    cov = LegalAnalytics.evaluate_clause_coverage(detected)
    assert cov["coverage_percentage"] > 0, "Clause coverage calculation error"
    assert "Confidentiality" in cov["present_clauses"], "Present clause missing"
    assert "Liability" in cov["missing_clauses"], "Missing clause not flagged"

    # Risk Index
    dummy_res = LegalAnalysisResult(
        document_type="Contract",
        parties="A and B",
        purpose="Test",
        duration="12 months",
        key_obligations=["Do X"],
        payment="100",
        termination="immediate without notice",
        important_dates={},
        potentially_important_clauses=["Termination"],
        summary="Test summary"
    )
    risk = LegalAnalytics.calculate_risk_index(dummy_res)
    assert risk["risk_score"] > 20, "High risk document not scored appropriately"
    assert len(risk["flags"]) > 0, "Risk flags should be populated"

    # Comparison metrics
    comp = LegalAnalytics.compare_summaries(sample_text, ["Verbatim sentence here."], "Abstractive sentence here.")
    assert "original_word_count" in comp, "Original word count missing"
    assert "abstractive_compression_ratio" in comp, "Abstractive compression missing"

    print("   [PASS] LegalAnalytics working correctly.")


def test_all_sample_contracts():
    print("[TEST 6/8] Testing LegalDocumentAnalyzer on all 5 sample contracts...")
    analyzer = LegalDocumentAnalyzer()
    samples = [
        ("employment", "data/employment_agreement.txt", "Employment"),
        ("nda", "data/non_disclosure_agreement.txt", "Non-Disclosure"),
        ("service", "data/service_agreement.txt", "Service"),
        ("rental", "data/rental_lease_agreement.txt", "Rental"),
        ("long", "data/long_commercial_contract.txt", "Agreement")
    ]

    for name, path, expected_type in samples:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        res = analyzer.analyze_document(content, backend="deterministic")
        assert expected_type.lower() in res.document_type.lower(), f"Contract {name} doc_type expected {expected_type}, got {res.document_type}"
        assert len(res.parties) > 0, f"Contract {name} missing parties"
        assert len(res.key_obligations) > 0, f"Contract {name} missing obligations"
        assert len(res.potentially_important_clauses) > 0, f"Contract {name} missing clauses"
        assert len(res.summary) > 0, f"Contract {name} missing summary"
        
        # Test standard format formatting
        formatted = res.to_standard_format()
        assert "LEGAL DOCUMENT ANALYSIS" in formatted, f"Standard header missing in {name}"
        assert "Document Type:" in formatted, f"Document Type field missing in {name}"
        assert "Parties:" in formatted, f"Parties field missing in {name}"
        assert "Key Obligations:" in formatted, f"Obligations missing in {name}"
        assert "Summary:" in formatted, f"Summary missing in {name}"

        # Test dictionary conversion
        data_dict = res.to_dict()
        assert isinstance(data_dict, dict), f"to_dict failed for {name}"
        assert "metadata" in data_dict, f"metadata missing in {name}"

    print("   [PASS] All 5 sample contracts analyzed successfully.")


def test_chunked_map_reduce_pipeline():
    print("[TEST 7/8] Testing Chunked Map-Reduce Pipeline (Section 9)...")
    analyzer = LegalDocumentAnalyzer()
    with open("data/long_commercial_contract.txt", "r", encoding="utf-8") as f:
        content = f.read()

    res = analyzer.analyze_document(
        document_text=content,
        backend="deterministic",
        enable_chunking=True,
        chunk_size=1500
    )
    assert res.chunk_count >= 3, f"Expected >= 3 chunks, got {res.chunk_count}"
    assert "24 months" in res.duration or "months" in res.duration, f"Duration extraction in chunked failed: {res.duration}"
    assert res.execution_time > 0, "Execution time should be tracked"

    print(f"   [PASS] Chunked Map-Reduce pipeline executed cleanly ({res.chunk_count} chunks).")


def test_export_formats():
    print("[TEST 8/8] Testing Result Exporting (TXT, JSON, MD)...")
    analyzer = LegalDocumentAnalyzer()
    with open("data/employment_agreement.txt", "r", encoding="utf-8") as f:
        content = f.read()
    res = analyzer.analyze_document(content, backend="deterministic")

    # 1. Export TXT
    txt_path = "test_export.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(res.to_standard_format())
    assert os.path.exists(txt_path) and os.path.getsize(txt_path) > 200, "TXT export failed"
    os.remove(txt_path)

    # 2. Export JSON
    json_path = "test_export.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(res.to_dict(), f, indent=2)
    assert os.path.exists(json_path) and os.path.getsize(json_path) > 200, "JSON export failed"
    # Verify valid JSON
    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["document_type"] == "Employment Agreement", "JSON content mismatch"
    os.remove(json_path)

    # 3. Export Markdown
    md_path = "test_export.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Legal Document Analysis\n\n```\n{res.to_standard_format()}\n```\n")
    assert os.path.exists(md_path) and os.path.getsize(md_path) > 200, "MD export failed"
    os.remove(md_path)

    print("   [PASS] All export formats verified.")


def main():
    print("=" * 70)
    print("STARTING FULL END-TO-END VERIFICATION SCAN FOR EXPERIMENT 5")
    print("=" * 70)
    t0 = time.time()
    
    test_text_extractor()
    test_chunker()
    test_prompt_builder()
    test_deterministic_engine()
    test_analytics()
    test_all_sample_contracts()
    test_chunked_map_reduce_pipeline()
    test_export_formats()

    elapsed = time.time() - t0
    print("=" * 70)
    print(f"🎉 ALL 8 TESTS PASSED SUCCESSFULLY IN {elapsed:.2f}s!")
    print("The Automated Legal Document Analyzer is fully functional and ready.")
    print("=" * 70)


if __name__ == "__main__":
    main()
