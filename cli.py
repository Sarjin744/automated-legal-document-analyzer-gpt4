"""
Command-Line Interface for Experiment 5: Automated Legal Document Analyzer
Course: 24CS2506 – Building Applications using GPT-4
Experiment 5: Automated Legal Document Analyzer using GPT-4
"""

import os
import sys
import argparse
import json
import time

# Ensure UTF-8 output in Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from engine.analyzer import LegalDocumentAnalyzer
from engine.text_extractor import LegalTextExtractor
from engine.chunker import LegalDocumentChunker
from engine.analytics import LegalAnalytics

SAMPLE_FILES = {
    "employment": "data/employment_agreement.txt",
    "nda": "data/non_disclosure_agreement.txt",
    "service": "data/service_agreement.txt",
    "rental": "data/rental_lease_agreement.txt",
    "long": "data/long_commercial_contract.txt"
}


def print_banner():
    banner = """
================================================================================
       24CS2506 – Building Applications using GPT-4
       Experiment 5: Automated Legal Document Analyzer using GPT-4
================================================================================
"""
    print(banner)


def main():
    parser = argparse.ArgumentParser(
        description="Automated Legal Document Analyzer using GPT-4 / Ollama / Deterministic NLP"
    )
    parser.add_argument(
        "-f", "--file",
        help="Path to legal document file (.txt, .pdf, .md)"
    )
    parser.add_argument(
        "-s", "--sample",
        choices=list(SAMPLE_FILES.keys()),
        default="employment",
        help="Run analysis on a built-in sample contract (default: employment)"
    )
    parser.add_argument(
        "-b", "--backend",
        choices=["auto", "openai", "gpt-4", "gpt-4o", "ollama", "deterministic"],
        default="auto",
        help="Execution backend (default: auto)"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model name for OpenAI or Ollama (default: gpt-4o)"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--chunk",
        action="store_true",
        help="Force document chunking and map-reduce pipeline (Section 9)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2500,
        help="Target chunk size in characters (default: 2500)"
    )
    parser.add_argument(
        "-o", "--export",
        help="File path to save the structured analysis (.txt, .json, or .md)"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Start interactive console mode"
    )

    args = parser.parse_args()
    print_banner()

    target_file = args.file
    if not target_file:
        target_file = SAMPLE_FILES.get(args.sample, SAMPLE_FILES["employment"])
        print(f"[*] No file specified. Using sample contract: '{args.sample}' ({target_file})")

    if not os.path.exists(target_file):
        print(f"[ERROR] Target file not found: {target_file}")
        sys.exit(1)

    print(f"[*] Extracting text from: {target_file} ...")
    raw_text, metadata = LegalTextExtractor.extract_from_file_path(target_file)
    print(f"[✓] Extracted: {metadata.get('format')} | Words: {metadata.get('word_count')} | Chars: {metadata.get('char_count')}")

    # Inspect chunks
    chunks = LegalDocumentChunker.chunk_document(raw_text, chunk_size=args.chunk_size, force_chunking=args.chunk)
    print(f"[*] Chunks generated: {len(chunks)} chunk(s)")

    print(f"[*] Initializing Legal Document Analyzer (Backend: {args.backend}) ...")
    analyzer = LegalDocumentAnalyzer(api_key=args.api_key)

    print("[*] Performing legal extraction and summarization ...")
    t0 = time.time()
    result = analyzer.analyze_document(
        document_text=raw_text,
        backend=args.backend,
        model_name=args.model,
        enable_chunking=args.chunk or (len(chunks) > 1),
        chunk_size=args.chunk_size
    )
    elapsed = time.time() - t0
    print(f"[✓] Completed in {elapsed:.2f}s using {result.engine_used}\n")

    # Display Standard Output Format (Section 10)
    print("=" * 60)
    print("RECOMMENDED LAB OUTPUT FORMAT (SECTION 10)")
    print("=" * 60)
    print(result.to_standard_format())
    print("=" * 60)

    # Compute Analytics
    readability = LegalAnalytics.analyze_readability(raw_text)
    risk = LegalAnalytics.calculate_risk_index(result)
    comp = LegalAnalytics.compare_summaries(raw_text, result.extractive_summary, result.summary)

    print("\n--- ANALYTICAL METRICS (SECTION 3 & 8) ---")
    print(f"• Readability Score: {readability['flesch_reading_ease']}/100 ({readability['complexity_rating']})")
    print(f"• Flesch-Kincaid Grade: {readability['fk_grade_level']}")
    print(f"• Risk Index: {risk['risk_score']}/100 [{risk['risk_level']}]")
    if risk['flags']:
        for flag in risk['flags']:
            print(f"  [!] {flag}")
    print(f"• Extractive Compression: {comp['extractive_compression_ratio']} ({comp['extractive_word_count']} words)")
    print(f"• Abstractive Compression: {comp['abstractive_compression_ratio']} ({comp['abstractive_word_count']} words)")

    # Export if requested
    if args.export:
        ext = os.path.splitext(args.export)[1].lower()
        if ext == ".json":
            with open(args.export, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            print(f"\n[✓] Results exported to JSON: {args.export}")
        elif ext == ".md":
            with open(args.export, "w", encoding="utf-8") as f:
                f.write(f"# Legal Document Analysis Report\n\n```\n{result.to_standard_format()}\n```\n")
            print(f"\n[✓] Results exported to Markdown: {args.export}")
        else:
            with open(args.export, "w", encoding="utf-8") as f:
                f.write(result.to_standard_format())
            print(f"\n[✓] Results exported to Text: {args.export}")


if __name__ == "__main__":
    main()
