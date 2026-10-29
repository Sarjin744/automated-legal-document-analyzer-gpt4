"""
Core Legal Document Analyzer Pipeline
Course: 24CS2506 – Building Applications using GPT-4
Experiment 5: Automated Legal Document Analyzer using GPT-4
"""

import json
import time
from typing import Optional, Dict, Any, List, Tuple

from .models import LegalAnalysisResult, KeyClause, DocumentChunk
from .text_extractor import LegalTextExtractor
from .chunker import LegalDocumentChunker
from .prompt_builder import LegalPromptBuilder
from .llm_client import LegalLLMClient, DeterministicLegalEngine


class LegalDocumentAnalyzer:
    """
    Executes the end-to-end Legal Document Analysis workflow described in Section 6:
    Legal Document -> Text Extraction -> Preprocessing -> Chunking ->
    Prompt Generation -> GPT-4 Execution -> Legal Information Extraction -> Structured Analysis -> Save Results.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.llm_client = LegalLLMClient(api_key=api_key)

    def analyze_document(
        self,
        document_text: str,
        backend: str = "auto",            # 'auto', 'gpt-4', 'ollama', 'deterministic'
        model_name: str = "gpt-4o",
        enable_chunking: bool = True,
        chunk_size: int = 2500,
        chunk_overlap: int = 300
    ) -> LegalAnalysisResult:
        """
        Orchestrates full analysis workflow.
        Handles both single document and chunked map-reduce pipelines.
        """
        start_time = time.time()
        cleaned_text = LegalTextExtractor.clean_text(document_text)

        if not cleaned_text:
            raise ValueError("Document text is empty or could not be extracted.")

        # Chunk document if requested
        chunks = LegalDocumentChunker.chunk_document(
            cleaned_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            force_chunking=False
        )

        # If backend is explicitly deterministic, run rule-based engine directly
        if backend == "deterministic":
            res = DeterministicLegalEngine.analyze(cleaned_text)
            res.execution_time = time.time() - start_time
            res.chunk_count = len(chunks)
            return res

        # If multi-chunk and chunking is enabled: Map-Reduce pipeline (Section 9)
        if len(chunks) > 1 and enable_chunking:
            return self._run_chunked_pipeline(
                chunks=chunks,
                backend=backend,
                model_name=model_name,
                full_text=cleaned_text,
                start_time=start_time
            )

        # Single document prompt generation (Section 5)
        system_prompt = LegalPromptBuilder.build_system_prompt()
        analysis_prompt = LegalPromptBuilder.build_analysis_prompt(cleaned_text)

        raw_response, engine_tag = self.llm_client.generate_chat(
            prompt=analysis_prompt,
            system_prompt=system_prompt,
            backend=backend,
            model_name=model_name
        )

        if raw_response:
            try:
                # Parse JSON
                data = self._clean_and_parse_json(raw_response)
                result = self._build_result_from_dict(data, engine_tag, time.time() - start_time, len(chunks))
                result.raw_response = raw_response
                return result
            except Exception:
                # If JSON parsing fails, fallback gracefully to deterministic engine
                pass

        # Fallback to Deterministic Engine
        fallback_res = DeterministicLegalEngine.analyze(cleaned_text)
        fallback_res.execution_time = time.time() - start_time
        fallback_res.chunk_count = len(chunks)
        fallback_res.engine_used = f"{fallback_res.engine_used} (Offline Fallback)"
        return fallback_res

    def _run_chunked_pipeline(
        self,
        chunks: List[DocumentChunk],
        backend: str,
        model_name: str,
        full_text: str,
        start_time: float
    ) -> LegalAnalysisResult:
        """
        Map-Reduce pipeline for long legal documents (Section 9):
        1. Map: Analyze each chunk individually
        2. Reduce: Combine chunk summaries into final structured synthesis
        """
        system_prompt = LegalPromptBuilder.build_system_prompt()
        chunk_summaries: List[Dict[str, Any]] = []
        engine_tag = "GPT-4 (Chunked Map-Reduce)"

        # Check if LLM is accessible
        client = self.llm_client.get_openai_client() if backend != "ollama" else None
        ollama_ok = self.llm_client.is_ollama_available()

        if not client and not ollama_ok and backend != "auto":
            res = DeterministicLegalEngine.analyze(full_text)
            res.execution_time = time.time() - start_time
            res.chunk_count = len(chunks)
            res.engine_used = "Deterministic Legal Engine (Chunked Batch)"
            return res

        for chunk in chunks:
            chunk_prompt = LegalPromptBuilder.build_chunk_analysis_prompt(
                chunk.text,
                chunk.chunk_id,
                chunk.total_chunks
            )
            raw, tag = self.llm_client.generate_chat(
                prompt=chunk_prompt,
                system_prompt=system_prompt,
                backend=backend,
                model_name=model_name
            )
            engine_tag = tag
            if raw:
                try:
                    c_data = self._clean_and_parse_json(raw)
                    chunk_summaries.append(c_data)
                except Exception:
                    chunk_summaries.append({"chunk_id": chunk.chunk_id, "section_summary": chunk.text[:200]})
            else:
                chunk_summaries.append({"chunk_id": chunk.chunk_id, "section_summary": chunk.text[:200]})

        # Reduce stage: Combine chunks
        reduce_prompt = LegalPromptBuilder.build_combine_chunks_prompt(chunk_summaries)
        raw_final, final_tag = self.llm_client.generate_chat(
            prompt=reduce_prompt,
            system_prompt=system_prompt,
            backend=backend,
            model_name=model_name
        )

        if raw_final:
            try:
                final_data = self._clean_and_parse_json(raw_final)
                res = self._build_result_from_dict(final_data, f"{final_tag} [Chunked Map-Reduce]", time.time() - start_time, len(chunks))
                res.raw_response = raw_final
                return res
            except Exception:
                pass

        # Fallback to deterministic
        fallback_res = DeterministicLegalEngine.analyze(full_text)
        fallback_res.execution_time = time.time() - start_time
        fallback_res.chunk_count = len(chunks)
        return fallback_res

    def _clean_and_parse_json(self, raw_str: str) -> Dict[str, Any]:
        """Strips markdown code fences and cleans invalid json strings."""
        text = raw_str.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Find first '{' and last '}'
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end+1]

        return json.loads(text)

    def _build_result_from_dict(
        self,
        data: Dict[str, Any],
        engine_name: str,
        exec_time: float,
        chunk_count: int
    ) -> LegalAnalysisResult:
        """Constructs LegalAnalysisResult from parsed dictionary."""
        # Convert important_dates
        dates_raw = data.get("important_dates", {})
        if not isinstance(dates_raw, dict):
            dates_raw = {"Key Date": str(dates_raw)}

        # Format dates as string keys/values
        cleaned_dates = {str(k): str(v) for k, v in dates_raw.items()}

        # Build detailed clauses
        clauses_raw = data.get("potentially_important_clauses", [])
        detailed = []
        for c in clauses_raw:
            detailed.append(
                KeyClause(
                    clause_type=str(c),
                    title=f"{c} Clause",
                    summary=f"Contractual provisions governing {c.lower()}.",
                    snippet=f"Document specifies explicit guidelines relating to {c.lower()}."
                )
            )

        return LegalAnalysisResult(
            document_type=str(data.get("document_type", "Legal Agreement")),
            parties=str(data.get("parties", "Specified Parties")),
            purpose=str(data.get("purpose", "Establish legal obligations.")),
            duration=str(data.get("duration", "As specified in agreement")),
            key_obligations=list(data.get("key_obligations", ["Maintain contractual terms."])),
            payment=str(data.get("payment", "Terms as defined in contract")),
            termination=str(data.get("termination", "Standard notice required")),
            important_dates=cleaned_dates,
            potentially_important_clauses=list(clauses_raw),
            summary=str(data.get("summary", "Summary of contract terms.")),
            extractive_summary=list(data.get("extractive_summary", [])),
            governing_law=str(data.get("governing_law", "Applicable Law")),
            liability=str(data.get("liability", "Not specified")),
            dispute_resolution=str(data.get("dispute_resolution", "Standard")),
            detailed_clauses=detailed,
            engine_used=engine_name,
            execution_time=exec_time,
            chunk_count=chunk_count
        )
