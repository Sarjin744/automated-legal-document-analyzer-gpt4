"""
Engineered Prompt Construction Module for Legal Document Analysis
Course: 24CS2506 – Building Applications using GPT-4
Experiment 5: Automated Legal Document Analyzer using GPT-4
"""

import json
from typing import List, Dict, Any, Optional


class LegalPromptBuilder:
    """
    Builds structured prompts strictly compliant with Section 5 (Components of a Good Legal Analysis Prompt)
    and Section 9 (Chunking and Map-Reduce Aggregation).
    """

    SYSTEM_PROMPT = (
        "You are a specialized legal document analysis assistant. "
        "Your task is to analyze contracts, agreements, and legal instruments with extreme precision, "
        "identifying obligations, covenants, liabilities, and key terms without hallucination."
    )

    @classmethod
    def build_system_prompt(cls) -> str:
        """Returns standard system role prompt."""
        return cls.SYSTEM_PROMPT

    @classmethod
    def build_analysis_prompt(cls, document_text: str, summarization_type: str = "both") -> str:
        """
        Constructs the main legal analysis prompt incorporating Section 5 specifications:
        Role, Task, Information to Extract, Constraints, and Output Format.
        """
        prompt = f"""### ROLE
You are a legal document analysis assistant.

### TASK
Analyze and summarize the provided legal document thoroughly.

### INFORMATION TO EXTRACT
Identify and extract:
1. Document Type (e.g., Employment Agreement, Non-Disclosure Agreement, Service Agreement, Rental Agreement, etc.)
2. Parties involved (names of individuals, companies, roles like Employer/Employee or Licensor/Licensee)
3. Purpose of the agreement (clear statement of what this document establishes)
4. Duration / Term (e.g., 12 months, ongoing, fixed term)
5. Key Obligations (explicit duties and obligations of each party)
6. Payment terms (compensation, rates, invoicing, deadlines, currency)
7. Termination conditions (notice period, grounds for immediate termination)
8. Important Dates (Start Date, End Date, Effective Date, Signing Date)
9. Potentially Important Clauses (Confidentiality, Liability, Intellectual Property, Dispute Resolution, Governing Law)
10. Summary:
    - Abstractive Summary: Plain language summary of the contract's core meaning and obligations.
    - Extractive Summary: Key verbatim sentences extracted directly from the original text.

### CONSTRAINTS
- Use ONLY information present in the document.
- Do NOT invent facts or hallucinate terms not mentioned.
- If a field is not specified in the document, explicitly write "Not specified in document".

### OUTPUT FORMAT
Return your response strictly as valid, parsable JSON matching this schema:
{{
  "document_type": "string",
  "parties": "string",
  "purpose": "string",
  "duration": "string",
  "key_obligations": [
    "string"
  ],
  "payment": "string",
  "termination": "string",
  "important_dates": {{
    "Start Date": "string",
    "End Date": "string"
  }},
  "potentially_important_clauses": [
    "Confidentiality",
    "Termination",
    "Intellectual Property",
    "Liability",
    "Governing Law"
  ],
  "summary": "string",
  "extractive_summary": [
    "verbatim sentence 1 from contract",
    "verbatim sentence 2 from contract"
  ],
  "governing_law": "string",
  "liability": "string",
  "dispute_resolution": "string"
}}

### LEGAL DOCUMENT TEXT:
\"\"\"
{document_text}
\"\"\"
"""
        return prompt

    @classmethod
    def build_chunk_analysis_prompt(cls, chunk_text: str, chunk_id: int, total_chunks: int) -> str:
        """Prompt to analyze an individual chunk in a multi-chunk document (Map stage)."""
        prompt = f"""### ROLE
You are a legal document analysis assistant.

### TASK
Analyze section {chunk_id} of {total_chunks} of a long legal document.
Extract all discernible entities, clauses, obligations, and dates present in this specific section.

### CONSTRAINTS
- Rely solely on the provided chunk text.
- Do not make assumptions beyond the text.

### OUTPUT FORMAT
Provide a concise JSON with:
{{
  "chunk_id": {chunk_id},
  "detected_clauses": ["Clause names or topics discussed"],
  "parties_mentioned": ["Any parties mentioned"],
  "obligations_found": ["Key duties stated in this section"],
  "dates_found": {{"Date Name": "Value"}},
  "payment_mentions": "Any fee/payment mention or null",
  "termination_mentions": "Any termination mention or null",
  "section_summary": "1-2 sentences summarizing this chunk"
}}

### CHUNK TEXT:
\"\"\"
{chunk_text}
\"\"\"
"""
        return prompt

    @classmethod
    def build_combine_chunks_prompt(cls, chunk_summaries: List[Dict[str, Any]]) -> str:
        """Prompt to combine intermediate chunk findings into the final analysis (Reduce stage)."""
        chunks_json_str = json.dumps(chunk_summaries, indent=2)
        prompt = f"""### ROLE
You are a senior legal document analysis assistant.

### TASK
Combine the extracted findings from all document sections into a single comprehensive legal document analysis.
Reconcile conflicting clauses, synthesize complete obligations, verify all parties and dates, and generate an overall summary.

### INTERMEDIATE SECTION FINDINGS:
{chunks_json_str}

### REQUIRED OUTPUT FORMAT:
Return strictly valid JSON matching:
{{
  "document_type": "string",
  "parties": "string",
  "purpose": "string",
  "duration": "string",
  "key_obligations": ["string"],
  "payment": "string",
  "termination": "string",
  "important_dates": {{"Start Date": "string"}},
  "potentially_important_clauses": ["string"],
  "summary": "string",
  "extractive_summary": ["string"],
  "governing_law": "string",
  "liability": "string",
  "dispute_resolution": "string"
}}
"""
        return prompt
