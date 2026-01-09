# =============================================================================
# Banka Dokümanları Preparation Script - PDF Only
# =============================================================================
"""
Banka Dokümanları Preparation Script - PDF Only

This script processes PDF documents to extract,
chunk, and prepare vector-ready data for AI applications.

Features:
- PDF document parsing with page tracking
- Text chunking for retrieval systems
- Vector-ready subchunking with embedding optimization
- Comprehensive quality control and logging

Author: Can Mergen
Date: 2026-01-02
Version: 1.0.0
"""

# =============================================================================
# IMPORTS
# =============================================================================

import sys
import logging
from pathlib import Path
import re
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from pathlib import Path

# Third-party imports
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

# Local imports
from scope_utils import assign_bank_document_scopes

# =============================================================================
# CONSTANTS
# =============================================================================

PROJECT_MARKERS = {"data", "src"}


# Chunking parameters
MAX_EMBEDDING_CHARS = 900
MIN_SUBCHUNK_CHARS = 80

# File paths (will be set after directory detection)
BANKA_DOKUMANLARI_RAW_DIR: Optional[Path] = None
BANKA_DOKUMANLARI_PARSED_DIR: Optional[Path] = None
BANKA_DOKUMANLARI_CHUNKS_DIR: Optional[Path] = None

# =============================================================================
# REGEX PATTERNS
# =============================================================================

# Header/footer patterns to remove
HEADER_FOOTER_RE = re.compile(
    r"^\d{14,}|"  # Long number sequences (14+ digits)
    r"^\d{2}:\d{2}:\d{2}|"  # Time stamps
    r"^Reference\s*:|"  # Reference lines
    r"^\d+/\d+$|"  # Page numbers like "1/9"
    r"^[|\s]+$|"  # Lines with only pipes and spaces
    r"^:ecnerefeR$"  # Specific garbage
)

# Inline noise tokens that sometimes leak into text
NOISE_TOKEN_RE = re.compile(
    r"(\b\d{8,}\b|"            # very long numeric strings
    r"\b\d{2}:\d{2}:\d{2}\b|" # timestamps
    r",\d{4}:\d{2}:\d{2}\b|"   # comma + timestamp-like
    r"\b\d{2,}-\d+\b|"         # patterns like 66-2
    r"Bireysel\s+Bankacılık\s+Hizmetleri\s+Sözleşmesi)"
    , re.IGNORECASE
)

# Section patterns - A., B., C. or I., II., III. or 1., 2., 3.
MAIN_SECTION_RE = re.compile(r"^([A-Z])\.([\w\sÇĞİÖŞÜçğıöşü]+)$")  # A.ÜRÜN VE HİZMETLER...
ROMAN_SECTION_RE = re.compile(r"^([IVX]+)\.([\w\sÇĞİÖŞÜçğıöşü]+)$")  # I.VADESİZ MEVDUAT

# Article patterns (MADDE, Madde - if exists)
MADDE_RE = re.compile(r"^(?:Madde|MADDE)\s+(\d+)[:\-–]?\s*(.*)$")

# Numbered/lettered item patterns
NUMBERED_ITEM_RE = re.compile(r"^(\d+)\.\s+(.+)$")
NUMERIC_SUBSECTION_RE = re.compile(r"^(\d+(?:\.\d+)+)\.?\s+(.+)$")  # 1.1 or 1.1.1
LETTERED_ITEM_RE = re.compile(r"^([a-zçğıöşü])\)\s+(.+)$", re.IGNORECASE)

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PDFUnit:
    """Represents a parsed unit from a PDF document."""
    doc_id: str
    unit_id: str
    unit_type: str  # 'section', 'article', 'numbered_item', 'lettered_item', 'paragraph'
    doc_type: str  # 'contract', 'agreement', 'policy'
    section_name: Optional[str]
    article_no: Optional[int]
    article_title: Optional[str]
    text: str
    page_start: int
    page_end: int
    metadata: Dict[str, Any]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def find_project_root(start_path: Optional[Path] = None) -> Path:
    """Find the project root directory by searching for marker directories."""
    start_path = (start_path or Path.cwd()).resolve()
    for path in [start_path, *start_path.parents]:
        if PROJECT_MARKERS.issubset({item.name for item in path.iterdir() if item.is_dir()}):
            return path
    raise RuntimeError(f"Project root not found from: {start_path}")

def setup_logging(log_file_path: Path) -> logging.Logger:
    """Configure logging with both file and console handlers."""
    log_file_path.parent.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def setup_directories(base_dir: Path) -> None:
    """Set up global directory paths."""
    global BANKA_DOKUMANLARI_RAW_DIR, BANKA_DOKUMANLARI_PARSED_DIR, BANKA_DOKUMANLARI_CHUNKS_DIR
    
    banka_dokumanlari_dir = base_dir / "data" / "banka_dokumanlari"
    BANKA_DOKUMANLARI_RAW_DIR = banka_dokumanlari_dir / "raw"
    BANKA_DOKUMANLARI_PARSED_DIR = banka_dokumanlari_dir / "parsed"
    BANKA_DOKUMANLARI_CHUNKS_DIR = banka_dokumanlari_dir / "chunks"
    
    # Create directories if they don't exist
    BANKA_DOKUMANLARI_PARSED_DIR.mkdir(parents=True, exist_ok=True)
    BANKA_DOKUMANLARI_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

def safe_text(text: str) -> str:
    """Clean and normalize text for processing."""
    if not text:
        return ""
    text = text.replace("\r", "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def normalize_for_embedding(text: str) -> str:
    """Normalize text for embedding by converting newlines to spaces."""
    if not text:
        return ""
    text = text.replace("\r", "")
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def clean_noise_tokens(text: str) -> str:
    """Remove known noise tokens (page refs, numeric garbage)."""
    if not text:
        return ""
    cleaned = NOISE_TOKEN_RE.sub("", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()

def determine_doc_type(doc_id: str) -> str:
    """Determine document type from doc_id."""
    doc_id_lower = doc_id.lower()
    if 'sozlesme' in doc_id_lower or 'sozlesmesi' in doc_id_lower:
        return 'contract'
    elif 'kredi' in doc_id_lower and 'kart' in doc_id_lower:
        return 'credit_card'
    elif 'ucret' in doc_id_lower or 'tarife' in doc_id_lower:
        return 'fee_schedule'
    elif 'politika' in doc_id_lower:
        return 'policy'
    return 'pdf_document'

# =============================================================================
# EXCEL ATTACHMENT HELPERS
# =============================================================================

def _excel_token_candidates(stem: str) -> List[str]:
    """Generate simple tokens from excel filename stem for matching in PDF text."""
    stem_lower = stem.lower()
    token_space = stem.replace("_", " ").lower()
    no_digits_space = re.sub(r"\d+", "", token_space).strip()
    no_digits_underscore = re.sub(r"\d+", "", stem_lower).strip()
    return [t for t in {stem_lower, token_space, no_digits_space, no_digits_underscore} if t]

def find_referenced_excel_records(pdf_full_text: str, base_dir: Path, current_doc_id: str) -> List[Dict[str, Any]]:
    """Return excel vector records whose filename is mentioned in PDF text."""
    chunks_dir = base_dir / "data" / "banka_dokumanlari" / "chunks"
    pdf_text_l = pdf_full_text.lower()
    records: List[Dict[str, Any]] = []

    for path in chunks_dir.glob("*_vector_ready.json"):
        if current_doc_id in path.stem:
            continue  # skip self
        stem = path.stem.replace("_vector_ready", "")
        tokens = _excel_token_candidates(stem)
        if not any(tok and tok in pdf_text_l for tok in tokens):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Tag source for clarity
            for rec in data:
                rec = dict(rec)
                rec.setdefault("metadata", {})
                rec["metadata"]["source"] = "excel_attachment"
                records.append(rec)
            logger.info(f"Attached excel vector records from {path.name} (match tokens: {tokens})")
        except Exception as exc:
            logger.warning(f"Could not load excel vector file {path}: {exc}")
            continue
    return records

# =============================================================================
# SCOPE MAPPING (now handled by scope_utils.py)
# =============================================================================

# Scope assignment is now handled by assign_bank_document_scopes()
# from scope_utils module for consistency across the project

# =============================================================================
# PDF PARSER
# =============================================================================

def is_header_footer(line: str) -> bool:
    """Check if line is header/footer garbage."""
    return bool(HEADER_FOOTER_RE.match(line.strip()))

def read_pdf_lines_with_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    """Extract text lines from PDF with page numbers, de-duplicating shadow lines."""
    lines_with_pages = []
    last_line = None
    try:
        with pdfplumber.open(pdf_path) as pdf: # pyright: ignore[reportOptionalMemberAccess]
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                for line in page_text.splitlines():
                    cleaned_line = line.strip()
                    if cleaned_line and not is_header_footer(cleaned_line):
                        # Shadow text protection: skip if line is identical to previous
                        if cleaned_line == last_line:
                            continue
                        lines_with_pages.append((page_num, cleaned_line))
                        last_line = cleaned_line
    except Exception as e:
        logger.error(f"Error reading PDF {pdf_path}: {e}")
        raise
    return lines_with_pages

def parse_pdf_document(pdf_path: Path, doc_id: str) -> List[Dict[str, Any]]:
    """Parse PDF document with structural awareness."""
    if pdfplumber is None:
        raise ImportError("pdfplumber is not installed. Install with: pip install pdfplumber")

    logger.info(f"Parsing PDF: {pdf_path}")

    # Read lines with page tracking
    lines_with_pages = read_pdf_lines_with_pages(pdf_path)
    logger.info(f"Extracted {len(lines_with_pages)} clean lines")

    units = []
    current_main_section = None  # A., B., C.
    current_main_section_name = None
    current_subsection = None  # I., II., III.
    current_subsection_name = None
    current_article_no = None
    current_article_title = None
    current_lines = []
    current_page_start = None
    unit_counter = 1

    for page_num, line in lines_with_pages:
        # Track page start
        if current_page_start is None:
            current_page_start = page_num

        if main_section_match := MAIN_SECTION_RE.match(line):
            # Save previous unit if exists
            if current_lines:
                text = clean_noise_tokens(" ".join(current_lines))
                if text.strip() and len(text.strip()) > 10:
                    units.append({
                        "doc_id": doc_id,
                        "unit_id": f"{doc_id}_unit_{unit_counter}",
                        "unit_type": "subsection" if current_subsection else "section",
                        "doc_type": determine_doc_type(doc_id),
                        "section_name": current_subsection_name or current_main_section_name,
                        "article_no": current_article_no,
                        "article_title": current_article_title,
                        "text": text,
                        "page_start": current_page_start,
                        "page_end": page_num - 1,
                        "metadata": {
                            "main_section": current_main_section,
                            "subsection": current_subsection
                        }
                    })
                    unit_counter += 1
                current_lines = []

            # Update context
            current_main_section = main_section_match.group(1)
            current_main_section_name = main_section_match.group(2).strip()
            current_subsection = None
            current_subsection_name = None
            current_article_no = None
            current_article_title = None
            current_page_start = page_num
            continue

        if roman_section_match := ROMAN_SECTION_RE.match(line):
            # Save previous unit if exists
            if current_lines:
                text = clean_noise_tokens(" ".join(current_lines))
                if text.strip() and len(text.strip()) > 10:
                    units.append({
                        "doc_id": doc_id,
                        "unit_id": f"{doc_id}_unit_{unit_counter}",
                        "unit_type": "subsection" if current_subsection else "section",
                        "doc_type": determine_doc_type(doc_id),
                        "section_name": current_subsection_name or current_main_section_name,
                        "article_no": current_article_no,
                        "article_title": current_article_title,
                        "text": text,
                        "page_start": current_page_start,
                        "page_end": page_num - 1,
                        "metadata": {
                            "main_section": current_main_section,
                            "subsection": current_subsection
                        }
                    })
                    unit_counter += 1
                current_lines = []

            # Update subsection context
            current_subsection = roman_section_match.group(1)
            current_subsection_name = roman_section_match.group(2).strip()
            current_article_no = None
            current_article_title = None
            current_page_start = page_num
            continue

        if madde_match := MADDE_RE.match(line):
            # Save previous unit if exists
            if current_lines:
                text = clean_noise_tokens(" ".join(current_lines))
                if text.strip() and len(text.strip()) > 10:
                    units.append({
                        "doc_id": doc_id,
                        "unit_id": f"{doc_id}_unit_{unit_counter}",
                        "unit_type": "article",
                        "doc_type": determine_doc_type(doc_id),
                        "section_name": current_subsection_name or current_main_section_name,
                        "article_no": current_article_no,
                        "article_title": current_article_title,
                        "text": text,
                        "page_start": current_page_start,
                        "page_end": page_num - 1,
                        "metadata": {
                            "main_section": current_main_section,
                            "subsection": current_subsection
                        }
                    })
                    unit_counter += 1
                current_lines = []

            # Update article context
            current_article_no = int(madde_match.group(1))
            current_article_title = madde_match.group(2).strip() if madde_match.group(2) else None
            current_page_start = page_num
            current_page_start = page_num
            continue

        # Check for numeric subsection (1.1, 1.1.1)
        if numeric_sub_match := NUMERIC_SUBSECTION_RE.match(line):
             # Save previous unit if exists
            if current_lines:
                text = clean_noise_tokens(" ".join(current_lines))
                if text.strip() and len(text.strip()) > 10:
                    units.append({
                        "doc_id": doc_id,
                        "unit_id": f"{doc_id}_unit_{unit_counter}",
                        "unit_type": "subsection",
                        "doc_type": determine_doc_type(doc_id),
                        "section_name": current_subsection_name or current_main_section_name,
                        "article_no": None,
                        "article_title": numeric_sub_match.group(2).strip(),
                        "text": text,
                        "page_start": current_page_start,
                        "page_end": page_num - 1,
                        "metadata": {
                            "main_section": current_main_section,
                            "subsection": current_subsection,
                            "numeric_subsection": numeric_sub_match.group(1)
                        }
                    })
                    unit_counter += 1
                current_lines = []

            # Update context for numeric subsection - treated as a type of subsection or item
            # We don't change 'current_subsection' (Roman) but invalidating article_no
            current_article_no = None
            current_article_title = numeric_sub_match.group(2).strip()
            current_page_start = page_num
            continue

        # Check for numbered item (1., 2., etc.) - only under subsection
        numbered_match = NUMBERED_ITEM_RE.match(line)
        if numbered_match and current_subsection is not None:
            # Save previous numbered item as separate unit
            if current_lines:
                text = clean_noise_tokens(" ".join(current_lines))
                if text.strip() and len(text.strip()) > 10:
                    units.append({
                        "doc_id": doc_id,
                        "unit_id": f"{doc_id}_unit_{unit_counter}",
                        "unit_type": "numbered_item",
                        "doc_type": determine_doc_type(doc_id),
                        "section_name": current_subsection_name,
                        "article_no": None,
                        "article_title": None,
                        "text": text,
                        "page_start": current_page_start,
                        "page_end": page_num - 1,
                        "metadata": {
                            "main_section": current_main_section,
                            "subsection": current_subsection,
                            "item_number": int(numbered_match.group(1)) - 1 if unit_counter > 1 else None
                        }
                    })
                    unit_counter += 1

            # Start new numbered item
            current_lines = [numbered_match.group(2)]
            current_page_start = page_num
            continue

        # Check for lettered item (a), b), etc.)
        lettered_match = LETTERED_ITEM_RE.match(line)
        if lettered_match and (current_subsection is not None or current_article_no is not None):
            # Save previous item as separate unit
            if current_lines:
                text = clean_noise_tokens(" ".join(current_lines))
                if text.strip() and len(text.strip()) > 10:
                    units.append({
                        "doc_id": doc_id,
                        "unit_id": f"{doc_id}_unit_{unit_counter}",
                        "unit_type": "lettered_item",
                        "doc_type": determine_doc_type(doc_id),
                        "section_name": current_subsection_name or current_main_section_name,
                        "article_no": current_article_no,
                        "article_title": current_article_title,
                        "text": text,
                        "page_start": current_page_start,
                        "page_end": page_num - 1,
                        "metadata": {
                            "main_section": current_main_section,
                            "subsection": current_subsection,
                            "item_letter": lettered_match.group(1).lower()
                        }
                    })
                    unit_counter += 1

            # Start new lettered item
            current_lines = [lettered_match.group(2)]
            current_page_start = page_num
            continue

        # Regular content line
        current_lines.append(line)

    # Save final unit
    if current_lines:
        text = clean_noise_tokens(" ".join(current_lines))
        if text.strip() and len(text.strip()) > 10:
            units.append({
                "doc_id": doc_id,
                "unit_id": f"{doc_id}_unit_{unit_counter}",
                "unit_type": "subsection" if current_subsection else "paragraph",
                "doc_type": determine_doc_type(doc_id),
                "section_name": current_subsection_name or current_main_section_name,
                "article_no": current_article_no,
                "article_title": current_article_title,
                "text": text,
                "page_start": current_page_start,
                "page_end": lines_with_pages[-1][0] if lines_with_pages else 1,
                "metadata": {
                    "main_section": current_main_section,
                    "subsection": current_subsection
                }
            })

    # Fallback: If very few units were created (less than 5 for a doc > 3 pages), 
    # it might mean regexes failed. Flatten and chunk by simple splitting.
    total_pages = lines_with_pages[-1][0] if lines_with_pages else 1
    if len(units) < 5 and total_pages > 3:
        logger.warning(f"Low unit count ({len(units)}) for {total_pages} pages. Running fallback chunking.")
        full_text = " ".join([line for _, line in lines_with_pages])
        full_text = clean_noise_tokens(full_text)
        
        # Split by simple paragraph boundaries or sentences to respect MAX_EMBEDDING_CHARS
        sentences = re.split(r'(?<=[.!?])\s+', full_text)
        
        fallback_units = []
        current_chunk = ""
        current_chunk_start_page = 1 # Approximation
        
        chunk_idx = 1
        for sentence in sentences:
             if len(current_chunk) + len(sentence) + 1 <= MAX_EMBEDDING_CHARS:
                 current_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
             else:
                 if current_chunk:
                     fallback_units.append({
                        "doc_id": doc_id,
                        "unit_id": f"{doc_id}_fallback_{chunk_idx}",
                        "unit_type": "fallback_paragraph",
                        "doc_type": determine_doc_type(doc_id),
                        "section_name": "General",
                        "article_no": None,
                        "article_title": None,
                        "text": current_chunk,
                        "page_start": 1, # Lost granularity in fallback
                        "page_end": total_pages,
                        "metadata": {"fallback": True}
                     })
                     chunk_idx += 1
                 current_chunk = sentence
        
        if current_chunk:
             fallback_units.append({
                "doc_id": doc_id,
                "unit_id": f"{doc_id}_fallback_{chunk_idx}",
                "unit_type": "fallback_paragraph",
                "doc_type": determine_doc_type(doc_id),
                "section_name": "General",
                "article_no": None,
                "article_title": None,
                "text": current_chunk,
                "page_start": 1, 
                "page_end": total_pages,
                "metadata": {"fallback": True}
             })
        
        # Merge or replace? If we found almost nothing, better to replace.
        if len(units) <= 2:
            logger.info(f"Replacing {len(units)} structured units with {len(fallback_units)} fallback units.")
            units = fallback_units
        else:
             logger.info(f"Appending {len(fallback_units)} fallback units to existing {len(units)} units? No, keeping distinct.")
             # Actually, if we found *some* structure but missed a lot, it is hard to merge without duplication.
             # For this hackathon, let's assume if < 5 units, the parsing mostly failed.
             units = fallback_units

    logger.info(f"Extracted {len(units)} units from PDF")
    return units

def create_chunks(parsed_units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create chunks from parsed units."""
    chunks = []
    
    for unit in parsed_units:
        unit_id = unit.get("unit_id")
        if not unit_id:
            continue
        
        text = unit.get("text", "")
        if not text or len(text) < 10:  # Skip very short texts
            continue

        scope = assign_bank_document_scopes(unit.get("doc_id", ""), text)
        
        chunk = {
            "chunk_id": unit_id,
            "doc_id": unit.get("doc_id"),
            "unit_id": unit_id,
            "unit_type": unit.get("unit_type"),
            "doc_type": unit.get("doc_type"),
            "section_name": unit.get("section_name"),
            "article_no": unit.get("article_no"),
            "article_title": unit.get("article_title"),
            "page_start": unit.get("page_start"),
            "page_end": unit.get("page_end"),
            "text": text,
            "scope": scope,
            "metadata": unit.get("metadata", {})
        }
        chunks.append(chunk)
    
    logger.info(f"Created {len(chunks)} chunks")
    return chunks

def create_vector_records(chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create vector-ready records from a chunk. Returns ONLY child chunks."""
    full_text = chunk.get("text", "")
    if not full_text:
        return []
    
    parent_id = chunk["chunk_id"]
    records = []
    
    # If text is short enough, create single child
    if len(full_text) <= MAX_EMBEDDING_CHARS:
        child = dict(chunk)
        child["chunk_id"] = f"{parent_id}__c1"
        child["chunk_kind"] = "child"
        child["parent_chunk_id"] = parent_id
        child["text_for_embedding"] = full_text
        child["text"] = full_text  # Fix: Ensure text matches embedding text
        records.append(child)
        return records
    
    # Split long text by sentences
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    
    current_chunk = ""
    child_idx = 1
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= MAX_EMBEDDING_CHARS:
            current_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
        else:
            if current_chunk:
                child = dict(chunk)
                child["chunk_id"] = f"{parent_id}__c{child_idx}"
                child["chunk_kind"] = "child"
                child["parent_chunk_id"] = parent_id
                child["text_for_embedding"] = current_chunk
                child["text"] = current_chunk  # Fix: Overwrite parent text with specific chunk content
                records.append(child)
                child_idx += 1
            current_chunk = sentence
    
    # Add remaining chunk
    if current_chunk:
        child = dict(chunk)
        child["chunk_id"] = f"{parent_id}__c{child_idx}"
        child["chunk_kind"] = "child"
        child["parent_chunk_id"] = parent_id
        child["text_for_embedding"] = current_chunk
        child["text"] = current_chunk  # Fix: Overwrite parent text
        records.append(child)
    
    return records

def perform_chunk_qc(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Perform quality control on chunks."""
    chunk_ids = [chunk.get("chunk_id") for chunk in chunks if chunk.get("chunk_id")]
    duplicate_ids = [k for k, v in Counter(chunk_ids).items() if v > 1]
    
    empty_texts = [chunk.get("chunk_id") for chunk in chunks if not (chunk.get("text") or "").strip()]
    
    text_lengths = [(chunk.get("chunk_id"), len(chunk.get("text", ""))) for chunk in chunks]
    sorted_lengths = sorted(text_lengths, key=lambda x: x[1], reverse=True)
    
    unit_types = Counter(chunk.get("unit_type") for chunk in chunks)
    
    return {
        "counts": {
            "chunks": len(chunks),
            "unit_types": dict(unit_types),
        },
        "duplicate_chunk_ids": duplicate_ids,
        "empty_text": empty_texts,
        "text_length_stats": {
            "min": min((length for _, length in sorted_lengths), default=0),
            "max": max((length for _, length in sorted_lengths), default=0),
            "top_10_longest": sorted_lengths[:10],
        },
    }

def perform_vector_qc(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Perform quality control on vector records."""
    chunk_ids = [record.get("chunk_id") for record in records if record.get("chunk_id")]
    duplicate_ids = [k for k, v in Counter(chunk_ids).items() if v > 1]
    
    children = [record for record in records if record.get("chunk_kind") == "child"]
    too_long = [record["chunk_id"] for record in children
               if len(record.get("text_for_embedding", "")) > MAX_EMBEDDING_CHARS]
    too_short = [record["chunk_id"] for record in children
                if 0 < len(record.get("text_for_embedding", "")) < MIN_SUBCHUNK_CHARS]
    missing_parent = [record["chunk_id"] for record in children
                     if not record.get("parent_chunk_id")]
    
    parents = {record["chunk_id"] for record in records if record.get("chunk_kind") == "parent"}
    orphan_children = [record["chunk_id"] for record in children
                      if record.get("parent_chunk_id") not in parents]
    
    return {
        "counts": {
            "records_total": len(records),
            "parents": len(parents),
            "children": len(children),
            "duplicate_chunk_ids": len(duplicate_ids),
            "child_too_long": len(too_long),
            "child_too_short": len(too_short),
            "child_missing_parent_id": len(missing_parent),
            "child_orphan": len(orphan_children),
        },
        "sample": {
            "duplicate_chunk_ids": duplicate_ids[:50],
            "child_too_long": too_long[:50],
            "child_too_short": too_short[:50],
            "child_missing_parent_id": missing_parent[:50],
            "child_orphan": orphan_children[:50],
        },
    }

# =============================================================================
# FILE I/O FUNCTIONS
# =============================================================================

def save_json(data: Any, file_path: Path) -> None:
    """Save data to JSON file."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved JSON to {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        raise

def save_jsonl(data: List[Dict[str, Any]], file_path: Path) -> None:
    """Save list of dictionaries to JSONL file."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Saved JSONL to {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSONL to {file_path}: {e}")
        raise

# =============================================================================
# MAIN EXECUTION FUNCTIONS
# =============================================================================

def process_pdf_document(file_path: Path) -> None:
    """Process a single PDF document."""
    logger.info(f"Processing PDF document: {file_path.name}")

    # Generate doc_id from filename
    doc_id = file_path.stem

    # Parse PDF
    units = parse_pdf_document(file_path, doc_id)

    # Save parsed units
    parsed_output = BANKA_DOKUMANLARI_PARSED_DIR / f"{doc_id}_units.json" # pyright: ignore[reportOptionalOperand]
    save_json(units, parsed_output)

    # Create chunks
    chunks = create_chunks(units)
    _extracted_from_process_pdf_document_17(
        doc_id, '_chunks.json', '_chunks.jsonl', chunks
    )
    # QC on chunks
    chunk_qc = perform_chunk_qc(chunks)
    chunk_qc_output = BANKA_DOKUMANLARI_CHUNKS_DIR / f"{doc_id}_chunks_qc.json" # pyright: ignore[reportOptionalOperand]
    save_json(chunk_qc, chunk_qc_output)

    # Create vector-ready records
    records = []
    for chunk in chunks:
        records.extend(create_vector_records(chunk))

    # Attach excel vector records when the PDF references an excel filename
    pdf_full_text = " ".join(normalize_for_embedding(c.get("text", "")) for c in chunks)
    try:
        base_dir = find_project_root(file_path.parent)
    except Exception:
        base_dir = find_project_root()
    if excel_records := find_referenced_excel_records(
        pdf_full_text, base_dir, doc_id
    ):
        records.extend(excel_records)
        logger.info(f"Attached {len(excel_records)} excel vector records for {doc_id}")

    _extracted_from_process_pdf_document_17(
        doc_id, '_vector_ready.json', '_vector_ready.jsonl', records
    )
    # QC on vector records
    vector_qc = perform_vector_qc(records)
    vector_qc_output = BANKA_DOKUMANLARI_CHUNKS_DIR / f"{doc_id}_vector_ready_qc.json" # pyright: ignore[reportOptionalOperand]
    save_json(vector_qc, vector_qc_output)

    logger.info(f"Completed processing: {file_path.name}")
    logger.info(f"  - Units: {len(units)}")
    logger.info(f"  - Chunks: {len(chunks)}")
    logger.info(f"  - Vector records: {len(records)}")
    logger.info(f"  - QC Summary: {chunk_qc['counts']}")


# TODO Rename this here and in `process_pdf_document`
def _extracted_from_process_pdf_document_17(doc_id, arg1, arg2, arg3):
    chunks_output = BANKA_DOKUMANLARI_CHUNKS_DIR / f"{doc_id}{arg1}" # pyright: ignore[reportOptionalOperand]
    chunks_jsonl_output = BANKA_DOKUMANLARI_CHUNKS_DIR / f"{doc_id}{arg2}" # pyright: ignore[reportOptionalOperand]
    save_json(arg3, chunks_output)
    save_jsonl(arg3, chunks_jsonl_output)

def main():
    """Main execution function."""
    try:
        # Setup
        base_dir = find_project_root()
        setup_directories(base_dir)
        log_file = base_dir / "logs" / "banka_dokumanlari_preparation_pdf.log"
        global logger
        logger = setup_logging(log_file)
        
        logger.info("Starting Banka Dokümanları Preparation - PDF Only")
        logger.info(f"Base directory: {base_dir}")
        logger.info(f"Raw directory: {BANKA_DOKUMANLARI_RAW_DIR}")
        
        # Process all PDF documents in raw directory
        if not BANKA_DOKUMANLARI_RAW_DIR.exists(): 
            logger.error(f"Raw directory does not exist: {BANKA_DOKUMANLARI_RAW_DIR}")
            return
        
        # Find all PDF documents
        pdf_documents = [f for f in BANKA_DOKUMANLARI_RAW_DIR.iterdir() 
                        if f.is_file() and f.suffix.lower() == '.pdf']
        
        if not pdf_documents:
            logger.warning("No PDF documents found in raw directory")
            return
        
        logger.info(f"Found {len(pdf_documents)} PDF documents to process")
        
        for doc_path in pdf_documents:
            try:
                process_pdf_document(doc_path)
            except Exception as e:
                logger.error(f"Failed to process {doc_path.name}: {e}")
                continue
        
        logger.info("All PDF documents processed successfully")
        
    except Exception as e:
        if logger:
            logger.error(f"Error in main execution: {e}")
        else:
            print(f"Error in main execution (logger not init): {e}")
        raise

if __name__ == "__main__":
    main()
