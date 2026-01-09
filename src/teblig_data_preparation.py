# =============================================================================
# Teblig Data Preparation Script
# =============================================================================
"""
Teblig Data Preparation Script

This script processes a Turkish Central Bank (TCMB) Teblig PDF document to extract,
chunk, and prepare vector-ready data for AI applications.

Features:
- PDF parsing with structured extraction of articles and paragraphs
- Text chunking for retrieval systems
- Vector-ready subchunking with embedding optimization
- Comprehensive quality control and logging

Author: Can Mergen
Date: 2026-01-01
Version: 1.0.0
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import sys
import logging
from pathlib import Path
import re
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

# Third-party imports
import pdfplumber
from collections import Counter

# Local imports
from scope_utils import assign_teblig_scopes

# =============================================================================
# CONSTANTS
# =============================================================================

# Project structure markers
PROJECT_MARKERS = {"data", "src"}

# ... (omitted lines)


# Document configuration
DOC_ID = "tcmb_teblig"
INCLUDE_GECICI_MADDE = True

# Chunking parameters
MAX_EMBEDDING_CHARS = 900
MIN_SUBCHUNK_CHARS = 80
CONTEXT_PREFIX_MAX_CHARS = 220
MIN_BENT_COUNT_FOR_SUBCHUNK = 3

# File paths (will be set after directory detection)
PDF_PATH: Optional[Path] = None
OUTPUT_PATH: Optional[Path] = None
PARSED_PATH: Optional[Path] = None
CHUNKS_JSONL_PATH: Optional[Path] = None
CHUNKS_JSON_PATH: Optional[Path] = None
CHUNKS_QC_JSON_PATH: Optional[Path] = None
VECTOR_READY_JSON_PATH: Optional[Path] = None
VECTOR_READY_JSONL_PATH: Optional[Path] = None
VECTOR_READY_QC_JSON_PATH: Optional[Path] = None

# =============================================================================
# SCOPE MAPPING (Now handled by scope_utils.py)
# =============================================================================

# Scope assignment is now handled by the assign_teblig_scopes() function
# from scope_utils module for consistency across the project

# =============================================================================
# REGEX PATTERNS
# =============================================================================

# Section and article patterns
BOLUM_RE = re.compile(r"^(BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ)\s+BÖLÜM\s*$")
MADDE_RE = re.compile(r"^MADDE\s+(\d+)\s+–\s*(.*)$")
GECICI_MADDE_RE = re.compile(r"^GEÇİCİ\s+MADDE\s+(\d+)\s+–\s*(.*)$")

# Paragraph and sub-paragraph patterns
FIKRA_MARK_RE = re.compile(r"\((\d+)\)")
FIRST_FIKRA_POS_RE = re.compile(r"\(\s*1\s*\)")

# Special headers
EK1_HEADER_RE = re.compile(r"^Ek-1\s+Ürün\s+veya\s+Hizmet\s+Sınıflandırmaları", re.IGNORECASE)

# Trailing heading patterns
TRAILING_HEADING_PATTERNS = [
    r"^Kapsam$",
    r"^Dayanak$",
    r"^Tanımlar$",
    r"^Bilgilendirme$",
    r"^Sözleşme esasları$",
    r"^Ücretlerin sınıflandırılması$",
    r"^Ücretlerin değiştirilmesi$",
    r"^Ücretlerin iadesi$",
    r"^Banka ve kredi kartı ücretleri$",
    r"^Para ve kıymetli maden transfer işlemleri$",
    r"^Mevduat ve katılım fonu işlemleri$",
    r"^ATM kullanımı ve kiralık kasa hizmeti$",
    r"^Kampanyalar ve özel hizmetler$",
    r"^Yürürlük$",
    r"^Yürütme$",
]
TRAILING_HEADING_RE = re.compile("|".join(TRAILING_HEADING_PATTERNS), re.IGNORECASE)

# Chunking patterns
HEADER_OK_RE = re.compile(
    r"^(MADDE\s+\d+\s+–\s+FIKRA\s+\d+|GEÇİCİ\s+MADDE\s+\d+\s+–\s+FIKRA\s+\d+):\s+",
    re.IGNORECASE
)
BENT_RE = re.compile(r"(?:(?<=\s)|^)([a-zçğıöşü])\)\s+", re.IGNORECASE)
NUM_BENT_RE = re.compile(r"(?:(?<=\s)|^)\(?(\d+)\)\s+")
HEADER_RE = re.compile(
    r"^(MADDE\s+\d+\s+–\s+FIKRA\s+\d+|GEÇİCİ\s+MADDE\s+\d+\s+–\s+FIKRA\s+\d+)\s*:\s*",
    re.IGNORECASE
)

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TebligUnit:
    """Represents a parsed unit from the Teblig document."""
    doc_id: str
    unit_id: str
    unit_type: str  # 'fikra' or 'gecici_madde'
    bolum_no: Optional[int]
    bolum_name: Optional[str]
    madde_no: Optional[int]
    madde_title: Optional[str]
    madde_header_raw: str
    fikra_no: Optional[int]
    text: str
    page_start: int
    page_end: int
    scope: List[str]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def find_project_root(start_path: Optional[Path] = None) -> Path:
    """
    Find the project root directory by searching for marker directories.

    Args:
        start_path: Starting path for search. Defaults to current working directory.

    Returns:
        Path to the project root directory.

    Raises:
        RuntimeError: If project root cannot be found.
    """
    start_path = (start_path or Path.cwd()).resolve()
    for path in [start_path, *start_path.parents]:
        if PROJECT_MARKERS.issubset({item.name for item in path.iterdir() if item.is_dir()}):
            return path
    raise RuntimeError(f"Project root not found from: {start_path}")

def setup_logging(log_file_path: Path) -> logging.Logger:
    """
    Configure logging with both file and console handlers.

    Args:
        log_file_path: Path to the log file.

    Returns:
        Configured logger instance.
    """
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
    global PDF_PATH, OUTPUT_PATH, PARSED_PATH, CHUNKS_JSONL_PATH, CHUNKS_JSON_PATH
    global CHUNKS_QC_JSON_PATH, VECTOR_READY_JSON_PATH, VECTOR_READY_JSONL_PATH, VECTOR_READY_QC_JSON_PATH

    dataset_dir = base_dir / "data"
    src_dir = base_dir / "src"
    teblig_dir = dataset_dir / "teblig"
    parsed_dir = teblig_dir / "parsed"
    chunks_dir = teblig_dir / "chunks"

    PDF_PATH = teblig_dir / "raw" / "tcmb_teblig.pdf"
    OUTPUT_PATH = parsed_dir / "teblig_units.json"
    PARSED_PATH = parsed_dir / "teblig_units.json"
    CHUNKS_JSONL_PATH = chunks_dir / "teblig_chunks.jsonl"
    CHUNKS_JSON_PATH = chunks_dir / "teblig_chunks.json"
    CHUNKS_QC_JSON_PATH = chunks_dir / "teblig_chunks_qc.json"
    VECTOR_READY_JSON_PATH = chunks_dir / "teblig_chunks_vector_ready.json"
    VECTOR_READY_JSONL_PATH = chunks_dir / "teblig_chunks_vector_ready.jsonl"
    VECTOR_READY_QC_JSON_PATH = chunks_dir / "teblig_chunks_vector_ready_qc.json"

def safe_text(text: str) -> str:
    """
    Clean and normalize text for processing.

    Args:
        text: Input text string.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""
    text = text.replace("\r", "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def normalize_for_embedding(text: str) -> str:
    """
    Normalize text for embedding by converting newlines to spaces and collapsing whitespace.

    Args:
        text: Input text string.

    Returns:
        Normalized text string suitable for embeddings.
    """
    if not text:
        return ""
    text = text.replace("\r", "")
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def get_scope_for_madde(madde_no: Optional[int]) -> List[str]:
    """
    Get the scope list for a given madde number using scope_utils.

    Args:
        madde_no: Madde number.

    Returns:
        List of scope strings.
    """
    if madde_no is None:
        return ["general_terms"]
    return assign_teblig_scopes(madde_no)

def read_pdf_lines_with_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    """
    Extract text lines from PDF with page numbers.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of tuples (page_number, line_text).
    """
    lines_with_pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                for line in page_text.splitlines():
                    cleaned_line = line.rstrip()
                    if cleaned_line.strip():
                        lines_with_pages.append((page_num, cleaned_line.strip()))
    except Exception as e:
        logger.error(f"Error reading PDF {pdf_path}: {e}")
        raise
    return lines_with_pages

def bolum_name_to_number(name: str) -> int:
    """
    Convert Turkish bolum name to number.

    Args:
        name: Turkish bolum name (BİRİNCİ, İKİNCİ, etc.).

    Returns:
        Corresponding number.
    """
    mapping = {"BİRİNCİ": 1, "İKİNCİ": 2, "ÜÇÜNCÜ": 3, "DÖRDÜNCÜ": 4}
    return mapping.get(name, 0)

def join_lines(lines: List[str]) -> str:
    """
    Join lines with proper newline handling and hyphenation fix.

    Args:
        lines: List of text lines.

    Returns:
        Joined text string.
    """
    text = "\n".join(ln.rstrip() for ln in lines if ln is not None)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # Fix hyphenation
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def is_header_line(line: str) -> bool:
    """
    Check if a line is a header (bolum, madde, etc.).

    Args:
        line: Text line to check.

    Returns:
        True if it's a header line.
    """
    stripped = line.strip()
    return bool(
        BOLUM_RE.match(stripped) or
        MADDE_RE.match(stripped) or
        GECICI_MADDE_RE.match(stripped) or
        EK1_HEADER_RE.search(stripped)
    )

def clean_trailing_headings(text: str) -> str:
    """
    Remove trailing structural headings from text.

    Args:
        text: Input text.

    Returns:
        Cleaned text without trailing headings.
    """
    lines = [line.rstrip() for line in (text or "").splitlines()]
    while lines:
        last = lines[-1].strip()
        if not last:
            lines.pop()
            continue
        if is_header_line(last):
            lines.pop()
            continue
        if TRAILING_HEADING_RE.match(last):
            lines.pop()
            continue
        break
    return "\n".join(lines).strip()

def drop_leading_structural_header(text: str) -> str:
    """
    Remove leading MADDE header from text body.

    Args:
        text: Input text.

    Returns:
        Text without leading header.
    """
    lines = [line.rstrip() for line in (text or "").splitlines()]
    if not lines:
        return ""
    first = lines[0].strip()
    if (MADDE_RE.match(first) or GECICI_MADDE_RE.match(first) or
        first.startswith("MADDE ") or first.startswith("GEÇİCİ MADDE ")):
        lines = lines[1:]
    return "\n".join(line for line in lines if line.strip()).strip()

def split_fikras(madde_text: str) -> List[Tuple[int, str]]:
    """
    Split madde text into fikra tuples.

    Args:
        madde_text: Text containing fikras marked with (1), (2), etc.

    Returns:
        List of (fikra_number, fikra_text) tuples.
    """
    parts = FIKRA_MARK_RE.split(madde_text)
    if len(parts) <= 2:
        return []
    fikras = []
    for i in range(1, len(parts), 2):
        num_str = parts[i]
        body = (parts[i + 1] if i + 1 < len(parts) else "").strip()
        try:
            num = int(num_str)
        except ValueError:
            continue
        body = clean_trailing_headings(body)
        if body := re.sub(r"\n{3,}", "\n\n", body).strip():
            fikras.append((num, body))
    return fikras

def split_madde_header_and_body(tail: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Split madde header into title and body parts.

    Args:
        tail: Text after "MADDE X – ".

    Returns:
        Tuple of (title_raw, title, body_tail).
    """
    tail = tail.strip()
    match = FIRST_FIKRA_POS_RE.search(tail)
    if not match:
        return tail, tail or None, None
    pos = match.start()
    title_part = tail[:pos].strip()
    body_part = tail[pos:].strip()
    return title_part, title_part or None, body_part

# =============================================================================
# PDF PARSING FUNCTIONS
# =============================================================================

def parse_teblig(pdf_path: Path, doc_id: str, include_gecici: bool) -> List[Dict[str, Any]]:
    """
    Parse the Teblig PDF and extract structured units.

    Args:
        pdf_path: Path to the PDF file.
        doc_id: Document identifier.
        include_gecici: Whether to include geçici maddeler.

    Returns:
        List of parsed unit dictionaries.
    """
    logger.info("Starting PDF parsing: %s", pdf_path)
    lines_with_pages = read_pdf_lines_with_pages(pdf_path)
    logger.info("Loaded %d lines from PDF", len(lines_with_pages))

    units: List[TebligUnit] = []

    # State variables for parsing
    current_bolum_no: Optional[int] = None
    current_bolum_name: Optional[str] = None
    waiting_for_bolum_name = False

    current_type: Optional[str] = None  # 'MADDE' or 'GECICI_MADDE'
    current_no: Optional[int] = None
    current_title: Optional[str] = None
    current_header_raw: Optional[str] = None
    current_lines: List[str] = []
    current_page_start: Optional[int] = None
    current_page_end: Optional[int] = None

    def flush_current_unit():
        """Flush the current unit buffer and create units."""
        nonlocal current_type, current_no, current_title, current_header_raw
        nonlocal current_lines, current_page_start, current_page_end

        if not current_type or not current_lines or current_page_start is None or current_no is None:
            reset_state()
            return

        raw_text = clean_trailing_headings(join_lines(current_lines))

        def create_fikra_units(is_gecici: bool):
            nonlocal raw_text
            fikras = split_fikras(raw_text)

            # Fallback: if no fikras, create single fikra
            if not fikras:
                body = drop_leading_structural_header(raw_text)
                if body:
                    fikras = [(1, body)]

            for fikra_no, fikra_text in fikras:
                if is_gecici:
                    units.append(TebligUnit(
                        doc_id=doc_id,
                        unit_id=f"GECICI_MADDE_{current_no}_{fikra_no}",
                        unit_type="gecici_madde",
                        bolum_no=current_bolum_no,
                        bolum_name=current_bolum_name,
                        madde_no=current_no,
                        madde_title=current_title,
                        madde_header_raw=f"GEÇİCİ MADDE {current_no} – FIKRA {fikra_no}",
                        fikra_no=fikra_no,
                        text=fikra_text,
                        page_start=current_page_start, # pyright: ignore[reportArgumentType]
                        page_end=current_page_end or current_page_start, # pyright: ignore[reportArgumentType]
                        scope=["transition"],
                    ))
                else:
                    units.append(TebligUnit(
                        doc_id=doc_id,
                        unit_id=f"MADDE_{current_no}_{fikra_no}",
                        unit_type="fikra",
                        bolum_no=current_bolum_no,
                        bolum_name=current_bolum_name,
                        madde_no=current_no,
                        madde_title=current_title,
                        madde_header_raw=f"MADDE {current_no} – FIKRA {fikra_no}",
                        fikra_no=fikra_no,
                        text=fikra_text,
                        page_start=current_page_start, # pyright: ignore[reportArgumentType]
                        page_end=current_page_end or current_page_start, # pyright: ignore[reportArgumentType]
                        scope=get_scope_for_madde(current_no),
                    ))

            return len(fikras)

        if current_type == "MADDE":
            fikra_count = create_fikra_units(is_gecici=False)
            logger.info("Parsed madde %d with %d fikras", current_no, fikra_count)
        elif current_type == "GECICI_MADDE" and include_gecici:
            fikra_count = create_fikra_units(is_gecici=True)
            logger.info("Parsed geçici madde %d with %d fikras", current_no, fikra_count)

        reset_state()

    def reset_state():
        """Reset all state variables."""
        nonlocal current_type, current_no, current_title, current_header_raw
        nonlocal current_lines, current_page_start, current_page_end
        current_type = None
        current_no = None
        current_title = None
        current_header_raw = None
        current_lines = []
        current_page_start = None
        current_page_end = None

    for page_no, line in lines_with_pages:
        if EK1_HEADER_RE.search(line):
            flush_current_unit()
            logger.info("Reached Ek-1 section, stopping parsing")
            break

        if current_type and is_header_line(line):
            flush_current_unit()

        bolum_match = BOLUM_RE.match(line)
        if bolum_match:
            flush_current_unit()
            current_bolum_no = bolum_name_to_number(bolum_match.group(1))
            current_bolum_name = None
            waiting_for_bolum_name = True
            logger.info("Started bölüm %d: %s", current_bolum_no, bolum_match.group(1))
            continue

        if waiting_for_bolum_name:
            if line.startswith("MADDE") or line.startswith("GEÇİCİ MADDE"):
                waiting_for_bolum_name = False
            else:
                current_bolum_name = line
                waiting_for_bolum_name = False
                logger.info("Bölüm name: %s", current_bolum_name)
                continue

        madde_match = MADDE_RE.match(line)
        gecici_match = GECICI_MADDE_RE.match(line)
        if madde_match or gecici_match:
            flush_current_unit()

            if madde_match:
                current_type = "MADDE"
                current_no = int(madde_match.group(1))
                tail = madde_match.group(2).strip()
                title_raw, title, body_tail = split_madde_header_and_body(tail)
                current_title = title
                current_header_raw = f"MADDE {current_no} – {title_raw}".strip() if title_raw else f"MADDE {current_no} –"
            else:
                current_type = "GECICI_MADDE"
                current_no = int(gecici_match.group(1)) # pyright: ignore[reportOptionalMemberAccess]
                tail = gecici_match.group(2).strip() # pyright: ignore[reportOptionalMemberAccess]
                title_raw, title, body_tail = split_madde_header_and_body(tail)
                current_title = title
                current_header_raw = f"GEÇİCİ MADDE {current_no} – {title_raw}".strip() if title_raw else f"GEÇİCİ MADDE {current_no} –"

            current_lines = [current_header_raw]
            if body_tail:
                current_lines.append(body_tail)

            current_page_start = page_no
            current_page_end = page_no
            continue

        if current_type:
            current_lines.append(line)
            current_page_end = page_no

    flush_current_unit()
    logger.info("Parsing completed, extracted %d units", len(units))
    return [asdict(unit) for unit in units]

# =============================================================================
# CHUNKING FUNCTIONS
# =============================================================================

def create_chunks(parsed_units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create chunks from parsed units.

    Args:
        parsed_units: List of parsed unit dictionaries.

    Returns:
        List of chunk dictionaries.
    """
    chunks = []
    for unit in parsed_units:
        unit_id = unit.get("unit_id")
        if not unit_id:
            continue

        header = safe_text(unit.get("madde_header_raw", ""))
        body = safe_text(unit.get("text", ""))

        full_text = f"{header}: {body}".strip() if header and body else (header or body)
        full_text = normalize_for_embedding(full_text)
        if not full_text:
            continue

        chunk = {
            "chunk_id": unit_id,  # 1:1 mapping
            "doc_id": unit.get("doc_id"),
            "unit_id": unit_id,  # Canonical anchor
            "unit_type": unit.get("unit_type"),  # 'fikra' or 'gecici_madde'
            "bolum_no": unit.get("bolum_no"),
            "bolum_name": unit.get("bolum_name"),
            "madde_no": unit.get("madde_no"),
            "fikra_no": unit.get("fikra_no"),
            "scope": unit.get("scope", []),
            "page_start": unit.get("page_start"),
            "page_end": unit.get("page_end"),
            "text": full_text,  # Embedding-friendly
        }
        chunks.append(chunk)

    logger.info("Created %d chunks", len(chunks))
    return chunks

def perform_chunk_qc(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Perform quality control on chunks.

    Args:
        chunks: List of chunk dictionaries.

    Returns:
        QC results dictionary.
    """
    chunk_ids = [chunk.get("chunk_id") for chunk in chunks if chunk.get("chunk_id")]
    duplicate_ids = [k for k, v in Counter(chunk_ids).items() if v > 1]

    empty_texts = [chunk.get("chunk_id") for chunk in chunks if not (chunk.get("text") or "").strip()]

    bad_pages = []
    for chunk in chunks:
        ps, pe = chunk.get("page_start"), chunk.get("page_end")
        if ps is None or pe is None or ps > pe:
            bad_pages.append({
                "chunk_id": chunk.get("chunk_id"),
                "page_start": ps,
                "page_end": pe
            })

    unknown_scopes = [chunk.get("chunk_id") for chunk in chunks
                     if not chunk.get("scope") or chunk.get("scope") == ["unknown"]]

    bad_unit_types = [chunk.get("chunk_id") for chunk in chunks
                     if chunk.get("unit_type") not in ("fikra", "gecici_madde")]

    header_warnings = [chunk.get("chunk_id") for chunk in chunks
                      if not HEADER_OK_RE.match(chunk.get("text", ""))]

    text_lengths = [(chunk.get("chunk_id"), len(chunk.get("text", ""))) for chunk in chunks]
    sorted_lengths = sorted(text_lengths, key=lambda x: x[1], reverse=True)

    return {
        "counts": {
            "chunks": len(chunks),
            "unit_types": dict(
                Counter(chunk.get("unit_type") for chunk in chunks)
            ),
        },
        "duplicate_chunk_ids": duplicate_ids,
        "empty_text": empty_texts,
        "bad_pages": bad_pages,
        "unknown_or_missing_scope": unknown_scopes,
        "bad_unit_type": bad_unit_types,
        "header_format_warn": header_warnings,
        "text_length_stats": {
            "min": min((length for _, length in sorted_lengths), default=0),
            "max": max((length for _, length in sorted_lengths), default=0),
            "top_10_longest": sorted_lengths[:10],
        },
    }

# =============================================================================
# VECTOR READY CHUNKING FUNCTIONS
# =============================================================================

def split_by_bents(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Split text by bent markers (a), b), 1), etc.).

    Args:
        text: Input text.

    Returns:
        Tuple of (prelude, items) where items is list of (label, item_text).
    """
    if matches := list(BENT_RE.finditer(text)):
        first_pos = matches[0].start()
        prelude = text[:first_pos].strip(" ;:\n\t")
        items = []
        for i, match in enumerate(matches):
            label = match.group(1).lower()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            item = text[start:end].strip(" ;\n\t,")
            items.append((label, item))
        return prelude, items

    if matches := list(NUM_BENT_RE.finditer(text)):
        first_pos = matches[0].start()
        prelude = text[:first_pos].strip(" ;:\n\t")
        items = []
        for i, match in enumerate(matches):
            label = match.group(1)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            item = text[start:end].strip(" ;\n\t,")
            items.append((label, item))
        return prelude, items

    return "", []

def build_context_prefix(full_text: str) -> str:
    """
    Build a context prefix for subchunks.

    Args:
        full_text: Full text of the chunk.

    Returns:
        Context prefix string.
    """
    header_match = HEADER_RE.match(full_text)
    header = ""
    rest = full_text
    if header_match:
        header = header_match.group(1).strip()
        rest = full_text[header_match.end():].strip()

    prelude, _ = split_by_bents(rest)
    prefix = f"{header}: {prelude}".strip() if prelude else f"{header}:".strip()
    prefix = normalize_for_embedding(prefix)
    return prefix[:CONTEXT_PREFIX_MAX_CHARS].strip()

def should_subchunk(full_text: str) -> bool:
    """
    Determine if text should be subchunked.

    Args:
        full_text: Text to check.

    Returns:
        True if subchunking is needed.
    """
    if len(full_text) > MAX_EMBEDDING_CHARS:
        return True
    body = HEADER_RE.sub("", full_text).strip()
    _, items = split_by_bents(body)
    return len(items) >= MIN_BENT_COUNT_FOR_SUBCHUNK

def chunk_long_text_by_sentences(text: str, prefix: str) -> List[str]:
    """
    Split long text by sentences as fallback.

    Args:
        text: Text to split.
        prefix: Context prefix.

    Returns:
        List of sentence chunks.
    """
    text = text.strip()
    parts = re.split(r"(?<=[\.\?;])\s+", text)
    chunks = []
    current = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue
        candidate = f"{current} {part}".strip() if current else part
        if len(prefix) + 3 + len(candidate) <= MAX_EMBEDDING_CHARS:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = part

    if current:
        chunks.append(current)

    # Merge small chunks
    merged = []
    for chunk in chunks:
        if (merged and len(chunk) < MIN_SUBCHUNK_CHARS and
            len(merged[-1]) + 1 + len(chunk) <= MAX_EMBEDDING_CHARS):
            merged[-1] = f"{merged[-1]} {chunk}".strip()
        else:
            merged.append(chunk)

    return merged

def create_vector_records(chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create vector-ready records from a chunk.

    Args:
        chunk: Chunk dictionary.

    Returns:
        List of vector records (parent + children).
    """
    full_text = normalize_for_embedding(chunk.get("text", ""))
    if not full_text:
        return []

    def finalize_child_text(child_text: str, parent_text: str) -> str:
        """Ensure child text ends with a complete clause using parent context."""
        if child_text.endswith(('.', '?', '!')):
            return child_text

        pos = parent_text.find(child_text)
        if pos != -1:
            tail = parent_text[pos + len(child_text):]

            # Find nearest boundary: sentence end or next bent label like " b)" / " ç)"
            sentence_pos = tail.find('.')
            semicolon_pos = tail.find(';')

            label_match = BENT_RE.search(tail)
            label_pos = label_match.start() if label_match else -1

            candidates = [p for p in (sentence_pos, semicolon_pos, label_pos) if p != -1]
            cut = min(candidates) if candidates else len(tail)
            addition = tail[:cut].strip()

            if addition:
                candidate = f"{child_text} {addition}".strip()
                child_text = candidate[:MAX_EMBEDDING_CHARS]

        # Fallback: close the definition with "ifade eder." if still no punctuation
        if not child_text.endswith(('.', '?', '!')):
            suffix = " ifade eder."
            room = MAX_EMBEDDING_CHARS - len(child_text)
            if room > len(suffix):
                child_text = (child_text + suffix)[:MAX_EMBEDDING_CHARS]
            else:
                child_text = f'{child_text}.'[:MAX_EMBEDDING_CHARS]

        return child_text.strip()

    parent_id = chunk["chunk_id"]
    prefix = build_context_prefix(full_text)

    # Create parent record
    parent = dict(chunk)
    parent["chunk_kind"] = "parent"
    parent["parent_chunk_id"] = None
    parent["text_full"] = full_text

    records = [parent]

    if not should_subchunk(full_text):
        # Single child
        child = dict(chunk)
        child["chunk_id"] = f"{parent_id}__c1"
        child["chunk_kind"] = "child"
        child["parent_chunk_id"] = parent_id
        child["text_for_embedding"] = full_text
        child["text_for_embedding"] = finalize_child_text(child["text_for_embedding"], full_text)
        records.append(child)
        return records

    # Extract body for processing
    body = HEADER_RE.sub("", full_text).strip()
    prelude, items = split_by_bents(body)

    if items:
        # Bent-based children
        prelude_norm = normalize_for_embedding(prelude)
        base_prefix = normalize_for_embedding(prefix)

        for label, item in items:
            item_norm = normalize_for_embedding(item)
            text_for_emb = f"{base_prefix} {label}) {item_norm}".strip()
            text_for_emb = text_for_emb[:MAX_EMBEDDING_CHARS].strip()

            if len(text_for_emb) < MIN_SUBCHUNK_CHARS:
                continue

            child = dict(chunk)
            child["chunk_id"] = f"{parent_id}__b{label}"
            child["chunk_kind"] = "child"
            child["parent_chunk_id"] = parent_id
            child["bent_label"] = str(label)
            child["prelude"] = prelude_norm
            child["text_for_embedding"] = text_for_emb
            child["text_for_embedding"] = finalize_child_text(child["text_for_embedding"], full_text)
            records.append(child)

        return records

    # Sentence-based fallback
    base_prefix = normalize_for_embedding(prefix)
    sentence_chunks = chunk_long_text_by_sentences(body, prefix=base_prefix)

    for i, sc in enumerate(sentence_chunks, 1):
        sc = normalize_for_embedding(sc)
        text_for_emb = f"{base_prefix} {sc}".strip()
        text_for_emb = text_for_emb[:MAX_EMBEDDING_CHARS].strip()

        child = dict(chunk)
        child["chunk_id"] = f"{parent_id}__s{i}"
        child["chunk_kind"] = "child"
        child["parent_chunk_id"] = parent_id
        child["text_for_embedding"] = text_for_emb
        child["text_for_embedding"] = finalize_child_text(child["text_for_embedding"], full_text)
        records.append(child)

    return records

def perform_vector_qc(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Perform quality control on vector records.

    Args:
        records: List of vector record dictionaries.

    Returns:
        QC results dictionary.
    """
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
    """
    Save data to JSON file.

    Args:
        data: Data to save.
        file_path: Output file path.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Saved JSON to %s", file_path)
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        raise

def save_jsonl(data: List[Dict[str, Any]], file_path: Path) -> None:
    """
    Save list of dictionaries to JSONL file.

    Args:
        data: List of dictionaries to save.
        file_path: Output file path.
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info("Saved JSONL to %s", file_path)
    except Exception as e:
        logger.error(f"Error saving JSONL to {file_path}: {e}")
        raise

def load_json(file_path: Path) -> Any:
    """
    Load data from JSON file.

    Args:
        file_path: Input file path.

    Returns:
        Loaded data.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Loaded JSON from %s", file_path)
        return data
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        raise

# =============================================================================
# MAIN EXECUTION FUNCTIONS
# =============================================================================

def run_pdf_parsing() -> List[Dict[str, Any]]:
    """Run PDF parsing and return units."""
    logger.info("Starting PDF parsing process")
    units = parse_teblig(PDF_PATH, DOC_ID, include_gecici=INCLUDE_GECICI_MADDE) # pyright: ignore[reportArgumentType]
    save_json(units, OUTPUT_PATH) # pyright: ignore[reportArgumentType]
    logger.info("PDF parsing completed: %s (count=%d) | include_gecici_madde=%s",
                OUTPUT_PATH, len(units), INCLUDE_GECICI_MADDE)
    return units

def run_chunking() -> List[Dict[str, Any]]:
    """Run chunking process and return chunks."""
    logger.info("Starting chunking process")
    units = load_json(PARSED_PATH) # pyright: ignore[reportArgumentType]
    logger.info("Loaded %d parsed units", len(units))

    chunks = create_chunks(units)

    # Save outputs
    save_jsonl(chunks, CHUNKS_JSONL_PATH) # pyright: ignore[reportArgumentType]
    save_json(chunks, CHUNKS_JSON_PATH) # pyright: ignore[reportArgumentType]

    # QC
    qc = perform_chunk_qc(chunks)
    save_json(qc, CHUNKS_QC_JSON_PATH) # pyright: ignore[reportArgumentType]

    logger.info("Chunk QC: Chunks=%d, Unit types=%s, Duplicate IDs=%d, Empty text=%d, Bad pages=%d, Unknown scope=%d, Bad unit type=%d, Header warnings=%d, Text length min/max=%d/%d",
                qc['counts']['chunks'], qc['counts']['unit_types'], len(qc['duplicate_chunk_ids']),
                len(qc['empty_text']), len(qc['bad_pages']), len(qc['unknown_or_missing_scope']),
                len(qc['bad_unit_type']), len(qc['header_format_warn']),
                qc['text_length_stats']['min'], qc['text_length_stats']['max'])

    for cid, length in qc["text_length_stats"]["top_10_longest"][:3]:
        logger.info("Top chunk: %s (%d chars)", cid, length)

    logger.info("Chunking completed")
    return chunks

def run_vector_ready_chunking() -> List[Dict[str, Any]]:
    """Run vector-ready chunking process and return records."""
    logger.info("Starting vector-ready chunking process")
    chunks = load_json(CHUNKS_JSON_PATH) # pyright: ignore[reportArgumentType]
    logger.info("Loaded %d chunks", len(chunks))

    records = []
    for chunk in chunks:
        if not chunk.get("chunk_id"):
            continue
        records.extend(create_vector_records(chunk))

    logger.info("Created %d vector-ready records (parents + children)", len(records))

    # Save outputs
    save_json(records, VECTOR_READY_JSON_PATH) # pyright: ignore[reportArgumentType]
    save_jsonl(records, VECTOR_READY_JSONL_PATH) # pyright: ignore[reportArgumentType]

    # QC
    qc = perform_vector_qc(records)
    save_json(qc, VECTOR_READY_QC_JSON_PATH) # pyright: ignore[reportArgumentType]

    logger.info("Vector-ready QC counts: %s", qc["counts"])

    logger.info("Vector-ready chunking completed")
    return records

# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    """Main execution function."""
    try:
        # Setup
        base_dir = find_project_root()
        setup_directories(base_dir)
        log_file = base_dir / "logs" / "teblig_data_preparation.log"
        global logger
        logger = setup_logging(log_file)

        logger.info("Starting Teblig Data Preparation")
        logger.info("Base directory: %s", base_dir)
        logger.info("Source directory: %s", base_dir / "src")
        logger.info("Dataset directory: %s", base_dir / "data")

        # Run processes
        run_pdf_parsing()
        run_chunking()
        run_vector_ready_chunking()

        logger.info("All processes completed successfully")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()

