"""LLM-based compliance checker for bank documents against Tebliğ regulations.

Reads the top-3 matches from chroma_tool.py outputs and performs detailed 
compliance analysis using LLM.

Phase 1 Optimizations:
- Hybrid approach: Rule-based numerical checks + LLM
- Chain-of-thought: Step-by-step reasoning
- Multi-step workflow: Scope check → Compliance check
"""
from __future__ import annotations

import argparse
import json
import re
import os
import requests
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple

import chromadb
from sentence_transformers import SentenceTransformer
from src.evaluate_model import compute_metrics, compute_reasoning_quality

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  psutil not installed. Using conservative defaults.")
    print("   Install with: pip install psutil")

# Use Ollama directly without importing all of llm_utils
_OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_CHAT_URL = f"{_OLLAMA_BASE.rstrip('/')}/api/chat"


# Global set to track hashes across multiple document processing calls
_GLOBAL_SEEN_HASHES = set()

def generate_with_single_input(
    prompt: str,
    role: str = "user",
    temperature: float = None, # pyright: ignore[reportArgumentType]
    max_tokens: int = 500,
    model: str = "gemma3:27b",
    **kwargs
) -> Dict[str, str]:
    """Call Ollama API for single-turn chat."""
    messages = [{"role": role, "content": prompt}]

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    options = {}
    if temperature is not None:
        options["temperature"] = temperature
    if max_tokens is not None:
        options["num_predict"] = max_tokens

    if options:
        payload["options"] = options

    try:
        return _send_ollama_request(payload)
    except Exception as e:
        raise RuntimeError(f"Ollama error: {e}")


def _send_ollama_request(payload: dict) -> dict:
    """Send request to Ollama API and parse response."""
    resp = requests.post(_CHAT_URL, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    
    role_out = data["message"].get("role", "assistant")
    content = data["message"].get("content", "")
    
    return {"role": role_out, "content": str(content)}


def parse_chroma_results(result_file: Path) -> List[Dict[str, Any]]:
    """Parse the chroma_tool output file to extract source chunks and their top-3 matches."""
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")
    
    with result_file.open("r", encoding="utf-8") as f:
        content = f.read()
    
    # Parse each source chunk section
    chunks = []
    # Pattern: ##### Source N/M | id=... | scope=... | madde=... | fikra=...
    # Capture multiline chunk text until the section separator. This prevents truncation when we fall back to parsed files.
    source_pattern = r"##### Source (\d+)/(\d+) \| id=([^\|]+) \| scope=([^\|]+) \| madde=([^\|]+) \| fikra=([^\n]+)\ntext=([\s\S]*?)(?:\n\n=== [^\n]+ ===\n([\s\S]*?))?(?=\n##### Source |\Z)"
    
    matches = re.finditer(source_pattern, content, re.DOTALL)
    
    for match in matches:
        idx, total, chunk_id, scope, madde, fikra, text, results_section = match.groups()
        
        # Parse top-3 results
        top_results = []
        if results_section and "No results." not in results_section:
            # Pattern for individual results
            result_pattern = r"\[(\d+)\] sim=([\d\.]+) \| id=([^\n]+)\n\s+scope=([^\|]+) \| madde=([^\|]+) \| fikra=([^\|]+) \| parent=([^\n]+)\n\s+text=([^\n]+)"
            for res_match in re.finditer(result_pattern, results_section):
                rank, sim, res_id, res_scope, res_madde, res_fikra, res_parent, res_text = res_match.groups()
                top_results.append({
                    "rank": int(rank),
                    "similarity": float(sim),
                    "id": res_id.strip(),
                    "scope": res_scope.strip(),
                    "madde": res_madde.strip(),
                    "fikra": res_fikra.strip(),
                    "parent": res_parent.strip(),
                    "text": res_text.strip()
                })
        
        chunks.append({
            "index": int(idx),
            "total": int(total),
            "chunk_id": chunk_id.strip(),
            "scope": scope.strip(),
            "madde": madde.strip(),
            "fikra": fikra.strip(),
            "text": text.strip(),
            "top_matches": top_results
        })
    
    return chunks


def pick_chunk_text(row: Dict[str, Any]) -> str:
    """Get best-available text for embedding and display."""
    return (
        row.get("text_for_embedding")
        or row.get("text")
        or row.get("text_full")
        or ""
    ).strip()


def load_bank_chunks(json_path: Path) -> List[Dict[str, Any]]:
    """Load bank document chunks (ALL chunks, both parent and child)."""
    if not json_path.exists():
        raise FileNotFoundError(f"Bank chunk JSON not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Bank chunk JSON must be a list")
    
    # Return ONLY child chunks (detailed clauses)
    # Parent chunks are headers/context, not actual compliance items
    child_chunks = [c for c in data if c.get("chunk_kind") == "child"]
    print(f"[load] Loaded {len(child_chunks)} child chunks (filtered from {len(data)} total)")
    return child_chunks


def query_live_matches(
    chunks: List[Dict[str, Any]],
    persist_dir: Path,
    collection: str,
    embedding_model: str,
    top_n: int,
    min_sim: float,
    scope_match: bool,
) -> List[Dict[str, Any]]:
    """For each chunk, run a live Chroma query and attach top matches."""

    client = chromadb.PersistentClient(path=str(persist_dir))
    col = client.get_collection(name=collection)
    model = SentenceTransformer(embedding_model)

    enriched: List[Dict[str, Any]] = []
    
    # Load config for hybrid search settings
    try:
        from src.config_loader import config
        use_hybrid = config.get("retrieval", {}).get("hybrid_search", {}).get("enabled", False)
    except ImportError:
        use_hybrid = False
        
    hybrid_tool = None
    if use_hybrid:
        try:
            from src.retrieval_utils import HybridRetriever, reciprocal_rank_fusion
            # Pre-fetch all documents for BM25 index (this might be memory intensive for huge datasets, but fine for Tebliğ)
            all_docs = col.get(include=["documents", "metadatas"])
            docs_list = all_docs["documents"]
            metas_list = all_docs["metadatas"]
            
            if docs_list:
                # print(f"Index: {len(docs_list)} docs loaded for BM25")
                hybrid_tool = HybridRetriever(docs_list, metas_list)
        except Exception as e:
            print(f"⚠️ Hybrid search init failed: {e}")

    # Force-load EK-1 chunk (Allowed Fee List) to ensure LLM knows what is permitted
    ek1_chunk_obj = None
    try:
        ek1_res = col.get(ids=["EK_1_LISTE"], include=["documents", "metadatas"])
        if ek1_res and ek1_res.get("documents") and ek1_res["documents"]:
            ek1_chunk_obj = {
                "text": ek1_res["documents"][0],
                "metadata": ek1_res["metadatas"][0],
                "similarity": 0.95,  # High relevance manually assigned
                "is_hybrid": False, 
                "forced_context": True 
            }
            # print("[RAG] ✅ Force-loaded EK-1 List for context injection.")
    except Exception as e:
        print(f"[RAG] ⚠️ Could not force-load EK-1: {e}")

    for chunk in chunks:
        q_text = pick_chunk_text(chunk)
        q_scope: Set[str] = set(chunk.get("scope") or [])
        
        # 1. Vector Search
        embedding = model.encode(q_text, normalize_embeddings=True, show_progress_bar=False)
        vec_res = col.query(
            query_embeddings=[embedding], # pyright: ignore[reportArgumentType]
            n_results=top_n * 2,
            include=["documents", "metadatas", "distances"],
        )
        
        vector_matches = []
        docs = vec_res.get("documents", [[]])[0]
        metas = vec_res.get("metadatas", [[]])[0]
        dists = vec_res.get("distances", [[]])[0]
        
        for doc, meta, dist in zip(docs, metas, dists):
            vector_matches.append({
                "text": doc,
                "metadata": meta,
                "score": 1 - dist # Similarity
            })
            
        # 2. Keyword Search (if enabled)
        bm25_matches = []
        if hybrid_tool:
            bm25_matches = hybrid_tool.search_bm25(q_text, top_n=top_n * 2)
            
        # 3. Fusion or Fallback
        if hybrid_tool and bm25_matches:
            # Helper to adapt vector format to fusion format
            start_fusion = time.time()
            fused_docs = reciprocal_rank_fusion(vector_matches, bm25_matches, k=60)
            
            # Convert back to matches list format ensuring metadata structure
            matches = []
            for doc in fused_docs[:top_n]: # Take top N from fusion
               matches.append({
                   "text": doc["text"],
                   "metadata": doc["metadata"],
                   "similarity": 0.99, # Dummy high score since RRF doesn't give probability
                   "is_hybrid": True
               })
        else:
            # Fallback to vector only
            matches = []
            for item in vector_matches:
                if item["score"] >= min_sim:
                    matches.append({
                        "text": item["text"],
                        "metadata": item["metadata"],
                        "similarity": item["score"],
                        "is_hybrid": False
                    })
                    
        # Filter by scope if needed
        final_matches = []
        for match in matches:
             target_scope = match["metadata"].get("scope")
             # ... (scope filtering logic same as before)
             if isinstance(target_scope, str):
                target_scope_set = {s.strip() for s in target_scope.split(",")}
             elif target_scope is None:
                target_scope_set = set()
             else:
                target_scope_set = set(target_scope)
             
             if not scope_match or not q_scope or not target_scope_set or not q_scope.isdisjoint(target_scope_set):
                 final_matches.append(match)
                 
             if len(final_matches) >= top_n:
                 break
        
        # Inject EK-1 chunk if not already present
        if ek1_chunk_obj:
            # Check if likely relevant (fee related) or just always add it
            # To be safe, we always add it, but only if not already top 1
            already_has_ek1 = any(m["metadata"].get("chunk_id") == "EK_1_LISTE" for m in final_matches)
            if not already_has_ek1:
                # Add it to the top or end? Add to end to provide context without overriding top vector match
                final_matches.append(ek1_chunk_obj)

        chunk["top_matches"] = final_matches
        enriched.append(chunk)

    return enriched


# ============================================================================
# PHASE 1 OPTIMIZATIONS: Hybrid Approach (Rule-Based + LLM)
# ============================================================================

def extract_amounts(text: str) -> List[float]:
    """
    Extract monetary amounts from text (TL, TRY, lira).
    Handles formats like:
    - "50 TL"
    - "(TL): 50"
    - "Ücret 10.5 TL"
    - "15 TRY"
    """
    patterns = [
        r'(\d+(?:[.,]\d+)?)\s*(?:TL|TRY|lira|Lira)',  # Standard: "X TL"
        r'(?:\(TL\)|TL):\s*(\d+(?:[.,]\d+)?)',  # Reverse: "(TL): X"
    ]
    
    amounts = []
    for pattern in patterns:
        for match in re.findall(pattern, text, re.IGNORECASE):
            try:
                value = match if isinstance(match, str) else match[-1]
                amounts.append(float(value.replace(',', '.')))
            except (ValueError, AttributeError):
                continue
    
    return amounts

def extract_limits(text: str) -> List[Tuple[str, float]]:
    """Extract limit keywords and their amounts from Tebliğ text."""
    limit_patterns = [
        (r'(azami|en\s+fazla|maksimum)\s+(\d+(?:[.,]\d+)?)\s*(?:TL|TRY|lira)', 'max', True),
        (r'(en\s+az|asgari|minimum)\s+(\d+(?:[.,]\d+)?)\s*(?:TL|TRY|lira)', 'min', True),
        (r'(\d+(?:[.,]\d+)?)\s*(?:TL|TRY|lira)[^\w]*(?:geçemez|fazla\s+olamaz|aşamaz)', 'max', False),
    ]
    
    limits = []
    for pattern, limit_type, has_keyword in limit_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                amount_str = match[1] if has_keyword else match
                amount = float(amount_str.replace(',', '.'))
                limits.append((limit_type, amount))
            except (ValueError, IndexError):
                continue
    
    return limits

def check_numerical_compliance(
    bank_text: str,
    teblig_matches: List[Dict[str, Any]]
) -> Optional[Tuple[str, str, str]]:
    """
    Rule-based numerical compliance check.
    Returns (status, reason, corrected_text) if violation found, else None.
    
    PRIORITY CHECKS (before limit checks):
    1. Percentage surcharges (e.g., "%5 fazlası") - forbidden for currency exchange
    """
    bank_lower = bank_text.lower()
    
    # Extract amounts first (needed for all checks)
    bank_amounts = extract_amounts(bank_text)
    
    # ========================================================================
    # CRITICAL CHECK 0: EFT transaction fees (MADDE 12 FIKRA 3)
    # Mobile: max 2 TL for 1000-50K TL transactions
    # Branch: max 10 TL for 1000-50K TL transactions
    # ========================================================================
    
    # First check if this is about EFT and has 1000-50K range
    is_eft_with_range = (
        'eft' in bank_lower and 
        ('1000' in bank_lower or '50' in bank_lower or '50k' in bank_lower.replace(' ', ''))
    )
    
    if is_eft_with_range and bank_amounts:
        # Determine channel type
        is_mobile = bool(re.search(r'mobil|mobile', bank_lower))
        is_branch = bool(re.search(r'şube|branch', bank_lower))
        
        if is_mobile:
            limit = 2.0
            channel_name = 'Mobil bankacılık'
            # Check all amounts in the text
            for amount in bank_amounts:
                if amount > limit:
                    return (
                        'NOT_OK',
                        f"[CRITICAL] {channel_name} EFT ücreti {amount} TL, ancak Tebliğ limiti {limit} TL. "
                        f"{amount - limit} TL fazla alınıyor. "
                        f"[MADDE 12 - FIKRA 3: İşlem tutarının 1.000 TL ile 50.000 TL arasında olması hâlinde "
                        f"mobil bankacılık ile yapılan işlemlerde 2 TL'yi geçemez]",
                        f"DÜZELTME ÖNERİSİ: Mobil Bankacılık EFT ücreti {limit} TL olarak belirlenmelidir. "
                        f"(Mevcut {amount} TL yerine {limit} TL yazılmalı)"
                    )
        elif is_branch:
            limit = 10.0
            channel_name = 'Şube'
            for amount in bank_amounts:
                if amount > limit:
                    return (
                        'NOT_OK',
                        f"[CRITICAL] {channel_name} EFT ücreti {amount} TL, ancak Tebliğ limiti {limit} TL. "
                        f"{amount - limit} TL fazla alınıyor. "
                        f"[MADDE 12 - FIKRA 3: İşlem tutarının 1.000 TL ile 50.000 TL arasında olması hâlinde "
                        f"diğer kanallar ile yapılan işlemlerde 10 TL'yi geçemez]",
                        f"DÜZELTME ÖNERİSİ: Şube EFT ücreti {limit} TL olarak belirlenmelidir. "
                        f"(Mevcut {amount} TL yerine {limit} TL yazılmalı)"
                    )
    
    # ========================================================================
    # CRITICAL CHECK 1: Percentage surcharges on exchange rates
    # Example: "Banka Gişe Satış kurunun %5 fazlası uygulanır"
    # This is a NOT_OK violation - cannot charge percentage above market rate
    # Works across ALL documents (KREDI_KARTI, Bireysel_Bankacilik, etc.)
    # ========================================================================
    percentage_surcharge_patterns = [
        r'%\s*(\d+(?:[.,]\d+)?)\s*fazla',  # "%5 fazla"
        r'kurunun\s*%\s*(\d+(?:[.,]\d+)?)',  # "kurunun %5"
        r'(\d+(?:[.,]\d+)?)\s*%\s*fazla',  # "5% fazla"
        r'yüzde\s*(\d+(?:[.,]\d+)?)\s*(?:fazla|artış|fazlası)',  # "yüzde 5 fazla"
        r'satış\s*kurunun\s*%\s*(\d+(?:[.,]\d+)?)',  # "satış kurunun %5"
        r'gişe.*?satış.*?kurunun\s*%\s*(\d+(?:[.,]\d+)?)',  # "gişe satış kurunun %5"
    ]
    
    for pattern in percentage_surcharge_patterns:
        if matches := re.findall(pattern, bank_lower):
            for percentage_str in matches:
                try:
                    percentage = float(percentage_str.replace(',', '.'))
                    
                    # Find position of percentage in text
                    percentage_pattern = pattern.replace(r'(\d+(?:[.,]\d+)?)', percentage_str)
                    if match_obj := re.search(percentage_pattern, bank_lower):
                        match_pos = match_obj.start()
                        
                        # STEP 1: Check VERY NARROW context (±50 chars) for currency
                        narrow_start = max(0, match_pos - 50)
                        narrow_end = min(len(bank_lower), match_pos + 50)
                        narrow_context = bank_lower[narrow_start:narrow_end]
                        
                        # Currency keywords (must be VERY close to percentage)
                        currency_keywords = [
                            'döviz', 'kur', 'yabancı para', 'foreign', 'exchange', 
                            'satış kuru', 'alış kuru', 'gişe satış', 'döviz işlem',
                            'para cinsinden', 'para birimi', 'foreign currency'
                        ]
                        
                        has_currency_narrow = any(kw in narrow_context for kw in currency_keywords)
                        
                        # STEP 2: If currency found, check for excludes in SAME narrow window
                        if has_currency_narrow:
                            # EXCLUDE keywords (sorumluluk, bildirim, etc.)
                            exclude_keywords = [
                                'sorumlu', 'bildirim', 'haber ver', 'bildirmeli', 
                                'kaybolma', 'çalınma', 'hukuka aykırı', 'sigorta',
                                'zarar', 'başvur', 'tebligat', 'risk'
                            ]
                            
                            # If exclude keywords in NARROW context, skip
                            if any(kw in narrow_context for kw in exclude_keywords):
                                continue  # False positive
                            
                            # VALID DETECTION: Currency + No exclude in narrow window
                            return (
                                'NOT_OK',
                                f"[CRITICAL] Döviz işlemlerinde yüzdesel fazlalık uygulanamaz. "
                                f"Banka %{percentage} fazlalık uygulayacağını belirtiyor. "
                                f"Bu, müşteri aleyhine keyfi ek maliyet oluşturur ve Tebliğ'e aykırıdır. "
                                f"[MADDE 14 - FIKRA 2: Ücretler şeffaf ve makul olmalıdır]",
                                f"DÜZELTME ÖNERİSİ: Yabancı para cinsinden işlemlerde Banka'nın ilan ettiği güncel döviz kuru aynen uygulanır, "
                                f"herhangi bir yüzdesel artış (%{percentage} gibi) uygulanmaz. "
                                f"(İfadeden '%{percentage} fazlası' kısmı çıkarılmalı)"
                            )
                except ValueError:
                    continue
    
    # ========================================================================
    # STANDARD LIMIT CHECKS
    # ========================================================================
    bank_amounts = extract_amounts(bank_text)
    if not bank_amounts:
        return None  # No amounts to check
    
    # Check each Tebliğ match for limits
    for match in teblig_matches:
        teblig_text = match.get('text', '')
        limits = extract_limits(teblig_text)
        
        if not limits:
            continue
        
        # Check violations
        for bank_amount in bank_amounts:
            for limit_type, limit_value in limits:
                if limit_type == 'max' and bank_amount > limit_value:
                    return (
                        'NOT_OK',
                        f"Sayısal ihlal: Banka {bank_amount} TL ücret alıyor, "
                        f"Tebliğ'de azami limit {limit_value} TL. "
                        f"Fark: {bank_amount - limit_value} TL fazla. "
                        f"[Madde {match.get('madde', 'N/A')} Fıkra {match.get('fikra', 'N/A')}]",
                        f"DÜZELTME ÖNERİSİ: Ücret {limit_value} TL'ye düşürülmelidir (Tebliğ azami limit)."
                    )
                elif limit_type == 'min' and bank_amount < limit_value:
                    return (
                        'NOT_OK',
                        f"Sayısal ihlal: Banka {bank_amount} TL belirtiyor, "
                        f"Tebliğ'de asgari {limit_value} TL gerekli. "
                        f"Fark: {limit_value - bank_amount} TL eksik. "
                        f"[Madde {match.get('madde', 'N/A')} Fıkra {match.get('fikra', 'N/A')}]",
                        f"DÜZELTME ÖNERİSİ: Asgari {limit_value} TL olarak güncellenmelidir (Tebliğ minimum şartı)."
                    )
    
    return None  # No violation found

def check_keyword_violations(bank_text: str, teblig_matches: List[Dict[str, Any]]) -> Optional[Tuple[str, str, str]]:
    """Check for keyword-based violations (e.g., 'onaysız', 'zorunlu').
    Returns (status, reason, corrected_text) if violation found, else None.
    """
    # Pattern: "onaysız", "müşteri onayı olmadan", "zorunlu"
    violation_keywords = [
        (r'onaysız', 'Müşteri onayı olmadan işlem yapılıyor'),
        (r'müşteri\s+onay[ıi]\s+olmadan', 'Müşteri onayı olmadan işlem yapılıyor'),
        (r'zorunlu.*onaysız', 'Zorunlu ve onaysız işlem'),
        (r'onay.*olmaksızın', 'Onay olmaksızın işlem yapılıyor'),
    ]
    
    # Simple Turkish lowercase normalization for basic matching
    # (Handling I/ı/İ/i distinction manually for critical keywords if needed, 
    # but here we rely on the regex patterns covering common variations or standard lower())
    bank_lower = bank_text.replace('İ', 'i').replace('I', 'ı').lower()
    
    for pattern, reason in violation_keywords:
        if re.search(pattern, bank_lower):
            # Check if Tebliğ matches are even relevant or if this is a standalone prohibition
            # For "onaysız", it's a general prohibition in usage, so we might not need a specific Tebliğ match 
            # to validate it IF the bank explicitly says "onaysız".
            # But let's keep the logic to fetch citation if available.
            
            # If no teblig matches, we can still flag it as a general violation if we are sure?
            # Ground truth implies it cites MADDE 9 - FIKRA 2.
            
            # Search for a relevant citation in the potential matches provided
            citation_found = "N/A"
            for match in teblig_matches:
                teblig_text = match.get('text', '').replace('İ', 'i').replace('I', 'ı').lower()

                if 'onay' in teblig_text or 'izin' in teblig_text:
                    return (
                        'NOT_OK',
                        f"İhlal tespit edildi: {reason}. "
                        "Tebliğ'de müşteri onayı gereklidir. "
                        f"[Madde {match.get('madde', 'N/A')} Fıkra {match.get('fikra', 'N/A')}]",
                        "İlgili işlem için müşteri onayı alınması gerekmektedir. "
                        "'Onaysız' veya 'zorunlu' ifadeleri kaldırılmalı ve 'müşteri onayı ile' ifadesi eklenmelidir."
                    )
    
    return None

def check_prohibition_compliance(bank_text: str, teblig_matches: List[Dict[str, Any]]) -> Optional[Tuple[str, str, str]]:
    """
    Check for prohibition-based violations using both:
    1. Prohibition keywords in Tebliğ matches ("yasak", "alamaz", "ücretsiz")
    2. Domain-specific rules for known prohibited fees (e.g., "hesap açılış", "kendi ATM")
    
    Returns (status, reason, corrected_text) if violation found, else None.
    
    Examples:
    - Bank: "Hesap açılış ücreti 50 TL" → NOT_OK (MADDE 13: alınamaz)
    - Bank: "Kendi ATM nakit çekim 5 TL" → NOT_OK (MADDE 14: alınamaz)
    - Bank: "Para yatırma 3 TL" → NOT_OK (MADDE 14: ücretsiz)
    """
    # Extract amounts from bank text
    bank_amounts = extract_amounts(bank_text)
    bank_lower = bank_text.lower()
    
    # ========================================================================
    # STRATEGY 1: Domain-specific prohibition rules (high-priority)
    # These catch cases where RAG similarity is too low to find the right Tebliğ article
    # ========================================================================
    
    domain_rules = [
        {
            'keywords': [r'hesap açılış.*ücret', r'hesap açma.*ücret', r'açılış.*ücret', 'yeni hesap.*ücret'],
            'exclude_keywords': ['ücret tarifeleri', 'ücret listesi', 'ücret tarifesi', 'ücret tablosu', 'sigorta', 'tmsf', 'sigortalanmıştır', 'sigortalanır'],
            'reason': 'Hesap açılış işlemlerinden ücret alınamaz',
            'citation': 'MADDE 13 - FIKRA 1',
            'full_text': 'Finansal tüketicilerin açtıkları mevduat ve katılım fonu hesaplarının açılış işlemlerinde... ücret alınamaz.'
        },
        {
            'keywords': ['kendi.*atm.*çekim', 'kendi bankamız atm', 'kendi atm.*nakit', 'atm.*nakit.*çekim'],
            'exclude_keywords': ['başka banka', 'diğer kuruluş', 'başka kuruluş', 'başka.*atm', 'ücret tarifeleri', 'ücret listesi'],
            'reason': 'Kendi ATM\'den para çekme işlemlerinden ücret alınamaz',
            'citation': 'MADDE 14 - FIKRA 1',
            'full_text': 'Finansal tüketicinin hesabının bulunduğu kuruluşa ait ATM\'lerden... para çekme işlemlerinden ücret alınamaz.'
        },
        {
            'keywords': ['kendi hesaba.*para yatırma', 'kendi hesap.*yatırma', 'para yatırma.*kendi', r'para yatırma.*\(şube\)'],
            'exclude_keywords': ['tarife', 'ücret listesi', 'başka', 'diğer'],
            'reason': 'Kendi hesaba para yatırma işlemlerinden ücret alınamaz',
            'citation': 'MADDE 14 - FIKRA 1 (para yatırma)',
            'full_text': 'Finansal tüketicinin... kendi hesabı için para yatırma... işlemlerinden ücret alınamaz.'
        },
        {
            'keywords': [r'hesap işletim.*ücret', r'hesap bakım.*ücret', r'işletim.*ücret', r'hesap.*aidat'],
            'exclude_keywords': ['kredi kartı', 'kart aidatı', 'üyelik ücreti'],
            'reason': 'Hesap işletim ücreti veya benzeri adlar altında ücret alınamaz',
            'citation': 'MADDE 13 - FIKRA 1',
            'full_text': 'Finansal tüketicilerin açtıkları mevduat ve katılım fonu hesaplarından... hesap işletim ücreti... alınamaz.'
        },
        {
            'keywords': ['bakiye sorgulama', 'bakiye sorgu'],
            'exclude_keywords': ['tarife', 'ücret listesi'],
            'reason': 'Bakiye sorgulama işlemlerinden ücret alınamaz',
            'citation': 'MADDE 14 - FIKRA 1 (bakiye sorgulama)',
            'full_text': 'Finansal tüketicinin... bakiye sorgulama... işlemlerinden ücret alınamaz.'
        },
        {
            'keywords': [r'sözleşme.*ilk yıl', r'ilk yıl.*sözleşme', r'tekrar.*basım'],
            'exclude_keywords': [],
            'reason': 'Sözleşme örneği ilk yıl ücretsizdir',
            'citation': 'MADDE 9 - FIKRA 4',
            'full_text': 'Finansal tüketicinin talep etmesi halinde... sözleşmenin bir örneğinin ilk yıl ücretsiz verilmesi zorunludur.'
        },

        {
            'keywords': ['hesap işletim ücreti', 'hesap bakım ücreti', 'hesap aidatı', 'işletim ücreti'],
            'exclude_keywords': ['yasal', 'vergi'],
            'reason': 'Hesap işletim ücreti alınamaz',
            'citation': 'Danıştay İDDK Kararı & Tebliğ Madde 13',
            'full_text': 'Mevduat ve katılım fonu hesaplarından hesap işletim ücreti tahsil edilemez.'
        },
        {
            'keywords': ['kredi kartı aidatı', 'kart aidatı', 'yıllık üyelik bedeli'],
            'exclude_keywords': ['aidatsız', 'ücretsiz', 'alınmaz'],
            'reason': 'Tüketiciye aidatsız kredi kartı seçeneği sunulmak zorundadır',
            'citation': 'MADDE 11 - FIKRA 1',
            'full_text': 'Kart çıkaran kuruluşlar, tüketicilere yıllık üyelik aidatı ve benzeri isim altında ücret tahsil etmedikleri bir kredi kartı türü sunmak zorundadır.'
        }
    ]
    
    # Check domain rules first (more reliable than Tebliğ keyword search)
    for rule in domain_rules:
        # First check if any exclude keywords exist
        if any(re.search(exclude_pattern, bank_lower) for exclude_pattern in rule.get('exclude_keywords', [])):
            continue  # Skip this rule if exclude keyword found
        
        # Check if any keyword matches
        for keyword_pattern in rule['keywords']:
            # Found prohibited operation + amounts = violation
            if re.search(keyword_pattern, bank_lower) and bank_amounts:
                # For ücret tarifesi (short fee tables), context check may fail
                # So we check: if text is short (<200 chars), skip context check
                if len(bank_lower) < 200:
                    # Short text - likely a fee table entry, accept it
                    amounts_str = ", ".join(f"{a:.1f} TL" for a in bank_amounts)
                    # Generate corrected text suggestion
                    if 'hesap açılış' in keyword_pattern.lower() or 'açılış' in keyword_pattern.lower():
                        corrected_suggestion = "Hesap Açılış Ücreti: 0 TL (Ücretsiz)"
                    elif 'atm' in keyword_pattern.lower():
                        corrected_suggestion = "Kendi ATM'den Nakit Çekim: 0 TL (Ücretsiz)"
                    elif 'para yatırma' in keyword_pattern.lower():
                        corrected_suggestion = "Kendi Hesaba Para Yatırma: 0 TL (Ücretsiz)"
                    elif 'para yatırma' in keyword_pattern.lower():
                        corrected_suggestion = "Kendi Hesaba Para Yatırma: 0 TL (Ücretsiz)"
                    elif 'bakiye' in keyword_pattern.lower():
                        corrected_suggestion = "Bakiye Sorgulama: 0 TL (Ücretsiz)"
                    elif 'işletim ücreti' in keyword_pattern.lower():
                        corrected_suggestion = "Hesap İşletim Ücreti: 0 TL (Yoktur)"
                    elif 'kart aidatı' in keyword_pattern.lower():
                        corrected_suggestion = "Aidatsız Kart Seçeneği Sunulmalıdır"
                    else:
                        corrected_suggestion = "Bu işlem ücretsiz olmalıdır"
                    
                    return (
                        'NOT_OK',
                        f"[DOMAIN-RULE] {rule['reason']}. "
                        f"Banka {amounts_str} ücret alıyor. "
                        f"Tebliğ: '{rule['full_text']}' [{rule['citation']}]",
                        f"DÜZELTME ÖNERİSİ: {corrected_suggestion}"
                    )
                else:
                    # Long text - do context check to avoid false positives
                    if (match_obj := re.search(keyword_pattern, bank_lower)):
                        match_pos = match_obj.start()
                        # Check 150 chars before and after for better context
                        context_start = max(0, match_pos - 150)
                        context_end = min(len(bank_lower), match_pos + 200)
                        context = bank_lower[context_start:context_end]
                        
                        # If amounts in text are not near the keyword match, likely false positive
                        # Check if amount appears within context window
                        has_amount_in_context = False
                        for amount in bank_amounts:
                            amount_pattern = str(amount).replace('.', r'[.,]')
                            if re.search(amount_pattern, context):
                                has_amount_in_context = True
                                break
                        
                        if has_amount_in_context:
                            amounts_str = ", ".join(f"{a:.1f} TL" for a in bank_amounts)
                            # Generate corrected text suggestion
                            if 'hesap açılış' in keyword_pattern.lower() or 'açılış' in keyword_pattern.lower():
                                corrected_suggestion = "Hesap açılış işlemleri ücretsizdir. (Bu cümleden ücret bilgisi çıkarılmalı)"
                            elif 'atm' in keyword_pattern.lower():
                                corrected_suggestion = "Kendi ATM'lerden nakit çekim işlemleri ücretsizdir. (Bu cümleden ücret bilgisi çıkarılmalı)"
                            elif 'para yatırma' in keyword_pattern.lower():
                                corrected_suggestion = "Kendi hesaba para yatırma işlemleri ücretsizdir. (Bu cümleden ücret bilgisi çıkarılmalı)"
                            elif 'para yatırma' in keyword_pattern.lower():
                                corrected_suggestion = "Kendi hesaba para yatırma işlemleri ücretsizdir. (Bu cümleden ücret bilgisi çıkarılmalı)"
                            elif 'bakiye' in keyword_pattern.lower():
                                corrected_suggestion = "Bakiye sorgulama işlemleri ücretsizdir. (Bu cümleden ücret bilgisi çıkarılmalı)"
                            elif 'işletim ücreti' in keyword_pattern.lower():
                                corrected_suggestion = "Hesap işletim ücreti alınamaz. (Bu madde çıkarılmalı)"
                            elif 'kart aidatı' in keyword_pattern.lower():
                                corrected_suggestion = "Banka, aidatsız bir kredi kartı türü sunmak zorundadır. (Sözleşmede aidatsız seçenek belirtilmeli)"
                            else:
                                corrected_suggestion = "Bu işlem ücretsiz olmalıdır, ücret ifadesi kaldırılmalıdır"
                            
                            return (
                                'NOT_OK',
                                f"[DOMAIN-RULE] {rule['reason']}. "
                                f"Banka {amounts_str} ücret alıyor. "
                                f"Tebliğ: '{rule['full_text']}' [{rule['citation']}]",
                                f"DÜZELTME ÖNERİSİ: {corrected_suggestion}"
                            )
    
    # If no amounts in bank text, can't determine violation
    if not bank_amounts:
        return None
    
    # ========================================================================
    # STRATEGY 2: Prohibition keywords in Tebliğ matches
    # Only trigger if similarity is VERY high (>0.75) to minimize false positives
    # ========================================================================
    
    prohibition_patterns = [
        (r'yasak(?:tır|lanmıştır)?', 'Yasak bir işlem için ücret alınıyor'),
        (r'alamaz|alınamaz', 'Ücret alınamaz denirken banka ücret alıyor'),
        (r'ücretsiz(?:dir)?', 'Ücretsiz olması gereken işlem için ücret alınıyor'),
        (r'geçemez|aşamaz|olamaz', 'Aşılamaz bir sınır aşılıyor'),
        (r'yapılamaz|mümkün\s+değil', 'Yapılamaz denilen işlem için ücret alınıyor'),
        (r'ücret.*talep\s+edilemez', 'Ücret talep edilemez denirken banka ücret alıyor'),
    ]
    
    # Check each Tebliğ match for prohibition keywords
    for match in teblig_matches:
        # INCREASED threshold to 0.75 to reduce false positives
        if match.get('similarity', 0) < 0.75:
            continue
            
        teblig_text = match.get('text', '').lower()
        
        for pattern, reason in prohibition_patterns:
            if re.search(pattern, teblig_text):
                # Additional validation: Check if this is really a violation
                # Skip if bank text contains positive phrases (ücret alınmayacak, ücretsiz, vs.)
                positive_phrases = ['alınmayacak', 'alınmaz', 'ücretsiz', 'ücret yok', 'talep edilmeyecek']
                if any(phrase in bank_lower for phrase in positive_phrases):
                    continue  # Bank text already says "no fee", so not a violation
                
                # Found prohibition in Tebliğ + amounts in bank text → violation
                amounts_str = ", ".join(f"{a:.1f} TL" for a in bank_amounts)
                return (
                    'NOT_OK',
                    f"[TEBLIG-PROHIBITION] {reason}. "
                    f"Banka {amounts_str} ücret alıyor ama Tebliğ bunu yasaklıyor. "
                    f"[Madde {match.get('madde', 'N/A')} Fıkra {match.get('fikra', 'N/A')}]",
                    "DÜZELTME ÖNERİSİ: Bu ücret Tebliğ tarafından yasaklandığı için tamamen kaldırılmalıdır."
                )
    
    return None


# ============================================================================
# PHASE 1: Chain-of-Thought Prompt with Multi-Step Reasoning
# ============================================================================

def build_compliance_prompt(chunk: Dict[str, Any]) -> str:
    """Build a structured prompt for LLM compliance checking."""
    
    # Build context from top matches
    context_items = []
    for m in chunk['top_matches'][:5]: # Limit to top 5 matches
        metadata = m.get('metadata', {})
        # Handle flattened or nested structure
        madde = metadata.get('madde', m.get('madde', 'N/A'))
        fikra = metadata.get('fikra', m.get('fikra', 'N/A'))
        text = m.get('text', '')
        similarity = m.get('similarity', 0.0)
        context_items.append(f"- [Tebliğ Madde {madde} Fıkra {fikra}] (Benzerlik: {similarity:.2f})\n   {text}")
    
    context = "\n\n".join(context_items) if context_items else "İlgili Tebliğ maddesi bulunamadı."

    return f"""Sen bir BDDK uyumluluk denetçisisin. Banka dokümanlarını Tebliğ ile karşılaştırıyorsun.

**OTOMATİK KONTROL SİSTEMİ UYARISI:**
{chunk.get('automated_warning', 'Yok')}
(Bu uyarı, metinde bazı sayısal veya kelime ihlalleri olabileceğini gösterir. Lütfen bu uyarıyı context ile birleştirerek değerlendir. Eğer context bu uyarıyı doğruluyorsa NOT_OK ver, ama context farklı bir durumdan bahsediyorsa kendi kararını ver.)

**KRİTİK: SADECE JSON FORMATINDA YANIT VER, HİÇBİR EK AÇIKLAMA YAPMA!**

**KRİTİK: SADECE JSON FORMATINDA YANIT VER, HİÇBİR EK AÇIKLAMA YAPMA!**

**TEBLİĞ KAPSAMI (ÖNEMLİ):**
Bu Tebliğ YALNIZCA şunları düzenler: "ücret, komisyon, masraf, limitler, yasaklar, müşteri koruma hükümleri"
⚠️ MÜŞTERİ KORUMA MADDELERİ KAPSAM İÇİNDEDİR! (bilgilendirme, şeffaflık, hesap kapatma hakları vs.)
Kapsam DIŞI: sadece faiz oranları, sadece vade süreleri, sadece genel tanımlar

**BANKA MADDESİ:**
{chunk['text']}

**İLGİLİ TEBLİĞ MADDELERİ:**
{context}

**FEW-SHOT ÖRNEKLER:**

✅ OK Örneği:
Banka: "Hesap kapatma masrafı 10 TL"
Tebliğ: "Hesap kapatma masrafı 10 TL ile sınırlıdır" (Madde 15-Fıkra 2)
→ {{"status": "OK", "reason": "Banka 10 TL ücret alıyor, Tebliğ limiti 10 TL. Uyumlu.", "citation": "MADDE 15 - FIKRA 2"}}

❌ NOT_OK Örneği 1:
Banka: "Hesap kapatma masrafı 15 TL"
Tebliğ: "Hesap kapatma masrafı 10 TL ile sınırlıdır" (Madde 15-Fıkra 2)
→ {{"status": "NOT_OK", "reason": "Banka 15 TL alıyor, Tebliğ limiti 10 TL. 5 TL fazla.", "citation": "MADDE 15 - FIKRA 2", "corrected_text": "Hesap kapatma masrafı 10 TL"}}

❌ NOT_OK Örneği 2:
Banka: "Yabancı para cinsinden yapılan işlemlerde Banka Gişe Satış kurunun %5 fazlası uygulanacaktır"
Tebliğ: "Ücretler şeffaf ve makul olmalıdır" (Madde 14-Fıkra 2)
→ {{"status": "NOT_OK", "reason": "Döviz işlemlerinde %5 yüzdesel fazlalık uygulanması müşteri aleyhine keyfi maliyet oluşturur. Tebliğ'e aykırıdır.", "citation": "MADDE 14 - FIKRA 2", "corrected_text": "Yabancı para cinsinden yapılan işlemlerde Banka'nın ilan ettiği güncel döviz kuru aynen uygulanır"}}

⚪ NA Örneği:
Banka: "Kredi notu bilgilerinizi takip edebilirsiniz"
Tebliğ: "Hesap kapatma masrafı 10 TL ile sınırlıdır" (Madde 15-Fıkra 2)
→ {{"status": "NA", "reason": "Kredi notu bilgilendirmesi, Tebliğ kapsamı dışı (ücret/komisyon değil).", "citation": ""}}

**CHAIN-OF-THOUGHT REASONING (Adım adım düşün, AMA JSON DIŞINDA YAZMA):**
1. Bu madde kapsam içinde mi? (ücret/komisyon/limit/yasak/müşteri koruma mı?)
2. Eğer KAPSAMDA: Sayılar var mı? Karşılaştır! (örn: 15 TL > 10 TL limit?)
3. Eğer KAPSAMDA ve sayı yoksa: Yasak bir durum mu? (örn: onaysız ücret?)
4. Son karar: OK (uyumlu), NOT_OK (ihlal), NA (kapsam dışı)

**GÖREV:** Aşağıdaki JSON'u AYNEN doldur:

{{
  "status": "OK veya NOT_OK veya NA",
  "reason": "ADIM ADIM AÇIKLA: 1) Kapsam? 2) Sayısal karşılaştırma (varsa)? 3) Sonuç?",
  "citation": "MADDE X - FIKRA Y",
  "corrected_text": "Düzeltilmiş metin (sadece NOT_OK ise)"
}}

**STATUS KURALLARI - ÖNCELİK SIRASI (MUTLAKA BU SIRAYLA KONTROL ET!):**

**KRİTİK İSTİSNA (HER ŞEYDEN ÖNCE OKU):**
- Eğer bir ücret kalemi (örn. Başka Banka ATM, EFT, Kart Aidatı) **Ek-1 Listesinde** yer alıyorsa, bu ücret TEBLİĞ TARAFINDAN İZİN VERİLMİŞTİR.
- Bu durumda:
  - Eğer Tebliğ metninde açıkça sayısal bir limit (örn. "10 TL'yi geçemez") varsa ve banka bunu aşıyorsa -> `NOT_OK`.
  - Eğer Tebliğ metninde sayısal bir limit YOKSA (veya limit enflasyona/TÜFE'ye bağlıysa ve metinde yazmıyorsa) -> **ASLA `NOT_OK` VERME!** Bunun yerine `NA` (veya `OK`) ver.
  - "Tebliğde limit yazmıyor o yüzden yasak" mantığı YANLIŞTIR. Ek-1'de varsa yasak değildir.


**ÖNEMLİ: OK ve NOT_OK karar verebiliyorsan, NA kullanma! NA sadece son çare.**

1. **NOT_OK KULLAN EĞER (EN YÜKSEK ÖNCELİK - HİÇBİR İHLAL KAÇMASIN!):**
   - Banka Tebliğ'in belirlediği ÜCRETİ/KOMİSYONU aşıyor (SAYILARI DİKKATLE KONTROL ET!)
   - Banka Tebliğ'in YASAK ettiği bir ücret alıyor (hesap açılış, işletim, vb.)
   - Tebliğ'de açıkça belirtilen limit/müşteri hakkı ihlal ediliyor
   - SAYISAL KARŞILAŞTIRMA: Banka değeri > Tebliğ limiti → MUTLAKA NOT_OK!
   - EFT ücretleri: Mobil >2 TL veya Şube >10 TL (1000-50K TL arası) → NOT_OK
   - Döviz işlemlerinde %X fazlalık → NOT_OK
   - **Kart Yenileme Ücreti:** "Kart yenileme bedeli", "kart aidatı" gibi ifadeleri ARA. Kart yenileme ÜCRETSİZDİR. Ücret varsa → NOT_OK!
   - **Zorunlu/Otomatik Onay (Bundling):** "Sözleşmeyi imzalamakla kartı kabul etmiş sayılırsınız" veya "Otomatik sigorta yapılır" gibi ifadeler → NOT_OK! (Ürünler ayrı talep edilmelidir).
   - ⚠️ Şüpheli durum: "Belki ihlal var" → NOT_OK de (ihlal kaçırma riski alma!)

    - **ANCAK - EK-1 İSTİSNASI:** Eğer ücret kalemi (örn: Başka Banka ATM, EFT, Kart Aidatı) Ek-1 listesinde veya genel olarak Tebliğ'de GEÇİYORSA (izin veriliyorsa), sadece limiti kontrol et. Limit yazmıyorsa veya enflasyon/TÜFE hesabı gerekiyorsa -> **NA VER** ("NOT_OK" verme!). "Tebliğ'de yazmıyor o yüzden yasak" deme!

2. **OK KULLAN EĞER:**
   - Banka maddesi ücret/komisyon/limit/müşteri hakkı içeriyor VE Tebliğ'e uyumlu
   - Banka Tebliğ limitinden düşük/eşit ücret alıyor (SAYILARI KARŞILAŞTIR!)
   - Banka müşteri koruma hükümlerini Tebliğ'e uygun tanımlıyor
   - İlgili Tebliğ maddesi var (benzerlik ≥ 0.30) VE kesin ihlal YOK
   - ⚠️ SADECE AYNI KONUDAN BAHSEDİYORSA OK KULLANABİLİRSİN!
   - ⚠️ Banka ve Tebliğ FARKLI KONULARDAN BAHSEDİYORSA → NA KULLAN!
   - Örnek OK: Banka "hesap kapatma 8 TL" diyor, Tebliğ "max 10 TL" diyor → OK
   - Örnek NA: Banka "kart limiti belirler" diyor, Tebliğ "EFT ücreti" diyor → NA (farklı konular!)
   - Müşteri bilgilendirme/koruma prosedürü Tebliğ'e uygun açıklanmış VE AYNI KONU
   - Faiz artışı bildirim prosedürü (30 gün önceden) VE benzer Tebliğ maddesi var → OK
   - Sözleşme feshi/cayma hakları müşteri lehine VE benzer Tebliğ düzenlemesi var → OK

3. **NA KULLAN SADECE EĞER (SON ÇARE - MÜMKÜNSE OK/NOT_OK TERCİH ET!):**
   - ⚠️ KONU UYUŞMAZLIĞI: Banka ve Tebliğ TAMAMEN FARKLI konulardan bahsediyor
   - Örnek: Banka "kart limiti" diyor, Tebliğ "ATM ücreti" diyor → FARKLI KONU → NA
   - Banka maddesi TAMAMEN genel tanım/terminoloji ("Kart Hamili: ..." gibi)
   - Banka maddesi SADECE prosedür açıklaması (ücret/limit/hak içermiyor)
   - Faiz oranları HAKKINDA BİLGİ (ücret değil, sadece faiz prosedürü)
   - KKDF/BSMV gibi yasal vergi açıklamaları (Tebliğ kapsamı dışı)
   - İlgili Tebliğ maddesi yok (benzerlik < 0.30)
   - Tebliğ maddesi banka maddesiyle TAMAMEN farklı konu (ilgisiz)
   - En yüksek benzerlik < 0.30 VE içerik tamamen farklı konular
   - "kabul eder", "beyan eder" gibi genel hüküm cümleleri (ücret/limit olmadan)
   - ⚠️ ÜCRET/KOMİSYON/LİMİT/HAK varsa ANCAK Tebliğ ile AYNI KONU DEĞİLSE → NA
   - ⚠️ AYNI KONU AMA İHLAL YOK/VAR → OK veya NOT_OK kullan, NA kullanma!

**KRİTİK UYARILAR:**
- Müşteri koruma/bilgilendirme/şeffaflık maddeleri → KAPSAM İÇİNDE!
- Sayısal limitler varsa MUTLAKA karşılaştır (15 TL > 10 TL → NOT_OK)
- "Belki uyumlu" durumunda → OK de (NA değil!)
- Emin değilsen reason'da "neden emin değilim" belirt ama gene de karar ver!

**ÖNEMLİ:** SADECE JSON yaz, hiçbir açıklama ekleme!
"""


def check_compliance_with_llm(
    chunk: Dict[str, Any],
    model: str = "gemma3:27b",
    temperature: float = 0.08,
    max_tokens: int = 500,
    min_sim: float = 0.30,
) -> Dict[str, Any]:
    # print("DEBUG: check_compliance_with_llm CALLED") # START DEBUG
    """
    v4.4 Two-Stage Approach: LLM first, rule-based validation only for NOT_OK.
    
    Stage 1: LLM makes initial prediction
    Stage 2: If LLM says NOT_OK, validate with rule-based checks
             If LLM says OK/NA, trust it (skip rule-based to avoid false positives)
    """
    
    # If no top matches, return NA
    if not chunk["top_matches"]:
        return {
            "chunk_id": chunk["chunk_id"],
            "chunk_text": chunk["text"],
            "status": "NA",
            "reason": "İlgili Tebliğ maddesi bulunamadı",
            "citation": "",
            "corrected_text": "",
            "top_matches": []
        }

    # Check if all similarities are too low
    max_sim = max(m["similarity"] for m in chunk["top_matches"])
    if max_sim < min_sim:
        return {
            "chunk_id": chunk["chunk_id"],
            "chunk_text": chunk["text"],
            "status": "NA",
            "reason": f"Benzerlik çok düşük (max: {max_sim:.2f}, eşik: {min_sim}). Tebliğ'de ilgili madde bulunamadı.",
            "citation": "",
            "corrected_text": "",
            "top_matches": chunk["top_matches"]
        }
    
    # ========================================================================
    # HYBRID APPROACH: Rule-based checks as WARNINGS only
    # ========================================================================
    
    bank_text = chunk.get('text', '')
    warnings = []
    
    # Step 1: Check numerical violations
    if numerical_result := check_numerical_compliance(bank_text, chunk["top_matches"]):
        status, reason, corrected_text = numerical_result
        warnings.append(f"[SAYISAL KURAL UYARISI]: {reason}")

    # Step 2: Check keyword violations
    if keyword_result := check_keyword_violations(bank_text, chunk["top_matches"]):
        status, reason, corrected_text = keyword_result
        warnings.append(f"[ANAHTAR KELİME UYARISI]: {reason}")
        
    # Step 3: Check prohibition violations
    if prohibition_result := check_prohibition_compliance(bank_text, chunk["top_matches"]):
        status, reason, corrected_text = prohibition_result
        warnings.append(f"[YASAK KURALI UYARISI]: {reason}")
        
    # Inject warnings into chunk for prompt builder
    if warnings:
        chunk['automated_warning'] = "\n".join(warnings)
    
    # ========================================================================
    # FALLBACK: LLM with Chain-of-Thought (if rule-based didn't catch it)
    # ========================================================================

    # Build prompt and call LLM
    prompt = build_compliance_prompt(chunk)

    raw_content = ""
    try:
        response = generate_with_single_input(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        raw_content = response["content"].strip()

        # Try multiple JSON extraction methods
        content = None
        
        # Method 1: Extract from code blocks
        if json_match := re.search(
            r'```(?:json)?\s*(\{.*?\})\s*```', raw_content, re.DOTALL
        ):
            content = json_match.group(1)
        # Method 2: Find JSON object directly
        elif json_match := re.search(r'\{[^{}]*"status"[^{}]*\}', raw_content, re.DOTALL):
            content = json_match.group(0)
        # Method 3: Try entire content as JSON
        else:
            content = raw_content

        # Clean common JSON issues
        content = content.strip()
        content = re.sub(r',\s*}', '}', content)  # Remove trailing commas
        content = re.sub(r',\s*]', ']', content)  # Remove trailing commas in arrays

        # Parse JSON
        result = json.loads(content)

        # Validate required fields
        required = ["status", "reason", "citation"]
        for field in required:
            if field not in result:
                result[field] = ""

        if "corrected_text" not in result:
            result["corrected_text"] = ""

        # Validate status
        if result["status"] not in ["OK", "NOT_OK", "NA"]:
            result["status"] = "NA"
            result["reason"] = f"Geçersiz status: {result.get('status', 'unknown')}, orijinal yanıt: {raw_content[:200]}"

    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        # Fallback if LLM response is not valid JSON
        # Try to extract status from raw text
        status = "NA"
        if "OK" in raw_content[:100] and "NOT_OK" not in raw_content[:100]:
            status = "OK"
        elif "NOT_OK" in raw_content[:100]:
            status = "NOT_OK"
        
        result = {
            "status": status,
            "reason": f"LLM yanıtı parse edilemedi: {str(e)}. Yanıt: {raw_content[:300]}",
            "citation": "",
            "corrected_text": "",
        }

    # Add original chunk info
    result["chunk_id"] = chunk["chunk_id"]
    result["chunk_text"] = chunk["text"]
    result["top_matches"] = chunk["top_matches"]
    result["regenerated_response"] = raw_content
    result["detection_method"] = "llm_chain_of_thought"

    # -------------------------------------------------------------------------
    # HEURISTIC OVERRIDE: Catch specific "Bundled Consent" violations
    # -------------------------------------------------------------------------
    lower_text = " ".join(chunk["text"].split()).lower()
    # print(f"DEBUG HEURISTIC: {lower_text[:50]}...") # Debug print active
    
    # Pattern 1: Automatic Card Issuance upon signing
    # "bu sözleşmeyi imzalamanızla ... kart ... gönderilmesine ... onay verdiğiniz"
    # Relaxed logic: Check for "imzalamanızla" + "kart" + "gönderilmesine" OR "onay verdiğiniz kabul edilir"
    cond1 = "imzalamanızla" in lower_text and "kart" in lower_text and "gönderilmesine" in lower_text
    cond2 = "onay verdiğiniz kabul edilir" in lower_text and "kart" in lower_text
    
    if cond1 or cond2:
         result["status"] = "NOT_OK"
         result["reason"] = "[HEURISTIC] Banka, sözleşme imzalanmasıyla birlikte kart gönderimini 'kabul edilmiş' saymaktadır (Paket Satış / Bundling). Ürünler ayrı ayrı talep edilmeli, otomatik tanımlanmamalıdır."
         if "Madde 6" not in result["citation"] and "Madde 11" not in result["citation"]:
             result["citation"] = "MADDE 5 - FIKRA 1: İyi niyet kurallarına aykırı olarak tüketicinin zararına dengesizliğe neden olan sözleşme şartları haksız şarttır."

    # Pattern 2: Explicit "Automatic" insurance/overdraft
    if "otomatik olarak" in lower_text and "yapılır" in lower_text and ("sigorta" in lower_text or "kredili" in lower_text):
        result["status"] = "NOT_OK"
        result["reason"] = "[HEURISTIC] Banka, ürünleri (sigorta/kredili mevduat) otomatik olarak tanımlamaktadır. Bu durum 'Zorunlu Satış/Bundling' yasağına aykırıdır."
    # -------------------------------------------------------------------------

    return result


def process_document(
    result_file: Path,
    output_file: Path,
    model: str = "gemma3:27b",
    temperature: float = 0.1,
    max_tokens: int = 500,
    limit: Optional[int] = None,
    live_rag: bool = False,
    persist_dir: Path = Path(".chroma/teblig"),
    collection: str = "teblig_chunks",
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
    scope_match: bool = True,
    min_sim: float = 0.3,
    top_n: int = 3,
) -> None:
    """Process entire document and save compliance results."""

    if live_rag:
        print(f"Loading bank chunks (live RAG) from: {result_file}")
        bank_chunks = load_bank_chunks(result_file)
        # print(f"Found {len(bank_chunks)} chunks to process")
        if limit:
            bank_chunks = bank_chunks[:limit]
            print(f"Limiting to first {limit} chunks for testing")
        chunks = query_live_matches(
            bank_chunks,
            persist_dir=persist_dir,
            collection=collection,
            embedding_model=embedding_model,
            top_n=top_n,
            min_sim=min_sim,
            scope_match=scope_match,
        )
    else:
        print(f"Loading results from: {result_file}")
        chunks = parse_chroma_results(result_file)
        # print(f"Found {len(chunks)} chunks to process")
        if limit:
            chunks = chunks[:limit]
            print(f"Limiting to first {limit} chunks for testing")

    # Load config and settings
    try:
        from src.config_loader import config
        concurrency = config.get("llm", {}).get("concurrency", 4)
    except ImportError:
        concurrency = 4

    # Load ground truth if exists
    ground_truth_map = {}
    gt_data = []
    gt_path = Path("data/ground_truth.json")
    if gt_path.exists():
        import json
        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
                ground_truth_map = {item["chunk_id"]: item.get("ground_truth_label") for item in gt_data}
            # print(f"Loaded ground truth for {len(ground_truth_map)} chunks")
        except Exception as e:
            print(f"Could not load ground truth: {e}")

    results = []
    start_time = time.time()
    
    # --- DE-DUPLICATION LOGIC (GLOBAL) ---
    import hashlib
    unique_chunks = []
    # seen_hashes = set() # Use global now
    skipped_count = 0
    global _GLOBAL_SEEN_HASHES
    
    # print(f"Starting de-duplication on {len(chunks)} chunks (Global History: {len(_GLOBAL_SEEN_HASHES)} items)...")
    for chunk in chunks:
        # Use normalized text for hashing (strip whitespace, maybe lower case)
        # CRITICAL FIX: Prioritize 'text_for_embedding' because 'text' might be the full parent text shared by siblings
        text_content = chunk.get("text_for_embedding") or chunk.get("text", "")
        text_content = text_content.strip()
        if not text_content:
            continue
            
        # Create MD5 hash of the content
        text_hash = hashlib.md5(text_content.encode("utf-8")).hexdigest()
        
        if text_hash in _GLOBAL_SEEN_HASHES:
            skipped_count += 1
            # Optional: Log duplicate found
            # print(f"  [Skip] Duplicate content found: {chunk.get('chunk_id')}")
            continue
        
        _GLOBAL_SEEN_HASHES.add(text_hash)
        unique_chunks.append(chunk)
        
    # print(f"De-duplication complete. Kept: {len(unique_chunks)}, Skipped: {skipped_count} duplicates.")
    chunks = unique_chunks
    # -----------------------------

    total = len(chunks)
    
    # print(f"🚀 Starting compliance analysis...")

    # Progress tracking
    completed = 0
    # Sequential processing (User request to remove threads)
    for chunk in chunks:
        # process_single_chunk equivalent inline
        chunk_id = chunk.get("chunk_id", "unknown")
        try:
            result = check_compliance_with_llm(
                chunk=chunk,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                min_sim=min_sim,
            )
        except Exception as e:
            print(f"Error processing {chunk_id}: {e}")
            result = {"status": "ERROR", "reason": str(e)}

        
        # CRITICAL: Ensure chunk_id is in the result dict
        result["chunk_id"] = chunk_id
        
        completed += 1
        
        # Print result immediately
        status = result.get("status", "UNKNOWN")
        method = result.get("detection_method", "unknown")
        status_icon = ""
        
        # Get Ground Truth
        gt_label = ground_truth_map.get(chunk_id, "Unknown")
        gt_str = f" | Truth: {gt_label}" if gt_label != "Unknown" else ""
        
        # Comparison visual
        # ANSI Colors
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        RESET = "\033[0m"

        if gt_label != "Unknown":
            if status == gt_label:
                match_icon = f"-> {GREEN}MATCH{RESET}"
                raw_match = "MATCH"
            elif status in ["OK", "NA"] and gt_label in ["OK", "NA"]:
                match_icon = f"-> {YELLOW}BENIGN DIFF{RESET}"
                raw_match = "BENIGN DIFF"
            else:
                match_icon = f"-> {RED}MISMATCH{RESET}"
                raw_match = "MISMATCH"
            # User format: [9/12] Pred: NOT_OK  | Truth: NOT_OK -> MATCH
            print(f"  [{completed}/{total}] Pred: {status} {gt_str} {match_icon}")
        else:
            raw_match = ""

        # User Request: Show context for ALL "NOT_OK" predictions AND for "MISMATCH" cases
        # This covers:
        # 1. True Positives (NOT_OK -> MATCH) -> User wants to see context
        # 2. False Positives (NOT_OK -> MISMATCH) -> User wants to see context
        # 3. False Negatives (OK/NA -> MISMATCH) -> Useful for debugging
        
        # User Request: Show context for ALL "NOT_OK" predictions AND for "MISMATCH" cases
        # This covers:
        # 1. True Positives (NOT_OK -> MATCH) -> User wants to see context
        # 2. False Positives (NOT_OK -> MISMATCH) -> User wants to see context
        # 3. False Negatives (OK/NA -> MISMATCH) -> Useful for debugging
        
        show_details = (status == "NOT_OK") or (raw_match == "MISMATCH")

        if show_details:
            print(f"    🆔 Chunk ID: {chunk_id}")
            raw_text = chunk.get("text", "")
            print(f"    📝 Banka Maddesi: {raw_text}")
            
            matches = chunk.get("top_matches", [])
            if matches:
                m = matches[0]
                citation = f"MADDE {m.get('metadata', {}).get('madde_no', '?')} - FIKRA {m.get('metadata', {}).get('fikra_no', '?')}"
                rag_text = m.get('content') or m.get('page_content') or m.get('text') or ''
                rag_text = rag_text.replace('\n', ' ')
                # User format: ⚖️  İlgili Tebliğ Maddesi (En İyi Eşleşme): MADDE 9... : ...
                print(f"    ⚖️  İlgili Tebliğ Maddesi (En İyi Eşleşme): {citation}: {rag_text}")
            else:
                print(f"    ⚖️  İlgili Tebliğ Maddesi: Bulunamadı")
            
            # Print reason here in YELLOW
            reason_text = result.get('reason', '-')
            print(f"    {YELLOW}⚠️ Reason: {reason_text}{RESET}")
            print("-" * 50)
        
        # Reason is already printed above if show_details is true.
        # Print reason for others only if NA (optional info)
        if not show_details and status == "NA":
             pass 
        else:
            pass
        
        results.append(result)
        
        # Progress summary every 10 chunks
        if completed % 10 == 0 or completed == total:
            elapsed = time.time() - start_time
            avg_time = elapsed / completed if completed > 0 else 0
            remaining = (total - completed) * avg_time # No concurrency div
            
            ok = sum(r["status"] == "OK" for r in results)
            not_ok = sum(r["status"] == "NOT_OK" for r in results)
            na = sum(r["status"] == "NA" for r in results)
            
        # Progress update removed as requested
        

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Summary
    ok = sum(r["status"] == "OK" for r in results)
    not_ok = sum(r["status"] == "NOT_OK" for r in results)
    na = sum(r["status"] == "NA" for r in results)

    print(f"\n{'='*60}")
    print(f"SUMMARY for {result_file.name}")
    print(f"{'='*60}")
    
    total_results = len(results)
    print(f"Total chunks: {total_results}")
    
    if total_results > 0:
        print(f"OK:      {ok} ({ok/total_results*100:.1f}%)")
        print(f"NOT_OK:  {not_ok} ({not_ok/total_results*100:.1f}%)")
        print(f"NA:      {na} ({na/total_results*100:.1f}%)")
    else:
        print(f"OK:      0 (0.0%)")
        print(f"NOT_OK:  0 (0.0%)")
        print(f"NA:      0 (0.0%)")
    
    # Calculate detailed metrics if ground truth exists
    if ground_truth_map:
        y_true = []
        y_pred = []
        for r in results:
            chunk_id = r.get("chunk_id")
            gt = ground_truth_map.get(chunk_id)
            status = r.get("status")
            if gt and gt != "Unknown":
                y_true.append(gt)
                y_pred.append(status)
        
        if y_true:
            try:
                metrics = compute_metrics(y_true, y_pred)
                print(f"{'-'*60}")
                print(f"EVALUATION METRICS (Based on {len(y_true)} ground truth matches)")
                print(f"{'-'*60}")
                print(f"Accuracy:  {metrics['accuracy']:.2%} (Strict Match)")
                if 'binary_accuracy' in metrics:
                    print(f"Compliance Accuracy: {metrics['binary_accuracy']:.2%} (Violation vs Non-Violation)")
                
                # Print specifically for NOT_OK (the detailed class)
                not_ok_metrics = metrics['per_class']['NOT_OK']
                print(f"NOT_OK Precision: {not_ok_metrics['precision']:.2%}")
                print(f"NOT_OK Recall:    {not_ok_metrics['recall']:.2%}")
                print(f"NOT_OK F1 Score:  {not_ok_metrics['f1']:.2%}")
                
                # print(f"Macro F1 Score:   {metrics['macro_f1']:.2%}")

                # Compute semantic reasoning quality
                if gt_data:
                    reason_score = 0.0 # compute_reasoning_quality(results, gt_data) - Skipping duplicate calc
                    # print(f"Reason Semantic Score: {reason_score:.1f}/100 (Similarity to Ground Truth)")

            except Exception as e:
                print(f"Error computing metrics: {e}")

    print(f"{'='*60}")
    print(f"Results saved to: {output_file}")


def combine_results_from_dir(directory: Path, combined_path: Path) -> Path:
    """Combine all compliance JSON files in a directory into one list."""
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

    combined: List[Dict[str, Any]] = []
    for json_file in sorted(directory.glob("*.json")):
        name_lower = json_file.name.lower()
        if "test_compliance" in name_lower:
            continue
        if combined_path.name.lower() in name_lower:
            continue
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                combined.extend(data)

    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with combined_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"Combined {len(combined)} records into: {combined_path}")
    return combined_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-based compliance checker for bank documents"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to chroma_tool result file (e.g., logs/banka_vs_teblig/...txt)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save compliance results JSON"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma3:27b",
        help="Ollama model name (bankada gemma3:27b kullanılacak)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="LLM temperature (lower = more deterministic)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="LLM max token limit"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of chunks to process (for testing)"
    )
    parser.add_argument(
        "--combine-dir",
        type=Path,
        default=Path("logs/compliance_results"),
        help="Dizin içindeki tüm compliance JSON'larını birleştir"
    )
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=Path("logs/compliance_results/combined_compliance.json"),
        help="Birleştirilmiş JSON çıkış yolu"
    )
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Sadece mevcut JSON'ları birleştir, LLM çalıştırma"
    )
    parser.add_argument(
        "--live-rag",
        action="store_true",
        help="Chroma üzerinden canlı sorgu yap (input olarak banka chunk JSON bekler)"
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path(".chroma/teblig"),
        help="Chroma persist dizini"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="teblig_chunks",
        help="Chroma koleksiyon adı"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="Sorgu embedding modeli"
    )
    parser.add_argument(
        "--min-sim",
        type=float,
        default=0.3,
        help="Benzerlik eşiği (NA kararı için)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="RAG sonucunda alınacak en alakalı madde sayısı"
    )
    parser.add_argument(
        "--no-scope-match",
        action="store_false",
        dest="scope_match",
        help="Scope eşleşmesini devre dışı bırak"
    )
    parser.set_defaults(scope_match=True)

    args = parser.parse_args()

    if args.combine_only:
        combine_results_from_dir(args.combine_dir, args.combined_output)
        return

    process_document(
        result_file=args.input,
        output_file=args.output,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        limit=args.limit,
        live_rag=args.live_rag or args.input.suffix.lower() == ".json",
        persist_dir=args.persist_dir,
        collection=args.collection,
        embedding_model=args.embedding_model,
        scope_match=args.scope_match,
        min_sim=args.min_sim,
        top_n=args.top_n,
    )

    combine_results_from_dir(args.combine_dir, args.combined_output)


if __name__ == "__main__":
    main()
