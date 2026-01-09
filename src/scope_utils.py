"""
Scope assignment utilities for Tebliğ and Bank documents.

This module provides scope categorization logic to enable proper matching
between bank documents and regulatory (Tebliğ) articles.
"""

from typing import List, Dict

# =============================================================================
# TEBLIG SCOPE MAPPING
# =============================================================================

TEBLIG_MADDE_SCOPE_MAP: Dict[int, List[str]] = {
    1: ["fees_disclosure"],  # Amaç, kapsam - sadece ücretlerle ilgili
    2: [],  # Kapsam - genel, matchlememeli (her şeyle eşleşmesin)
    3: [],  # Dayanak - genel, matchlememeli
    4: [],  # Tanımlar - genel, matchlememeli (sadece tanım yapıyor)
    5: ["account_management"],  # Sözleşme esasları
    6: ["fees_disclosure"],  # Ücret kategorileri
    7: ["fees_calculation"],  # Ücret güncellemeleri
    8: ["refund_procedures"],  # İadeler
    9: ["fees_disclosure"],  # Bilgilendirme gereklilikleri
    10: ["fees_calculation", "account_management"],  # Kredi işletme ücretleri
    11: ["credit_card", "fees_disclosure"],  # Kredi kartı ücretleri
    12: ["account_management"],  # Para yatırma işlemleri
    13: ["account_management"],  # Hesap açma/yönetimi
    14: ["account_management"],  # ATM, kiralık kasa
    15: ["fees_calculation"],  # Paket fiyatlandırma
    16: [],  # Yürürlük tarihi - genel
    17: [],  # Yürütme - genel
}


def assign_teblig_scopes(madde_no: int) -> List[str]:
    """
    Assign scope categories to a Tebliğ article (madde) based on content.
    
    Args:
        madde_no: Article number
        
    Returns:
        List of scope category strings (empty list if no specific scope)
    """
    return TEBLIG_MADDE_SCOPE_MAP.get(madde_no, [])


# =============================================================================
# BANK DOCUMENT SCOPE ASSIGNMENT
# =============================================================================

def assign_bank_document_scopes(doc_id: str, text: str) -> List[str]:
    """
    Assign scope categories to a bank document chunk based on document type and content.
    
    Args:
        doc_id: Document identifier
        text: Text content of the chunk (lowercase for matching)
        
    Returns:
        List of scope category strings
    """
    scopes = []
    text_lower = text.lower()
    
    # Document type-based scopes
    if "kredi_karti" in doc_id.lower() or "kredi kartı" in text_lower or "kart" in text_lower:
        scopes.append("credit_card")
    
    if "bireysel" in doc_id.lower() or "bireysel bankacılık" in text_lower:
        scopes.append("account_management")
    
    if "ucret" in doc_id.lower() or "ücret" in text_lower or "tarife" in text_lower:
        scopes.append("fees_disclosure")
    
    # Content-based scopes (more specific)
    if any(kw in text_lower for kw in ["ücret hesapl", "faiz hesapl", "oranı", "asgari"]) and "fees_calculation" not in scopes:
        scopes.append("fees_calculation")
    
    if any(kw in text_lower for kw in ["hesap aç", "mevduat", "bakiye yönet", "atm işlem"]) and "account_management" not in scopes:
        scopes.append("account_management")
    
    if any(kw in text_lower for kw in ["iade", "geri ödeme", "cayma hakkı"]) and "refund_procedures" not in scopes:
        scopes.append("refund_procedures")
    
    # NOT adding general_terms anymore - let chunks have empty scopes if truly general
    # This will cause NA for chunks that don't match specific regulatory areas
    
    return list(set(scopes))  # Remove duplicates
