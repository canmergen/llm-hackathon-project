import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any

class HybridRetriever:
    """Combines BM25 and Vector Search using Reciprocal Rank Fusion (RRF)."""
    
    def __init__(self, documents: List[str], metadatas: List[Dict[str, Any]]):
        self.documents = documents
        self.metadatas = metadatas
        self.tokenizer = lambda text: text.lower().split()
        
        # Initialize BM25
        tokenized_corpus = [self.tokenizer(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
    def search_bm25(self, query: str, top_n: int = 20) -> List[Dict[str, Any]]:
        """Get top_n results using BM25."""
        tokenized_query = self.tokenizer(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get indices of top scores
        top_indices = np.argsort(scores)[::-1][:top_n]
        
        results = []
        for idx in top_indices:
            results.append({
                "index": int(idx),
                "score": float(scores[idx]),
                "text": self.documents[idx],
                "metadata": self.metadatas[idx]
            })
        return results

def reciprocal_rank_fusion(
    vector_results: List[Dict[str, Any]], 
    bm25_results: List[Dict[str, Any]], 
    k: int = 60
) -> List[Dict[str, Any]]:
    """Merge results using RRF."""
    
    # Map doc content or ID to score
    fused_scores = {}
    
    # Process vector results
    for rank, doc in enumerate(vector_results):
        doc_signature = doc["metadata"].get("id") or doc["text"][:100]
        if doc_signature not in fused_scores:
            fused_scores[doc_signature] = {"score": 0.0, "doc": doc}
        fused_scores[doc_signature]["score"] += 1 / (k + rank + 1)
        
    # Process BM25 results
    for rank, doc in enumerate(bm25_results):
        doc_signature = doc["metadata"].get("id") or doc["text"][:100]
        if doc_signature not in fused_scores:
            fused_scores[doc_signature] = {"score": 0.0, "doc": doc}
        fused_scores[doc_signature]["score"] += 1 / (k + rank + 1)
        
    # Sort by fused score
    sorted_docs = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_docs]
