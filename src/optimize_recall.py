
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List
# Explicitly import the checker function
try:
    from src.llm_compliance_check import check_compliance_with_llm
except ImportError:
    import sys
    sys.path.append('.') # Ensure cwd is in path if needed, though running 'python src/optimize.py' adds src to path
    from llm_compliance_check import check_compliance_with_llm

def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    print("🚀 Starting Recall Optimization Loop...")
    
    # 1. Load Ground Truth
    gt_path = Path("data/ground_truth.json")
    ground_truth = load_json(gt_path)
    
    # Filter for NOT_OK violations only (Essential Recall Items)
    target_violations = [item for item in ground_truth if item["ground_truth_label"] == "NOT_OK"]
    print(f"🎯 Total Violations to Catch: {len(target_violations)}")
    
    # 2. Load All Chunks to get text content & RAG matches
    # We prefer logs/compliance_results because they contain 'top_matches' (RAG context).
    chunks_dir = Path("data/banka_dokumanlari/chunks")
    logs_dir = Path("logs/compliance_results")
    chunk_map = {}
    
    # Load from logs first (Best source: has RAG context)
    for log_file in logs_dir.glob("*_compliance.json"):
        if log_file.name == "combined_compliance.json": continue
        try:
            chunks = load_json(log_file)
            for c in chunks:
                chunk_map[c["chunk_id"]] = c
        except Exception as e:
            print(f"Error loading log {log_file}: {e}")

    # Fallback: Load from vector_ready files (Has text but no RAG context)
    for chunk_file in chunks_dir.glob("*_vector_ready.json"):
        chunks = load_json(chunk_file)
        for c in chunks:
            if c["chunk_id"] not in chunk_map:
                chunk_map[c["chunk_id"]] = c
            
    print(f"📚 Loaded {len(chunk_map)} total chunks from disk.")
    
    # 3. Run Compliance Check on Targets
    passed = 0
    failed = 0
    failed_items = []
    
    print("\n⚡️ Running Compliance Checks on VIOLATIONS only...\n")
    
    for i, target in enumerate(target_violations, 1):
        chunk_id = target["chunk_id"]
        
        if chunk_id not in chunk_map:
            print(f"⚠️ Chunk not found in files: {chunk_id}")
            continue
            
        chunk = chunk_map[chunk_id]
        
        # Normalize Keys (Logs use 'chunk_text', Function needs 'text')
        if "text" not in chunk and "chunk_text" in chunk:
            chunk["text"] = chunk["chunk_text"]
            
        # Ensure proper keys exist
        if "text" not in chunk:
            print(f"⚠️ Skipped {chunk_id}: No text content")
            continue
            
        # Ensure top_matches exists even if empty (for code safety)
        if "top_matches" not in chunk:
            chunk["top_matches"] = []

        # Run Check
        try:
            result = check_compliance_with_llm(chunk, model="gemma3:27b")
            
            status = result["status"]
            
            if status == "NOT_OK":
                print(f"✅ [{i}/{len(target_violations)}] CAUGHT: {chunk_id}")
                passed += 1
            else:
                print(f"❌ [{i}/{len(target_violations)}] MISSED: {chunk_id} (Pred: {status})")
                print(f"   Reason: {result['reason'][:150]}...")
                failed += 1
                failed_items.append(result)
                
        except Exception as e:
            print(f"💥 Error on {chunk_id}: {e}")
            
    # 4. Summary
    print("\n" + "="*40)
    print(f"RESULTS: {passed}/{len(target_violations)} Caught")
    if len(target_violations) > 0:
        print(f"Recall: {passed/len(target_violations):.1%}")
    print("="*40)
    
    if failed_items:
        print("\n📝 Missed Items Reasons:")
        for f in failed_items:
            print(f"- {f['chunk_id']}: {f['reason']}")

if __name__ == "__main__":
    main()
