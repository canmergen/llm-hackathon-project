"""End-to-end runner for ingestion, live RAG compliance, and result combine.

🎯 FULL WORKFLOW (Recommended):
    python main.py --skip-preparation --skip-ingest --with-readme
    
    Pipeline Akışı:
    1. Compliance check (tüm dökümanlar)
    2. Streamlit otomatik açılır → Dashboard'ı incele
    3. Screenshot al: Cmd+Shift+4 → docs/streamlit_dashboard.png
    4. Streamlit'i kapat: Ctrl+C
    5. ✨ README.md OTOMATIK OLUŞUR (sadece --with-readme flag'i varsa)

Alternative Usage:
    python main.py --skip-preparation --skip-ingest     # Pipeline + Streamlit (README yok)
    python main.py --only-streamlit                     # Only launch Streamlit viewer
    python main.py --readme                             # Only generate README.md
    python main.py --no-streamlit                       # Pipeline without Streamlit

Options:
    --skip-preparation     Skip data preparation step
    --skip-ingest          Skip Tebliğ ingest if you already have .chroma/teblig
    --force-ingest         Re-ingest Tebliğ even if .chroma/teblig exists
    --no-streamlit         Skip Streamlit auto-launch after pipeline
    --only-streamlit       Only launch Streamlit viewer (skip all processing)
    --readme               Generate comprehensive README.md (requires completed pipeline)
    --model gemma3:27b     Override LLM model (default: gemma3:27b)
    --temperature 0.08     Override temperature (default: 0.08) [optimized for better recall]
    --min-sim 0.30         Similarity threshold for NA (default: 0.30) [lowered for more context]
    --top-n 8              Number of Tebliğ matches to fetch per chunk (default: 8) [increased for richer context]
    --max-tokens 500       Max tokens for LLM response (default: 500)
    --limit N              Process only N chunks per document (for quick tests)

Recent Optimizations (v3):
    - Improved prompt with few-shot examples (OK, NOT_OK, NA)
    - Emphasized customer protection clauses (in scope!)
    - Added numerical comparison instructions
    - Reduced NA bias (more confident OK/NOT_OK predictions)
    - Lowered min_sim 0.35→0.30, raised top_n 5→8, temp 0.05→0.08
    - Expected improvement: 69% → 75-78% accuracy
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

from src.chroma_tool import ingest
from src.llm_compliance_check import process_document, combine_results_from_dir
from src.evaluate_model import compute_metrics, analyze_errors, compute_reasoning_quality, load_json
try:
    from src.report_generator_pdf import generate_pdf_report
except ImportError:
    generate_pdf_report = None

# Paths
TEBLIG_JSON = Path("data/teblig/chunks/teblig_chunks_vector_ready.json")
PERSIST_DIR = Path(".chroma/teblig")
COLLECTION = "teblig_chunks"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

BANK_DOCS: List[Tuple[Path, Path]] = [
    # Excel first (most NOT_OK cases here)
    (
        Path("data/banka_dokumanlari/chunks/ucret_tarifeleri_2025_vector_ready.json"),
        Path("logs/compliance_results/ucret_tarifeleri_compliance.json"),
    ),
    # PDF second
    (
        Path("data/banka_dokumanlari/chunks/Bireysel_Bankacilik_Hizmetleri_Sozlesmesi_2025_vector_ready.json"),
        Path("logs/compliance_results/bireysel_bankacilik_sozlesmesi_compliance.json"),
    ),
    # Word last
    (
        Path("data/banka_dokumanlari/chunks/KREDI_KARTI_SOZLESMESI_HIBRIT_01.08.2025_GUNCEL_vector_ready.json"),
        Path("logs/compliance_results/kredi_karti_sozlesmesi_compliance.json"),
    ),
]

COMBINED_PATH = Path("logs/compliance_results/combined_compliance.json")


def print_section(title: str) -> None:
    """Print a section header."""
    print("=" * 60)
    print(f"[{title}]")
    print("=" * 60)


def launch_streamlit() -> None:
    """Launch Streamlit viewer."""
    print("\n[streamlit] Launching Streamlit viewer...")
    print("[streamlit] Streamlit will open automatically in your browser.")
    print("[streamlit] Press Ctrl+C to stop Streamlit when done.")
    print("=" * 60)
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "src/streamlit_compliance_viewer.py", "--server.port", "8501"])
    except KeyboardInterrupt:
        print("\n[streamlit] Streamlit stopped by user.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end compliance runner")
    parser.add_argument("--skip-preparation", action="store_true", help="Skip data preparation step")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip Tebliğ ingest step")
    parser.add_argument("--force-ingest", action="store_true", help="Force Tebliğ re-ingest")
    parser.add_argument("--no-streamlit", action="store_true", help="Skip Streamlit launch (by default Streamlit opens automatically)")
    parser.add_argument("--only-streamlit", action="store_true", help="Only launch Streamlit viewer (skip all processing)")
    parser.add_argument("--readme", action="store_true", help="Generate README.md with Streamlit screenshot and metrics (standalone)")
    parser.add_argument("--with-readme", action="store_true", help="Auto-generate README after Streamlit closes (requires screenshot)")
    parser.add_argument("--model", default="gemma3:27b", help="LLM model name")
    parser.add_argument("--temperature", type=float, default=0.08, help="LLM temperature")
    parser.add_argument("--min-sim", type=float, default=0.30, help="Similarity threshold for NA")
    parser.add_argument("--top-n", type=int, default=8, help="Top-N Tebliğ matches to fetch per chunk")
    parser.add_argument("--max-tokens", type=int, default=500, help="LLM max tokens")
    parser.add_argument("--limit", type=int, default=None, help="Optional chunk limit per document (for quick tests)")
    return parser.parse_args()


def run_preparation() -> None:
    """Run all data preparation scripts."""
    print("=" * 60)
    print("[preparation] Running data preparation scripts...")
    print("=" * 60)
    
    scripts = [
        ("Tebliğ data", "src/teblig_data_preparation.py"),
        ("PDF bank documents", "src/banka_dokumanlari_preparation_pdf.py"),
        ("Word bank documents", "src/banka_dokumanlari_preparation_word.py"),
        ("Excel fee schedules", "src/banka_dokumanlari_preparation_excel.py"),
    ]
    
    total_scripts = len(scripts)
    for idx, (name, script) in enumerate(scripts, 1):
        print(f"\n[{idx}/{total_scripts}] Processing {name}...")
        print(f"Running: {script}")
        import time
        start_time = time.time()
        result = subprocess.run([sys.executable, script], capture_output=False)
        elapsed = time.time() - start_time
        if result.returncode != 0:
            print(f"✗ Error in {script} (took {elapsed:.1f}s)")
            sys.exit(1)
        print(f"✓ Completed {name} in {elapsed:.1f}s")
    
    print("\n" + "=" * 60)
    print("[preparation] All preparation scripts completed")
    print("=" * 60)


def maybe_ingest_teblig(force: bool, skip: bool) -> None:
    if skip:
        print("[ingest] Skipping Tebliğ ingest as requested.")
        return
    if PERSIST_DIR.exists() and not force:
        print(f"[ingest] Chroma persist dir already exists: {PERSIST_DIR} (use --force-ingest to rebuild)")
        return
    
    print("=" * 60)
    print("[ingest] Ingesting Tebliğ chunks into Chroma...")
    print("[ingest] This may take 2-5 minutes...")
    print("=" * 60)
    
    import time
    start_time = time.time()
    
    ingest(
        json_path=TEBLIG_JSON,
        persist_dir=PERSIST_DIR,
        collection_name=COLLECTION,
        model_name=EMBED_MODEL,
        include_parents=False,
        batch_size=64,
    )
    
    elapsed = time.time() - start_time
    print(f"\n✓ Tebliğ ingest completed in {elapsed:.1f}s")


def run_compliance(args: argparse.Namespace) -> None:
    total_docs = len(BANK_DOCS)
    
    for idx, (src_json, out_path) in enumerate(BANK_DOCS, 1):
        print("\n" + "=" * 60)
        print(f"[{idx}/{total_docs}] Processing {src_json.name}")
        print("[compliance] Estimated time: ~45-60 seconds per item (Chunks are processed carefully)")
        print("=" * 60)
        
        import time
        start_time = time.time()
        
        process_document(
            result_file=src_json,
            output_file=out_path,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            limit=args.limit,
            live_rag=True,
            persist_dir=PERSIST_DIR,
            collection=COLLECTION,
            embedding_model=EMBED_MODEL,
            scope_match=True,
            min_sim=args.min_sim,
            top_n=args.top_n,
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ Completed {src_json.name} in {elapsed/60:.1f} minutes")


def combine_outputs() -> None:
    print("[combine] Merging all compliance JSON files...")
    combine_results_from_dir(Path("logs/compliance_results"), COMBINED_PATH)


def run_evaluation() -> None:
    """Run evaluation against ground truth if available."""
    gt_path = Path("data/ground_truth.json")
    pred_path = COMBINED_PATH
    
    if not gt_path.exists():
        print("[evaluation] Ground truth not found, skipping evaluation.")
        return
    
    if not pred_path.exists():
        print("[evaluation] Combined predictions not found, skipping evaluation.")
        return
    
    print_section("evaluation: Evaluating model against ground truth...")
    
    # Load data
    gt_data = load_json(gt_path)
    pred_data = load_json(pred_path)
    
    # Create lookup
    gt_dict = {item["chunk_id"]: item for item in gt_data}
    pred_dict = {item["chunk_id"]: item for item in pred_data}
    
    # Match samples
    common_ids = set(gt_dict.keys()) & set(pred_dict.keys())
    
    if not common_ids:
        print("[evaluation] No common chunk IDs found between ground truth and predictions.")
        return
    
    print(f"[evaluation] Found {len(common_ids)} chunks with ground truth labels.")
    
    # Extract labels
    y_true = [gt_dict[cid]["ground_truth_label"] for cid in common_ids]
    y_pred = [pred_dict[cid]["status"] for cid in common_ids]
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)
    errors = analyze_errors([gt_dict[cid] for cid in common_ids], pred_dict)
    
    # Compute reasoning semantic score
    reason_score = 0.0
    try:
        # Prepare lists for reasoning check (needs list of dicts)
        pred_list_for_reason = [pred_dict[cid] for cid in common_ids]
        gt_list_for_reason = [gt_dict[cid] for cid in common_ids]
        reason_score = compute_reasoning_quality(pred_list_for_reason, gt_list_for_reason)
    except Exception as e:
        print(f"⚠️ Reasoning score calculation failed: {e}")
    
    # Print results
    print("\n" + "=" * 60)
    print(f"Overall Accuracy: {metrics['accuracy']:.2%}")
    if reason_score > 0:
        print(f"Reasoning Quality: {reason_score:.2%} (Semantic Similarity)")
    print("=" * 60)
    
    print("\nPer-Class Metrics:")
    print("-" * 60)
    for label in ["OK", "NOT_OK", "NA"]:
        if label in metrics:
            m = metrics[label]
            print(f"\n{label}:")
            print(f"  Precision: {m['precision']:.2%}")
            print(f"  Recall:    {m['recall']:.2%}")
            print(f"  F1-Score:  {m['f1']:.2%}")
    
    print("\n" + "=" * 60)
    print("Confusion Matrix:")
    print("=" * 60)
    confusion = metrics.get("confusion_matrix", {})
    print("\n         Predicted →")
    print("      OK  NOT_OK   NA")
    for true_label in ["OK", "NOT_OK", "NA"]:
        row = confusion.get(true_label, {})
        print(f"{true_label:>6} {row.get('OK', 0):>3}  {row.get('NOT_OK', 0):>6}  {row.get('NA', 0):>3}")
    
    # Save results
    output_path = Path("logs/evaluation_results.json")
    results = {
        "metrics": metrics,
        "reasoning_score": reason_score,
        "confusion_matrix": confusion,
        "matched_samples": len(common_ids),
        "error_count": len(errors),
        "errors": errors[:50] if errors else []  # Save first 50 errors for review
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        import json
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Evaluation results saved to: {output_path}")
    
    # Generate PDF Report
    if generate_pdf_report:
        pdf_path = Path("logs/compliance_report.pdf")
        try:
            generate_pdf_report(results, pdf_path)
        except Exception as e:
            print(f"⚠️ Failed to generate PDF report: {e}")
            
    print("=" * 60)


def _print_readme_section() -> None:
    """Print README generation section header."""
    print("\n" + "=" * 60)
    print("[readme] Generating comprehensive README.md...")
    print("=" * 60)


def _generate_readme() -> None:
    """Import and run README generator."""
    from src.generate_readme import main as generate_readme_main
    generate_readme_main()


def main() -> None:
    args = parse_args()

    # If README generation requested, run it and exit
    if args.readme:
        _print_readme_section()
        _generate_readme()
        return

    # If only-streamlit flag is set, skip everything and launch Streamlit
    if args.only_streamlit:
        print("\n" + "=" * 60)
        print("[only-streamlit] Launching Streamlit with existing results...")
        print("=" * 60)
        launch_streamlit()
        return

    import time
    pipeline_start = time.time()

    print("\n" + "#" * 60)
    print("#" + " " * 58 + "#")
    print("#" + "  BANKA UYUM ANALİZİ PİPELİNE".center(58) + "#")
    print("#" + " " * 58 + "#")
    print("#" * 60)
    print(f"\nModel: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Min Similarity: {args.min_sim}")
    print(f"Chunk Limit: {args.limit or 'None (all chunks)'}")
    print("\n" + "#" * 60)

    # Step 1: Data preparation (skip if requested)
    if not args.skip_preparation:
        run_preparation()
    else:
        print("[preparation] Skipping data preparation as requested.")

    # Step 2: Chroma ingest
    maybe_ingest_teblig(force=args.force_ingest, skip=args.skip_ingest)

    # Step 3: Compliance check
    run_compliance(args)

    # Step 4: Combine results
    combine_outputs()

    # Step 5: Evaluate against ground truth
    run_evaluation()
    
    # Ingest final results for chatbot
    print(f"\n{'='*60}")
    print("[chatbot: Ingesting insights...]")
    print(f"{'='*60}")
    
    from src.chroma_tool import ingest_results
    from src.config_loader import config
    
    ingest_results(
        json_path=Path("logs/compliance_results/combined_compliance.json"),
        persist_dir=Path(config["paths"]["chroma_persist_dir"]),
        collection_name="compliance_insights"
    )

    # Calculate total elapsed time
    total_elapsed = time.time() - pipeline_start

    print_section("✓ Pipeline completed successfully!")
    print(f"\nTotal pipeline time: {total_elapsed/60:.1f} minutes")
    print("=" * 60)

    # Step 6: Launch Streamlit (default: auto-launch)
    if not args.no_streamlit:
        launch_streamlit()  # Blocking - kullanıcı Ctrl+C ile kapatana kadar bekler

        # Step 7: Streamlit kapandıktan sonra README oluştur (sadece --with-readme flag'i varsa)
        if args.with_readme:
            _print_readme_section()
            _generate_readme()
        else:
            print("\n[readme] Skipped. README oluşturmak için: python main.py --readme")
    else:
        print("\n[streamlit] Skipped (--no-streamlit flag used)")
        print("To view results manually, run:")
        print("  streamlit run src/streamlit_compliance_viewer.py --server.port 8501")
    print()

if __name__ == "__main__":
    main()
