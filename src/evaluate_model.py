"""
Model Evaluation Script - Ground Truth Comparison

Compares LLM predictions against ground truth annotations to compute:
- Accuracy
- Precision, Recall, F1 per class (OK, NOT_OK, NA)
- Confusion matrix
- Detailed error analysis

Usage:
    python evaluate_model.py --predictions logs/compliance_results/combined_compliance.json --ground-truth data/ground_truth_samples.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import numpy as np


def load_json(path: Path) -> List[Dict[str, Any]]:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def print_header(title: str, char: str = "=") -> None:
    """Print formatted header."""
    separator = char * 60
    print(f"\n{separator}")
    print(title)
    print(separator)


def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """
    Compute classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with metrics
    """
    labels = ["OK", "NOT_OK", "NA"]
    
    # Confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))
    for true, pred in zip(y_true, y_pred):
        confusion[true][pred] += 1
    
    # Overall accuracy
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / len(y_true) if y_true else 0
    
    # Per-class metrics
    metrics_per_class = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": y_true.count(label)
        }
    
    # Binary Metrics (Compliance vs Violation)
    # Map OK and NA to "COMPLIANT", NOT_OK to "VIOLATION"
    binary_y_true = ["VIOLATION" if y == "NOT_OK" else "COMPLIANT" for y in y_true]
    binary_y_pred = ["VIOLATION" if y == "NOT_OK" else "COMPLIANT" for y in y_pred]
    
    binary_correct = sum(t == p for t, p in zip(binary_y_true, binary_y_pred))
    binary_accuracy = binary_correct / len(binary_y_true) if binary_y_true else 0
    
    # Binary Precision/Recall for VIOLATION class (which is the critical class)
    # This should match the NOT_OK metrics above but recalculated strictly on the binary map
    # to ensure consistency.
    tp_bin = sum(1 for t, p in zip(binary_y_true, binary_y_pred) if t == "VIOLATION" and p == "VIOLATION")
    fp_bin = sum(1 for t, p in zip(binary_y_true, binary_y_pred) if t == "COMPLIANT" and p == "VIOLATION")
    fn_bin = sum(1 for t, p in zip(binary_y_true, binary_y_pred) if t == "VIOLATION" and p == "COMPLIANT")
    
    binary_precision = tp_bin / (tp_bin + fp_bin) if (tp_bin + fp_bin) > 0 else 0
    binary_recall = tp_bin / (tp_bin + fn_bin) if (tp_bin + fn_bin) > 0 else 0
    binary_f1 = 2 * binary_precision * binary_recall / (binary_precision + binary_recall) if (binary_precision + binary_recall) > 0 else 0

    # Macro averages (Calculate only for classes present in Ground Truth)
    present_classes = [label for label, m in metrics_per_class.items() if m["support"] > 0]
    
    if present_classes:
        # Updated Macro Logic based on User Request:
        # Treat OK and NA as "COMPLIANT" (merged). Treat NOT_OK as "VIOLATION".
        # Macro F1 = Average(F1_VIOLATION, F1_COMPLIANT)
        
        # We already have VIOLATION metrics (binary_precision, etc.)
        # Now calculate COMPLIANT metrics
        tp_comp = sum(1 for t, p in zip(binary_y_true, binary_y_pred) if t == "COMPLIANT" and p == "COMPLIANT")
        fp_comp = sum(1 for t, p in zip(binary_y_true, binary_y_pred) if t == "VIOLATION" and p == "COMPLIANT")
        fn_comp = sum(1 for t, p in zip(binary_y_true, binary_y_pred) if t == "COMPLIANT" and p == "VIOLATION")
        
        comp_precision = tp_comp / (tp_comp + fp_comp) if (tp_comp + fp_comp) > 0 else 0
        comp_recall = tp_comp / (tp_comp + fn_comp) if (tp_comp + fn_comp) > 0 else 0
        comp_f1 = 2 * comp_precision * comp_recall / (comp_precision + comp_recall) if (comp_precision + comp_recall) > 0 else 0
        
        # New Binary Macro
        macro_precision = (binary_precision + comp_precision) / 2
        macro_recall = (binary_recall + comp_recall) / 2
        macro_f1 = (binary_f1 + comp_f1) / 2
    else:
        macro_precision = 0
        macro_recall = 0
        macro_f1 = 0

    return {
        "accuracy": accuracy,
        "binary_accuracy": binary_accuracy,
        "binary_metrics": {
            "precision": binary_precision,
            "recall": binary_recall,
            "f1": binary_f1
        },
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": metrics_per_class,
        "confusion_matrix": dict(confusion),
        "total_samples": len(y_true)
    }

def compute_reasoning_quality(
    predictions: List[Dict[str, Any]], 
    ground_truth: List[Dict[str, Any]],
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
) -> float:
    """
    Compute semantic similarity using LLM-as-a-Judge (Gemma 3).
    """
    # We are forcing gemma3:27b inside the function, so we should print that.
    actual_judge_model = "gemma3:27b"
    print(f"[eval] Using LLM-as-a-Judge ({actual_judge_model}) for reasoning check...")
    
    # Create lookup map
    gt_map = {item["chunk_id"]: item.get("ground_truth_reason", "") for item in ground_truth}
    
    scores = []
    
    import requests
    import os
    _OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    _CHAT_URL = f"{_OLLAMA_BASE.rstrip('/')}/api/chat"

    for pred in predictions:
        chunk_id = pred.get("chunk_id")
        pred_reason = pred.get("reason", "") if pred.get("reason") else pred.get("llm_reasoning", "")
        gt_reason = gt_map.get(chunk_id, "")
        
        # Only compare if we have both
        if chunk_id and pred_reason and gt_reason:
            prompt = (
                f"Compare the following two explanations for a compliance check.\n"
                f"Ground Truth: \"{gt_reason}\"\n"
                f"Prediction: \"{pred_reason}\"\n\n"
                f"Rate the semantic similarity and factual alignment on a scale of 0 to 100.\n"
                f"0 = Completely different or contradictory.\n"
                f"100 = Same meaning and conclusion.\n"
                f"Output ONLY the integer score."
            )
            
            payload = {
                "model": "gemma3:27b",  # Force user requested model
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.0}
            }
            
            try:
                resp = requests.post(_CHAT_URL, json=payload, timeout=60)
                if resp.status_code == 200:
                    content = resp.json().get("message", {}).get("content", "").strip()
                    # Extract number
                    import re
                    match = re.search(r"\d+", content)
                    if match:
                        score = int(match.group())
                        scores.append(score)
            except Exception as e:
                print(f"Warning: reasoning eval failed for {chunk_id}: {e}")
            
    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"Average LLM-based Reasoning Score: {avg_score:.1f}/100")
        return avg_score
    return 0.0



def analyze_errors(ground_truth: List[Dict], predictions: Dict[str, Any]) -> List[Dict]:
    """Find and analyze prediction errors."""
    errors = []
    
    for gt in ground_truth:
        chunk_id = gt["chunk_id"]
        true_label = gt["ground_truth_label"]
        
        # Find prediction
        pred_item = predictions.get(chunk_id)
        if not pred_item:
            errors.append({
                "chunk_id": chunk_id,
                "error_type": "missing_prediction",
                "true_label": true_label,
                "predicted_label": None,
                "text": gt.get("text", "")[:100] + "..."
            })
            continue
        
        pred_label = pred_item.get("status")  # Changed from "compliance_status"
        
        if true_label != pred_label:
            errors.append({
                "chunk_id": chunk_id,
                "error_type": "misclassification",
                "true_label": true_label,
                "predicted_label": pred_label,
                "llm_reasoning": pred_item.get("llm_reasoning", ""),
                "ground_truth_reason": gt.get("ground_truth_reason", ""),
                "text": gt.get("text", "")[:100] + "..."
            })
    
    return errors


def main():
    parser = argparse.ArgumentParser(description="Evaluate compliance model predictions")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions JSON")
    parser.add_argument("--ground-truth", type=str, required=True, help="Path to ground truth JSON")
    parser.add_argument("--output", type=str, default="logs/evaluation_results.json", help="Output path")
    args = parser.parse_args()
    
    # Load data
    print(f"Loading ground truth from {args.ground_truth}...")
    ground_truth = load_json(Path(args.ground_truth))
    
    print(f"Loading predictions from {args.predictions}...")
    predictions_list = load_json(Path(args.predictions))
    
    # Convert predictions to dict for faster lookup
    predictions = {p["chunk_id"]: p for p in predictions_list}
    
    print(f"\nGround truth samples: {len(ground_truth)}")
    print(f"Predictions available: {len(predictions)}")
    
    # Match ground truth with predictions
    y_true = []
    y_pred = []
    matched_samples = []
    
    for gt in ground_truth:
        chunk_id = gt["chunk_id"]
        true_label = gt["ground_truth_label"]
        
        pred_item = predictions.get(chunk_id)
        if pred_item:
            pred_label = pred_item.get("status")  # Changed from "compliance_status"
            y_true.append(true_label)
            y_pred.append(pred_label)
            matched_samples.append({
                "chunk_id": chunk_id,
                "true": true_label,
                "pred": pred_label,
                "match": true_label == pred_label
            })
    
    print(f"Matched samples: {len(matched_samples)}")
    
    if not matched_samples:
        print("\n⚠️  No matching samples found between ground truth and predictions!")
        print("Make sure chunk IDs in ground truth match those in predictions.")
        return
    
    # Compute metrics
    print_header("EVALUATION RESULTS")
    
    metrics = compute_metrics(y_true, y_pred)
    
    print(f"\nCompliance Accuracy: {metrics['binary_accuracy']:.2%} (OK/NA Merged)")
    print(f"Strict Accuracy:     {metrics['accuracy']:.2%} (Exact Label Match)")
    
    if "binary_metrics" in metrics:
        bin_m = metrics["binary_metrics"]
        print(f"\nCompliance Metrics (Target: Violations):")
        print(f"  Precision: {bin_m['precision']:.2%}")
        print(f"  Recall:    {bin_m['recall']:.2%}")
        print(f"  F1 Score:  {bin_m['f1']:.2%}")

    # User Request: Hide Macro Metrics as they are misleading for skewed datasets (e.g. 1 sample)
    # print(f"\nMacro Metrics (Binary: Violation vs Compliant):")
    # print(f"  Precision:  {metrics['macro_precision']:.2%}")
    # print(f"  Recall:     {metrics['macro_recall']:.2%}")
    # print(f"  F1:         {metrics['macro_f1']:.2%}")
    
    print_header("Per-Class Metrics:", "-")
    for label, m in metrics["per_class"].items():
        print(f"\n{label}:")
        print(f"  Precision: {m['precision']:.2%}")
        print(f"  Recall:    {m['recall']:.2%}")
        print(f"  F1 Score:  {m['f1']:.2%}")
        print(f"  Support:   {m['support']}")
    
    print_header("Confusion Matrix:", "-")
    header = f"{'True/Pred':<12} {'OK':>10} {'NOT_OK':>10} {'NA':>10}"
    print(header)
    for true_label in ["OK", "NOT_OK", "NA"]:
        row = metrics["confusion_matrix"].get(true_label, {})
        print(f"{true_label:<12} {row.get('OK', 0):>10} {row.get('NOT_OK', 0):>10} {row.get('NA', 0):>10}")
    
    # Error analysis
    errors = analyze_errors(ground_truth, predictions)
    
    if errors:
        print("\n" + "-" * 60)
        print_header(f"Error Analysis ({len(errors)} errors):", "-")
        
        error_types = defaultdict(int)
        for err in errors:
            error_types[err["error_type"]] += 1
        
        for err_type, count in error_types.items():
            print(f"  {err_type}: {count}")
        
        print("\nSample Errors:")
        for i, err in enumerate(errors[:3], 1):
            print(f"\n  Error {i}:")
            print(f"    Chunk ID: {err['chunk_id']}")
            print(f"    True: {err['true_label']} → Predicted: {err.get('predicted_label', 'N/A')}")
            if err.get("ground_truth_reason"):
                print(f"    Expected: {err['ground_truth_reason'][:100]}...")
            if err.get("llm_reasoning"):
                print(f"    LLM Said: {err['llm_reasoning'][:100]}...")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        "metrics": metrics,
        "matched_samples": matched_samples,
        "errors": errors
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
