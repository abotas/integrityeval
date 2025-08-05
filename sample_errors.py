"""
Sample and display model errors by cue_id.

Usage:
    python sample_errors.py --run-id 010
    python sample_errors.py --run-id 010 --seed 42
"""

import json
import random
import argparse
from pathlib import Path
from typing import List
from collections import defaultdict

from eval import EvalRecord


def load_run_results(run_dir: Path) -> List[EvalRecord]:
    """Load all model results from a run directory."""
    all_records = []
    
    for result_file in run_dir.glob("*.json"):
        with open(result_file, 'r') as f:
            data = json.load(f)
        records = [EvalRecord(**record) for record in data]
        all_records.extend(records)
    
    print(f"Loaded {len(all_records)} total records from {len(list(run_dir.glob('*.json')))} files")
    return all_records


def find_cue_with_errors(records: List[EvalRecord]) -> tuple[str, str]:
    """
    Find a model and cue_id where the model has at least one incorrect answer.
    Returns (model_id, cue_id).
    """
    # Group by model and cue_id
    model_cue_errors = defaultdict(lambda: defaultdict(list))
    
    for record in records:
        if record.cue_type != "neutral":  # Skip neutral cues
            is_incorrect = record.generated_model_answer.lower() != record.question_best_option.lower()
            model_cue_errors[record.model_id_answer_generator][record.cue_id].append(is_incorrect)
    
    # Find model/cue pairs with at least one error
    candidates = []
    for model_id, cue_dict in model_cue_errors.items():
        for cue_id, error_list in cue_dict.items():
            if any(error_list):  # Has at least one error
                candidates.append((model_id, cue_id))
    
    if not candidates:
        raise ValueError("No model/cue pairs with errors found")
    
    # Randomly select one
    return random.choice(candidates)


def display_cue_records(model_id: str, cue_id: str, all_records: List[EvalRecord]):
    """Display all records for a specific model and cue_id."""
    # Get all records for this model and cue
    cue_records = [
        r for r in all_records 
        if r.model_id_answer_generator == model_id and r.cue_id == cue_id
    ]
    
    if not cue_records:
        print("No records found")
        return
    
    # Use first record for common info
    first = cue_records[0]
    question_id = first.question_id
    
    # Get neutral cue performance for this model and question
    neutral_records = [
        r for r in all_records
        if r.model_id_answer_generator == model_id 
        and r.question_id == question_id 
        and r.cue_type == "neutral"
    ]
    
    print("\n" + "="*80)
    print("SAMPLED ERROR ANALYSIS")
    print("="*80)
    
    print(f"\nMODEL: {model_id}")
    print(f"QUESTION ID: {question_id}")
    print(f"CUE ID: {cue_id}")
    
    # Show baseline performance
    if neutral_records:
        neutral_correct = sum(1 for r in neutral_records 
                             if r.generated_model_answer.lower() == r.question_best_option.lower())
        print(f"\nBASELINE PERFORMANCE (neutral cue): {neutral_correct}/{len(neutral_records)} correct")
    else:
        print(f"\nBASELINE PERFORMANCE (neutral cue): No neutral records found")
    
    print(f"\nQUESTION: {first.question_baseline}")
    print(f"OBVIOUSNESS: {first.question_obviousness}")
    
    print("\nMULTIPLE CHOICE:")
    for option, text in first.multiple_choice.items():
        marker = " ← CORRECT" if option == first.question_best_option else ""
        print(f"  {option.upper()}: {text}{marker}")
    
    print(f"\nCUE TYPE: {first.cue_type}")
    print(f"CUE SEVERITY: {first.cue_severity}")
    print(f"CUE DIRECTION: {first.cue_direction}")
    
    # Summary of model performance with this cue
    correct_count = sum(1 for r in cue_records 
                       if r.generated_model_answer.lower() == r.question_best_option.lower())
    print(f"\nCUE PERFORMANCE: {correct_count}/{len(cue_records)} correct")
    
    print("\n" + "-"*80)
    print("INDIVIDUAL SAMPLES:")
    print("-"*80)
    
    # Display each record
    for i, record in enumerate(cue_records, 1):
        is_correct = record.generated_model_answer.lower() == record.question_best_option.lower()
        result_marker = "✓ CORRECT" if is_correct else "✗ INCORRECT"
        
        print(f"\n[Sample {i}/{len(cue_records)}] {result_marker}")
        
        if record.generated_context_with_cue:
            print(f"\nCONTEXT WITH CUE:")
            print(record.generated_context_with_cue)
        
        print(f"\nMODEL ANSWERED: {record.generated_model_answer.upper()}")
        
        print(f"\nMODEL EXPLANATION:")
        print(record.generated_model_explanation)
        
        if record.generated_ctrl_unfairness is not None:
            fairness = "UNFAIR" if record.generated_ctrl_unfairness else "FAIR"
            print(f"\nFAIRNESS ASSESSMENT: {fairness}")
            if record.generated_ctrl_unfairness_explanation:
                print(f"WHY: {record.generated_ctrl_unfairness_explanation}")
        
        if i < len(cue_records):
            print("\n" + "-"*40)
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Sample and display model errors by cue_id"
    )
    parser.add_argument(
        "--run-id", 
        type=str, 
        required=True,
        help="Run ID (e.g., 010)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible sampling"
    )
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
    
    # Load data
    run_dir = Path("data/eval_results") / args.run_id
    if not run_dir.exists():
        print(f"Error: Run directory {run_dir} does not exist")
        return
    
    print(f"Loading results from {run_dir}...")
    all_records = load_run_results(run_dir)
    
    # Find a model/cue pair with errors
    try:
        model_id, cue_id = find_cue_with_errors(all_records)
        print(f"\nSampled model: {model_id}")
        print(f"Sampled cue_id: {cue_id}")
        
        # Display all records for this model/cue
        display_cue_records(model_id, cue_id, all_records)
        
    except ValueError as e:
        print(f"\nError: {e}")
        return


if __name__ == "__main__":
    main()