#!/usr/bin/env python3
"""
Human review script for dataset cues.
Allows manual approval/rejection of generated cues.
Supports resumable review sessions.
Creates a log of all reviewed cues with full context.
"""

import json
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any
import sys
from datetime import datetime

from generate_dataset import DatasetRecord, CueRecord
from prompt_templates import EVAL_SAMPLE_SUFFIX_TEMPLATE


def load_dataset(dataset_path: Path) -> List[DatasetRecord]:
    """Load dataset from JSON file."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    return [DatasetRecord(**record) for record in data]


def save_dataset(dataset_path: Path, records: List[DatasetRecord]):
    """Save dataset back to JSON file."""
    with open(dataset_path, 'w') as f:
        json.dump([record.model_dump() for record in records], f, indent=2)


def load_review_log(log_path: Path) -> List[Dict[str, Any]]:
    """Load existing review log if it exists."""
    if not log_path.exists():
        return []
    
    with open(log_path, 'r') as f:
        return json.load(f)


def save_review_log(log_path: Path, log_entries: List[Dict[str, Any]]):
    """Save review log to JSON file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(log_entries, f, indent=2)


def create_log_entry(record: DatasetRecord, cue: CueRecord, decision: bool) -> Dict[str, Any]:
    """Create a log entry for a reviewed cue."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question_id": record.question_id,
        "cue_id": cue.cue_id,
        "cue_type": cue.cue_type,
        "cue_severity": cue.cue_severity,
        "decision": "approved" if decision else "rejected",
        "question_baseline": record.generated_question_baseline,
        "multiple_choice": record.generated_multiple_choice,
        "question_topic": record.question_topic,
        "question_obviousness": record.question_obviousness,
        "question_best_option": record.question_best_option,
        "context_with_cues": cue.generated_context_with_cues
    }
    return log_entry


def clear_screen():
    """Clear terminal screen."""
    print("\033[2J\033[H", end="")


def display_question(record: DatasetRecord):
    """Display basic question metadata."""
    print("\n" + "="*80)
    print(f"Question ID: {record.question_id}")
    print(f"Topic: {record.question_topic}")
    print(f"Obviousness: {record.question_obviousness}/10")
    print(f"Correct Answer: {record.question_best_option.upper()}")
    print("="*80)


def display_cue(cue: CueRecord, record: DatasetRecord):
    """Display a single cue for review with complete prompts."""
    print(f"\nReviewing Cue ID: {cue.cue_id}")
    print(f"Cue Type: {cue.cue_type}")
    if cue.cue_severity is not None:
        print(f"Severity: {cue.cue_severity}/10")
    
    # Create the question+options suffix that will be used
    options_text = "\n".join([
        f"{label.upper()}: {text}" 
        for label, text in sorted(record.generated_multiple_choice.items())
    ])
    prompt_suffix = EVAL_SAMPLE_SUFFIX_TEMPLATE.format(
        question_baseline=record.generated_question_baseline,
        options_text=options_text
    )
    
    if cue.generated_context_with_cues:
        print("\n" + "="*80)
        print("This cue has different contexts to bias toward each option.")
        print("Review all variations below to decide if this cue should be approved.")
        print("="*80)
        
        for option, context in sorted(cue.generated_context_with_cues.items()):
            print(f"\n>>> Context when biasing toward OPTION {option.upper()} <<<")
            print("-" * 40)
            print(f"{context}")
            print(f"\n{prompt_suffix}")
            print("="*80)
        
        print("\nShould this cue (with all its option variations) be approved?")
    else:
        print(f"\n>>> NEUTRAL PROMPT (no biasing context) <<<")
        print("-" * 40)
        print(prompt_suffix)


def get_user_decision() -> Optional[bool]:
    """Get approval decision from user."""
    while True:
        print("\n[A]pprove / [R]eject / [S]kip / [Q]uit: ", end="")
        choice = input().strip().lower()
        
        if choice == 'a':
            return True
        elif choice == 'r':
            return False
        elif choice == 's':
            return None
        elif choice == 'q':
            print("\nExiting (progress saved)...")
            sys.exit(0)
        else:
            print("Invalid choice. Please enter A, R, S, or Q.")


def count_reviews(records: List[DatasetRecord]) -> tuple[int, int]:
    """Count reviewed and total reviewable cues."""
    reviewed = 0
    total = 0
    
    for record in records:
        for cue in record.cues:
            # Skip neutral cues
            if cue.cue_type == "neutral":
                continue
            total += 1
            if cue.human_approved is not None:
                reviewed += 1
    
    return reviewed, total


def main():
    parser = argparse.ArgumentParser(description="Human review for dataset cues")
    parser.add_argument(
        "--dataset-id",
        type=str,
        required=True,
        help="Dataset ID to review (e.g., 004)"
    )
    parser.add_argument(
        "--show-approved",
        action="store_true",
        help="Also show already approved/rejected cues for re-review"
    )
    
    args = parser.parse_args()
    
    dataset_path = Path(f"data/datasets/{args.dataset_id}.json")
    log_path = Path(f"data/review_logs/{args.dataset_id}_review_log.json")
    
    if not dataset_path.exists():
        print(f"Error: Dataset {dataset_path} not found")
        sys.exit(1)
    
    # Load dataset and review log
    records = load_dataset(dataset_path)
    review_log = load_review_log(log_path)
    print(f"Loaded {len(records)} questions from dataset {args.dataset_id}")
    if review_log:
        print(f"Found existing review log with {len(review_log)} entries")
    
    # Count progress
    reviewed, total = count_reviews(records)
    print(f"Progress: {reviewed}/{total} cues reviewed")
    
    if reviewed == total and not args.show_approved:
        print("All cues have been reviewed! Use --show-approved to re-review.")
        return
    
    # Review each question and its cues
    for record_idx, record in enumerate(records):
        has_unreviewed = any(
            cue.cue_type != "neutral" and (cue.human_approved is None or args.show_approved)
            for cue in record.cues
        )
        
        if not has_unreviewed:
            continue
        
        for cue_idx, cue in enumerate(record.cues):
            # Skip neutral cues
            if cue.cue_type == "neutral":
                continue
            
            # Skip already reviewed unless requested
            if cue.human_approved is not None and not args.show_approved:
                continue
            
            clear_screen()
            print(f"Question {record_idx + 1}/{len(records)}")
            display_question(record)
            
            # Count only non-neutral cues for display
            non_neutral_cues = [c for c in record.cues if c.cue_type != "neutral"]
            cue_num = len([c for c in record.cues[:cue_idx+1] if c.cue_type != "neutral"])
            
            print(f"\n{'='*80}")
            print(f"Cue {cue_num}/{len(non_neutral_cues)} for Question {record_idx + 1}")
            
            if cue.human_approved is not None:
                status = "APPROVED" if cue.human_approved else "REJECTED"
                print(f"Current status: {status}")
            
            display_cue(cue, record)
            
            decision = get_user_decision()
            
            if decision is not None:
                cue.human_approved = decision
                
                # Add to review log
                log_entry = create_log_entry(record, cue, decision)
                review_log.append(log_entry)
                
                # Save both dataset and log after each decision
                save_dataset(dataset_path, records)
                save_review_log(log_path, review_log)
                
                reviewed, total = count_reviews(records)
                print(f"\nSaved! Progress: {reviewed}/{total} cues reviewed")
    
    print("\n" + "="*80)
    print("Review complete!")
    reviewed, total = count_reviews(records)
    print(f"Final: {reviewed}/{total} cues reviewed")
    
    # Show summary
    approved = sum(
        1 for record in records
        for cue in record.cues
        if cue.cue_type != "neutral" and cue.human_approved is True
    )
    rejected = sum(
        1 for record in records
        for cue in record.cues
        if cue.cue_type != "neutral" and cue.human_approved is False
    )
    print(f"Approved: {approved}, Rejected: {rejected}, Skipped: {total - approved - rejected}")
    
    if review_log:
        print(f"\nReview log saved to: {log_path}")
        print(f"Log contains {len(review_log)} review decisions with full context")


if __name__ == "__main__":
    main()