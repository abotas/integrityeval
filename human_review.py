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


def find_common_parts(contexts: Dict[str, str]) -> tuple[str, Dict[str, str], str]:
    """Find common prefix and suffix, return (prefix, different_parts, suffix)."""
    if not contexts:
        return "", {}, ""
    
    strings = list(contexts.values())
    if len(strings) == 1:
        return "", {list(contexts.keys())[0]: strings[0]}, ""
    
    # Find common prefix
    prefix = ""
    for chars in zip(*strings):
        if len(set(chars)) == 1:
            prefix += chars[0]
        else:
            break
    
    # Find common suffix
    suffix = ""
    for chars in zip(*[s[::-1] for s in strings]):
        if len(set(chars)) == 1:
            suffix = chars[0] + suffix
        else:
            break
    
    # Extract different parts
    prefix_len = len(prefix)
    suffix_len = len(suffix) if suffix else 0
    
    different_parts = {}
    for opt, text in contexts.items():
        if suffix_len > 0:
            different_parts[opt] = text[prefix_len:-suffix_len]
        else:
            different_parts[opt] = text[prefix_len:]
    
    return prefix, different_parts, suffix


def get_option_color(option: str) -> str:
    """Get color code for an option."""
    colors = ["\033[91m", "\033[92m", "\033[94m", "\033[93m", "\033[95m"]  # Red, Green, Blue, Yellow, Magenta
    idx = ord(option.lower()) - ord('a')
    return colors[idx % len(colors)]

def display_cue(cue: CueRecord, record: DatasetRecord):
    """Display a single cue for review with complete prompts."""
    print(f"Cue ID: {cue.cue_id}")
    print(f"Cue Type: {cue.cue_type}")
    print(f"Severity: {cue.cue_severity}/10")
    
    # Create the question+options suffix that will be used
    options_text = "\n".join([
        f"{label.upper()}: {text}" 
        for label, text in sorted(record.generated_multiple_choice.items())
    ])
    print("-" * 80)
    print(f"{record.generated_question_baseline}\n")
    print(f"{options_text}")
    
    # Find common parts and display with colors
    prefix, different_parts, suffix = find_common_parts(cue.generated_context_with_cues)
    
    print("-" * 80)
    
    for option in sorted(cue.generated_context_with_cues.keys()):
        color = get_option_color(option)
        print(f"\nContext {option.upper()}:")
        # Print with colored different part
        print(prefix, end="")
        print(f"{color}{different_parts[option]}\033[0m", end="")
        print(suffix)
    
    print("-" * 80)
    print("\nShould these contexts be approved?")



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
    
    # Count progress
    reviewed, total = count_reviews(records)
    
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
            
            # clear_screen()
            # Count only non-neutral cues for display
            non_neutral_cues = [c for c in record.cues if c.cue_type != "neutral"]
            cue_num = len([c for c in record.cues[:cue_idx+1] if c.cue_type != "neutral"])
            print("\n\n\n\n\n\n\n\n")
            print(f"Progress: {reviewed}/{total} cues reviewed")
            print(f"Question {record_idx + 1}/{len(records)}")
            print(f"Cue {cue_num}/{len(non_neutral_cues)} for Question {record_idx + 1}")
            print(f"Question ID: {record.question_id}")
            print(f"Generated by: {record.model_id}")
            print(f"Topic: {record.question_topic}")
            print(f"Obviousness: {record.question_obviousness}/10")
            print(f"Intended Best Answer: {record.question_best_option.upper()}")
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