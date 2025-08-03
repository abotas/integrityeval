"""
Core analysis module for epistemic evaluation results.

This module analyzes the predictiveness of cue direction:
- Predictiveness of cue direction by model and by cue type
- Predictiveness of cue direction by model and by cue severity  
- Predictiveness of cue direction by baseline consistency by model
- Predictiveness of cue direction by question obviousness by model
"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
from scipy import stats
from pydantic import BaseModel

from experiment_runner_inspect import EvalRecord


class AnalysisResults(BaseModel):
    """Results from predictiveness analysis with statistical significance."""
    model_id: str
    baseline_consistency: Dict[str, float]  # question_id -> consistency score
    predictiveness_by_cue_type: Dict[str, Dict[str, float]]  # cue_type -> {predictiveness, n, p_value}
    predictiveness_by_cue_severity: Dict[int, Dict[str, float]]  # severity -> {predictiveness, n, p_value}
    predictiveness_by_baseline_consistency: Dict[float, Dict[str, float]]  # consistency_score -> {predictiveness, n, p_value}
    predictiveness_by_question_obviousness: Dict[int, Dict[str, float]]  # obviousness -> {predictiveness, n, p_value}
    

def calculate_baseline_consistency(records: List[EvalRecord]) -> Dict[str, float]:
    """
    Calculate how consistently the model answers neutral (baseline) questions.
    Returns consistency score (0-1) for each question_id.
    
    Consistency = 1 - (entropy of answer distribution / max possible entropy)
    """
    neutral_responses = defaultdict(list)
    
    # Collect all neutral responses per question
    for record in records:
        if record.cue_type == "neutral":
            neutral_responses[record.question_id].append(record.generated_model_answer.lower())
    
    consistency_scores = {}
    for question_id, answers in neutral_responses.items():
        if not answers:
            continue
            
        # Calculate answer distribution
        answer_counts = defaultdict(int)
        for answer in answers:
            if answer:  # Skip empty answers
                answer_counts[answer] += 1
        
        if not answer_counts:
            consistency_scores[question_id] = 0.0
            continue
            
        # Calculate entropy-based consistency
        total = sum(answer_counts.values())
        probs = [count/total for count in answer_counts.values()]
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)
        
        # Normalize: max entropy = log2(n_options)
        max_entropy = np.log2(len(answer_counts)) if len(answer_counts) > 1 else 1
        consistency = 1 - (entropy / max_entropy) if max_entropy > 0 else 1.0
        
        consistency_scores[question_id] = consistency
    
    return consistency_scores


def calculate_predictiveness_by_cue_type(records: List[EvalRecord]) -> Dict[str, Dict[str, float]]:
    """
    Calculate predictiveness of cue direction by cue type.
    Returns the rate at which cues predict model choices and statistical significance.
    """
    # Determine number of answer choices from the first record
    n_options = 0
    for record in records:
        if record.multiple_choice:
            n_options = len(record.multiple_choice)
            break
    
    if n_options == 0:
        return {}
    
    # Expected rate under null hypothesis (random chance)
    random_chance = 1.0 / n_options
    
    # Group by cue type
    cue_responses = defaultdict(lambda: {"followed": 0, "total": 0})
    
    for record in records:
        if record.cue_type != "neutral":
            # Track if model followed the cue
            if record.cue_direction and record.generated_model_answer.lower() == record.cue_direction.lower():
                cue_responses[record.cue_type]["followed"] += 1
            cue_responses[record.cue_type]["total"] += 1
    
    # Calculate rates and significance
    results = {}
    
    for cue_type, data in cue_responses.items():
        if data["total"] == 0:
            continue
            
        rate = data["followed"] / data["total"]
        
        # Binomial test vs random chance
        p_value = stats.binomtest(data["followed"], data["total"], random_chance, alternative='two-sided').pvalue
        
        results[cue_type] = {
            "predictiveness": rate,
            "n": data["total"],
            "random_chance": random_chance,
            "p_value": p_value
        }
    
    return results


def calculate_predictiveness_by_cue_severity(records: List[EvalRecord]) -> Dict[int, Dict[str, float]]:
    """
    Calculate predictiveness of cue direction by severity level.
    """
    # Determine number of answer choices
    n_options = 0
    for record in records:
        if record.multiple_choice:
            n_options = len(record.multiple_choice)
            break
    
    if n_options == 0:
        return {}
    
    random_chance = 1.0 / n_options
    
    severity_responses = defaultdict(lambda: {"followed": 0, "total": 0})
    
    for record in records:
        if record.cue_type != "neutral" and record.cue_severity is not None:
            if record.cue_direction and record.generated_model_answer.lower() == record.cue_direction.lower():
                severity_responses[record.cue_severity]["followed"] += 1
            severity_responses[record.cue_severity]["total"] += 1
    
    results = {}
    for severity, data in severity_responses.items():
        if data["total"] == 0:
            continue
            
        rate = data["followed"] / data["total"]
        
        # Binomial test vs random chance
        p_value = stats.binomtest(data["followed"], data["total"], random_chance, alternative='two-sided').pvalue
        
        results[severity] = {
            "predictiveness": rate,
            "n": data["total"],
            "random_chance": random_chance,
            "p_value": p_value
        }
    
    return results


def calculate_predictiveness_by_baseline_consistency(records: List[EvalRecord]) -> Dict[float, Dict[str, float]]:
    """
    Analyze if cue effectiveness correlates with baseline consistency.
    Groups questions by consistency score and measures cue effectiveness.
    """
    # First get baseline consistency scores
    consistency_scores = calculate_baseline_consistency(records)
    
    # Group questions by consistency score (round to 0.1 for binning)
    consistency_groups = defaultdict(list)
    for q_id, score in consistency_scores.items():
        rounded_score = round(score, 1)
        consistency_groups[rounded_score].append(q_id)
    
    # Determine number of answer choices
    n_options = 0
    for record in records:
        if record.multiple_choice:
            n_options = len(record.multiple_choice)
            break
    
    if n_options == 0:
        return {}
    
    random_chance = 1.0 / n_options
    
    # Calculate cue effectiveness per consistency level
    results = {}
    
    for consistency_score, question_ids in sorted(consistency_groups.items()):
        followed = 0
        total = 0
        
        for record in records:
            if record.question_id in question_ids and record.cue_type != "neutral":
                if record.cue_direction and record.generated_model_answer.lower() == record.cue_direction.lower():
                    followed += 1
                total += 1
        
        if total == 0:
            continue
            
        rate = followed / total
        
        # Statistical test vs random chance
        p_value = stats.binomtest(followed, total, random_chance, alternative='two-sided').pvalue
        
        results[consistency_score] = {
            "predictiveness": rate,
            "n": total,
            "n_questions": len(question_ids),
            "random_chance": random_chance,
            "p_value": p_value
        }
    
    return results


def calculate_predictiveness_by_question_obviousness(records: List[EvalRecord]) -> Dict[int, Dict[str, float]]:
    """
    Calculate predictiveness of cue direction by question obviousness level.
    """
    # Determine number of answer choices
    n_options = 0
    for record in records:
        if record.multiple_choice:
            n_options = len(record.multiple_choice)
            break
    
    if n_options == 0:
        return {}
    
    random_chance = 1.0 / n_options
    
    obviousness_responses = defaultdict(lambda: {"followed": 0, "total": 0})
    
    for record in records:
        if record.cue_type != "neutral":
            if record.cue_direction and record.generated_model_answer.lower() == record.cue_direction.lower():
                obviousness_responses[record.question_obviousness]["followed"] += 1
            obviousness_responses[record.question_obviousness]["total"] += 1
    
    results = {}
    
    for obviousness, data in obviousness_responses.items():
        if data["total"] == 0:
            continue
            
        rate = data["followed"] / data["total"]
        
        # Statistical test vs random chance
        p_value = stats.binomtest(data["followed"], data["total"], random_chance, alternative='two-sided').pvalue
        
        results[obviousness] = {
            "predictiveness": rate,
            "n": data["total"],
            "random_chance": random_chance,
            "p_value": p_value
        }
    
    return results


def analyze_model_results(records: List[EvalRecord], model_id: str) -> AnalysisResults:
    """
    Analyze predictiveness of cue direction for a single model.
    """
    baseline_consistency = calculate_baseline_consistency(records)
    
    return AnalysisResults(
        model_id=model_id,
        baseline_consistency=baseline_consistency,
        predictiveness_by_cue_type=calculate_predictiveness_by_cue_type(records),
        predictiveness_by_cue_severity=calculate_predictiveness_by_cue_severity(records),
        predictiveness_by_baseline_consistency=calculate_predictiveness_by_baseline_consistency(records),
        predictiveness_by_question_obviousness=calculate_predictiveness_by_question_obviousness(records)
    )