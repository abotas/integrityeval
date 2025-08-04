"""
Shared schemas and templates for dataset generation.

This module contains all the Pydantic models and prompt templates used
across the dataset generation pipeline.
"""

from typing import Dict, List, Literal, Optional, Set
from pydantic import BaseModel as PydanticModel


# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================

class GeneratedMultipleChoice(PydanticModel):
    """Schema for multiple choice options that matches prompt expectations."""
    question: str  # The actual question being asked
    options: Dict[str, str]  # {"a": "option text", "b": "option text", ...}

class CuePrompts(PydanticModel):
    """Schema for cue prompts for each option."""
    questions_with_cues: Dict[str, str]  # {"a": "cue for option a", "b": "cue for option b", ...}

class CueRecord(PydanticModel):
    """Schema for a single cue within a question record."""
    cue_type: Literal["neutral", "preference", "consequence", "self_preservation"]
    n_samples: int  # Will just be 1 for non-neutral cue types for now
    cue_severity: Optional[int]
    prompt_for_cue_generation: Optional[str]
    generated_altered_questions_with_cues: Optional[Dict[str, str]]

class DatasetRecord(PydanticModel):
    """Schema for a dataset record - one per question."""
    question_id: str
    question_obviousness: int
    question_best_option: str
    question_topic: str
    prompt_for_question_generation: str
    generated_question_baseline: str  # The actual question text
    generated_multiple_choice: Dict[str, str]
    cues: List[CueRecord]  # List of all cue types/severities for this question
    model_id: str


class QuestionConfig(PydanticModel):
    """Configuration for generating a single question and its cues."""
    question_id: str
    topic: str
    obviousness: int
    correct_answer: str
    num_options: int
    model_id: str
    cue_types: Set[Literal["preference", "consequence", "self_preservation"]]
    cue_severities: List[int]
    n_neutral_samples: int
