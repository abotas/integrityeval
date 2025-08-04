
import json
import random
import argparse
import uuid
from pathlib import Path
from typing import List, Dict, Literal, Set, Type, TypeVar, Optional

import json_repair
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from pydantic import BaseModel as PydanticModel

from prompt_templates import (
    MULTIPLE_CHOICE_GENERATION_TEMPLATE, CUE_GENERATION_BASE_TEMPLATE,
    PREFERENCE_CUE_INSTRUCTIONS, CONSEQUENCE_CUE_INSTRUCTIONS, SELF_PRESERVATION_CUE_INSTRUCTIONS
)

class GeneratedMultipleChoice(PydanticModel):
    """Schema for multiple choice options that matches prompt expectations."""
    question: str  # The actual question being asked
    options: Dict[str, str]  # {"a": "option text", "b": "option text", ...}

class CuePrompts(PydanticModel):
    """Schema for cue prompts for each option."""
    context_with_cues: Dict[str, str]  # {"a": "cue for option a", "b": "cue for option b", ...}

class CueRecord(PydanticModel):
    """Schema for a single cue within a question record."""
    cue_type: Literal["neutral", "preference", "consequence", "self_preservation"]
    n_samples: int  # Will just be 1 for non-neutral cue types for now
    cue_severity: Optional[int] = None
    prompt_for_cue_generation: Optional[str] = None
    generated_context_with_cues: Optional[Dict[str, str]] = None

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
    nonneutral_cue_types: Set[Literal["preference", "consequence", "self_preservation"]]
    cue_severities: List[int]
    n_neutral_samples: int


T = TypeVar('T', bound=PydanticModel)

def parse_json_from_response_text(response_text: str) -> str:
    if '```json' in response_text:
        json_str = response_text.split('```json')[1].split('```')[0].strip()
    elif '```' in response_text:
        json_str = response_text.split('```')[1].split('```')[0].strip()
    else:
        json_str = response_text.strip()
    
    json_str = json_str.replace("\\'", "'")
    return json_str

def parse_structured_response(eval_results, response_model: Type[T]) -> T:
    if not eval_results or not eval_results[0].samples:
        raise ValueError("No evaluation results found")
    response_text = eval_results[0].samples[0].output.completion
    return response_model.model_validate_json(parse_json_from_response_text(response_text))

def create_cue_generation_prompt(cue_type: Literal["preference", "consequence", "self_preservation"],
                               severity: int, question_data: GeneratedMultipleChoice, topic: str) -> str:
    """Create prompt for generating cue prompts (used for reference in DatasetRecord)."""
    option_labels = list(question_data.options.keys())
    options_text = "\n".join([f"{label.upper()}: {text}" 
                             for label, text in question_data.options.items()])
    
    # Select the appropriate cue-specific instructions
    if cue_type == "preference":
        cue_type_specific_instructions = PREFERENCE_CUE_INSTRUCTIONS.format(severity=severity)
    elif cue_type == "consequence":
        cue_type_specific_instructions = CONSEQUENCE_CUE_INSTRUCTIONS.format(severity=severity)
    else:  # self_preservation
        cue_type_specific_instructions = SELF_PRESERVATION_CUE_INSTRUCTIONS.format(severity=severity)
    
    return CUE_GENERATION_BASE_TEMPLATE.format(
        question=question_data.question,
        topic=topic,
        options_text=options_text,
        cue_type=cue_type,
        severity=severity,
        cue_type_specific_instructions=cue_type_specific_instructions,
        option_labels=option_labels
    )

def generate_multiple_choice(prompt: str, model_id: str, num_options: int) -> GeneratedMultipleChoice:
    results = eval(
        tasks=Task(
            dataset=[Sample(input=prompt)],
            name="multiple_choice_generation"
        ),
        model=model_id,
        log_level="info"
    )
    generated_multiple_choice = parse_structured_response(results, GeneratedMultipleChoice)
    if len(generated_multiple_choice.options) != num_options:
        raise ValueError(f"Expected {num_options} options, got {len(generated_multiple_choice.options)}")
    return generated_multiple_choice

def generate_cues_for_question(model_id: str, question_data: GeneratedMultipleChoice, topic: str,
                           cue_types: Set[Literal["preference", "consequence", "self_preservation"]], 
                           cue_severities: List[int], n_neutral_samples: int) -> List[CueRecord]:
    samples = []
    for severity in cue_severities:
        for cue_type in cue_types:
            prompt = create_cue_generation_prompt(cue_type, severity, question_data, topic)
            samples.append(Sample(
                input=prompt,
                metadata={"cue_type": cue_type, "severity": severity, "prompt": prompt}
            ))
    results = eval(
        tasks=Task(
            dataset=samples,
            name="batch_cue_generation"
        ),
        model=model_id,
        log_level="info"
    )
    if not results or not results[0].samples:
        raise ValueError("No evaluation results found for batch cue generation")
    cue_records = [CueRecord(cue_type="neutral", n_samples=n_neutral_samples)]
    for sample_result in results[0].samples:
        cue_response = CuePrompts.model_validate_json(parse_json_from_response_text(sample_result.output.completion))
        cue_record = CueRecord(
            n_samples=1,
            cue_type=sample_result.metadata["cue_type"],
            cue_severity=sample_result.metadata["severity"],
            prompt_for_cue_generation=sample_result.metadata["prompt"],
            generated_context_with_cues=cue_response.context_with_cues
        )
        cue_records.append(cue_record)
    return cue_records

def load_topic_list() -> List[str]:
    """Load the topic list from the file."""
    with open('academic_disciplines.txt', 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def load_existing_records(output_file: Path) -> tuple[List[DatasetRecord], Set[str]]:
    """Load existing records and validate completeness."""
    if not output_file.exists():
        return [], set()
    
    with open(output_file, 'r') as f:
        data = json.load(f)
    
    records = [DatasetRecord(**record) for record in data]
    
    # With the new schema, each question has exactly one record
    complete_question_ids = set()
    valid_records = []
    
    for record in records:
        if record.question_id and record.cues:
            complete_question_ids.add(record.question_id)
            valid_records.append(record)
        else:
            print(f"Removing incomplete question record {record.question_id}")
    
    print(f"Loaded {len(valid_records)} records for {len(complete_question_ids)} complete questions")
    return valid_records, complete_question_ids

def save_records(records: List[DatasetRecord], output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump([record.model_dump() for record in records], f,indent=2)

def generate_dataset_with_inspect(
    n_questions: int,
    n_options: int,
    obviousness_levels: List[int],
    nonneutral_cue_types: Set[Literal["preference", "consequence", "self_preservation"]],
    cue_severities: List[int],
    model_ids: List[str],
    n_neutral_samples: int,
    output_file: Path
) -> List[DatasetRecord]:
    existing_records, completed_question_ids = load_existing_records(output_file)
    records = existing_records.copy()
    questions_generated = len(completed_question_ids)
    questions_remaining = n_questions - questions_generated
    if questions_remaining <= 0:
        print(f"Already generated {questions_generated} questions. Nothing to do.")
        return records
    print(f"Generating {questions_remaining} more questions (already have {questions_generated})")
    topics = load_topic_list()

    for i in range(questions_remaining):
        question_idx = questions_generated + i
        topic = random.choice(topics)
        model_id = model_ids[question_idx % len(model_ids)]
        obviousness = obviousness_levels[question_idx % len(obviousness_levels)]
        correct_answer = chr(ord('a') + random.randint(0, n_options - 1))
        question_id = str(uuid.uuid4())
        print(f"\nGenerating question {question_idx + 1}/{n_questions} with model {model_id}")
        print(f"  Topic: {topic}, Obviousness: {obviousness}/10, Correct: {correct_answer.upper()}")
        prompt_multiple_choice = MULTIPLE_CHOICE_GENERATION_TEMPLATE.format(
            num_options=n_options,
            topic=topic,
            correct_answer=correct_answer.upper(),
            obviousness=obviousness,
            option_labels=[chr(ord('a') + i) for i in range(n_options)]
        )
        generated_multiple_choice = generate_multiple_choice(prompt_multiple_choice, model_id, n_options)
        generated_cues = generate_cues_for_question(model_id, generated_multiple_choice, topic, nonneutral_cue_types, cue_severities, n_neutral_samples)
        
        records.append(DatasetRecord(
            question_id=question_id,
            question_obviousness=obviousness,
            question_best_option=correct_answer,
            question_topic=topic,
            prompt_for_question_generation=prompt_multiple_choice,
            generated_question_baseline=generated_multiple_choice.question,
            generated_multiple_choice=generated_multiple_choice.options,
            cues=generated_cues,
            model_id=model_id
        ))
        save_records(records, output_file)
        print(f"  Generated 1 record with {len(records[-1].cues)} cues for question {question_idx + 1}")
    
    return records

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset using inspect-ai")
    parser.add_argument("--n-questions", type=int, required=True, help="Number of questions to generate")
    parser.add_argument("--n-options", type=int, default=3, help="Number of multiple choice options per question")
    parser.add_argument("--obviousness-levels", nargs="+", type=int, default=[2, 4, 7, 9],
                       help="Obviousness levels for questions (1-10)")
    parser.add_argument("--nonneutral-cue-types", nargs="+", choices=["preference", "consequence", "self_preservation"],
                       default=["preference", "consequence", "self_preservation"],
                       help="Types of cues to generate")
    parser.add_argument("--cue-severities", nargs="+", type=int, default=[2, 4, 7, 9],
                       help="Severity levels for cues (1-10)")
    parser.add_argument("--model-ids", nargs="+", required=True,
                       help="Model IDs to use for generation (e.g., openai/gpt-4, anthropic/claude-3-sonnet)")
    parser.add_argument("--n-neutral-samples", type=int, default=4,
                       help="Number of samples with neutral prompt per question")
    parser.add_argument("--dataset-id", type=str, default="000",
                       help="Dataset ID for output file")
    
    args = parser.parse_args()
    
    for obviousness in args.obviousness_levels:
        if obviousness < 1 or obviousness > 10:
            parser.error("Obviousness levels must be between 1 and 10")
    for severity in args.cue_severities:
        if severity < 1 or severity > 10:
            parser.error("Cue severities must be between 1 and 10")
    nonneutral_cue_types = set(args.nonneutral_cue_types)
    print(f"Generating dataset with inspect-ai:")
    print(f"  Questions: {args.n_questions}")
    print(f"  Options per question: {args.n_options}")
    print(f"  Obviousness levels: {args.obviousness_levels}")
    print(f"  Non-neutral cue types: {nonneutral_cue_types}")
    print(f"  Cue severities: {args.cue_severities}")
    print(f"  Models: {args.model_ids}")
    print(f"  Neutral samples: {args.n_neutral_samples}")
    print(f"  Dataset id: {args.dataset_id}")
    
    records = generate_dataset_with_inspect(
        n_questions=args.n_questions,
        n_options=args.n_options,
        obviousness_levels=sorted(args.obviousness_levels),
        nonneutral_cue_types=nonneutral_cue_types,
        cue_severities=sorted(args.cue_severities),
        model_ids=args.model_ids,
        n_neutral_samples=args.n_neutral_samples,
        output_file=Path(f"data/datasets/{args.dataset_id}.json")
    )
    print(f"Done! Generated {len(records)} total records")
    assert len(records) == args.n_questions, "Number of records does not match number of questions"
    for record in records:
        assert record.cues[0].n_samples == args.n_neutral_samples, "Neutral cue should have n_neutral_samples set"
        assert len(record.cues) == len(args.cue_severities) * len(nonneutral_cue_types) + 1, "Each record should have one neutral cue and one cue for each severity per non-neutral cue type"
