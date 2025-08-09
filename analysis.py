"""
Analyze predictiveness of cue direction across models.

Usage:
    python run_analysis.py --run-dir run_001
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
from io import BytesIO
import base64

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from eval import EvalRecord
from analysis_utils import (
    analyze_model_results, AnalysisResults, get_cue_unfairness_votes,
    filter_records_by_model_unfairness_vote, 
    get_human_approved_cue_ids, filter_records_by_human_approval
)
from generate_dataset import DatasetRecord

# Constants
FIGURE_SIZE = (10, 6)
DPI = 150
MAJORITY_THRESHOLD = 0.60  # threshold for majority consensus filtering
P_VALUE_THRESHOLDS = {
    0.001: '***',
    0.01: '**', 
    0.05: '*'
}
CUE_TYPE_ORDER = ["preference", "consequence", "self_preservation"]
MODEL_ORDER = [
    "anthropic/claude-3-7-sonnet-20250219",
    "anthropic/claude-sonnet-4-20250514", 
    "anthropic/claude-opus-4-20250514",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "grok/grok-3-mini",
    "grok/grok-3",
    "grok/grok-4-0709",
    "openai/gpt-4o-2024-11-20",
    "openai/gpt-4.1-2025-04-14",
    "openai/o3-2025-04-16",
    "openai/gpt-5-2025-08-07",
]


# ============================================================================
# CORE LOGIC: Model ordering and data loading
# ============================================================================

def sort_model_list(models: List[str]) -> List[str]:
    """Sort models by predefined capability order."""
    position_map = {model: i for i, model in enumerate(MODEL_ORDER)}
    return sorted(models, key=lambda model: position_map[model])


def load_run_results(run_dir: Path) -> Dict[str, List[EvalRecord]]:
    """Load all model results from a run directory."""
    model_results = {}
    
    for result_file in run_dir.glob("*.json"):
        with open(result_file, 'r') as f:
            data = json.load(f)
        records = [EvalRecord(**record) for record in data]
        model_id = records[0].model_id_answer_generator
        model_results[model_id] = records
        print(f"Loaded {len(records)} records for {model_id}")

    return model_results


def load_dataset_records(run_id: str) -> List[DatasetRecord]:
    """Load dataset records to access human approval data."""
    dataset_path = Path(f"data/datasets/{run_id}.json")
    if not dataset_path.exists():
        print(f"Warning: Dataset file {dataset_path} not found. Human approval filtering will be skipped.")
        return []
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    return [DatasetRecord(**record) for record in data]


# ============================================================================
# VISUALIZATION: Plot generation
# ============================================================================

def _get_random(all_results: Dict[str, AnalysisResults]) -> float:
    """Get random value from first available result."""
    results = next(iter(all_results.values()))
    data = next(iter(results.predictiveness_by_cue_type.values()))
    return data['random']

def _get_significance_marker(p_value: float) -> str:
    """Return statistical significance marker."""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    return ''

def generate_predictiveness_plots(all_results: Dict[str, AnalysisResults]) -> Dict[str, str]:
    """Generate all plots as base64 strings."""
    plots = {}
    
    if not all_results:
        return plots
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Sort models by provider and capability
    sorted_models = sort_model_list(list(all_results.keys()))
    colors = sns.color_palette("husl", len(sorted_models))
    
    random = _get_random(all_results)
    plots['cue_type'] = _plot_cue_type(sorted_models, all_results, colors, random)
    plots['severity'] = _plot_severity(sorted_models, all_results, colors, random)
    plots['consistency'] = _plot_consistency(sorted_models, all_results, colors, random)
    plots['obviousness'] = _plot_obviousness(sorted_models, all_results, colors, random)
    return plots


def _plot_cue_type_combined(all_data: Dict[str, Dict[str, AnalysisResults]], 
                           sorted_models: List[str], colors: List, random: float) -> str:
    """Create combined bar plot for predictiveness by cue type across all data categories."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Define bar patterns for each category
    patterns = {
        'all': None,  # No pattern for all data
        'human_approved': '///',  # Diagonal hatching for human approved
        'model_filtered': '...'  # Dots for self-reviewed
    }
    
    # Define alphas for visual distinction
    alphas = {
        'all': 0.9,
        'human_approved': 0.7,
        'model_filtered': 0.7
    }
    
    bar_width = 0.25
    x_pos = 0
    x_labels = []
    x_positions = []
    
    for i, model in enumerate(sorted_models):
        model_name = model.split('/')[-1]
        is_first_cue_type_for_model = True
        
        for cue_type in CUE_TYPE_ORDER:
            # Check if this cue type has data in any category for this model
            has_data_anywhere = False
            for category_name, category_results in all_data.items():
                if model in category_results:
                    results = category_results[model]
                    if cue_type in results.predictiveness_by_cue_type:
                        has_data_anywhere = True
                        break
            
            # Skip this cue type if it has no data anywhere for this model
            if not has_data_anywhere:
                continue
            
            # Track if we have data for this specific model/cue combination
            has_data = False
            
            # Plot bars for each category side by side
            category_offset = 0
            for category_name, category_results in all_data.items():
                if model in category_results:
                    results = category_results[model]
                    if cue_type in results.predictiveness_by_cue_type:
                        has_data = True
                        data = results.predictiveness_by_cue_type[cue_type]
                        
                        # Create label only for first cue type of each model/category combo
                        label = None
                        if is_first_cue_type_for_model:
                            if category_name == 'all':
                                label = f"{model_name} (All)"
                            elif category_name == 'human_approved':
                                label = f"{model_name} (Human)"
                            elif category_name == 'model_filtered':
                                label = f"{model_name} (Self)"
                        
                        ax.bar(x_pos + category_offset, data["predictiveness"], 
                              width=bar_width, color=colors[i], 
                              alpha=alphas[category_name],
                              hatch=patterns[category_name],
                              label=label, edgecolor='black', linewidth=0.5)
                        
                        # Statistical significance marker
                        sig_marker = _get_significance_marker(data["p_value"])
                        if sig_marker:
                            ax.text(x_pos + category_offset, data["predictiveness"] + 0.02, 
                                   sig_marker, ha='center', fontsize=8)
                
                category_offset += bar_width
            
            if has_data:
                x_labels.append(cue_type)
                x_positions.append(x_pos + bar_width)  # Center position
                x_pos += bar_width * 3 + 0.1  # Space for 3 bars plus gap
                is_first_cue_type_for_model = False  # No longer the first cue type
        
        x_pos += 0.3  # Extra space between models
    
    ax.axhline(y=random, color='red', linestyle='--', alpha=0.7, 
              label=f'Random ({random:.2f})')
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Predictiveness of Cue Direction')
    ax.set_title('Predictiveness of Cue Direction by Model and Cue Type')
    ax.set_ylim(0, 1.1)
    
    # Create legend with better positioning
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def _plot_cue_type(sorted_models: List[str], all_results: Dict[str, AnalysisResults], 
                   colors: List, random: float) -> str:
    """Create bar plot for predictiveness by cue type."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    x_pos = 0
    x_labels = []
    x_positions = []
    
    for i, model in enumerate(sorted_models):
        results = all_results[model]
        model_name = model.split('/')[-1]
        
        for j, cue_type in enumerate(CUE_TYPE_ORDER):
            if cue_type in results.predictiveness_by_cue_type:
                data = results.predictiveness_by_cue_type[cue_type]
                
                ax.bar(x_pos, data["predictiveness"], color=colors[i], 
                       alpha=0.8, label=model_name if j == 0 else "")
                
                # Statistical significance marker
                sig_marker = _get_significance_marker(data["p_value"])
                if sig_marker:
                    ax.text(x_pos, data["predictiveness"] + 0.02, sig_marker, 
                           ha='center', fontsize=10)
                
                x_labels.append(cue_type)
                x_positions.append(x_pos)
                x_pos += 1
        
        x_pos += 0.5
    
    ax.axhline(y=random, color='red', linestyle='--', alpha=0.7, 
              label=f'Random ({random:.2f})')
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel('Predictiveness of Cue Direction')
    ax.set_title('Predictiveness of Cue Direction by Model and Cue Type')
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    return _fig_to_base64(fig)


def _plot_severity_combined(all_data: Dict[str, Dict[str, AnalysisResults]], 
                           sorted_models: List[str], colors: List, random: float) -> str:
    """Create combined line plot for predictiveness by cue severity across all data categories."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define line styles for each category
    line_styles = {
        'all': '-',  # Solid line for all data
        'human_approved': '--',  # Dashed for human approved
        'model_filtered': ':'  # Dotted for self-reviewed
    }
    
    # Define marker styles
    markers = {
        'all': 'o',
        'human_approved': 's',
        'model_filtered': '^'
    }
    
    for i, model in enumerate(sorted_models):
        model_name = model.split('/')[-1]
        
        for category_name, category_results in all_data.items():
            if model in category_results:
                results = category_results[model]
                
                severities = sorted(results.predictiveness_by_cue_severity.keys())
                if severities:  # Only plot if we have data
                    predictiveness = [results.predictiveness_by_cue_severity[s]["predictiveness"] 
                                     for s in severities]
                    
                    # Create label with category
                    if category_name == 'all':
                        label = f"{model_name} (All)"
                    elif category_name == 'human_approved':
                        label = f"{model_name} (Human)"
                    elif category_name == 'model_filtered':
                        label = f"{model_name} (Self)"
                    
                    ax.plot(severities, predictiveness, 
                           linestyle=line_styles[category_name],
                           marker=markers[category_name],
                           color=colors[i], 
                           label=label, markersize=6, linewidth=1.5, alpha=0.8)
    
    ax.axhline(y=random, color='red', linestyle='--', alpha=0.7, 
              label=f'Random ({random:.2f})')
    
    ax.set_xlabel('Cue Severity')
    ax.set_ylabel('Predictiveness of Cue Direction')
    ax.set_title('Predictiveness of Cue Direction by Model and Cue Severity')
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(range(1, 11))
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def _plot_severity(sorted_models: List[str], all_results: Dict[str, AnalysisResults], 
                   colors: List, random: float) -> str:
    """Create line plot for predictiveness by cue severity."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    for i, model in enumerate(sorted_models):
        results = all_results[model]
        model_name = model.split('/')[-1]
        
        severities = sorted(results.predictiveness_by_cue_severity.keys())
        predictiveness = [results.predictiveness_by_cue_severity[s]["predictiveness"] 
                         for s in severities]
        
        ax.plot(severities, predictiveness, 'o-', color=colors[i], 
               label=model_name, markersize=8, linewidth=2)
    
    ax.axhline(y=random, color='red', linestyle='--', alpha=0.7, 
              label=f'Random ({random:.2f})')
    
    ax.set_xlabel('Cue Severity')
    ax.set_ylabel('Predictiveness of Cue Direction')
    ax.set_title('Predictiveness of Cue Direction by Model and Cue Severity')
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(range(1, 11))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return _fig_to_base64(fig)


def _plot_consistency_combined(all_data: Dict[str, Dict[str, AnalysisResults]], 
                              sorted_models: List[str], colors: List, random: float) -> str:
    """Create combined scatter plot for predictiveness by baseline consistency."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define marker styles for each category
    markers = {
        'all': 'o',
        'human_approved': 's',
        'model_filtered': '^'
    }
    
    # Define sizes for visual distinction
    sizes = {
        'all': 80,
        'human_approved': 60,
        'model_filtered': 60
    }
    
    for i, model in enumerate(sorted_models):
        model_name = model.split('/')[-1]
        
        for category_name, category_results in all_data.items():
            if model in category_results:
                results = category_results[model]
                
                consistency_scores = sorted(results.predictiveness_by_baseline_consistency.keys())
                if consistency_scores:  # Only plot if we have data
                    predictiveness = [results.predictiveness_by_baseline_consistency[c]["predictiveness"] 
                                     for c in consistency_scores]
                    
                    # Create label with category
                    if category_name == 'all':
                        label = f"{model_name} (All)"
                    elif category_name == 'human_approved':
                        label = f"{model_name} (Human)"
                    elif category_name == 'model_filtered':
                        label = f"{model_name} (Self)"
                    
                    ax.scatter(consistency_scores, predictiveness, 
                              marker=markers[category_name],
                              s=sizes[category_name],
                              color=colors[i], 
                              label=label, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax.axhline(y=random, color='red', linestyle='--', alpha=0.7, 
              label=f'Random ({random:.2f})')
    
    ax.set_xlabel('Baseline Consistency Score')
    ax.set_ylabel('Predictiveness of Cue Direction')
    ax.set_title('Predictiveness of Cue Direction by Baseline Consistency')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1.1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def _plot_consistency(sorted_models: List[str], all_results: Dict[str, AnalysisResults], 
                      colors: List, random: float) -> str:
    """Create scatter plot for predictiveness by baseline consistency."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    for i, model in enumerate(sorted_models):
        results = all_results[model]
        model_name = model.split('/')[-1]
        
        consistency_scores = sorted(results.predictiveness_by_baseline_consistency.keys())
        predictiveness = [results.predictiveness_by_baseline_consistency[c]["predictiveness"] 
                         for c in consistency_scores]
        
        ax.scatter(consistency_scores, predictiveness, color=colors[i], 
                  label=model_name, s=100, alpha=0.7)
        
        # Add linear trend line (only if we have data)
        if len(consistency_scores) > 1:
            z = np.polyfit(consistency_scores, predictiveness, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(0, 1, 100)
            ax.plot(x_trend, p(x_trend), color=colors[i], alpha=0.5, linestyle='--')
    
    ax.axhline(y=random, color='red', linestyle='--', alpha=0.7, 
              label=f'Random ({random:.2f})')
    
    ax.set_xlabel('Baseline Consistency Score')
    ax.set_ylabel('Predictiveness of Cue Direction')
    ax.set_title('Predictiveness of Cue Direction by Baseline Consistency')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return _fig_to_base64(fig)


def _plot_obviousness_combined(all_data: Dict[str, Dict[str, AnalysisResults]], 
                              sorted_models: List[str], colors: List, random: float) -> str:
    """Create combined line plot for predictiveness by question obviousness."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define line styles for each category
    line_styles = {
        'all': '-',  # Solid line for all data
        'human_approved': '--',  # Dashed for human approved
        'model_filtered': ':'  # Dotted for self-reviewed
    }
    
    # Define marker styles
    markers = {
        'all': 'o',
        'human_approved': 's',
        'model_filtered': '^'
    }
    
    for i, model in enumerate(sorted_models):
        model_name = model.split('/')[-1]
        
        for category_name, category_results in all_data.items():
            if model in category_results:
                results = category_results[model]
                
                obviousness_levels = sorted(results.predictiveness_by_question_obviousness.keys())
                if obviousness_levels:  # Only plot if we have data
                    predictiveness = [results.predictiveness_by_question_obviousness[o]["predictiveness"] 
                                     for o in obviousness_levels]
                    
                    # Create label with category
                    if category_name == 'all':
                        label = f"{model_name} (All)"
                    elif category_name == 'human_approved':
                        label = f"{model_name} (Human)"
                    elif category_name == 'model_filtered':
                        label = f"{model_name} (Self)"
                    
                    ax.plot(obviousness_levels, predictiveness, 
                           linestyle=line_styles[category_name],
                           marker=markers[category_name],
                           color=colors[i], 
                           label=label, markersize=6, linewidth=1.5, alpha=0.8)
    
    ax.axhline(y=random, color='red', linestyle='--', alpha=0.7, 
              label=f'Random ({random:.2f})')
    
    ax.set_xlabel('Question Obviousness')
    ax.set_ylabel('Predictiveness of Cue Direction')
    ax.set_title('Predictiveness of Cue Direction by Question Obviousness')
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(range(1, 11))
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def _plot_obviousness(sorted_models: List[str], all_results: Dict[str, AnalysisResults], 
                      colors: List, random: float) -> str:
    """Create line plot for predictiveness by question obviousness."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    for i, model in enumerate(sorted_models):
        results = all_results[model]
        model_name = model.split('/')[-1]
        
        obviousness_levels = sorted(results.predictiveness_by_question_obviousness.keys())
        predictiveness = [results.predictiveness_by_question_obviousness[o]["predictiveness"] 
                         for o in obviousness_levels]
        
        ax.plot(obviousness_levels, predictiveness, 'o-', color=colors[i], 
               label=model_name, markersize=8, linewidth=2)
    
    ax.axhline(y=random, color='red', linestyle='--', alpha=0.7, 
              label=f'Random ({random:.2f})')
    
    ax.set_xlabel('Question Obviousness')
    ax.set_ylabel('Predictiveness of Cue Direction')
    ax.set_title('Predictiveness of Cue Direction by Question Obviousness')
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(range(1, 11))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return _fig_to_base64(fig)


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string for HTML embedding."""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=DPI, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"


def _generate_human_approved_section(plot_type: str, human_approved_results: Dict[str, AnalysisResults], 
                                    human_approved_plots: Dict[str, str], include_table: bool = False) -> str:
    """Generate HTML for human-approved section."""
    if not human_approved_results:
        return ""
    
    if not human_approved_plots or plot_type not in human_approved_plots:
        return f'''
                <div class="subset-section">
                    <h3>Human-Approved Cues Only</h3>
                    <p><em>Only includes cues that have been explicitly approved by human review.</em></p>
                    <p><em>No data available (all records filtered out).</em></p>
                </div>
                '''
    
    table_html = _generate_cue_type_table(human_approved_results) if include_table else ""
    return f'''
                <div class="subset-section">
                    <h3>Human-Approved Cues Only</h3>
                    <p><em>Only includes cues that have been explicitly approved by human review.</em></p>
                    <div class="plot">
                        <img src="{human_approved_plots[plot_type]}" alt="Human Approved - Predictiveness by {plot_type.replace('_', ' ').title()}">
                    </div>
                    {table_html}
                </div>
                '''


# ============================================================================
# HTML GENERATION: Report creation
# ============================================================================

def generate_html_report_combined(run_id: str, 
                                 all_results: Dict[str, AnalysisResults],
                                 model_filtered_results: Dict[str, AnalysisResults],
                                 human_approved_results: Dict[str, AnalysisResults],
                                 combined_plots: Dict[str, str]) -> str:
    """Generate self-contained HTML report with combined visualizations."""
    
    # CSS styling
    css = """
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        h1 { color: #333; }
        h2 { color: #555; margin-top: 30px; }
        h3 { color: #666; margin-top: 20px; }
        .container { max-width: 1400px; margin: 0 auto; background-color: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
        th { background-color: #f2f2f2; font-weight: bold; }
        .plot { margin: 20px 0; text-align: center; }
        img { max-width: 100%; height: auto; }
        .metric-section { margin: 40px 0; border-top: 2px solid #eee; padding-top: 20px; }
        .legend-explanation { background-color: #f9f9f9; padding: 15px; margin: 20px 0; border-left: 3px solid #007bff; }
        .legend-explanation h4 { margin-top: 0; color: #007bff; }
        .legend-explanation ul { margin: 10px 0; }
        .legend-explanation li { margin: 5px 0; }
    </style>
    """
    
    # Build HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Predictiveness Analysis - {run_id}</title>
        {css}
    </head>
    <body>
        <div class="container">
            <h1>Predictiveness of Cue Direction Analysis</h1>
            <p><strong>Run ID:</strong> {run_id}</p>
            <p><strong>Models analyzed:</strong> {', '.join(all_results.keys())}</p>
            <p><strong>Refusals:</strong> {', '.join([f"{model}: {results.refusal_count}" for model, results in all_results.items()])}</p>
            
            <div class="legend-explanation">
                <h4>Figure Legend</h4>
                <p>Each figure combines three data filtering approaches:</p>
                <ul>
                    <li><strong>All:</strong> Complete dataset without filtering</li>
                    <li><strong>Human:</strong> Only cues explicitly approved by human review</li>
                    <li><strong>Self:</strong> Only cues the answering model voted as fair</li>
                </ul>
            </div>
            
            <div class="metric-section">
                <h1>1. Predictiveness by Cue Type</h1>
                <div class="plot">
                    <img src="{combined_plots['cue_type']}" alt="Combined - Predictiveness by Cue Type">
                </div>
                {_generate_combined_cue_type_table(all_results, model_filtered_results, human_approved_results)}
            </div>
            
            <div class="metric-section">
                <h1>2. Predictiveness by Cue Severity</h1>
                <div class="plot">
                    <img src="{combined_plots['severity']}" alt="Combined - Predictiveness by Cue Severity">
                </div>
            </div>
            
            <div class="metric-section">
                <h1>3. Predictiveness by Baseline Consistency</h1>
                <div class="plot">
                    <img src="{combined_plots['consistency']}" alt="Combined - Predictiveness by Baseline Consistency">
                </div>
            </div>
            
            <div class="metric-section">
                <h1>4. Predictiveness by Question Obviousness</h1>
                <div class="plot">
                    <img src="{combined_plots['obviousness']}" alt="Combined - Predictiveness by Question Obviousness">
                </div>
            </div>
            
            <hr>
            <p><em>Statistical significance: * p<0.05, ** p<0.01, *** p<0.001</em></p>
        </div>
    </body>
    </html>
    """
    
    return html


def generate_html_report(run_id: str, all_results: Dict[str, AnalysisResults], 
                        plots: Dict[str, str], 
                        model_filtered_results: Dict[str, AnalysisResults],
                        model_filtered_plots: Dict[str, str],
                        majority_filtered_results: Dict[str, AnalysisResults],
                        majority_filtered_plots: Dict[str, str],
                        human_approved_results: Dict[str, AnalysisResults] = None,
                        human_approved_plots: Dict[str, str] = None) -> str:
    """Generate self-contained HTML report."""
    
    # CSS styling
    css = """
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        h1 { color: #333; }
        h2 { color: #555; margin-top: 30px; }
        h3 { color: #666; margin-top: 20px; }
        .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
        th { background-color: #f2f2f2; font-weight: bold; }
        .plot { margin: 20px 0; text-align: center; }
        img { max-width: 100%; height: auto; }
        .metric-section { margin: 40px 0; border-top: 2px solid #eee; padding-top: 20px; }
        .subset-section { margin: 30px 0; }
    </style>
    """
    
    # Build HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Predictiveness Analysis - {run_id}</title>
        {css}
    </head>
    <body>
        <div class="container">
            <h1>Predictiveness of Cue Direction Analysis</h1>
            <p><strong>Run ID:</strong> {run_id}</p>
            <p><strong>Models analyzed:</strong> {', '.join(all_results.keys())}</p>
            <p><strong>Refusals:</strong> {', '.join([f"{model}: {results.refusal_count}" for model, results in all_results.items()])}</p>
            
            <div class="metric-section">
                <h1>1. Predictiveness by Cue Type</h1>
                
                <div class="subset-section">
                    <h3>All Data</h3>
                    <div class="plot">
                        <img src="{plots['cue_type']}" alt="All Data - Predictiveness by Cue Type">
                    </div>
                    {_generate_cue_type_table(all_results)}
                </div>
                
                {_generate_human_approved_section('cue_type', human_approved_results, human_approved_plots, include_table=True)}
                
                <div class="subset-section">
                    <h3>Model Self-Filtered (Cues voted fair by answering model)</h3>
                    <p><em>Only includes records where the answering model voted the cue as fair (unfairness==False unanimously for that cue_id).</em></p>
                    {"<p><em>No data available (all records filtered out).</em></p>" if not model_filtered_plots or 'cue_type' not in model_filtered_plots else f'''
                    <div class="plot">
                        <img src="{model_filtered_plots['cue_type']}" alt="Model Filtered - Predictiveness by Cue Type">
                    </div>
                    {_generate_cue_type_table(model_filtered_results)}
                    '''}
                </div>
                
                <div class="subset-section">
                    <h3>Majority Consensus Filtered (>={int(MAJORITY_THRESHOLD*100)}% of models voted cue fair)</h3>
                    <p><em>Only includes records where {int(MAJORITY_THRESHOLD*100)}% or more of models voted the cue_id as fair.</em></p>
                    {"<p><em>No data available (all records filtered out).</em></p>" if not majority_filtered_plots or 'cue_type' not in majority_filtered_plots else f'''
                    <div class="plot">
                        <img src="{majority_filtered_plots['cue_type']}" alt="Majority Filtered - Predictiveness by Cue Type">
                    </div>
                    {_generate_cue_type_table(majority_filtered_results)}
                    '''}
                </div>
            </div>
            
            <div class="metric-section">
                <h1>2. Predictiveness by Cue Severity</h1>
                
                <div class="subset-section">
                    <h3>All Data</h3>
                    <div class="plot">
                        <img src="{plots['severity']}" alt="All Data - Predictiveness by Cue Severity">
                    </div>
                </div>
                
                {_generate_human_approved_section('severity', human_approved_results, human_approved_plots)}
                
                <div class="subset-section">
                    <h3>Model Self-Filtered</h3>
                    {"<p><em>No data available (all records filtered out).</em></p>" if not model_filtered_plots or 'severity' not in model_filtered_plots else f'''
                    <div class="plot">
                        <img src="{model_filtered_plots['severity']}" alt="Model Filtered - Predictiveness by Cue Severity">
                    </div>
                    '''}
                </div>
                
                <div class="subset-section">
                    <h3>Majority Consensus Filtered</h3>
                    {"<p><em>No data available (all records filtered out).</em></p>" if not majority_filtered_plots or 'severity' not in majority_filtered_plots else f'''
                    <div class="plot">
                        <img src="{majority_filtered_plots['severity']}" alt="Majority Filtered - Predictiveness by Cue Severity">
                    </div>
                    '''}
                </div>
            </div>
            
            <div class="metric-section">
                <h1>3. Predictiveness by Baseline Consistency</h1>
                
                <div class="subset-section">
                    <h3>All Data</h3>
                    <div class="plot">
                        <img src="{plots['consistency']}" alt="All Data - Predictiveness by Baseline Consistency">
                    </div>
                </div>
                
                {_generate_human_approved_section('consistency', human_approved_results, human_approved_plots)}
                
                <div class="subset-section">
                    <h3>Model Self-Filtered</h3>
                    {"<p><em>No data available (all records filtered out).</em></p>" if not model_filtered_plots or 'consistency' not in model_filtered_plots else f'''
                    <div class="plot">
                        <img src="{model_filtered_plots['consistency']}" alt="Model Filtered - Predictiveness by Baseline Consistency">
                    </div>
                    '''}
                </div>
                
                <div class="subset-section">
                    <h3>Majority Consensus Filtered</h3>
                    {"<p><em>No data available (all records filtered out).</em></p>" if not majority_filtered_plots or 'consistency' not in majority_filtered_plots else f'''
                    <div class="plot">
                        <img src="{majority_filtered_plots['consistency']}" alt="Majority Filtered - Predictiveness by Baseline Consistency">
                    </div>
                    '''}
                </div>
            </div>
            
            <div class="metric-section">
                <h1>4. Predictiveness by Question Obviousness</h1>
                
                <div class="subset-section">
                    <h3>All Data</h3>
                    <div class="plot">
                        <img src="{plots['obviousness']}" alt="All Data - Predictiveness by Question Obviousness">
                    </div>
                </div>
                
                {_generate_human_approved_section('obviousness', human_approved_results, human_approved_plots)}
                
                <div class="subset-section">
                    <h3>Model Self-Filtered</h3>
                    {"<p><em>No data available (all records filtered out).</em></p>" if not model_filtered_plots or 'obviousness' not in model_filtered_plots else f'''
                    <div class="plot">
                        <img src="{model_filtered_plots['obviousness']}" alt="Model Filtered - Predictiveness by Question Obviousness">
                    </div>
                    '''}
                </div>
                
                <div class="subset-section">
                    <h3>Majority Consensus Filtered</h3>
                    {"<p><em>No data available (all records filtered out).</em></p>" if not majority_filtered_plots or 'obviousness' not in majority_filtered_plots else f'''
                    <div class="plot">
                        <img src="{majority_filtered_plots['obviousness']}" alt="Majority Filtered - Predictiveness by Question Obviousness">
                    </div>
                    '''}
                </div>
            </div>
            
            <hr>
            <p><em>Statistical significance: * p<0.05, ** p<0.01, *** p<0.001</em></p>
        </div>
    </body>
    </html>
    """
    
    return html




def _generate_combined_cue_type_table(all_results: Dict[str, AnalysisResults],
                                     model_filtered_results: Dict[str, AnalysisResults],
                                     human_approved_results: Dict[str, AnalysisResults]) -> str:
    """Generate HTML table for combined cue type data."""
    html = """
                <table>
                    <tr>
                        <th rowspan="2">Model</th>
                        <th rowspan="2">Cue Type</th>
                        <th colspan="3">All Data</th>
                        <th colspan="3">Human Approved</th>
                        <th colspan="3">Self-Reviewed</th>
                    </tr>
                    <tr>
                        <th>Pred.</th>
                        <th>N</th>
                        <th>p-val</th>
                        <th>Pred.</th>
                        <th>N</th>
                        <th>p-val</th>
                        <th>Pred.</th>
                        <th>N</th>
                        <th>p-val</th>
                    </tr>
    """
    
    sorted_models = sort_model_list(list(all_results.keys()))
    
    for model in sorted_models:
        model_name = model.split('/')[-1]
        
        for cue_type in CUE_TYPE_ORDER:
            # Check if this cue type has data in any of the result sets for this model
            has_data_in_all = model in all_results and cue_type in all_results[model].predictiveness_by_cue_type
            has_data_in_human = model in human_approved_results and cue_type in human_approved_results[model].predictiveness_by_cue_type
            has_data_in_self = model in model_filtered_results and cue_type in model_filtered_results[model].predictiveness_by_cue_type
            
            # Only include row if there's data in at least one category
            if not (has_data_in_all or has_data_in_human or has_data_in_self):
                continue
            
            html += f"""
                        <tr>
                            <td>{model_name}</td>
                            <td>{cue_type}</td>
            """
            
            # All data columns
            if has_data_in_all:
                data = all_results[model].predictiveness_by_cue_type[cue_type]
                sig_marker = _get_significance_marker(data["p_value"])
                html += f"""
                            <td>{data['predictiveness']:.3f}{sig_marker}</td>
                            <td>{data['n']}</td>
                            <td>{data['p_value']:.4f}</td>
                """
            else:
                html += "<td>-</td><td>-</td><td>-</td>"
            
            # Human approved columns
            if has_data_in_human:
                data = human_approved_results[model].predictiveness_by_cue_type[cue_type]
                sig_marker = _get_significance_marker(data["p_value"])
                html += f"""
                            <td>{data['predictiveness']:.3f}{sig_marker}</td>
                            <td>{data['n']}</td>
                            <td>{data['p_value']:.4f}</td>
                """
            else:
                html += "<td>-</td><td>-</td><td>-</td>"
            
            # Self-reviewed columns
            if has_data_in_self:
                data = model_filtered_results[model].predictiveness_by_cue_type[cue_type]
                sig_marker = _get_significance_marker(data["p_value"])
                html += f"""
                            <td>{data['predictiveness']:.3f}{sig_marker}</td>
                            <td>{data['n']}</td>
                            <td>{data['p_value']:.4f}</td>
                """
            else:
                html += "<td>-</td><td>-</td><td>-</td>"
            
            html += "</tr>"
    
    html += """
                </table>
    """
    return html


def _generate_cue_type_table(all_results: Dict[str, AnalysisResults]) -> str:
    """Generate HTML table for cue type data."""
    html = """
                <table>
                    <tr>
                        <th>Model</th>
                        <th>Cue Type</th>
                        <th>Predictiveness</th>
                        <th>N</th>
                        <th>p-value</th>
                    </tr>
    """
    
    sorted_models = sort_model_list(list(all_results.keys()))
    
    for model in sorted_models:
        results = all_results[model]
        
        for cue_type in CUE_TYPE_ORDER:
            if cue_type in results.predictiveness_by_cue_type:
                data = results.predictiveness_by_cue_type[cue_type]
                sig_marker = _get_significance_marker(data["p_value"])
                html += f"""
                        <tr>
                            <td>{model}</td>
                            <td>{cue_type}</td>
                            <td>{data['predictiveness']:.3f}{sig_marker}</td>
                            <td>{data['n']}</td>
                            <td>{data['p_value']:.4f}</td>
                        </tr>
                """
    
    html += """
                </table>
    """
    return html


# ============================================================================
# MAIN: CLI and orchestration
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze predictiveness of cue direction across models"
    )
    parser.add_argument(
        "--run-id", 
        type=str, 
        required=True,
        help="Run ID (e.g., run_001)"
    )
    
    args = parser.parse_args()
    
    run_dir = Path("data/eval_results") / f"{args.run_id}"
    run_id = run_dir.name
    
    print(f"\nLoading results from {run_dir}...")
    model_results = load_run_results(run_dir)
    
    print("\nLoading dataset for human approval data...")
    dataset_records = load_dataset_records(run_id)
    human_approved_cue_ids = get_human_approved_cue_ids(dataset_records) if dataset_records else set()
    print(f"Found {len(human_approved_cue_ids)} human-approved cues")
    
    print("\nAnalyzing cue unfairness votes...")
    cue_unfairness_votes = get_cue_unfairness_votes(model_results)
    
    print("\nAnalyzing predictiveness for all data...")
    all_analysis_results = {}
    for model_id, records in model_results.items():
        results = analyze_model_results(records, model_id)
        all_analysis_results[model_id] = results
    
    print("\nAnalyzing predictiveness for model-filtered data...")
    model_filtered_results = {}
    for model_id, records in model_results.items():
        filtered_records = filter_records_by_model_unfairness_vote(records, cue_unfairness_votes)
        if filtered_records:  # Only analyze if we have filtered data
            results = analyze_model_results(filtered_records, model_id)
            model_filtered_results[model_id] = results
    
    # Human-approved filtering (only if we have approval data)
    human_approved_results = {}
    if human_approved_cue_ids:
        print("\nAnalyzing predictiveness for human-approved data...")
        for model_id, records in model_results.items():
            filtered_records = filter_records_by_human_approval(records, human_approved_cue_ids)
            if filtered_records:  # Only analyze if we have filtered data
                results = analyze_model_results(filtered_records, model_id)
                human_approved_results[model_id] = results
    
    # Generate combined plots
    print("\nGenerating combined visualizations...")
    
    # Sort models and get colors
    sorted_models = sort_model_list(list(all_analysis_results.keys()))
    colors = sns.color_palette("husl", len(sorted_models))
    
    if all_analysis_results:
        random = _get_random(all_analysis_results)
        
        # Prepare data dictionary for combined plots
        combined_data = {
            'all': all_analysis_results,
            'human_approved': human_approved_results,
            'model_filtered': model_filtered_results
        }
        
        # Generate combined plots
        combined_plots = {
            'cue_type': _plot_cue_type_combined(combined_data, sorted_models, colors, random),
            'severity': _plot_severity_combined(combined_data, sorted_models, colors, random),
            'consistency': _plot_consistency_combined(combined_data, sorted_models, colors, random),
            'obviousness': _plot_obviousness_combined(combined_data, sorted_models, colors, random)
        }
    
    # Generate HTML report with combined plots
    print("\nGenerating HTML report...")
    html_content = generate_html_report_combined(
        run_id, all_analysis_results, 
        model_filtered_results,
        human_approved_results,
        combined_plots
    )
    
    # Save report
    report_path = run_dir / "predictiveness_analysis.html"
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"\nAnalysis complete!")
    print(f"Report saved to: {report_path.absolute()}. Open in your browser.")
    print(f"Cue unfairness votes analyzed for {len(cue_unfairness_votes)} cue_ids")
    print(f"Model-filtered results: {len(model_filtered_results)} models")
    if human_approved_results:
        print(f"Human-approved results: {len(human_approved_results)} models")


if __name__ == "__main__":
    main()