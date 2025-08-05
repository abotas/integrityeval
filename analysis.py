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
    filter_records_by_model_unfairness_vote, filter_records_by_majority_unfairness_vote
)

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


# ============================================================================
# HTML GENERATION: Report creation
# ============================================================================

def generate_html_report(run_id: str, all_results: Dict[str, AnalysisResults], 
                        plots: Dict[str, str], 
                        model_filtered_results: Dict[str, AnalysisResults],
                        model_filtered_plots: Dict[str, str],
                        majority_filtered_results: Dict[str, AnalysisResults],
                        majority_filtered_plots: Dict[str, str]) -> str:
    """Generate self-contained HTML report."""
    
    # CSS styling
    css = """
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
        h1 { color: #333; }
        h2 { color: #555; margin-top: 30px; }
        .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
        th { background-color: #f2f2f2; font-weight: bold; }
        .plot { margin: 20px 0; text-align: center; }
        img { max-width: 100%; height: auto; }
        .metric-section { margin: 40px 0; }
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
            
            <h1>Set 1: All Data (Original Analysis)</h1>
            
            <div class="metric-section">
                <h2>1. Predictiveness by Cue Type</h2>
                <div class="plot">
                    <img src="{plots['cue_type']}" alt="Predictiveness by Cue Type">
                </div>
                {_generate_cue_type_table(all_results)}
            </div>
            
            <div class="metric-section">
                <h2>2. Predictiveness by Cue Severity</h2>
                <div class="plot">
                    <img src="{plots['severity']}" alt="Predictiveness by Cue Severity">
                </div>
            </div>
            
            <div class="metric-section">
                <h2>3. Predictiveness by Baseline Consistency</h2>
                <div class="plot">
                    <img src="{plots['consistency']}" alt="Predictiveness by Baseline Consistency">
                </div>
            </div>
            
            <div class="metric-section">
                <h2>4. Predictiveness by Question Obviousness</h2>
                <div class="plot">
                    <img src="{plots['obviousness']}" alt="Predictiveness by Question Obviousness">
                </div>
            </div>
            
            <h1>Set 2: Model Self-Filtered (Cues voted fair by answering model)</h1>
            <p><em>Only includes records where the answering model voted the cue as fair (unfairness==False unanimously for that cue_id).</em></p>
            
            {_generate_filtered_section(model_filtered_results, model_filtered_plots, "Model Filtered")}
            
            <h1>Set 3: Majority Consensus Filtered (≥{int(MAJORITY_THRESHOLD*100)}% of models voted cue fair)</h1>
            <p><em>Only includes records where {int(MAJORITY_THRESHOLD*100)}% or more of models voted the cue_id as fair.</em></p>
            
            {_generate_filtered_section(majority_filtered_results, majority_filtered_plots, "Majority Filtered")}
            
            <hr>
            <p><em>Statistical significance: * p<0.05, ** p<0.01, *** p<0.001</em></p>
        </div>
    </body>
    </html>
    """
    
    return html


def _generate_filtered_section(results: Dict[str, AnalysisResults], plots: Dict[str, str], section_name: str) -> str:
    """Generate HTML section for filtered results."""
    if not results or not plots:
        return f"<p><em>No data available for {section_name} analysis (all records filtered out).</em></p>"
    
    html = f"""
            <div class="metric-section">
                <h2>1. Predictiveness by Cue Type</h2>
                <div class="plot">
                    <img src="{plots['cue_type']}" alt="{section_name} - Predictiveness by Cue Type">
                </div>
                {_generate_cue_type_table(results)}
            </div>
            
            <div class="metric-section">
                <h2>2. Predictiveness by Cue Severity</h2>
                <div class="plot">
                    <img src="{plots['severity']}" alt="{section_name} - Predictiveness by Cue Severity">
                </div>
            </div>
            
            <div class="metric-section">
                <h2>3. Predictiveness by Baseline Consistency</h2>
                <div class="plot">
                    <img src="{plots['consistency']}" alt="{section_name} - Predictiveness by Baseline Consistency">
                </div>
            </div>
            
            <div class="metric-section">
                <h2>4. Predictiveness by Question Obviousness</h2>
                <div class="plot">
                    <img src="{plots['obviousness']}" alt="{section_name} - Predictiveness by Question Obviousness">
                </div>
            </div>
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
    
    print("\nAnalyzing predictiveness for majority-filtered data...")
    majority_filtered_results = {}
    for model_id, records in model_results.items():
        filtered_records = filter_records_by_majority_unfairness_vote(records, cue_unfairness_votes, MAJORITY_THRESHOLD)
        if filtered_records:  # Only analyze if we have filtered data
            results = analyze_model_results(filtered_records, model_id)
            majority_filtered_results[model_id] = results
    
    # Generate plots for all three sets
    print("\nGenerating visualizations...")
    plots = generate_predictiveness_plots(all_analysis_results)
    model_filtered_plots = generate_predictiveness_plots(model_filtered_results) if model_filtered_results else {}
    majority_filtered_plots = generate_predictiveness_plots(majority_filtered_results) if majority_filtered_results else {}
    
    # Generate HTML report
    print("\nGenerating HTML report...")
    html_content = generate_html_report(
        run_id, all_analysis_results, plots,
        model_filtered_results, model_filtered_plots,
        majority_filtered_results, majority_filtered_plots
    )
    
    # Save report
    report_path = run_dir / "predictiveness_analysis.html"
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"\nAnalysis complete!")
    print(f"Report saved to: {report_path.absolute()}. Open in your browser.")
    print(f"Cue unfairness votes analyzed for {len(cue_unfairness_votes)} cue_ids")
    print(f"Model-filtered results: {len(model_filtered_results)} models")
    print(f"Majority-filtered results: {len(majority_filtered_results)} models")


if __name__ == "__main__":
    main()