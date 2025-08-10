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

def generate_plots(data_categories: Dict[str, Dict[str, AnalysisResults]]) -> Dict[str, str]:
    """Generate all plots as base64 strings for multiple data categories."""
    plots = {}
    
    # Get all unique models across all categories
    all_models = set()
    for category_results in data_categories.values():
        all_models.update(category_results.keys())
    
    if not all_models:
        return plots
    
    plt.style.use('seaborn-v0_8-whitegrid')
    sorted_models = sort_model_list(list(all_models))
    colors = sns.color_palette("husl", len(sorted_models))
    
    # Get random value from first available result
    random = None
    for category_results in data_categories.values():
        if category_results:
            random = _get_random(category_results)
            break
    
    if random is None:
        return plots
    
    plots['cue_type'] = _create_plot('cue_type', data_categories, sorted_models, colors, random)
    plots['severity'] = _create_plot('severity', data_categories, sorted_models, colors, random) 
    plots['consistency'] = _create_plot('consistency', data_categories, sorted_models, colors, random)
    plots['obviousness'] = _create_plot('obviousness', data_categories, sorted_models, colors, random)
    
    return plots


def _create_plot(plot_type: str, data_categories: Dict[str, Dict[str, AnalysisResults]], 
                sorted_models: List[str], colors: List, random: float) -> str:
    """Create unified plot for any metric type across data categories."""
    
    # Plot configuration
    patterns = {'all': None, 'human_approved': '///', 'model_filtered': '...'}
    alphas = {'all': 0.9, 'human_approved': 0.7, 'model_filtered': 0.7}
    
    if plot_type == 'cue_type':
        return _create_cue_type_plot(data_categories, sorted_models, colors, random, patterns, alphas)
    elif plot_type in ['severity', 'obviousness']:
        return _create_line_plot(plot_type, data_categories, sorted_models, colors, random)
    elif plot_type == 'consistency':
        return _create_scatter_plot(data_categories, sorted_models, colors, random)
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")


def _create_cue_type_plot(data_categories: Dict[str, Dict[str, AnalysisResults]], 
                         sorted_models: List[str], colors: List, random: float,
                         patterns: Dict[str, str], alphas: Dict[str, float]) -> str:
    """Create bar plot for cue type predictiveness."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    bar_width = 0.25
    x_pos = 0
    x_labels = []
    x_positions = []
    
    for i, model in enumerate(sorted_models):
        model_name = model.split('/')[-1]
        is_first = True
        
        for cue_type in CUE_TYPE_ORDER:
            # Check if any category has this data
            has_data = any(
                model in category_results and cue_type in category_results[model].predictiveness_by_cue_type
                for category_results in data_categories.values()
            )
            
            if not has_data:
                continue
            
            # Plot bars for each category
            category_offset = 0
            for category_name, category_results in data_categories.items():
                if model in category_results and cue_type in category_results[model].predictiveness_by_cue_type:
                    data = category_results[model].predictiveness_by_cue_type[cue_type]
                    
                    label = None
                    if is_first:
                        label = f"{model_name} ({category_name.replace('_', ' ').title()})"
                    
                    ax.bar(x_pos + category_offset, data["predictiveness"], 
                          width=bar_width, color=colors[i], 
                          alpha=alphas[category_name],
                          hatch=patterns[category_name],
                          label=label, edgecolor='black', linewidth=0.5)
                    
                    # Add significance marker
                    sig_marker = _get_significance_marker(data["p_value"])
                    if sig_marker:
                        ax.text(x_pos + category_offset, data["predictiveness"] + 0.02, 
                               sig_marker, ha='center', fontsize=8)
                
                category_offset += bar_width
            
            x_labels.append(cue_type)
            x_positions.append(x_pos + bar_width)
            x_pos += bar_width * 3 + 0.1
            is_first = False
        
        x_pos += 0.3
    
    ax.axhline(y=random, color='red', linestyle='--', alpha=0.7, 
              label=f'Random ({random:.2f})')
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Predictiveness of Cue Direction')
    ax.set_title('Predictiveness by Cue Type')
    ax.set_ylim(0, 1.1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def _create_line_plot(plot_type: str, data_categories: Dict[str, Dict[str, AnalysisResults]], 
                     sorted_models: List[str], colors: List, random: float) -> str:
    """Create line plot for severity or obviousness."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    line_styles = {'all': '-', 'human_approved': '--', 'model_filtered': ':'}
    markers = {'all': 'o', 'human_approved': 's', 'model_filtered': '^'}
    
    # Get the right data field
    if plot_type == 'severity':
        data_field = 'predictiveness_by_cue_severity'
        x_label = 'Cue Severity'
        title = 'Predictiveness by Cue Severity'
    else:  # obviousness
        data_field = 'predictiveness_by_question_obviousness'
        x_label = 'Question Obviousness'
        title = 'Predictiveness by Question Obviousness'
    
    for i, model in enumerate(sorted_models):
        model_name = model.split('/')[-1]
        
        for category_name, category_results in data_categories.items():
            if model in category_results:
                results = category_results[model]
                data_dict = getattr(results, data_field)
                
                x_values = sorted(data_dict.keys())
                if x_values:
                    y_values = [data_dict[x]["predictiveness"] for x in x_values]
                    
                    label = f"{model_name} ({category_name.replace('_', ' ').title()})"
                    
                    ax.plot(x_values, y_values, 
                           linestyle=line_styles[category_name],
                           marker=markers[category_name],
                           color=colors[i], 
                           label=label, markersize=6, linewidth=1.5, alpha=0.8)
    
    ax.axhline(y=random, color='red', linestyle='--', alpha=0.7, 
              label=f'Random ({random:.2f})')
    
    ax.set_xlabel(x_label)
    ax.set_ylabel('Predictiveness of Cue Direction')
    ax.set_title(title)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(range(1, 11))
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def _create_scatter_plot(data_categories: Dict[str, Dict[str, AnalysisResults]], 
                        sorted_models: List[str], colors: List, random: float) -> str:
    """Create scatter plot for baseline consistency."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    markers = {'all': 'o', 'human_approved': 's', 'model_filtered': '^'}
    sizes = {'all': 80, 'human_approved': 60, 'model_filtered': 60}
    
    for i, model in enumerate(sorted_models):
        model_name = model.split('/')[-1]
        
        for category_name, category_results in data_categories.items():
            if model in category_results:
                results = category_results[model]
                data_dict = results.predictiveness_by_baseline_consistency
                
                x_values = sorted(data_dict.keys())
                if x_values:
                    y_values = [data_dict[x]["predictiveness"] for x in x_values]
                    
                    label = f"{model_name} ({category_name.replace('_', ' ').title()})"
                    
                    ax.scatter(x_values, y_values, 
                              marker=markers[category_name],
                              s=sizes[category_name],
                              color=colors[i], 
                              label=label, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax.axhline(y=random, color='red', linestyle='--', alpha=0.7, 
              label=f'Random ({random:.2f})')
    
    ax.set_xlabel('Baseline Consistency Score')
    ax.set_ylabel('Predictiveness of Cue Direction')
    ax.set_title('Predictiveness by Baseline Consistency')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 1.1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
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

def generate_html_report(run_id: str, 
                        data_categories: Dict[str, Dict[str, AnalysisResults]],
                        plots: Dict[str, str]) -> str:
    """Generate self-contained HTML report with combined visualizations."""
    
    # Get info from first category (all should have same basic structure)
    all_results = data_categories.get('all', {})
    model_filtered_results = data_categories.get('model_filtered', {})
    human_approved_results = data_categories.get('human_approved', {})
    
    all_models = set()
    for category_results in data_categories.values():
        all_models.update(category_results.keys())
    
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
            <p><strong>Models analyzed:</strong> {', '.join(sorted(all_models))}</p>
            
            <div class="legend-explanation">
                <h4>Figure Legend</h4>
                <p>Each figure combines three data filtering approaches:</p>
                <ul>
                    <li><strong>All:</strong> Complete dataset without filtering</li>
                    <li><strong>Human Approved:</strong> Only cues explicitly approved by human review</li>
                    <li><strong>Model Filtered:</strong> Only cues the answering model voted as fair</li>
                </ul>
            </div>
            
            <div class="metric-section">
                <h1>1. Predictiveness by Cue Type</h1>
                <div class="plot">
                    <img src="{plots['cue_type']}" alt="Predictiveness by Cue Type">
                </div>
                {_generate_combined_cue_type_table(all_results, model_filtered_results, human_approved_results) if all_results else ''}
            </div>
            
            <div class="metric-section">
                <h1>2. Predictiveness by Cue Severity</h1>
                <div class="plot">
                    <img src="{plots['severity']}" alt="Predictiveness by Cue Severity">
                </div>
            </div>
            
            <div class="metric-section">
                <h1>3. Predictiveness by Baseline Consistency</h1>
                <div class="plot">
                    <img src="{plots['consistency']}" alt="Predictiveness by Baseline Consistency">
                </div>
            </div>
            
            <div class="metric-section">
                <h1>4. Predictiveness by Question Obviousness</h1>
                <div class="plot">
                    <img src="{plots['obviousness']}" alt="Predictiveness by Question Obviousness">
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
    
    # Prepare data dictionary for combined plots
    combined_data = {
        'all': all_analysis_results,
        'human_approved': human_approved_results,
        'model_filtered': model_filtered_results
    }
    
    # Generate plots
    combined_plots = generate_plots(combined_data) if all_analysis_results else {}
    
    # Generate HTML report
    print("\nGenerating HTML report...")
    html_content = generate_html_report(
        run_id, combined_data, combined_plots
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