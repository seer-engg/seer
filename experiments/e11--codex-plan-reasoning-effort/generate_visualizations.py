"""
Generate visualizations for Experiment E11: Codex Plan Reasoning Effort Impact
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

RESULTS_DIR = Path(__file__).parent / "results_e11"

def load_latest_results():
    """Load the most recent results file."""
    result_files = list(RESULTS_DIR.glob("e11_results_*.json"))
    if not result_files:
        raise FileNotFoundError("No results files found")
    
    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading results from: {latest_file.name}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def create_summary_table(data):
    """Create a summary statistics table."""
    aggregated = data['aggregated_results']
    
    summary_data = []
    for level in ['minimal', 'medium', 'high']:
        stats = aggregated[level]
        summary_data.append({
            'Reasoning Effort': level.upper(),
            'Mean Score': f"{stats['mean_score']:.2f}",
            'Mean Time (s)': f"{stats['mean_time']:.1f}",
            'Mean Correctness': f"{stats['mean_correctness']:.2f}",
            'Mean Code Quality': f"{stats['mean_code_quality']:.2f}",
            'Test Pass Rate': f"{stats['test_passing_rate']:.0%}",
            'Rounds': stats['num_rounds']
        })
    
    df = pd.DataFrame(summary_data)
    return df

def generate_visualizations(data):
    """Generate all visualizations."""
    aggregated = data['aggregated_results']
    individual = data['individual_rounds']
    
    # Prepare data for plotting
    levels = ['minimal', 'medium', 'high']
    colors = {'minimal': '#e74c3c', 'medium': '#3498db', 'high': '#2ecc71'}
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Mean Score Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    scores = [aggregated[level]['mean_score'] for level in levels]
    stds = [aggregated[level]['std_score'] for level in levels]
    bars = ax1.bar(levels, scores, yerr=stds, capsize=5, 
                   color=[colors[l] for l in levels], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Mean Score', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 0.6)
    ax1.set_title('Mean Quality Score by Reasoning Effort', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Mean Time Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    times = [aggregated[level]['mean_time'] for level in levels]
    time_stds = [aggregated[level]['std_time'] for level in levels]
    bars = ax2.bar(levels, times, yerr=time_stds, capsize=5,
                   color=[colors[l] for l in levels], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Mean Time (seconds)', fontsize=11, fontweight='bold')
    ax2.set_title('Mean Execution Time by Reasoning Effort', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{time_val:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Correctness Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    correctness = [aggregated[level]['mean_correctness'] for level in levels]
    corr_stds = [aggregated[level]['std_correctness'] for level in levels]
    bars = ax3.bar(levels, correctness, yerr=corr_stds, capsize=5,
                   color=[colors[l] for l in levels], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Mean Correctness', fontsize=11, fontweight='bold')
    ax3.set_ylim(0, 0.6)
    ax3.set_title('Mean Correctness by Reasoning Effort', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, corr in zip(bars, correctness):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{corr:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Code Quality Comparison
    ax4 = fig.add_subplot(gs[1, 0])
    code_quality = [aggregated[level]['mean_code_quality'] for level in levels]
    cq_stds = [aggregated[level]['std_code_quality'] for level in levels]
    bars = ax4.bar(levels, code_quality, yerr=cq_stds, capsize=5,
                   color=[colors[l] for l in levels], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Mean Code Quality', fontsize=11, fontweight='bold')
    ax4.set_ylim(0, 0.6)
    ax4.set_title('Mean Code Quality by Reasoning Effort', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, cq in zip(bars, code_quality):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{cq:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 5. Score Distribution (Individual Rounds)
    ax5 = fig.add_subplot(gs[1, 1])
    all_scores = []
    all_labels = []
    for level in levels:
        scores = [r['evaluation']['score'] for r in individual[level]]
        all_scores.extend(scores)
        all_labels.extend([level] * len(scores))
    
    df_scores = pd.DataFrame({'Score': all_scores, 'Reasoning Effort': all_labels})
    sns.violinplot(data=df_scores, x='Reasoning Effort', y='Score', 
                   order=levels, palette=[colors[l] for l in levels], ax=ax5)
    ax5.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax5.set_title('Score Distribution Across Rounds', fontsize=12, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 6. Time vs Score Scatter
    ax6 = fig.add_subplot(gs[1, 2])
    for level in levels:
        times = [r['time'] for r in individual[level]]
        scores = [r['evaluation']['score'] for r in individual[level]]
        ax6.scatter(times, scores, label=level.upper(), 
                   color=colors[level], s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
    ax6.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax6.set_title('Time vs Quality Score', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(alpha=0.3, linestyle='--')
    
    # 7. Score Over Rounds (Line Plot)
    ax7 = fig.add_subplot(gs[2, :])
    for level in levels:
        rounds = [r['round'] for r in individual[level]]
        scores = [r['evaluation']['score'] for r in individual[level]]
        ax7.plot(rounds, scores, 'o-', label=level.upper(), 
                color=colors[level], linewidth=2, markersize=8, alpha=0.8)
    ax7.set_xlabel('Round Number', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax7.set_title('Score Progression Across Rounds', fontsize=12, fontweight='bold')
    ax7.set_xticks([1, 2, 3])
    ax7.legend(fontsize=10, loc='best')
    ax7.grid(alpha=0.3, linestyle='--')
    ax7.set_ylim(0, 0.6)
    
    # Add overall title
    fig.suptitle('Experiment E11: Codex Plan Reasoning Effort Impact\n' + 
                 f"Task: {data['task']} | Rounds: {data['num_rounds']} per level",
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Save figure
    timestamp = data['timestamp']
    output_path = RESULTS_DIR / f"e11_results_{timestamp}_visualization.png"
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Visualization saved to: {output_path}")
    
    return fig

def generate_summary_table(data):
    """Generate and save summary table as image."""
    df = create_summary_table(data)
    
    # Create table visualization
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.12, 0.12, 0.15, 0.15, 0.12, 0.1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Experiment E11: Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    timestamp = data['timestamp']
    output_path = RESULTS_DIR / f"e11_results_{timestamp}_table.png"
    plt.savefig(output_path, bbox_inches='tight', facecolor='white', dpi=300)
    print(f"✅ Summary table saved to: {output_path}")
    
    return df

def main():
    """Main function to generate all visualizations."""
    print("="*60)
    print("EXPERIMENT E11: VISUALIZATION GENERATION")
    print("="*60)
    
    # Load data
    data = load_latest_results()
    
    # Generate summary table
    print("\n📊 Generating summary table...")
    summary_df = generate_summary_table(data)
    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))
    
    # Generate visualizations
    print("\n📈 Generating visualizations...")
    fig = generate_visualizations(data)
    
    # Print key findings
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    best_level = data['best_implementation']
    best_stats = data['aggregated_results'][best_level]
    
    print(f"\n🏆 Best Implementation: {best_level.upper()}")
    print(f"   Mean Score: {best_stats['mean_score']:.2f}")
    print(f"   Mean Time: {best_stats['mean_time']:.1f}s")
    print(f"   Mean Correctness: {best_stats['mean_correctness']:.2f}")
    print(f"   Mean Code Quality: {best_stats['mean_code_quality']:.2f}")
    
    print("\n📊 Comparison:")
    for level in ['minimal', 'medium', 'high']:
        stats = data['aggregated_results'][level]
        print(f"\n{level.upper()}:")
        print(f"  Score: {stats['mean_score']:.2f} (±{stats['std_score']:.2f})")
        print(f"  Time: {stats['mean_time']:.1f}s (±{stats['std_time']:.1f}s)")
        print(f"  Correctness: {stats['mean_correctness']:.2f} (±{stats['std_correctness']:.2f})")
        print(f"  Code Quality: {stats['mean_code_quality']:.2f} (±{stats['std_code_quality']:.2f})")
    
    print("\n" + "="*60)
    print("✅ Visualization generation complete!")
    print("="*60)

if __name__ == "__main__":
    main()

