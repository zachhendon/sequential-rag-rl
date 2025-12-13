#!/usr/bin/env python3
"""
Generate academic-quality visualizations for 1.5B model results.
Creates publication-ready figures for the sequential RL-based retrieval experiments.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2,
    'patch.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'baseline': '#8B8B8B',
    'ac': '#2E86AB',
    'ppo_ind': '#A23B72',
    'rwb': '#F18F01',
    'ppo': '#C73E1D',
    'train': '#4A90A4',
    'val': '#E94F37',
    'test': '#6A994E',
}

baseline_random = 62.0

methods_data = {
    'Actor-Critic\n(LSTM)': {
        'val': 77.5,
        'train': 78.5,
        'test': None,
        'color': COLORS['ac'],
        'abbrev': 'AC'
    },
    'PPO\nIndependent': {
        'val': 77.0,
        'train': 76.1,
        'test': 79.0,
        'color': COLORS['ppo_ind'],
        'abbrev': 'PPO-Ind'
    },
    'REINFORCE\nw/ Baseline': {
        'val': 74.5,
        'train': 74.6,
        'test': None,
        'color': COLORS['rwb'],
        'abbrev': 'RWB'
    },
    'PPO\n(LSTM)': {
        'val': 70.0,
        'train': 75.6,
        'test': None,
        'color': COLORS['ppo'],
        'abbrev': 'PPO'
    },
}

def create_figure_1_comparison():
    """Figure 1: Comparison of all methods vs baseline"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(methods_data.keys())
    val_accs = [methods_data[m]['val'] for m in methods]
    colors = [methods_data[m]['color'] for m in methods]
    
    all_methods = ['Random\nBaseline'] + methods
    all_accs = [baseline_random] + val_accs
    all_colors = [COLORS['baseline']] + colors
    
    bars = ax.bar(range(len(all_methods)), all_accs, color=all_colors, 
                  edgecolor='black', linewidth=1.2, alpha=0.85)
    
    for i, (bar, acc) in enumerate(zip(bars, all_accs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for i, (method, acc) in enumerate(zip(methods, val_accs), 1):
        improvement = acc - baseline_random
        ax.annotate(f'+{improvement:.1f}%', 
                   xy=(i, acc), xytext=(i, acc + 3),
                   ha='center', va='bottom',
                   fontsize=9, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    ax.set_xticks(range(len(all_methods)))
    ax.set_xticklabels(all_methods, fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Comparison: RL Methods vs Baseline\n(Qwen2.5-1.5B on GSM8K)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim([55, 85])
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.axhline(y=baseline_random, color='gray', linestyle=':', linewidth=2, 
               alpha=0.7, label='Baseline')
    
    legend_elements = [
        mpatches.Patch(color=COLORS['baseline'], label='Baseline'),
        mpatches.Patch(color=COLORS['ac'], label='RL Methods'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=True, 
              fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig('results/figures/figure1_method_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('results/figures/figure1_method_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved Figure 1: Method Comparison")
    plt.close()


def create_figure_2_train_val():
    """Figure 2: Train vs Validation accuracy comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(methods_data.keys())
    train_accs = [methods_data[m]['train'] for m in methods]
    val_accs = [methods_data[m]['val'] for m in methods]
    colors = [methods_data[m]['color'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_accs, width, label='Training', 
                   color=[COLORS['train']] * len(methods), 
                   edgecolor='black', linewidth=1.2, alpha=0.8)
    bars2 = ax.bar(x + width/2, val_accs, width, label='Validation', 
                   color=colors, edgecolor='black', linewidth=1.2, alpha=0.8)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9)
    
    for i, (train, val, method) in enumerate(zip(train_accs, val_accs, methods)):
        gap = train - val
        if abs(gap) > 0.5:
            y_pos = max(train, val) + 2
            ax.annotate(f'Δ={gap:+.1f}%', 
                       xy=(i, y_pos), xytext=(i, y_pos + 1.5),
                       ha='center', va='bottom',
                       fontsize=8, style='italic',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Training vs Validation Accuracy\n(Generalization Analysis)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim([65, 82])
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig('results/figures/figure2_train_val.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('results/figures/figure2_train_val.png', dpi=300, bbox_inches='tight')
    print("✓ Saved Figure 2: Train vs Validation")
    plt.close()


def create_figure_3_architecture():
    """Figure 3: Architecture comparison (Sequential vs Independent)"""
    fig, ax = plt.subplots(figsize=(9, 6))
    
    sequential_methods = ['Actor-Critic\n(LSTM)', 'REINFORCE\nw/ Baseline', 'PPO\n(LSTM)']
    independent_methods = ['PPO\nIndependent']
    
    seq_accs = [methods_data[m]['val'] for m in sequential_methods]
    ind_accs = [methods_data[m]['val'] for m in independent_methods]
    
    x = np.arange(2)
    width = 0.6
    
    seq_mean = np.mean(seq_accs)
    ind_mean = np.mean(ind_accs)
    seq_std = np.std(seq_accs)
    ind_std = np.std(ind_accs) if len(ind_accs) > 1 else 0
    
    bars = ax.bar(x, [seq_mean, ind_mean], width, 
                  color=[COLORS['ac'], COLORS['ppo_ind']],
                  edgecolor='black', linewidth=1.5, alpha=0.7,
                  yerr=[seq_std, ind_std], capsize=10,
                  error_kw={'elinewidth': 1.5, 'capthick': 1.5})
    
    for acc in seq_accs:
        ax.scatter(0, acc, color='white', s=80, zorder=6, 
                  edgecolors='black', linewidths=1.5)
    for acc in ind_accs:
        ax.scatter(1, acc, color='white', s=80, zorder=6, 
                  edgecolors='black', linewidths=1.5)
    
    for i, (bar, mean) in enumerate(zip(bars, [seq_mean, ind_mean])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{mean:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.text(0, seq_mean - 3, f'Methods:\n{", ".join([methods_data[m]["abbrev"] for m in sequential_methods])}',
            ha='center', va='top', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(1, ind_mean - 3, f'Method:\n{methods_data[independent_methods[0]]["abbrev"]}',
            ha='center', va='top', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Sequential\n(LSTM)', 'Independent'], fontsize=11)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Architecture Comparison: Sequential vs Independent\n(Qwen2.5-1.5B)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim([68, 80])
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    plt.savefig('results/figures/figure3_architecture.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('results/figures/figure3_architecture.png', dpi=300, bbox_inches='tight')
    print("✓ Saved Figure 3: Architecture Comparison")
    plt.close()


def create_figure_4_improvement():
    """Figure 4: Improvement over baseline"""
    fig, ax = plt.subplots(figsize=(9, 6))
    
    methods = list(methods_data.keys())
    improvements = [methods_data[m]['val'] - baseline_random for m in methods]
    colors = [methods_data[m]['color'] for m in methods]
    
    sorted_data = sorted(zip(methods, improvements, colors), key=lambda x: x[1], reverse=True)
    methods, improvements, colors = zip(*sorted_data)
    
    bars = ax.barh(range(len(methods)), improvements, color=colors,
                   edgecolor='black', linewidth=1.2, alpha=0.85)
    
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        width = bar.get_width()
        ax.text(width + 0.3, bar.get_y() + bar.get_height()/2.,
                f'+{imp:.1f}%',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=10)
    ax.set_xlabel('Improvement over Baseline (%)', fontsize=12, fontweight='bold')
    ax.set_title('Absolute Improvement over Random Baseline\n(Qwen2.5-1.5B)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim([0, 18])
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    plt.savefig('results/figures/figure4_improvement.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('results/figures/figure4_improvement.png', dpi=300, bbox_inches='tight')
    print("✓ Saved Figure 4: Improvement over Baseline")
    plt.close()


def create_figure_5_summary():
    """Figure 5: Comprehensive summary with multiple subplots"""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])
    methods = list(methods_data.keys())
    val_accs = [methods_data[m]['val'] for m in methods]
    colors = [methods_data[m]['color'] for m in methods]
    all_methods = ['Random\nBaseline'] + methods
    all_accs = [baseline_random] + val_accs
    all_colors = [COLORS['baseline']] + colors
    bars = ax1.bar(range(len(all_methods)), all_accs, color=all_colors,
                   edgecolor='black', linewidth=1.2, alpha=0.85)
    for i, (bar, acc) in enumerate(zip(bars, all_accs)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.set_xticks(range(len(all_methods)))
    ax1.set_xticklabels(all_methods, fontsize=10)
    ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('(a) All Methods Comparison', fontsize=12, fontweight='bold', pad=10)
    ax1.set_ylim([55, 85])
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    ax2 = fig.add_subplot(gs[1, 0])
    train_accs = [methods_data[m]['train'] for m in methods]
    val_accs = [methods_data[m]['val'] for m in methods]
    x = np.arange(len(methods))
    width = 0.35
    ax2.bar(x - width/2, train_accs, width, label='Training', 
            color=COLORS['train'], edgecolor='black', linewidth=1.2, alpha=0.8)
    ax2.bar(x + width/2, val_accs, width, label='Validation', 
            color=[methods_data[m]['color'] for m in methods],
            edgecolor='black', linewidth=1.2, alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([methods_data[m]['abbrev'] for m in methods], fontsize=9)
    ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title('(b) Generalization Analysis', fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylim([65, 82])
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.legend(loc='upper right', fontsize=9)
    
    ax3 = fig.add_subplot(gs[1, 1])
    improvements = [methods_data[m]['val'] - baseline_random for m in methods]
    sorted_data = sorted(zip(methods, improvements, [methods_data[m]['color'] for m in methods]), 
                        key=lambda x: x[1], reverse=True)
    methods_sorted, improvements_sorted, colors_sorted = zip(*sorted_data)
    bars = ax3.barh(range(len(methods_sorted)), improvements_sorted, color=colors_sorted,
                    edgecolor='black', linewidth=1.2, alpha=0.85)
    for i, (bar, imp) in enumerate(zip(bars, improvements_sorted)):
        width = bar.get_width()
        ax3.text(width + 0.3, bar.get_y() + bar.get_height()/2.,
                f'+{imp:.1f}%', ha='left', va='center', fontsize=9, fontweight='bold')
    ax3.set_yticks(range(len(methods_sorted)))
    ax3.set_yticklabels([methods_data[m]['abbrev'] for m in methods_sorted], fontsize=9)
    ax3.set_xlabel('Improvement (%)', fontsize=11, fontweight='bold')
    ax3.set_title('(c) Improvement over Baseline', fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlim([0, 18])
    ax3.grid(True, alpha=0.3, linestyle='--', axis='x')
    
    ax4 = fig.add_subplot(gs[2, 0])
    seq_methods = ['Actor-Critic\n(LSTM)', 'REINFORCE\nw/ Baseline', 'PPO\n(LSTM)']
    ind_methods = ['PPO\nIndependent']
    seq_accs = [methods_data[m]['val'] for m in seq_methods]
    ind_accs = [methods_data[m]['val'] for m in ind_methods]
    seq_mean = np.mean(seq_accs)
    ind_mean = np.mean(ind_accs)
    bars = ax4.bar([0, 1], [seq_mean, ind_mean], width=0.6,
                   color=[COLORS['ac'], COLORS['ppo_ind']],
                   edgecolor='black', linewidth=1.5, alpha=0.7)
    for i, (bar, mean) in enumerate(zip(bars, [seq_mean, ind_mean])):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax4.set_xticks([0, 1])
    ax4.set_xticklabels(['Sequential\n(LSTM)', 'Independent'], fontsize=10)
    ax4.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax4.set_title('(d) Architecture Comparison', fontsize=12, fontweight='bold', pad=10)
    ax4.set_ylim([70, 78])
    ax4.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    ax5 = fig.add_subplot(gs[2, 1])
    test_methods = [m for m in methods if methods_data[m]['test'] is not None]
    test_accs = [methods_data[m]['test'] for m in test_methods]
    val_accs_test = [methods_data[m]['val'] for m in test_methods]
    x = np.arange(len(test_methods))
    width = 0.35
    ax5.bar(x - width/2, val_accs_test, width, label='Validation', 
            color=[methods_data[m]['color'] for m in test_methods],
            edgecolor='black', linewidth=1.2, alpha=0.8)
    ax5.bar(x + width/2, test_accs, width, label='Test', 
            color=COLORS['test'], edgecolor='black', linewidth=1.2, alpha=0.8)
    for i, (val, test) in enumerate(zip(val_accs_test, test_accs)):
        ax5.text(i - width/2, val + 0.5, f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        ax5.text(i + width/2, test + 0.5, f'{test:.1f}%', ha='center', va='bottom', fontsize=9)
    ax5.set_xticks(x)
    ax5.set_xticklabels([methods_data[m]['abbrev'] for m in test_methods], fontsize=9)
    ax5.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax5.set_title('(e) Test Set Performance', fontsize=12, fontweight='bold', pad=10)
    ax5.set_ylim([75, 81])
    ax5.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax5.legend(loc='upper left', fontsize=9)
    
    fig.suptitle('Comprehensive Results Summary: Sequential RL-based Retrieval\n(Qwen2.5-1.5B on GSM8K)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig('results/figures/figure5_summary.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('results/figures/figure5_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Saved Figure 5: Comprehensive Summary")
    plt.close()


def main():
    """Generate all visualizations"""
    import os
    os.makedirs('results/figures', exist_ok=True)
    
    print("Generating academic-quality visualizations for 1.5B model results...")
    print("=" * 60)
    
    create_figure_1_comparison()
    create_figure_2_train_val()
    create_figure_3_architecture()
    create_figure_4_improvement()
    create_figure_5_summary()
    
    print("=" * 60)
    print("✓ All visualizations saved to results/figures/")
    print("\nGenerated figures:")
    print("  - figure1_method_comparison.pdf/png")
    print("  - figure2_train_val.pdf/png")
    print("  - figure3_architecture.pdf/png")
    print("  - figure4_improvement.pdf/png")
    print("  - figure5_summary.pdf/png")


if __name__ == '__main__':
    main()

