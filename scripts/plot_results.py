import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import os

FUNC_NAMES = {1: "De Jong", 2: "Ackley", 3: "Langermann"}
FUNC_COLORS = {1: "#2ecc71", 2: "#3498db", 3: "#e74c3c"}

def setup_plot_style():
    """Setup matplotlib style for publication-ready plots"""
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.figsize'] = (12, 8)

def load_results(json_file: str) -> dict:
    """Load results from JSON file"""
    with open(json_file, 'r') as f:
        return json.load(f)

def plot_time_vs_points(results: dict, output_dir: str = "."):
    """Plot execution time vs number of points for each function"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('CUDA Integration Performance: Time vs Points', fontsize=16, fontweight='bold')

    for idx, func_id in enumerate([1, 2, 3]):
        ax = axes[idx]
        func_results = [r for r in results['results'] if r['func_id'] == func_id]

        if not func_results:
            ax.text(0.5, 0.5, f'No data for {FUNC_NAMES[func_id]}',
                    transform=ax.transAxes, ha='center', va='center')
            continue

        # Sort by points
        func_results.sort(key=lambda x: x['total_points'])
        points = [r['total_points'] for r in func_results]
        times = [r['time_ms'] for r in func_results]

        ax.plot(points, times, 'o-', linewidth=2, markersize=8, color=FUNC_COLORS[func_id])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Number of Points')
        ax.set_ylabel('Time (ms)')
        ax.set_title(FUNC_NAMES[func_id])
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add annotations
        for p, t in zip(points, times):
            ax.annotate(f'{t:.0f}ms', (p, t), textcoords="offset points",
                        xytext=(5, 5), fontsize=9, alpha=0.7)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'cuda_time_vs_points.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def plot_error_convergence(results: dict, output_dir: str = "."):
    """Plot error convergence with increasing points"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Numerical Error Convergence', fontsize=16, fontweight='bold')

    for idx, func_id in enumerate([1, 2, 3]):
        ax = axes[idx]
        func_results = [r for r in results['results'] if r['func_id'] == func_id]

        if not func_results:
            ax.text(0.5, 0.5, f'No data for {FUNC_NAMES[func_id]}',
                    transform=ax.transAxes, ha='center', va='center')
            continue

        func_results.sort(key=lambda x: x['total_points'])
        points = [r['total_points'] for r in func_results]
        errors = [r['abs_error'] for r in func_results]

        ax.loglog(points, errors, 'o-', linewidth=2, markersize=8, color=FUNC_COLORS[func_id])
        ax.set_xlabel('Number of Points')
        ax.set_ylabel('Absolute Error')
        ax.set_title(FUNC_NAMES[func_id])
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add reference line O(1/N)
        if len(points) > 1:
            ref_error = errors[0] * (points[0] / np.array(points))
            ax.loglog(points, ref_error, 'k--', alpha=0.5, label='O(1/N)')
            ax.legend(fontsize=9)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'cuda_error_convergence.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def plot_scalability(results: dict, output_dir: str = "."):
    """Plot scalability (throughput) with problem size"""
    fig, ax = plt.subplots(figsize=(10, 6))

    for func_id in [1, 2, 3]:
        func_results = [r for r in results['results'] if r['func_id'] == func_id]

        if not func_results:
            continue

        func_results.sort(key=lambda x: x['total_points'])
        points = [r['total_points'] for r in func_results]
        # Throughput = points per second
        throughput = [r['total_points'] / (r['time_ms'] / 1000) for r in func_results]

        ax.plot(points, throughput, 'o-', linewidth=2, markersize=8,
                label=FUNC_NAMES[func_id], color=FUNC_COLORS[func_id])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Points')
    ax.set_ylabel('Throughput (points/second)')
    ax.set_title('CUDA Throughput Scaling')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend()

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'cuda_throughput.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def plot_blocks_vs_time(results: dict, output_dir: str = "."):
    """Plot time vs number of CUDA blocks"""
    fig, ax = plt.subplots(figsize=(10, 6))

    for func_id in [1, 2, 3]:
        func_results = [r for r in results['results'] if r['func_id'] == func_id]

        if not func_results or func_results[0]['blocks'] == 0:
            continue

        func_results.sort(key=lambda x: x['blocks'])
        blocks = [r['blocks'] for r in func_results]
        times = [r['time_ms'] for r in func_results]

        ax.plot(blocks, times, 'o-', linewidth=2, markersize=8,
                label=FUNC_NAMES[func_id], color=FUNC_COLORS[func_id])

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of CUDA Blocks')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Performance vs GPU Occupancy')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend()

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'cuda_blocks_vs_time.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def plot_summary_table(results: dict, output_dir: str = "."):
    """Create a summary table plot"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    table_data = []
    headers = ['Function', 'Steps', 'Points', 'Blocks', 'Time (ms)', 'Abs Error']

    for func_id in [1, 2, 3]:
        func_results = [r for r in results['results'] if r['func_id'] == func_id]
        for r in sorted(func_results, key=lambda x: x['steps']):
            table_data.append([
                FUNC_NAMES[func_id],
                r['steps'],
                f"{r['total_points']:,}",
                r['blocks'],
                f"{r['time_ms']:.0f}",
                f"{r['abs_error']:.2e}"
            ])

    if table_data:
        table = ax.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center',
                         colWidths=[0.12, 0.08, 0.15, 0.10, 0.10, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Color header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

    plt.title('CUDA Integration Benchmark Summary', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'cuda_summary_table.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot CUDA performance results")
    parser.add_argument("--input", type=str, default="cuda_benchmark_results.json",
                        help="Input JSON file with benchmark results")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory for plots")
    parser.add_argument("--all", action="store_true", default=True,
                        help="Generate all plots")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found")
        print("Please run run_cuda.py first to generate benchmark data")
        sys.exit(1)

    setup_plot_style()
    results = load_results(args.input)

    print(f"Loaded results from {args.input}")
    print(f"Timestamp: {results['timestamp']}")
    print(f"Total runs: {len(results['results'])}")

    # Generate plots
    plot_time_vs_points(results, args.output_dir)
    plot_error_convergence(results, args.output_dir)
    plot_scalability(results, args.output_dir)
    plot_blocks_vs_time(results, args.output_dir)
    plot_summary_table(results, args.output_dir)

    print("\nAll plots generated successfully!")

if __name__ == "__main__":
    main()