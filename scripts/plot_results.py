import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import os

FUNC_NAMES = {1: "De Jong", 2: "Ackley", 3: "Langermann"}
FUNC_COLORS = {1: "#2ecc71", 2: "#3498db", 3: "#e74c3c"}
CUDA_COLOR = "#2ecc71"
OPENCL_COLOR = "#e74c3c"

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
    if not os.path.exists(json_file):
        return None
    with open(json_file, 'r') as f:
        return json.load(f)

def plot_cuda_results(results: dict, output_dir: str = "."):
    """Plot CUDA execution time vs number of points"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('CUDA Integration Performance: Time vs Points', fontsize=16, fontweight='bold')

    for idx, func_id in enumerate([1, 2, 3]):
        ax = axes[idx]
        func_results = [r for r in results['results'] if r['func_id'] == func_id]

        if not func_results:
            ax.text(0.5, 0.5, f'No data for {FUNC_NAMES[func_id]}',
                    transform=ax.transAxes, ha='center', va='center')
            continue

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

        for p, t in zip(points, times):
            ax.annotate(f'{t:.0f}ms', (p, t), textcoords="offset points",
                        xytext=(5, 5), fontsize=9, alpha=0.7)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'cuda_time_vs_points.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def plot_opencl_results(results: dict, output_dir: str = "."):
    """Plot OpenCL execution time vs number of points"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('OpenCL Integration Performance: Time vs Points', fontsize=16, fontweight='bold')

    for idx, func_id in enumerate([1, 2, 3]):
        ax = axes[idx]
        func_results = [r for r in results['results'] if r['func_id'] == func_id]

        if not func_results:
            ax.text(0.5, 0.5, f'No data for {FUNC_NAMES[func_id]}',
                    transform=ax.transAxes, ha='center', va='center')
            continue

        func_results.sort(key=lambda x: x['total_points'])
        points = [r['total_points'] for r in func_results]
        times = [r['time_ms'] for r in func_results]

        ax.plot(points, times, 's-', linewidth=2, markersize=8, color=FUNC_COLORS[func_id])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Number of Points')
        ax.set_ylabel('Time (ms)')
        ax.set_title(FUNC_NAMES[func_id])
        ax.grid(True, alpha=0.3, linestyle='--')

        for p, t in zip(points, times):
            ax.annotate(f'{t:.0f}ms', (p, t), textcoords="offset points",
                        xytext=(5, 5), fontsize=9, alpha=0.7)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'opencl_time_vs_points.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def plot_cuda_vs_opencl(cuda_results: dict, opencl_results: dict, output_dir: str = "."):
    """Plot CUDA vs OpenCL comparison for each function"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('CUDA vs OpenCL Performance Comparison', fontsize=16, fontweight='bold')

    for idx, func_id in enumerate([1, 2, 3]):
        ax = axes[idx]

        cuda_func = [r for r in cuda_results['results'] if r['func_id'] == func_id]
        opencl_func = [r for r in opencl_results['results'] if r['func_id'] == func_id]

        if not cuda_func or not opencl_func:
            ax.text(0.5, 0.5, f'No data for {FUNC_NAMES[func_id]}',
                    transform=ax.transAxes, ha='center', va='center')
            continue

        cuda_func.sort(key=lambda x: x['total_points'])
        opencl_func.sort(key=lambda x: x['total_points'])

        points = [r['total_points'] for r in cuda_func]
        cuda_times = [r['time_ms'] for r in cuda_func]
        opencl_times = [r['time_ms'] for r in opencl_func]

        ax.plot(points, cuda_times, 'o-', linewidth=2, markersize=8,
                label='CUDA', color=CUDA_COLOR)
        ax.plot(points, opencl_times, 's-', linewidth=2, markersize=8,
                label='OpenCL', color=OPENCL_COLOR)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Number of Points')
        ax.set_ylabel('Time (ms)')
        ax.set_title(FUNC_NAMES[func_id])
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend()

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'cuda_vs_opencl_comparison.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def plot_speedup(cuda_results: dict, opencl_results: dict, output_dir: str = "."):
    """Plot CUDA speedup over OpenCL"""
    fig, ax = plt.subplots(figsize=(10, 6))

    for func_id in [1, 2, 3]:
        cuda_func = [r for r in cuda_results['results'] if r['func_id'] == func_id]
        opencl_func = [r for r in opencl_results['results'] if r['func_id'] == func_id]

        if not cuda_func or not opencl_func:
            continue

        cuda_func.sort(key=lambda x: x['total_points'])
        opencl_func.sort(key=lambda x: x['total_points'])

        points = [r['total_points'] for r in cuda_func]
        speedup = [opencl_func[i]['time_ms'] / cuda_func[i]['time_ms']
                   for i in range(len(cuda_func))]

        ax.plot(points, speedup, 'o-', linewidth=2, markersize=8,
                label=FUNC_NAMES[func_id], color=FUNC_COLORS[func_id])

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Equal performance')
    ax.set_xscale('log')
    ax.set_xlabel('Number of Points')
    ax.set_ylabel('Speedup (OpenCL / CUDA)')
    ax.set_title('CUDA Speedup over OpenCL (Higher is better for CUDA)', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend()

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'cuda_speedup.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def plot_error_comparison(cuda_results: dict, opencl_results: dict, output_dir: str = "."):
    """Plot error comparison between CUDA and OpenCL"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Numerical Error Comparison: CUDA vs OpenCL', fontsize=16, fontweight='bold')

    for idx, func_id in enumerate([1, 2, 3]):
        ax = axes[idx]

        cuda_func = [r for r in cuda_results['results'] if r['func_id'] == func_id]
        opencl_func = [r for r in opencl_results['results'] if r['func_id'] == func_id]

        if not cuda_func or not opencl_func:
            ax.text(0.5, 0.5, f'No data for {FUNC_NAMES[func_id]}',
                    transform=ax.transAxes, ha='center', va='center')
            continue

        cuda_func.sort(key=lambda x: x['total_points'])
        opencl_func.sort(key=lambda x: x['total_points'])

        points = [r['total_points'] for r in cuda_func]
        cuda_errors = [r['abs_error'] for r in cuda_func]
        opencl_errors = [r['abs_error'] for r in opencl_func]

        ax.loglog(points, cuda_errors, 'o-', linewidth=2, markersize=8,
                  label='CUDA', color=CUDA_COLOR)
        ax.loglog(points, opencl_errors, 's-', linewidth=2, markersize=8,
                  label='OpenCL', color=OPENCL_COLOR)

        ax.set_xlabel('Number of Points')
        ax.set_ylabel('Absolute Error')
        ax.set_title(FUNC_NAMES[func_id])
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend()

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'error_comparison.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def create_summary_table(cuda_results: dict, opencl_results: dict, output_dir: str = "."):
    """Create a summary table comparing CUDA and OpenCL"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    table_data = []
    headers = ['Function', 'Points', 'CUDA (ms)', 'OpenCL (ms)', 'Speedup', 'CUDA Error', 'OpenCL Error']

    for func_id in [1, 2, 3]:
        cuda_func = [r for r in cuda_results['results'] if r['func_id'] == func_id and r['steps'] == 1000]
        opencl_func = [r for r in opencl_results['results'] if r['func_id'] == func_id and r['steps'] == 1000]

        if cuda_func and opencl_func:
            cuda_time = cuda_func[0]['time_ms']
            opencl_time = opencl_func[0]['time_ms']
            speedup = opencl_time / cuda_time

            table_data.append([
                FUNC_NAMES[func_id],
                '1,000,000',
                f'{cuda_time:.0f}',
                f'{opencl_time:.0f}',
                f'{speedup:.2f}x',
                f'{cuda_func[0]["abs_error"]:.2e}',
                f'{opencl_func[0]["abs_error"]:.2e}'
            ])

    if table_data:
        table = ax.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center',
                         colWidths=[0.12, 0.10, 0.10, 0.10, 0.10, 0.18, 0.18])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Color header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Color speedup column
        for i, row in enumerate(table_data, start=1):
            speedup = float(row[4].replace('x', ''))
            if speedup > 1.3:
                table[(i, 4)].set_facecolor('#2ecc71')
                table[(i, 4)].set_text_props(weight='bold')
            elif speedup > 1.0:
                table[(i, 4)].set_facecolor('#f39c12')
            else:
                table[(i, 4)].set_facecolor('#e74c3c')

    plt.title('CUDA vs OpenCL Performance Summary (1M points)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'cuda_vs_opencl_summary.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot CUDA/OpenCL performance results")
    parser.add_argument("--cuda", type=str, default="cuda_benchmark_results.json",
                        help="CUDA benchmark results JSON file")
    parser.add_argument("--opencl", type=str, default="opencl_benchmark_results.json",
                        help="OpenCL benchmark results JSON file")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory for plots")

    args = parser.parse_args()

    setup_plot_style()

    # Load results
    cuda_results = load_results(args.cuda)
    opencl_results = load_results(args.opencl)

    if cuda_results:
        print(f"Loaded CUDA results from {args.cuda} ({len(cuda_results['results'])} runs)")
        plot_cuda_results(cuda_results, args.output_dir)
    else:
        print(f"CUDA results file not found: {args.cuda}")

    if opencl_results:
        print(f"Loaded OpenCL results from {args.opencl} ({len(opencl_results['results'])} runs)")
        plot_opencl_results(opencl_results, args.output_dir)
    else:
        print(f"OpenCL results file not found: {args.opencl}")

    # Comparison plots (if both available)
    if cuda_results and opencl_results:
        print("\nGenerating comparison plots...")
        plot_cuda_vs_opencl(cuda_results, opencl_results, args.output_dir)
        plot_speedup(cuda_results, opencl_results, args.output_dir)
        plot_error_comparison(cuda_results, opencl_results, args.output_dir)
        create_summary_table(cuda_results, opencl_results, args.output_dir)

    print("\nAll plots generated successfully!")

if __name__ == "__main__":
    main()