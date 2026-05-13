import subprocess
import sys
import os
import statistics
import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class Result:
    func_id: int
    steps: int
    total_points: int
    value: float
    abs_error: float
    rel_error: float
    time_ms: int
    exact_value: float

class OpenCLBenchmark:
    def __init__(self, binary_path: str = "./integrate_opencl"):
        self.binary_path = binary_path
        self.configs = {
            1: "configs/func1.cfg",
            2: "configs/func2.cfg",
            3: "configs/func3.cfg"
        }
        self.exact_values = {
            1: 4545447.652,
            2: 857208.2414,
            3: -1.604646665
        }

    def run_single(self, func_id: int, steps: int, runs: int = 3) -> Result:
        """Run OpenCL integration multiple times and return average stats"""
        config_file = self.configs[func_id]

        times = []
        values = []
        abs_errors = []

        for _ in range(runs):
            try:
                result = subprocess.run(
                    [self.binary_path, str(func_id), config_file, str(steps)],
                    capture_output=True, text=True, timeout=120
                )

                if result.returncode != 0:
                    print(f"  Warning: return code {result.returncode}")
                    continue

                output_lines = result.stdout.strip().split('\n')
                if len(output_lines) >= 4:
                    values.append(float(output_lines[0]))
                    abs_errors.append(float(output_lines[1]))
                    times.append(int(output_lines[3]))

            except subprocess.TimeoutExpired:
                print(f"  Timeout for func={func_id}, steps={steps}")
            except Exception as e:
                print(f"  Error: {e}")

        if not times:
            raise RuntimeError(f"No successful runs for func={func_id}, steps={steps}")

        return Result(
            func_id=func_id,
            steps=steps,
            total_points=steps * steps,
            value=statistics.mean(values),
            abs_error=statistics.mean(abs_errors),
            rel_error=statistics.mean(abs_errors) / abs(self.exact_values[func_id]),
            time_ms=statistics.mean(times),
            exact_value=self.exact_values[func_id]
        )

    def benchmark_scale(self, func_ids: list, step_sizes: list, runs: int = 3) -> list:
        """Run benchmarks across different step sizes"""
        results = []
        for func_id in func_ids:
            for steps in step_sizes:
                print(f"Benchmarking Function {func_id}, Steps={steps}...")
                try:
                    result = self.run_single(func_id, steps, runs)
                    results.append(result)
                    print(f"  Time: {result.time_ms}ms, Error: {result.abs_error:.6e}")
                except Exception as e:
                    print(f"  Failed: {e}")
        return results

def save_results(results: list, filename: str = "opencl_benchmark_results.json"):
    """Save results to JSON file"""
    data = {
        "timestamp": datetime.now().isoformat(),
        "results": [asdict(r) for r in results]
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {filename}")

def print_table(results: list):
    """Print formatted results table"""
    print("\n" + "=" * 110)
    print(f"{'Func':6} {'Steps':8} {'Points':12} {'Time(ms)':10} {'Abs Error':15} {'Rel Error':15}")
    print("=" * 110)

    for r in results:
        print(f"F{r.func_id:<5} {r.steps:<8} {r.total_points:<12} {r.time_ms:<10.0f} {r.abs_error:<15.6e} {r.rel_error:<15.6e}")

    print("=" * 110)

def main():
    parser = argparse.ArgumentParser(description="OpenCL Integration Benchmark")
    parser.add_argument("--func", type=int, nargs="+", default=[1, 2, 3], help="Function IDs to test")
    parser.add_argument("--steps", type=int, nargs="+", default=[100, 500, 1000, 2000, 4000], help="Step sizes")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per configuration")
    parser.add_argument("--binary", type=str, default="./integrate_opencl", help="Path to OpenCL binary")
    parser.add_argument("--output", type=str, default="opencl_benchmark_results.json", help="Output JSON file")

    args = parser.parse_args()

    # Check if binary exists
    if not os.path.exists(args.binary):
        print(f"Error: Binary not found at {args.binary}")
        print("Please compile first: g++ -std=c++17 -o integrate_opencl src/opencl/integrate_opencl.cpp src/config_parser/config_parser.cpp -lOpenCL -I./src -I./src/config_parser -O3")
        sys.exit(1)

    # Run benchmarks
    benchmark = OpenCLBenchmark(args.binary)
    results = benchmark.benchmark_scale(args.func, args.steps, args.runs)

    # Print results
    print_table(results)

    # Save results
    save_results(results, args.output)

    print(f"\nTo plot results, run: python3 scripts/plot_results.py --opencl {args.output} --cuda cuda_benchmark_results.json")

if __name__ == "__main__":
    main()