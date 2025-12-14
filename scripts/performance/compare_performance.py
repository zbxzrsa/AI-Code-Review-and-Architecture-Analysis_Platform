"""
Performance Comparison Script

Compare performance metrics before and after optimization,
verify 20% performance improvement target.
"""

import json
import sys
from pathlib import Path


def load_baseline(file_path: str) -> dict:
    """Load baseline performance data"""
    with open(file_path, 'r') as f:
        return json.load(f)


def compare_performance(baseline: dict, current: dict) -> dict:
    """
    Compare performance data
    
    Returns:
        Comparison results, including improvement percentage
    """
    results = {}
    
    # Compare scan performance
    if 'avg_duration' in baseline and 'avg_duration' in current:
        baseline_duration = baseline['avg_duration']
        current_duration = current['avg_duration']
        improvement = ((baseline_duration - current_duration) / baseline_duration) * 100
        results['scan_duration_improvement'] = improvement
        results['scan_duration_baseline'] = baseline_duration
        results['scan_duration_current'] = current_duration
    
    # Compare throughput
    if 'throughput_files_per_sec' in baseline and 'throughput_files_per_sec' in current:
        baseline_throughput = baseline['throughput_files_per_sec']
        current_throughput = current['throughput_files_per_sec']
        improvement = ((current_throughput - baseline_throughput) / baseline_throughput) * 100
        results['throughput_improvement'] = improvement
        results['throughput_baseline'] = baseline_throughput
        results['throughput_current'] = current_throughput
    
    # Compare concurrency efficiency
    if 'efficiency' in baseline and 'efficiency' in current:
        baseline_efficiency = baseline['efficiency']
        current_efficiency = current['efficiency']
        improvement = ((current_efficiency - baseline_efficiency) / baseline_efficiency) * 100
        results['efficiency_improvement'] = improvement
        results['efficiency_baseline'] = baseline_efficiency
        results['efficiency_current'] = current_efficiency
    
    return results


def print_comparison(results: dict):
    """Print comparison results"""
    print("="*60)
    print("Performance Comparison Results")
    print("="*60)
    
    if 'scan_duration_improvement' in results:
        improvement = results['scan_duration_improvement']
        print(f"\nScan Performance:")
        print(f"  Baseline: {results['scan_duration_baseline']:.2f}s")
        print(f"  Current: {results['scan_duration_current']:.2f}s")
        print(f"  Improvement: {improvement:.1f}%")
        if improvement >= 20:
            print(f"  ✅ Achieved 20% performance improvement target!")
        else:
            print(f"  ⚠️  Did not reach 20% performance improvement target")
    
    if 'throughput_improvement' in results:
        improvement = results['throughput_improvement']
        print(f"\nThroughput:")
        print(f"  Baseline: {results['throughput_baseline']:.2f} files/sec")
        print(f"  Current: {results['throughput_current']:.2f} files/sec")
        print(f"  Improvement: {improvement:.1f}%")
    
    if 'efficiency_improvement' in results:
        improvement = results['efficiency_improvement']
        print(f"\nConcurrency Efficiency:")
        print(f"  Baseline: {results['efficiency_baseline']*100:.1f}%")
        print(f"  Current: {results['efficiency_current']*100:.1f}%")
        print(f"  Improvement: {improvement:.1f}%")
    
    print("="*60)


def main():
    """Main function"""
    if len(sys.argv) < 3:
        print("Usage: python compare_performance.py <baseline.json> <current.json>")
        sys.exit(1)
    
    baseline_file = sys.argv[1]
    current_file = sys.argv[2]
    
    baseline = load_baseline(baseline_file)
    current = load_baseline(current_file)
    
    results = compare_performance(baseline, current)
    print_comparison(results)
    
    # Save comparison results
    output_file = Path("performance_comparison.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nComparison results saved to: {output_file}")


if __name__ == "__main__":
    main()

