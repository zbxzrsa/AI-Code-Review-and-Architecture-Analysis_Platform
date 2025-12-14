"""
Project Scan Performance Benchmark

Test performance differences before and after optimization,
verify 20% performance improvement target.
"""

import asyncio
import time
import statistics
from pathlib import Path
from ai_core.version_control.project_self_update_engine import ProjectSelfUpdateEngine


async def benchmark_scan_performance(project_root: str, iterations: int = 5):
    """
    Benchmark project scan performance
    
    Args:
        project_root: Project root directory
        iterations: Number of iterations
        
    Returns:
        Performance statistics
    """
    engine = ProjectSelfUpdateEngine(
        project_root=project_root,
        auto_apply=False,
        create_pr=False,
    )
    
    durations = []
    file_counts = []
    issue_counts = []
    
    print(f"Starting performance benchmark test ({iterations} iterations)...")
    
    for i in range(iterations):
        print(f"\nIteration {i+1}/{iterations}")
        
        start_time = time.time()
        result = await engine.scan_project()
        duration = time.time() - start_time
        
        durations.append(duration)
        file_counts.append(result.total_files_scanned)
        issue_counts.append(len(result.issues_found))
        
        print(f"  Duration: {duration:.2f}s")
        print(f"  Files: {result.total_files_scanned}")
        print(f"  Issues: {len(result.issues_found)}")
    
    # Calculate statistics
    stats = {
        "iterations": iterations,
        "avg_duration": statistics.mean(durations),
        "median_duration": statistics.median(durations),
        "min_duration": min(durations),
        "max_duration": max(durations),
        "std_duration": statistics.stdev(durations) if len(durations) > 1 else 0,
        "avg_files": statistics.mean(file_counts),
        "avg_issues": statistics.mean(issue_counts),
        "throughput_files_per_sec": statistics.mean(file_counts) / statistics.mean(durations),
    }
    
    print("\n" + "="*60)
    print("Performance Benchmark Results")
    print("="*60)
    print(f"Iterations: {stats['iterations']}")
    print(f"Average Duration: {stats['avg_duration']:.2f}s")
    print(f"Median Duration: {stats['median_duration']:.2f}s")
    print(f"Min Duration: {stats['min_duration']:.2f}s")
    print(f"Max Duration: {stats['max_duration']:.2f}s")
    print(f"Std Deviation: {stats['std_duration']:.2f}s")
    print(f"Average Files: {stats['avg_files']:.0f}")
    print(f"Average Issues: {stats['avg_issues']:.0f}")
    print(f"Throughput: {stats['throughput_files_per_sec']:.2f} files/sec")
    print("="*60)
    
    return stats


async def benchmark_concurrent_scan(project_root: str, concurrent_scans: int = 5):
    """
    测试并发扫描性能
    
    Args:
        project_root: 项目根目录
        concurrent_scans: 并发扫描数
        
    Returns:
        并发性能结果
    """
    print(f"\n测试并发扫描性能（{concurrent_scans}个并发）...")
    
    engine = ProjectSelfUpdateEngine(
        project_root=project_root,
        auto_apply=False,
        create_pr=False,
    )
    
    start_time = time.time()
    
    # 并发扫描
    tasks = [engine.scan_project() for _ in range(concurrent_scans)]
    results = await asyncio.gather(*tasks)
    
    total_duration = time.time() - start_time
    avg_duration = sum(r.scan_duration_seconds for r in results) / len(results)
    
    print(f"总耗时: {total_duration:.2f}秒")
    print(f"平均单次耗时: {avg_duration:.2f}秒")
    print(f"并发效率: {(avg_duration * concurrent_scans / total_duration * 100):.1f}%")
    
    return {
        "concurrent_scans": concurrent_scans,
        "total_duration": total_duration,
        "avg_single_duration": avg_duration,
        "efficiency": avg_duration * concurrent_scans / total_duration,
    }


async def benchmark_patch_generation(project_root: str):
    """
    测试补丁生成性能
    
    Args:
        project_root: 项目根目录
        
    Returns:
        补丁生成性能结果
    """
    print("\n测试补丁生成性能...")
    
    engine = ProjectSelfUpdateEngine(
        project_root=project_root,
        auto_apply=False,
        create_pr=False,
    )
    
    # 先扫描
    scan_result = await engine.scan_project()
    print(f"扫描完成，发现 {len(scan_result.issues_found)} 个问题")
    
    # 生成补丁
    start_time = time.time()
    patches = await engine.generate_improvement_patches(
        scan_result,
        max_patches=10,
    )
    duration = time.time() - start_time
    
    print(f"生成 {len(patches)} 个补丁，耗时: {duration:.2f}秒")
    if patches:
        print(f"平均每个补丁: {duration/len(patches):.2f}秒")
    
    return {
        "patches_generated": len(patches),
        "duration": duration,
        "avg_per_patch": duration / len(patches) if patches else 0,
    }


async def main():
    """主函数"""
    import sys
    
    project_root = sys.argv[1] if len(sys.argv) > 1 else "."
    
    print("="*60)
    print("项目自更新引擎性能基准测试")
    print("="*60)
    
    # 1. 基本扫描性能
    scan_stats = await benchmark_scan_performance(project_root, iterations=3)
    
    # 2. 并发扫描性能
    concurrent_stats = await benchmark_concurrent_scan(project_root, concurrent_scans=5)
    
    # 3. 补丁生成性能
    patch_stats = await benchmark_patch_generation(project_root)
    
    # 总结
    print("\n" + "="*60)
    print("性能测试总结")
    print("="*60)
    print(f"扫描性能: {scan_stats['avg_duration']:.2f}秒/次")
    print(f"并发效率: {concurrent_stats['efficiency']*100:.1f}%")
    print(f"补丁生成: {patch_stats['avg_per_patch']:.2f}秒/个")
    print("="*60)
    
    # 性能提升验证
    # 注意：这里需要对比优化前后的数据
    # 实际使用时应该保存基准数据，然后对比
    print("\n性能提升验证:")
    print("(需要对比优化前后的基准数据)")


if __name__ == "__main__":
    asyncio.run(main())

