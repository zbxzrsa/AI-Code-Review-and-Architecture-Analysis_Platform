#!/usr/bin/env python3
"""
Comprehensive Project Optimization Script

运行全面的项目优化分析，包括：
1. 代码质量分析
2. 性能瓶颈检测
3. 安全漏洞扫描
4. 测试覆盖率分析
5. 生成优化报告
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.shared.optimization.code_quality_analyzer import CodeQualityAnalyzer
from backend.shared.optimization.performance_optimizer import PerformanceProfiler


def run_code_quality_analysis(project_root: str) -> Dict[str, Any]:
    """运行代码质量分析"""
    print("=" * 80)
    print("Running Code Quality Analysis...")
    print("=" * 80)
    
    analyzer = CodeQualityAnalyzer(project_root)
    report = analyzer.analyze_project()
    
    print(f"\nCode Quality Report:")
    print(f"  Total Files: {report.total_files}")
    print(f"  Total Lines: {report.total_lines}")
    print(f"  Issues Found: {len(report.issues)}")
    print(f"  Duplicate Blocks: {len(report.duplicate_blocks)}")
    print(f"  Code Duplication Rate: {report.code_duplication_rate:.2f}%")
    print(f"  Maintainability Index: {report.maintainability_index:.2f}")
    
    # 按严重性分组问题
    critical = [i for i in report.issues if i.severity == "critical"]
    high = [i for i in report.issues if i.severity == "high"]
    medium = [i for i in report.issues if i.severity == "medium"]
    low = [i for i in report.issues if i.severity == "low"]
    
    print(f"\nIssues by Severity:")
    print(f"  Critical: {len(critical)}")
    print(f"  High: {len(high)}")
    print(f"  Medium: {len(medium)}")
    print(f"  Low: {len(low)}")
    
    if report.recommendations:
        print(f"\nRecommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")
    
    return {
        "type": "code_quality",
        "timestamp": report.timestamp.isoformat(),
        "metrics": {
            "total_files": report.total_files,
            "total_lines": report.total_lines,
            "issues_count": len(report.issues),
            "duplicate_blocks": len(report.duplicate_blocks),
            "duplication_rate": report.code_duplication_rate,
            "maintainability_index": report.maintainability_index
        },
        "issues_by_severity": {
            "critical": len(critical),
            "high": len(high),
            "medium": len(medium),
            "low": len(low)
        },
        "recommendations": report.recommendations
    }


def run_performance_analysis() -> Dict[str, Any]:
    """运行性能分析"""
    print("\n" + "=" * 80)
    print("Running Performance Analysis...")
    print("=" * 80)
    
    profiler = PerformanceProfiler()
    report = profiler.generate_report()
    
    print(f"\nPerformance Report:")
    print(f"  Total Measurements: {report.summary.get('total_measurements', 0)}")
    print(f"  Bottlenecks Found: {len(report.bottlenecks)}")
    
    if report.summary:
        print(f"  Avg Response Time: {report.summary.get('avg_response_time_ms', 0):.2f}ms")
        print(f"  Max Response Time: {report.summary.get('max_response_time_ms', 0):.2f}ms")
        print(f"  Avg Query Time: {report.summary.get('avg_query_time_ms', 0):.2f}ms")
    
    if report.bottlenecks:
        print(f"\nTop Bottlenecks:")
        for i, bottleneck in enumerate(report.bottlenecks[:5], 1):
            print(f"  {i}. {bottleneck.location}")
            print(f"     Impact: {bottleneck.impact}")
            print(f"     Current: {bottleneck.current_value:.2f}{bottleneck.metric_type.value}")
            print(f"     Recommendation: {bottleneck.recommendation}")
    
    return {
        "type": "performance",
        "timestamp": report.timestamp.isoformat(),
        "summary": report.summary,
        "bottlenecks_count": len(report.bottlenecks),
        "recommendations": report.recommendations
    }


def generate_optimization_report(results: list, output_file: str) -> None:
    """生成优化报告"""
    report = {
        "report_id": f"opt_report_{datetime.now().isoformat()}",
        "generated_at": datetime.now().isoformat(),
        "analyses": results,
        "summary": {
            "total_analyses": len(results),
            "critical_issues": sum(
                r.get("issues_by_severity", {}).get("critical", 0)
                for r in results if "issues_by_severity" in r
            ),
            "bottlenecks": sum(
                r.get("bottlenecks_count", 0)
                for r in results if "bottlenecks_count" in r
            )
        }
    }
    
    # 保存报告
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n" + "=" * 80)
    print(f"Optimization Report saved to: {output_path}")
    print("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Run comprehensive project optimization analysis"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(project_root),
        help="Project root directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/optimization_report.json",
        help="Output report file path"
    )
    parser.add_argument(
        "--skip-code-quality",
        action="store_true",
        help="Skip code quality analysis"
    )
    parser.add_argument(
        "--skip-performance",
        action="store_true",
        help="Skip performance analysis"
    )
    
    args = parser.parse_args()
    
    results = []
    
    # 1. 代码质量分析
    if not args.skip_code_quality:
        try:
            result = run_code_quality_analysis(args.project_root)
            results.append(result)
        except Exception as e:
            print(f"Error in code quality analysis: {e}", file=sys.stderr)
    
    # 2. 性能分析
    if not args.skip_performance:
        try:
            result = run_performance_analysis()
            results.append(result)
        except Exception as e:
            print(f"Error in performance analysis: {e}", file=sys.stderr)
    
    # 3. 生成报告
    if results:
        generate_optimization_report(results, args.output)
    else:
        print("No analyses completed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

