"""
Mutation Testing Configuration

Mutation testing helps find gaps in test coverage by introducing
small changes (mutations) to the code and checking if tests catch them.

Tools:
- mutmut: Primary mutation testing tool for Python
- pytest-mutmut: Integration with pytest

Usage:
    # Run mutation testing
    mutmut run --paths-to-mutate=backend/shared
    
    # View results
    mutmut results
    
    # Generate HTML report
    mutmut html
"""

import os
from pathlib import Path
from typing import Dict, List, Set

# ============================================================================
# Mutation Testing Configuration
# ============================================================================

MUTATION_CONFIG = {
    # Target directories for mutation
    "paths_to_mutate": [
        "backend/shared/auth",
        "backend/shared/security",
        "backend/shared/middleware",
        "backend/services",
        "ai_core",
        "services",
    ],
    
    # Files to skip
    "skip_files": [
        "**/test_*.py",
        "**/*_test.py",
        "**/conftest.py",
        "**/migrations/**",
        "**/__pycache__/**",
        "**/fixtures/**",
    ],
    
    # Functions to skip (too simple or known issues)
    "skip_functions": [
        "__repr__",
        "__str__",
        "__init__",  # Only if trivial
    ],
    
    # Mutation operators to use
    "operators": [
        "AOR",  # Arithmetic Operator Replacement
        "ASR",  # Assignment Operator Replacement
        "BCR",  # Boolean Comparison Replacement
        "COI",  # Conditional Operator Insertion
        "ROR",  # Relational Operator Replacement
        "SIR",  # Statement Insertion Replacement
        "SDR",  # Statement Deletion Replacement
    ],
    
    # Timeout per mutation (seconds)
    "timeout": 30,
    
    # Parallel execution
    "parallel": True,
    "workers": 4,
    
    # Minimum mutation score (percentage of killed mutants)
    "min_score": 80.0,
}


# ============================================================================
# Critical Modules for Mutation Testing
# ============================================================================

# These modules MUST have high mutation scores
CRITICAL_MODULES = {
    "backend/shared/security/secure_auth.py": 90.0,
    "backend/shared/middleware/rate_limiter.py": 85.0,
    "backend/shared/auth/password.py": 95.0,
    "backend/shared/utils/validators.py": 90.0,
    "ai_core/distributed_vc/rollback.py": 85.0,
}


# ============================================================================
# Mutation Test Runner
# ============================================================================

class MutationTestRunner:
    """
    Runs mutation testing and generates reports.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or MUTATION_CONFIG
        self.results: Dict[str, Dict] = {}
    
    def generate_mutmut_config(self) -> str:
        """Generate mutmut configuration file content."""
        paths = " ".join(self.config["paths_to_mutate"])
        
        return f"""
[mutmut]
paths_to_mutate = {paths}
backup = False
runner = python -m pytest -x --tb=no -q
tests_dir = tests/
dict_synonyms = Struct, NamedStruct

# Timeout per mutation
timeout = {self.config['timeout']}

# Parallel execution
parallel = {str(self.config['parallel']).lower()}
"""
    
    def run(self, target_path: str = None) -> Dict:
        """
        Run mutation testing on target path.
        
        Args:
            target_path: Specific path to test, or None for all
            
        Returns:
            Mutation testing results
        """
        import subprocess
        
        cmd = ["mutmut", "run"]
        
        if target_path:
            cmd.extend(["--paths-to-mutate", target_path])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        
        return self._parse_results(result.stdout)
    
    def _parse_results(self, output: str) -> Dict:
        """Parse mutmut output."""
        results = {
            "total_mutants": 0,
            "killed": 0,
            "survived": 0,
            "timeout": 0,
            "suspicious": 0,
            "score": 0.0,
        }
        
        # Parse output (simplified)
        lines = output.split("\n")
        for line in lines:
            if "killed" in line.lower():
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.isdigit():
                        results["killed"] = int(part)
                        break
        
        if results["total_mutants"] > 0:
            results["score"] = (results["killed"] / results["total_mutants"]) * 100
        
        return results
    
    def check_critical_modules(self) -> Dict[str, bool]:
        """
        Check if critical modules meet mutation score requirements.
        
        Returns:
            Dict mapping module to pass/fail status
        """
        status = {}
        
        for module, min_score in CRITICAL_MODULES.items():
            results = self.run(module)
            passed = results.get("score", 0) >= min_score
            status[module] = {
                "passed": passed,
                "score": results.get("score", 0),
                "required": min_score,
            }
        
        return status
    
    def generate_report(self) -> str:
        """Generate mutation testing report."""
        report = ["# Mutation Testing Report", ""]
        report.append(f"Generated: {__import__('datetime').datetime.now().isoformat()}")
        report.append("")
        
        # Summary
        report.append("## Summary")
        report.append(f"- Total Mutants: {self.results.get('total_mutants', 0)}")
        report.append(f"- Killed: {self.results.get('killed', 0)}")
        report.append(f"- Survived: {self.results.get('survived', 0)}")
        report.append(f"- Mutation Score: {self.results.get('score', 0):.2f}%")
        report.append("")
        
        # Critical modules
        report.append("## Critical Modules")
        for module, status in self.check_critical_modules().items():
            emoji = "✅" if status["passed"] else "❌"
            report.append(f"- {emoji} `{module}`: {status['score']:.2f}% (required: {status['required']}%)")
        
        return "\n".join(report)


# ============================================================================
# Pytest Plugin for Mutation Testing
# ============================================================================

def pytest_addoption(parser):
    """Add mutation testing options to pytest."""
    group = parser.getgroup("mutation")
    group.addoption(
        "--run-mutation",
        action="store_true",
        default=False,
        help="Run mutation testing after regular tests",
    )
    group.addoption(
        "--mutation-target",
        action="store",
        default=None,
        help="Specific path to run mutation testing on",
    )
    group.addoption(
        "--mutation-score",
        action="store",
        type=float,
        default=80.0,
        help="Minimum required mutation score",
    )


def pytest_sessionfinish(session, exitstatus):
    """Run mutation testing after test session if requested."""
    if session.config.getoption("--run-mutation"):
        runner = MutationTestRunner()
        target = session.config.getoption("--mutation-target")
        min_score = session.config.getoption("--mutation-score")
        
        print("\n" + "=" * 60)
        print("Running Mutation Testing...")
        print("=" * 60)
        
        results = runner.run(target)
        
        print(f"\nMutation Score: {results['score']:.2f}%")
        print(f"Killed: {results['killed']}/{results['total_mutants']}")
        
        if results['score'] < min_score:
            print(f"\n❌ Mutation score {results['score']:.2f}% is below required {min_score}%")
            session.exitstatus = 1
        else:
            print(f"\n✅ Mutation score meets requirement")


# ============================================================================
# Utility Functions
# ============================================================================

def get_uncovered_mutants(report_path: str = "html") -> List[Dict]:
    """
    Get list of survived mutants for investigation.
    
    Args:
        report_path: Path to mutmut HTML report
        
    Returns:
        List of survived mutant details
    """
    import subprocess
    
    result = subprocess.run(
        ["mutmut", "results"],
        capture_output=True,
        text=True,
    )
    
    survived = []
    lines = result.stdout.split("\n")
    
    for line in lines:
        if "survived" in line.lower():
            # Parse mutant ID
            parts = line.split()
            if parts:
                mutant_id = parts[0]
                survived.append({
                    "id": mutant_id,
                    "line": line,
                })
    
    return survived


def prioritize_test_improvements(survived_mutants: List[Dict]) -> List[str]:
    """
    Suggest test improvements based on survived mutants.
    
    Args:
        survived_mutants: List of survived mutants
        
    Returns:
        List of suggestions
    """
    suggestions = []
    
    # Group by file
    by_file: Dict[str, List] = {}
    for mutant in survived_mutants:
        # Extract file from mutant ID (format: file.py:line)
        if ":" in mutant.get("id", ""):
            file = mutant["id"].split(":")[0]
            by_file.setdefault(file, []).append(mutant)
    
    # Generate suggestions
    for file, mutants in by_file.items():
        suggestions.append(
            f"Add more tests for `{file}` - {len(mutants)} mutants survived"
        )
    
    return suggestions
